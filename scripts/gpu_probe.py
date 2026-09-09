"""
Find out exactly where the GPU stack dies, when it dies without saying anything.

A Python-level failure prints a traceback. A *process-level* one does not: if
the HIP runtime hits an access violation or cannot load a DLL while it is
enumerating devices, Windows kills the interpreter outright. There is no
exception to catch and nothing on stdout, so

    python -c "import torch;print(torch.cuda.device_count())"

prints nothing at all, and any diagnostic that calls torch.cuda in its own
process dies at the same line and reports nothing.

So every step here runs in a fresh subprocess. A step that crashes takes only
its own child with it, and the parent reports the exit code -- which is the
actual diagnosis, because Windows encodes the reason in it.

    python scripts/gpu_probe.py
    python scripts/gpu_probe.py --verbose     # ask the HIP runtime to log too
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _report import tee_to  # noqa: E402


# Windows NTSTATUS values, as returned to the shell (negative on POSIX-style
# reporting, the unsigned form is what cmd's ERRORLEVEL shows).
CRASH_CODES = {
    0xC0000005: ("ACCESS_VIOLATION",
                 "The HIP runtime dereferenced bad memory. Usually a driver "
                 "that does not match the ROCm wheels, or a gfx target the "
                 "wheels have no code objects for."),
    0xC0000135: ("DLL_NOT_FOUND",
                 "A DLL is missing. The ROCm SDK wheels and the torch wheel "
                 "must come from the same build date."),
    0xC0000139: ("ENTRYPOINT_NOT_FOUND",
                 "A DLL was found but is the wrong version -- mismatched ROCm "
                 "SDK and torch wheels."),
    0xC000007B: ("INVALID_IMAGE_FORMAT",
                 "A 32/64-bit mismatch, or a corrupt DLL."),
    0xC0000409: ("FAIL_FAST",
                 "The runtime aborted deliberately, often on an unsupported "
                 "GPU target."),
    0xC0000374: ("HEAP_CORRUPTION", "The runtime corrupted its own heap."),
}

STEPS = [
    ("import torch", "import torch; print(torch.__version__)"),
    ("torch.version.hip", "import torch; print(torch.version.hip)"),
    ("device_count()", "import torch; print(torch.cuda.device_count())"),
    ("is_available()", "import torch; print(torch.cuda.is_available())"),
    ("device name",
     "import torch; print(torch.cuda.get_device_properties(0).name)"),
    ("gcnArchName",
     "import torch; print(getattr(torch.cuda.get_device_properties(0),"
     " 'gcnArchName', '(none)'))"),
    ("allocate on GPU",
     "import torch; x = torch.zeros(8, device='cuda'); print('ok', x.sum().item())"),
    ("matmul on GPU",
     "import torch; x = torch.randn(256, 256, device='cuda');"
     " print('ok', float((x @ x).sum()))"),
]


def describe_exit(code: int) -> str:
    """Turn a child exit code into something a person can act on."""
    if code == 0:
        return ""
    unsigned = code & 0xFFFFFFFF
    if unsigned in CRASH_CODES:
        name, why = CRASH_CODES[unsigned]
        return f"CRASHED: {name} (0x{unsigned:08X})\n         {why}"
    if code < 0:
        return f"killed by signal {-code}"
    return f"exited {code}"


def run_step(source: str, verbose: bool, timeout: int = 180):
    env = dict(os.environ)
    env['PYTHONFAULTHANDLER'] = '1'      # C-level traceback on a fatal signal
    env['PYTHONUNBUFFERED'] = '1'
    if verbose:
        env['AMD_LOG_LEVEL'] = '4'       # make the HIP runtime narrate
    try:
        done = subprocess.run([sys.executable, '-X', 'faulthandler', '-c', source],
                              capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return None, "", f"TIMED OUT after {timeout}s (the runtime hung)"
    return done.returncode, (done.stdout or "").strip(), (done.stderr or "").strip()



# Environment knobs that are known to decide whether the HIP runtime survives
# device enumeration. Tried as a matrix rather than one per attempt, because
# each guess otherwise costs a full round trip.
SWEEP = [
    ({}, "baseline"),
    ({'HIP_VISIBLE_DEVICES': '0'}, "HIP device 0 only"),
    ({'HIP_VISIBLE_DEVICES': '1'}, "HIP device 1 only"),
    ({'ROCR_VISIBLE_DEVICES': '0'}, "ROCR device 0 only"),
    ({'ROCR_VISIBLE_DEVICES': '1'}, "ROCR device 1 only"),
    ({'HSA_OVERRIDE_GFX_VERSION': '10.3.0'}, "report as gfx1030"),
    ({'HSA_OVERRIDE_GFX_VERSION': '10.3.0', 'HIP_VISIBLE_DEVICES': '0'},
     "gfx1030 + HIP device 0"),
    ({'HSA_OVERRIDE_GFX_VERSION': '10.3.0', 'HIP_VISIBLE_DEVICES': '1'},
     "gfx1030 + HIP device 1"),
    ({'HSA_OVERRIDE_GFX_VERSION': '10.3.0', 'ROCR_VISIBLE_DEVICES': '0'},
     "gfx1030 + ROCR device 0"),
    ({'HSA_ENABLE_SDMA': '0'}, "SDMA disabled"),
    ({'HSA_OVERRIDE_GFX_VERSION': '10.3.0', 'HSA_ENABLE_SDMA': '0'},
     "gfx1030 + SDMA disabled"),
    ({'GPU_MAX_HW_QUEUES': '1'}, "single hardware queue"),
    ({'AMD_SERIALIZE_KERNEL': '3', 'HSA_OVERRIDE_GFX_VERSION': '10.3.0'},
     "gfx1030 + serialized kernels"),
]

SWEEP_KEYS = sorted({k for case, _ in SWEEP for k in case})


def run_case(source: str, overrides: dict, timeout: int = 120):
    """Run one snippet with a clean slate plus `overrides`."""
    env = {k: v for k, v in os.environ.items() if k not in SWEEP_KEYS}
    env.update(overrides)
    env['PYTHONFAULTHANDLER'] = '1'
    env['PYTHONUNBUFFERED'] = '1'
    try:
        done = subprocess.run([sys.executable, '-X', 'faulthandler', '-c', source],
                              capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return None, "", "timed out"
    return done.returncode, (done.stdout or "").strip(), (done.stderr or "").strip()


def sweep(timeout: int) -> int:
    """Find an environment where device enumeration survives, if one exists."""
    print("Environment sweep")
    print("-" * 17)
    print("Each row runs in its own process, so a crash only ends that row.")
    print()
    count_src = "import torch; print(torch.cuda.device_count())"
    work_src = ("import torch; p = torch.cuda.get_device_properties(0);"
                " x = torch.randn(256, 256, device='cuda');"
                " print(p.name, '|', getattr(p, 'gcnArchName', '?'),"
                " '| matmul', float((x @ x).sum()) == float((x @ x).sum()))")
    winners = []
    for overrides, label in SWEEP:
        code, out, _ = run_case(count_src, overrides, timeout)
        if code != 0:
            print(f"  crash   {label:28} {describe_exit(code).splitlines()[0] if code is not None else 'timed out'}")
            continue
        if out.strip() in ('0', ''):
            print(f"  0 gpus  {label:28} survived, but enumerated nothing")
            continue
        code2, out2, err2 = run_case(work_src, overrides, timeout)
        if code2 == 0:
            print(f"  WORKS   {label:28} {out}, {out2}")
            winners.append((overrides, label))
        else:
            detail = describe_exit(code2).splitlines()[0] if code2 is not None else 'timed out'
            print(f"  partial {label:28} {out} device(s), but using one: {detail}")
            if err2:
                print(f"          {err2.splitlines()[-1][:100]}")

    print()
    if not winners:
        print("No combination worked, so this is the driver or the wheels rather")
        print("than a setting. In order:")
        print("  1. Pin the build known to work on RDNA2 instead of the newest:")
        print("       python scripts\\install_rocm_torch.py --known-good")
        print("  2. Install the latest Microsoft Visual C++ Redistributable.")
        print("  3. Change the Adrenalin driver. 26.1.1 is named in several")
        print("     access-violation reports (ROCm/ROCm#5871); both newer and")
        print("     older builds are worth trying.")
        return 1

    overrides, label = winners[0]
    print(f"Use this: {label}")
    if overrides:
        for key, value in overrides.items():
            print(f"  set {key}={value}")
        print()
        print("Set them in the same shell before running Pixal3D, or once for")
        print("your account so every new window inherits them:")
        for key, value in overrides.items():
            print(f"  setx {key} {value}")
    else:
        print("  (no environment changes needed)")
    return 0


def report_environment() -> None:
    """Driver versions and installed ROCm wheels -- the two usual culprits."""
    amd_adapters = []
    print("Display adapters (from the driver registry)")
    print("-" * 42)
    try:
        import winreg
        key_path = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as root:
            index = 0
            while True:
                try:
                    sub = winreg.EnumKey(root, index)
                except OSError:
                    break
                index += 1
                if not sub.isdigit():
                    continue
                try:
                    with winreg.OpenKey(root, sub) as adapter:
                        desc = winreg.QueryValueEx(adapter, "DriverDesc")[0]
                        version = winreg.QueryValueEx(adapter, "DriverVersion")[0]
                        print(f"  {desc}  driver {version}")
                        if any(k in desc.lower() for k in ('amd', 'radeon')):
                            amd_adapters.append((desc, version))
                except OSError:
                    continue
    except ImportError:
        print("  (not Windows)")
    except Exception as exc:
        print(f"  could not read: {type(exc).__name__}: {exc}")

    # An integrated Radeon alongside a discrete one is a known breaker: the
    # runtime enumerates both, the gfx10 3X-dgpu wheels carry no kernels for
    # the integrated part, and it fails before the discrete card is ever used.
    # HIP_VISIBLE_DEVICES does not reliably help, because the damage is done
    # during enumeration. The ROCm ComfyUI fork tells people to disable the
    # iGPU outright for this reason.
    if len(amd_adapters) > 1:
        integrated = [d for d, _ in amd_adapters
                      if 'rx' not in d.lower() and 'pro w' not in d.lower()]
        print()
        print(f"  !! {len(amd_adapters)} AMD adapters. This is the most common cause of")
        print("     enumeration crashes and hipErrorInvalidImage on laptops.")
        if integrated:
            print(f"     The integrated one looks like: {integrated[0]}")
        print("     Disable it and retry -- Device Manager > Display adapters >")
        print("     right-click it > Disable device. Reversible, no reboot needed.")
        print("     (BIOS works too, and is what the ROCm ComfyUI fork recommends.)")
    print()

    # An access violation inside hipGetDeviceCount is usually a version
    # mismatch across a DLL boundary. The driver installs a HIP runtime into
    # System32 and the wheels ship their own; if the wrong one wins the loader
    # search, the struct layouts disagree and the runtime reads a bad pointer.
    print("HIP runtime DLLs on this machine")
    print("-" * 32)
    try:
        import glob
        seen = []
        for root in [os.path.join(os.environ.get('SystemRoot', r'C:\\Windows'), 'System32')] + \
                    [os.path.dirname(os.path.dirname(os.__file__)) + os.sep + 'site-packages']:
            for name in ('amdhip64*.dll', 'amd_comgr*.dll', 'hiprtc*.dll'):
                for hit in glob.glob(os.path.join(root, '**', name), recursive=True)[:6]:
                    seen.append(hit)
        if seen:
            for hit in seen:
                try:
                    size = os.path.getsize(hit)
                except OSError:
                    size = -1
                print(f"  {size/1e6:8.1f} MB  {hit}")
            roots = {('System32' if 'System32' in h else 'wheel') for h in seen}
            if len(roots) > 1:
                print()
                print("  Both the driver's copy and the wheels' copy are present.")
                print("  Whichever the loader picks must match the other components;")
                print("  a driver older than the wheels is the usual reason this")
                print("  crashes rather than reporting an error.")
        else:
            print("  none found (not Windows, or the wheels are elsewhere)")
    except Exception as exc:
        print(f"  could not scan: {type(exc).__name__}: {exc}")
    print()

    print("Installed ROCm / torch wheels")
    print("-" * 29)
    try:
        from importlib.metadata import distributions
        rows = sorted({(d.metadata['Name'], d.version) for d in distributions()
                       if (d.metadata['Name'] or '').lower().startswith(('rocm', 'torch'))})
        for name, version in rows:
            print(f"  {name:34} {version}")
        # The SDK and torch must come from one build; a date mismatch is the
        # classic cause of a torch that loads and then crashes in the runtime.
        stamps = {v.split('a')[-1][:8] for _, v in rows if 'a20' in v}
        if len(stamps) > 1:
            print()
            print(f"  MISMATCH: build dates {sorted(stamps)} are not all the same.")
            print("  Reinstall one consistent set:")
            print("    python scripts\\install_rocm_torch.py --rocm-version <one build>")
    except Exception as exc:
        print(f"  could not list: {type(exc).__name__}: {exc}")
    print()

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--verbose', action='store_true',
                        help="Set AMD_LOG_LEVEL=4 so the HIP runtime logs what it does.")
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--sweep', action='store_true',
                        help="Try the environment settings that decide whether the "
                             "HIP runtime survives enumeration, and report which work.")
    args = parser.parse_args()

    tee_to(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gpu_probe_report.txt"))
    print("Pixal3D GPU probe")
    print("=" * 17)
    if args.sweep:
        report_environment()
        return sweep(args.timeout)
    print(f"python: {sys.executable}")
    for name in ('HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES',
                 'HSA_OVERRIDE_GFX_VERSION', 'GPU_DEVICE_ORDINAL'):
        if name in os.environ:
            print(f"env:    {name}={os.environ[name]}")
    print()

    failed_at = None
    for label, source in STEPS:
        code, out, err = run_step(source, args.verbose, args.timeout)
        if code == 0:
            print(f"  ok    {label:20} {out}")
            continue
        print(f"  FAIL  {label:20} {describe_exit(code) if code is not None else err}")
        if out:
            print(f"        stdout: {out}")
        if err:
            for line in err.splitlines()[-12:]:
                print(f"        {line}")
        failed_at = label
        break

    print()
    if failed_at is None:
        print("The GPU stack works. If generation still fails, the problem is")
        print("further up -- run scripts/check_env.py.")
        return 0

    print(f"First failure: {failed_at}")
    if failed_at in ("device_count()", "is_available()", "device name",
                     "gcnArchName", "allocate on GPU", "matmul on GPU"):
        print()
        print("This is the GPU stack, not Pixal3D. In order:")
        print("  1. AMD Adrenalin driver 26.1.1 or newer.")
        print("  2. RX 6700/6750/6800M/6600 are gfx1031/gfx1032. If the wheels")
        print("     carry only gfx1030 code objects, in the SAME shell:")
        print("       set HSA_OVERRIDE_GFX_VERSION=10.3.0")
        print("       python scripts\\gpu_probe.py")
        print("  3. The ROCm SDK wheels and the torch wheel must share a build")
        print("     date. To see what is on offer and pin one:")
        print("       python scripts\\install_rocm_torch.py --list")
        print("  4. Pin the build the community reports as working for RDNA2")
        print("     rather than the newest nightly -- Windows ROCm has had")
        print("     repeated enumeration regressions on this family:")
        print("       python scripts\\install_rocm_torch.py --known-good")
        print("  5. Install the latest Microsoft Visual C++ Redistributable.")
        print("     The HIP DLLs need it, and a missing one faults like this.")
        print("  6. Re-run with --verbose to get the HIP runtime's own log.")
        print()
        print("  Or let it try all of them at once:")
        print("    gpu_probe.bat --sweep")
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
