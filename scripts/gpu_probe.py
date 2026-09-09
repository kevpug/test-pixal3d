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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--verbose', action='store_true',
                        help="Set AMD_LOG_LEVEL=4 so the HIP runtime logs what it does.")
    parser.add_argument('--timeout', type=int, default=180)
    args = parser.parse_args()

    print("Pixal3D GPU probe")
    print("=" * 17)
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
        print("  4. Re-run with --verbose to get the HIP runtime's own log.")
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
