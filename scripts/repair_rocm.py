"""
Try, in order, everything known to fix a ROCm GPU stack that will not
enumerate on Windows -- re-testing after each step and stopping at the first
thing that works.

The steps, cheapest first:

1. Test what is installed. If it already works, change nothing.
2. Sweep the environment variables that decide whether the HIP runtime
   survives enumeration, and if one works, persist it.
3. Check the Visual C++ runtime, which the HIP DLLs need and whose absence
   faults rather than errors.
4. Reinstall a ROCm build known to work for this GPU family instead of the
   newest nightly. Windows ROCm has repeatedly regressed here: HIP SDK 7.1.1
   stopped detecting gfx1030 (ROCm/hip#3899) and 7.13 nightlies broke gfx1100
   outright (ROCm/TheRock#5543).
5. Fall back through older builds, re-testing each.

Every GPU call runs in a subprocess, because a broken HIP runtime kills the
process rather than raising, and a repair tool that dies mid-repair is useless.

    python scripts/repair_rocm.py
    python scripts/repair_rocm.py --deep 3     # try more builds (GBs each)
    python scripts/repair_rocm.py --dry-run
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gpu_probe import SWEEP, SWEEP_KEYS, describe_exit, run_case  # noqa: E402
from _report import elapsed, tee_to  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE = os.path.join(ROOT, 'pixal3d_env.bat')
VC_REDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"

COUNT_SRC = "import torch; print(torch.cuda.device_count())"
WORK_SRC = ("import torch; p = torch.cuda.get_device_properties(0);"
            " x = torch.randn(512, 512, device='cuda'); y = (x @ x).sum().item();"
            " print(p.name, '|', getattr(p, 'gcnArchName', '?'), '| ok', y == y)")


def step(text):
    print()
    print(f"[{elapsed()}] ==> {text}")


def works(overrides, timeout=180):
    """Does the GPU enumerate *and* compute under these settings?"""
    code, out, _ = run_case(COUNT_SRC, overrides, timeout)
    if code != 0 or out.strip() in ('', '0'):
        return False, (describe_exit(code).splitlines()[0] if code not in (0, None)
                       else f"{out or '0'} devices")
    code, out, err = run_case(WORK_SRC, overrides, timeout)
    if code != 0:
        return False, (describe_exit(code).splitlines()[0] if code is not None
                       else 'timed out')
    return True, out.strip()


def persist(overrides):
    """Write the winning variables where the .bat wrappers will pick them up."""
    if not overrides:
        if os.path.exists(ENV_FILE):
            os.remove(ENV_FILE)
        return
    lines = ["@echo off",
             "REM Written by scripts/repair_rocm.py -- the settings this GPU needs.",
             "REM The run_*.bat / check_env.bat wrappers call this if it exists."]
    lines += [f"set {k}={v}" for k, v in sorted(overrides.items())]
    with open(ENV_FILE, 'w', newline='\r\n') as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"    saved to {ENV_FILE}")
    for key, value in sorted(overrides.items()):
        print(f"      {key}={value}")


def vc_redist_present():
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64") as key:
            return bool(winreg.QueryValueEx(key, "Installed")[0])
    except Exception:
        return False


def install_vc_redist(dry_run):
    if vc_redist_present():
        print("    already installed")
        return False
    print("    missing -- the HIP DLLs need it")
    if dry_run:
        print(f"    would download and run {VC_REDIST_URL}")
        return False
    try:
        import urllib.request
        target = os.path.join(os.environ.get('TEMP', ROOT), 'vc_redist.x64.exe')
        print("    downloading ...")
        urllib.request.urlretrieve(VC_REDIST_URL, target)
        print("    installing (accept the Windows prompt if it appears) ...")
        subprocess.run([target, '/install', '/passive', '/norestart'], timeout=900)
        return True
    except Exception as exc:
        print(f"    could not install automatically: {type(exc).__name__}: {exc}")
        print(f"    install it by hand from {VC_REDIST_URL}")
        return False


def installed_build():
    """The ROCm build id of the torch that is installed, or None."""
    code, out, _ = run_case("import torch; print(torch.__version__)", {}, 120)
    if code != 0 or 'rocm' not in (out or ''):
        return None
    return out.split('rocm')[-1].strip()


def candidate_builds(limit, exclude=()):
    """
    Known-good build first, then a spread of others, newest first.

    Builds from a different release series than the installed one come first:
    if 7.13 is what is crashing, another 7.13 nightly is unlikely to help, and
    each attempt is a multi-gigabyte download.
    """
    try:
        import install_rocm_torch as installer
    except Exception as exc:
        print(f"    could not load the installer: {exc}")
        return []
    family = installer.detect_family() or 'gfx103X-dgpu'
    index = f"{installer.BASE}/{family}"
    try:
        builds = installer.find_torch_builds(index, installer.python_tag(),
                                             installer.platform_tag())
    except SystemExit as exc:
        print(f"    could not read the index: {exc}")
        return []
    all_versions = sorted({b[0] for b in builds}, reverse=True)
    skip = set(exclude)
    versions = []
    known = installer.KNOWN_GOOD.get(family)
    if known and known in all_versions and known not in skip:
        versions.append(known)

    bad_series = {v.split('a')[0] for v in skip}
    rest = [v for v in all_versions if v not in skip and v not in versions]
    other_series = [v for v in rest if v.split('a')[0] not in bad_series]
    same_series = [v for v in rest if v.split('a')[0] in bad_series]
    versions += other_series + same_series
    return versions[:limit]


def install_build(version, dry_run):
    cmd = [sys.executable, os.path.join(ROOT, 'scripts', 'install_rocm_torch.py'),
           '--rocm-version', version]
    if dry_run:
        print("    would run: " + " ".join(cmd))
        return True
    print("    downloading and installing -- this is several GB and takes a")
    print("    while. pip's progress follows; the window is not frozen.")
    sys.stdout.flush()
    return subprocess.call(cmd) == 0


def try_all(label, timeout):
    """Baseline, then the sweep. Returns the winning overrides or None."""
    ok, detail = works({}, timeout)
    if ok:
        print(f"    {label}: WORKS -- {detail}")
        return {}
    print(f"    {label}: {detail}")
    for number, (overrides, name) in enumerate(SWEEP, 1):
        if not overrides:
            continue
        print(f"      trying {name} ({number}/{len(SWEEP)}) ...")
        ok, detail = works(overrides, timeout)
        if ok:
            print(f"    {label} + {name}: WORKS -- {detail}")
            return overrides
    print(f"    {label}: no environment setting helped")
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--deep', type=int, default=2,
                        help="How many ROCm builds to try. Each is several GB. Default 2.")
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--dry-run', action='store_true',
                        help="Say what would happen without installing anything.")
    parser.add_argument('--skip-vcredist', action='store_true')
    args = parser.parse_args()

    tee_to(os.path.join(ROOT, 'repair_report.txt'))
    print("Pixal3D ROCm repair")
    print("=" * 19)
    print("Each step is tested before moving on. Steps 4 and 5 download several")
    print("GB each. Everything here is also saved to repair_report.txt.")
    print(f"python: {sys.executable}")
    # Inherited variables would make every result a lie about the baseline.
    for key in SWEEP_KEYS:
        if key in os.environ:
            print(f"note:   ignoring inherited {key}={os.environ.pop(key)}")

    step("1/5  Testing what is installed")
    winner = try_all("as installed", args.timeout)
    if winner is not None:
        step("Done")
        persist(winner)
        print("The GPU works. Nothing else needed." if not winner else
              "The GPU works with the settings above.")
        return 0

    if not args.skip_vcredist:
        step("2/5  Checking the Visual C++ runtime")
        if install_vc_redist(args.dry_run):
            winner = try_all("after VC++ redist", args.timeout)
            if winner is not None:
                step("Done")
                persist(winner)
                return 0

    step(f"3/5  Reinstalling ROCm (up to {args.deep} build(s), several GB each)")
    current = installed_build()
    if current:
        print(f"    currently installed: rocm {current} (will not be retried)")
    versions = candidate_builds(args.deep, exclude={current} if current else ())
    if not versions:
        print("    no builds to try")
    for number, version in enumerate(versions, 1):
        print()
        print(f"    --- build {number}/{len(versions)}: rocm {version} ---")
        if not install_build(version, args.dry_run):
            print("    install failed, moving on")
            continue
        if args.dry_run:
            continue
        winner = try_all(f"rocm {version}", args.timeout)
        if winner is not None:
            step("Done")
            persist(winner)
            print(f"Fixed by rocm {version}.")
            return 0

    step("5/5  Out of options")
    print("None of the builds or settings made the GPU usable. At this point the")
    print("problem is below Python entirely -- the driver, or Windows ROCm on this")
    print("GPU generation. Worth trying, in order:")
    print("  1. A different Adrenalin driver. 26.1.1 is named in several")
    print("     access-violation reports (ROCm/ROCm#5871); try newer and older.")
    print("  2. Dual-boot Linux, where this GPU family works well and the native")
    print("     extensions build too. Set HSA_OVERRIDE_GFX_VERSION=10.3.0 there.")
    print("  3. A ZLUDA setup, if you already run image models that way.")
    print()
    print("Pixal3D itself is fine either way -- run_tests.bat passes on CPU.")
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
