"""
Install AMD's ROCm build of PyTorch for a specific GPU family.

AMD's *released* Windows PyTorch wheels only cover RDNA3 and RDNA4. RDNA2 cards
- the RX 6000 series, gfx103x - are built by the same CI but published only to
the nightly index, so they have to be installed by hand from there. This script
resolves a self-consistent set (the ROCm SDK and torch must come from the same
build date) and installs it.

    python scripts/install_rocm_torch.py                      # gfx103X, auto-detect Python tag
    python scripts/install_rocm_torch.py --family gfx110X-dgpu
    python scripts/install_rocm_torch.py --list               # show available builds
    python scripts/install_rocm_torch.py --dry-run

Stdlib only: it has to run before anything is installed.
"""

import argparse
import html
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple


# AMD's current channel. One torch wheel plus a per-architecture extra
# (torch[device-gfx1031]) rather than a per-family bundle, so the exact chip
# gets its own code objects -- including parts the old "gfx103X-dgpu" bundle
# covered badly or not at all.
NEW_INDEX = "https://nightly.repo.amd.com/rocm/whl-next/"

# The previous channel, kept for --legacy. Its gfx103X-dgpu bundle crashes
# during device enumeration on gfx1031.
BASE = "https://rocm.nightlies.amd.com/v2-staging"

# GPU family -> the marketing names that map onto it, for --detect.
FAMILIES: Dict[str, Tuple[str, ...]] = {
    'gfx103X-dgpu': ('RX 6800', 'RX 6900', 'RX 6950', 'RX 6750', 'RX 6700',
                     'RX 6650', 'RX 6600', 'RX 6500', 'RX 6400'),
    'gfx110X-dgpu': ('RX 7900', 'RX 7800', 'RX 7700', 'RX 7600'),
    'gfx120X-all': ('RX 9070', 'RX 9060'),
    'gfx1151': ('Ryzen AI Max', 'Strix Halo'),
}

# Newest is not safest. The Windows ROCm stack has had repeated regressions in
# device enumeration on RDNA2 -- HIP SDK 7.1.1 stopped detecting gfx1030
# (ROCm/hip#3899), and 7.13 nightlies have broken other families outright
# (ROCm/TheRock#5543). These builds are the ones the community reports as
# actually working for each family; --known-good pins them.
KNOWN_GOOD: Dict[str, str] = {
    'gfx103X-dgpu': '7.12.0a20260204',
}


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode('utf-8', 'replace')


def list_files(index: str, package: str) -> List[str]:
    """Filenames a PEP 503 index page offers for one package."""
    try:
        page = fetch(f"{index}/{package}/")
    except Exception as exc:
        raise SystemExit(f"Could not read {index}/{package}/ : {exc}")
    return [html.unescape(m) for m in re.findall(r'>([^<>]+\.(?:whl|tar\.gz))<', page)]


def python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def platform_tag() -> str:
    if sys.platform.startswith('win'):
        return 'win_amd64'
    if sys.platform.startswith('linux'):
        return 'linux_x86_64'
    raise SystemExit(f"No ROCm wheels are published for {sys.platform}. "
                     f"Use --platform-tag to resolve a wheel set for another machine.")


def _version_key(version: str) -> Tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r'\d+', version))


def find_torch_builds(index: str, tag: str, plat: str) -> List[Tuple[str, str, str]]:
    """
    ``(rocm_version, torch_version, filename)`` for every matching torch wheel,
    worst candidate first.

    The index carries several torch versions per ROCm build, including
    prereleases. Sorting puts the newest ROCm build last and, within it,
    prefers a released torch over an ``a0`` - the surrounding ecosystem
    (transformers, diffusers, trimesh) is only tested against releases.
    """
    pattern = re.compile(
        rf'^torch-([0-9][^+]*)\+rocm([0-9][0-9a-zA-Z.]*)-{tag}-{tag}-{plat}\.whl$')
    builds = []
    for name in list_files(index, 'torch'):
        m = pattern.match(name)
        if m:
            builds.append((m.group(2), m.group(1), name))
    builds.sort(key=lambda b: (b[0], 'a' not in b[1] and 'rc' not in b[1], _version_key(b[1])))
    return builds


def companion_prefix(package: str, torch_version: str) -> str:
    """
    Filename prefix of the torchvision/torchaudio release that goes with a
    given torch version. torchaudio tracks torch exactly; torchvision's minor
    runs fifteen ahead (torch 2.9 -> torchvision 0.24).
    """
    major, minor = _version_key(torch_version)[:2]
    if package == 'torchaudio':
        return f"torchaudio-{major}.{minor}."
    return f"torchvision-0.{minor + 15}."


def match_version(index: str, package: str, rocm_version: str,
                  tag: Optional[str], plat: str,
                  prefix: Optional[str] = None) -> Optional[str]:
    """The file for ``package`` belonging to the same ROCm build."""
    for name in list_files(index, package.replace('_', '-')):
        if rocm_version not in name:
            continue
        if prefix is not None and not name.startswith(prefix):
            continue
        if name.endswith('.tar.gz'):
            return name
        if tag is None and 'py3-none' in name and plat in name:
            return name
        if tag is not None and f'-{tag}-{tag}-{plat}.whl' in name:
            return name
    return None


def adapter_names() -> List[str]:
    """Display adapter names from the driver registry -- no torch needed."""
    names = []
    try:
        import winreg
    except ImportError:
        return names
    key_path = (r"SYSTEM\CurrentControlSet\Control\Class"
                r"\{4d36e968-e325-11ce-bfc1-08002be10318}")
    try:
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
                        names.append(winreg.QueryValueEx(adapter, "DriverDesc")[0])
                except OSError:
                    continue
    except Exception:
        pass
    return names


def detect_arch() -> Optional[str]:
    """
    The gfx target to install for: the discrete Radeon if there is one.

    A laptop with an integrated Radeon reports both, and the discrete card is
    the one worth building for.
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from pixal3d.compat.probe import arch_from_device_name
    except Exception:
        return None
    integrated_hint = ('graphics', '680m', '780m', '660m', '760m', 'vega')
    found = []
    for name in adapter_names():
        arch = arch_from_device_name(name)
        if arch:
            found.append((arch, any(h in name.lower() for h in integrated_hint), name))
    if not found:
        return None
    found.sort(key=lambda row: row[1])        # discrete first
    arch, _, name = found[0]
    print(f"gpu    : {name} -> {arch}")
    return arch


def install_new(arch: str, dry_run: bool, audio: bool) -> int:
    """
    Install from AMD's current index using the per-architecture extras.

    This is what the maintained Windows ROCm ComfyUI fork does, and it is the
    difference between a working install and one that access-violates during
    enumeration.
    """
    packages = [f"torch[device-{arch}]", f"torchvision[device-{arch}]"]
    if audio:
        packages.append("torchaudio")
    packages.append("rocm-sdk-devel")
    cmd = [sys.executable, '-m', 'pip', 'install', '--pre',
           '--index-url', NEW_INDEX, '--no-warn-script-location', *packages]
    print(f"index  : {NEW_INDEX}")
    print(f"arch   : {arch}")
    print()
    for name in packages:
        print(f"  {name}")
    print()
    if dry_run:
        print(' '.join(cmd))
        return 0
    print("Installing (this downloads a few GB) ...")
    return subprocess.call(cmd)


def detect_family() -> Optional[str]:
    """Best-effort GPU family from the Windows device name."""
    if not sys.platform.startswith('win'):
        return None
    try:
        out = subprocess.run(
            ['powershell', '-NoProfile', '-Command',
             '(Get-CimInstance Win32_VideoController).Name'],
            capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return None
    for family, names in FAMILIES.items():
        if any(n.lower() in out.lower() for n in names):
            return family
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--family', default=None,
                        help=f"GPU family. One of: {', '.join(FAMILIES)}. "
                             "Default: detected from the installed GPU, else gfx103X-dgpu.")
    parser.add_argument('--index', default=BASE, help="Nightly index base URL")
    parser.add_argument('--rocm-version', default=None,
                        help="Pin a build, e.g. 7.13.0a20260421. Default: newest available.")
    parser.add_argument('--list', action='store_true',
                        help="List available builds on the legacy index and exit")
    parser.add_argument('--arch', default=None,
                        help="gfx target to install for, e.g. gfx1031. "
                             "Default: detected from the discrete adapter.")
    parser.add_argument('--legacy', action='store_true',
                        help="Use the old per-family index instead of AMD's current "
                             "one. The gfx103X-dgpu bundle there crashes on gfx1031.")
    parser.add_argument('--known-good', action='store_true',
                        help="Pin the build reported to work for this family rather than "
                             "the newest. Try this if the newest one crashes.")
    parser.add_argument('--dry-run', action='store_true', help="Print the pip command only")
    parser.add_argument('--audio', action='store_true', help="Also install torchaudio")
    parser.add_argument('--python-tag', default=None,
                        help="Override the interpreter tag, e.g. cp312. Implies --dry-run.")
    parser.add_argument('--platform-tag', default=None,
                        help="Override the platform tag, e.g. win_amd64. Implies --dry-run.")
    args = parser.parse_args()

    cross = args.python_tag is not None or args.platform_tag is not None

    # AMD's current channel first. The old per-family bundles are still
    # reachable with --legacy, but gfx103X-dgpu crashes on gfx1031.
    if not args.legacy and not args.list and not cross:
        arch = args.arch or detect_arch()
        if arch:
            return install_new(arch, args.dry_run, args.audio)
        print("Could not identify the gfx target from the adapter list.")
        print("Pass it explicitly, e.g. --arch gfx1031, or use --legacy.")
        return 1

    family = args.family or detect_family() or 'gfx103X-dgpu'
    index = f"{args.index}/{family}"
    tag = args.python_tag or python_tag()
    plat = args.platform_tag or platform_tag()

    print(f"family : {family}")
    print(f"index  : {index}")
    print(f"python : {tag} / {plat}")

    builds = find_torch_builds(index, tag, plat)
    if not builds:
        raise SystemExit(
            f"No torch wheels for {tag}/{plat} under {index}.\n"
            f"AMD publishes cp312/cp313/cp314 Windows builds; Python "
            f"{sys.version_info.major}.{sys.version_info.minor} is not among them.")

    if args.list:
        for rocm_version, torch_version, name in builds:
            print(f"  rocm {rocm_version}  torch {torch_version}  {name}")
        return 0

    wanted = args.rocm_version
    if args.known_good and not wanted:
        wanted = KNOWN_GOOD.get(family)
        if wanted is None:
            print(f"note: no known-good build recorded for {family}; using the newest")
        else:
            print(f"known-good pin for {family}: {wanted}")
    args.rocm_version = wanted

    if args.rocm_version:
        selected = [b for b in builds if b[0] == args.rocm_version]
        if not selected:
            raise SystemExit(f"Build {args.rocm_version} not found. Use --list to see options.")
        rocm_version, torch_version, torch_file = selected[-1]
    else:
        rocm_version, torch_version, torch_file = builds[-1]
    print(f"build  : rocm {rocm_version}, torch {torch_version}")

    # The SDK runtime, the per-family kernel libraries and torch itself are
    # built together; mixing dates gives a torch that loads but sees no GPU.
    files = [torch_file]
    library_package = 'rocm_sdk_libraries_' + family.lower().replace('-', '_')
    for package, wheel_tag in (
        ('rocm', None),
        ('rocm_sdk_core', None),
        ('rocm_sdk_devel', None),
        (library_package, None),
    ):
        name = match_version(index, package, rocm_version, wheel_tag, plat)
        if name is None:
            raise SystemExit(f"{package} has no file for rocm {rocm_version}; "
                             f"try another --rocm-version.")
        files.append(name)

    # torchvision and torchaudio must match the *torch* version, not just the
    # ROCm build: several torch versions share one build date, and picking the
    # wrong companion makes pip pull a second torch over the top of this one.
    for package, prefix in (('torchvision', companion_prefix('torchvision', torch_version)),
                            ('torchaudio', companion_prefix('torchaudio', torch_version))):
        if package == 'torchaudio' and not args.audio:
            continue
        name = match_version(index, package, rocm_version, tag, plat, prefix=prefix)
        if name is None:
            print(f"  note: no {package} matching torch {torch_version}, skipping")
            continue
        files.append(name)

    urls = [f"{index}/{urllib.parse.quote(name)}" for name in files]
    cmd = [sys.executable, '-m', 'pip', 'install', '--no-cache-dir', *urls]

    print()
    for name in files:
        print(f"  {name}")
    print()
    if args.dry_run or cross:
        if cross and not args.dry_run:
            print("(resolved for another machine - not installing)")
        print(' '.join(cmd))
        return 0

    print("Installing (this downloads a few GB) ...")
    return subprocess.call(cmd)


if __name__ == '__main__':
    raise SystemExit(main())
