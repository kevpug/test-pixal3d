"""
Find the working GPU setup already on this machine and copy what makes it work.

If ComfyUI drives this GPU but Pixal3D cannot, the hardware and the driver are
fine and the difference is the stack. On Windows AMD there are three, and they
are not interchangeable:

  ROCm wheels   torch built against HIP; torch.version.hip is set. What
                Pixal3D installs.
  ZLUDA         a CUDA build of torch with a translation layer on top, using
                the HIP SDK installed system-wide. torch.version.cuda is set
                and the device is an AMD card. Common for RDNA2, because the
                official ROCm nightlies have been unreliable there.
  DirectML      a separate backend entirely; torch.cuda is not involved.

So: point this at the interpreter that works, and it reports which stack that
is and the exact package set behind it.

    python scripts/adopt_env.py                        # search for one
    python scripts/adopt_env.py --python "C:\\ComfyUI\\venv\\Scripts\\python.exe"
"""

import argparse
import glob
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _report import tee_to  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IDENTIFY = r'''
import json, sys
info = {"exe": sys.executable, "py": "%d.%d" % sys.version_info[:2]}
try:
    import torch
    info["torch"] = torch.__version__
    info["hip"] = getattr(torch.version, "hip", None)
    info["cuda"] = getattr(torch.version, "cuda", None)
    try:
        info["count"] = torch.cuda.device_count()
        info["name"] = torch.cuda.get_device_name(0) if info["count"] else None
        p = torch.cuda.get_device_properties(0) if info["count"] else None
        info["arch"] = getattr(p, "gcnArchName", None) if p else None
    except Exception as exc:
        info["count_error"] = "%s: %s" % (type(exc).__name__, exc)
except Exception as exc:
    info["torch_error"] = "%s: %s" % (type(exc).__name__, exc)
try:
    from importlib.metadata import distributions
    info["packages"] = sorted({
        (d.metadata["Name"], d.version) for d in distributions()
        if (d.metadata["Name"] or "").lower().startswith(("torch", "rocm", "zluda", "amdsmi"))
    })
except Exception:
    info["packages"] = []
print("PIXAL3D_JSON:" + json.dumps(info))
'''

WORK = ("import torch; x = torch.randn(256, 256, device='cuda');"
        " print('COMPUTE_OK', float((x @ x).sum()) == float((x @ x).sum()))")


def candidates():
    """Interpreters worth asking, newest-looking first."""
    roots = []
    for base in (os.path.expanduser("~"), os.path.expanduser("~/Desktop"),
                 "C:\\", "D:\\", os.path.dirname(ROOT)):
        for pattern in ("ComfyUI*", "comfyui*", "stable-diffusion*", "*Zluda*", "*rocm*"):
            try:
                roots.extend(glob.glob(os.path.join(base, pattern)))
            except Exception:
                pass
    found = []
    for root in dict.fromkeys(roots):
        for rel in (r"venv\Scripts\python.exe", r".venv\Scripts\python.exe",
                    r"python_embeded\python.exe", r"venv/bin/python"):
            path = os.path.join(root, rel)
            if os.path.exists(path) and os.path.abspath(path) != os.path.abspath(sys.executable):
                found.append(path)
    return found


def interrogate(python, timeout=180):
    try:
        done = subprocess.run([python, '-c', IDENTIFY], capture_output=True,
                              text=True, timeout=timeout)
    except Exception as exc:
        return {"exe": python, "error": f"{type(exc).__name__}: {exc}"}
    for line in (done.stdout or "").splitlines():
        if line.startswith("PIXAL3D_JSON:"):
            import json
            return json.loads(line[len("PIXAL3D_JSON:"):])
    return {"exe": python,
            "error": f"no answer (exit {done.returncode}) {(done.stderr or '')[-200:]}"}


def stack_of(info):
    if info.get("hip"):
        return "rocm"
    if info.get("cuda"):
        name = (info.get("name") or "").lower()
        if any(k in name for k in ("amd", "radeon", "gfx")):
            return "zluda"
        return "cuda"
    if info.get("torch"):
        return "cpu"
    return "unknown"


def describe(info):
    print(f"  {info.get('exe')}")
    if info.get("error"):
        print(f"    unreachable: {info['error']}")
        return None
    if info.get("torch_error"):
        print(f"    no torch: {info['torch_error']}")
        return None
    stack = stack_of(info)
    print(f"    python {info.get('py')}  torch {info.get('torch')}  stack: {stack}")
    if info.get("count_error"):
        print(f"    device query failed: {info['count_error']}")
    else:
        print(f"    devices {info.get('count')}  {info.get('name') or ''}"
              f"  {info.get('arch') or ''}")
    return stack


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--python', action='append', default=[],
                        help="Interpreter to inspect. Repeatable. Default: search.")
    parser.add_argument('--timeout', type=int, default=180)
    args = parser.parse_args()

    tee_to(os.path.join(ROOT, 'adopt_report.txt'))
    print("Pixal3D: what already works on this machine")
    print("=" * 43)

    targets = args.python or candidates()
    if not targets:
        print("\nFound no other Python environments to inspect. If ComfyUI (or")
        print("anything else) drives this GPU, point at its interpreter:")
        print(r'  python scripts\adopt_env.py --python "C:\ComfyUI\venv\Scripts\python.exe"')
        return 1

    print(f"\nInspecting {len(targets)} interpreter(s). Each runs in its own")
    print("process, so one that crashes does not stop the rest.\n")

    working = []
    for python in targets:
        info = interrogate(python, args.timeout)
        stack = describe(info)
        if stack in ("rocm", "zluda", "cuda") and info.get("count"):
            code = subprocess.run([python, '-c', WORK], capture_output=True,
                                  text=True, timeout=args.timeout)
            if 'COMPUTE_OK True' in (code.stdout or ""):
                print("    -> this one actually computes on the GPU")
                working.append((python, info, stack))
            else:
                print("    -> enumerates but cannot compute")
        print()

    if not working:
        print("Nothing here drives the GPU successfully, so there is no working")
        print("setup to copy. The driver remains the leading suspect.")
        return 1

    python, info, stack = working[0]
    print("=" * 43)
    print(f"Working setup: {stack}")
    print("=" * 43)
    for name, version in info.get("packages", []):
        print(f"  {name:34} {version}")
    print()

    if stack == "rocm":
        print("This is the same kind of stack Pixal3D wants, so it can be copied")
        print("exactly. In the Pixal3D folder:")
        print()
        pinned = [f"{n}=={v}" for n, v in info.get("packages", [])
                  if n.lower().startswith(("torch", "rocm"))]
        print(f"  .venv\\Scripts\\python.exe -m pip install {' '.join(pinned[:4])} ...")
        print()
        print("Those versions came from wheels, not PyPI, so the surest route is")
        print("to reuse that environment directly instead of rebuilding it:")
        print(f"  {python} inference.py --image assets\\images\\0_img.png --output out.glb")
        print()
        print("Pixal3D's own dependencies would need installing there first:")
        print(f"  {python} -m pip install -r requirements-rocm.txt")
    elif stack == "zluda":
        print("This is ZLUDA: a CUDA build of torch with a translation layer over")
        print("the system HIP SDK. It does not use the ROCm pip wheels at all,")
        print("which is why Pixal3D's install crashes while this one works --")
        print("the official ROCm nightlies have been unreliable on RDNA2.")
        print()
        print("Pixal3D detects and supports ZLUDA. The simplest path is to install")
        print("Pixal3D's dependencies into that environment and run from there:")
        print(f"  {python} -m pip install -r requirements-rocm.txt")
        print(f"  {python} -m pip install --no-deps git+https://github.com/microsoft/MoGe.git")
        print(f"  {python} scripts\\check_env.py")
        print(f"  {python} inference.py --image assets\\images\\0_img.png --output out.glb")
        print()
        print("Add --dtype float16 when you run: on RDNA2 there is no bfloat16")
        print("hardware, and ZLUDA reports no gfx target for auto to read.")
    else:
        print("That is an NVIDIA/CUDA setup, so it is not relevant to this port.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
