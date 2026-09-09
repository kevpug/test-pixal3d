"""
Environment doctor for the ROCm / Windows port.

Run this first when something does not work - it reports what torch was built
against, whether the GPU is actually visible, which backends will be selected,
and whether the CPU packages the GLB fallback needs are installed. Finally it
runs a short GPU self-test that exercises the exact code paths a generation
uses, so a broken install fails here in seconds rather than twenty minutes into
a run.

    python scripts/check_env.py
    python scripts/check_env.py --no-gpu-test
"""

import argparse
import os
import platform
import textwrap
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OK = "  [ok]  "
WARN = "  [warn]"
BAD = "  [FAIL]"


def section(title):
    print()
    print(title)
    print("-" * len(title))


def check_torch():
    section("PyTorch")
    try:
        import torch
    except ImportError as exc:
        print(f"{BAD} torch is not installed ({exc})")
        return None

    print(f"{OK} torch {torch.__version__}")
    hip = getattr(torch.version, 'hip', None)
    cuda = getattr(torch.version, 'cuda', None)

    from pixal3d.compat import probe
    kind = probe.platform()
    labels = {
        'rocm': f"built for ROCm/HIP {hip} - the supported path on Windows AMD",
        'zluda': f"CUDA {cuda} build running on an AMD GPU (ZLUDA or similar shim)",
        'cuda': f"built for CUDA {cuda}",
        'directml': "torch-directml, not a torch.cuda backend",
        'cpu': "CPU-only build - generation will be unusably slow",
    }
    marker = OK if kind in ('rocm', 'cuda') else WARN
    print(f"{marker} {labels[kind]}")
    note = probe.accelerator_note() if kind != 'cpu' else None
    if note:
        for line in textwrap.wrap(note, 68):
            print(f"         {line}")

    # Enumerating devices can kill the process rather than raise: if the HIP
    # runtime hits an access violation or a missing DLL, Windows terminates the
    # interpreter and nothing after this point would run or print. So ask a
    # child process first, and only touch torch.cuda here if it survived.
    try:
        from gpu_probe import run_step, describe_exit
        code, out, err = run_step("import torch; print(torch.cuda.device_count())",
                                  verbose=False, timeout=180)
    except Exception:
        code, out, err = 0, "", ""
    if code not in (0, None):
        print(f"{BAD} device enumeration {describe_exit(code)}")
        print("         torch.cuda crashed the process instead of raising, so")
        print("         nothing below could have run. Full detail:")
        print("           python scripts\\gpu_probe.py --verbose")
        for line in (err or "").splitlines()[-8:]:
            print(f"         {line}")
        return torch

    # Every path below prints a verdict. An earlier version only looped over
    # device_count(), so a torch that reported is_available() = True with zero
    # devices said nothing at all about the GPU.
    try:
        available = bool(torch.cuda.is_available())
    except Exception as exc:
        available = False
        print(f"{BAD} torch.cuda.is_available() raised: {type(exc).__name__}: {exc}")
    try:
        count = int(torch.cuda.device_count())
    except Exception as exc:
        count = 0
        print(f"{BAD} torch.cuda.device_count() raised: {type(exc).__name__}: {exc}")

    if available and count:
        for i in range(count):
            try:
                props = torch.cuda.get_device_properties(i)
            except Exception as exc:
                print(f"{BAD} device {i}: properties unreadable "
                      f"({type(exc).__name__}: {exc})")
                continue
            reported = getattr(props, 'gcnArchName', None)
            arch = probe.gpu_arch() if i == 0 else None
            arch = reported.split(':')[0] if reported else (arch or f"sm_{props.major}{props.minor}")
            inferred = "" if reported else "  (inferred from the device name)"
            print(f"{OK} device {i}: {props.name} [{arch}]{inferred} "
                  f"{props.total_memory / 1024 ** 3:.1f} GB")
    else:
        if available and not count:
            # HIP loaded but enumerated nothing: usually a gfx target the wheel
            # carries no code objects for, or a visibility variable hiding it.
            print(f"{BAD} torch reports a GPU backend but zero devices "
                  f"(is_available=True, device_count=0)")
        else:
            print(f"{BAD} no GPU visible to torch "
                  f"(is_available={available}, device_count={count})")

        if kind == 'directml':
            print("         DirectML works for many image models but not for this one:")
            print("         it is a separate backend, and the sparse ops here need HIP.")
            print("         Make a second venv and run scripts/install_rocm_torch.py.")
        elif platform.system() == 'Windows' and hip:
            print("         Things to try, in order:")
            print("         1. Driver must be AMD Adrenalin 26.1.1 or newer.")
            print("         2. If the card is gfx1031/gfx1032 (RX 6700/6800M/6600),")
            print("            the wheels may only carry gfx1030 code objects:")
            print("              set HSA_OVERRIDE_GFX_VERSION=10.3.0")
            print("            Set it in the SAME cmd window, then run:")
            print("              .venv\\Scripts\\python.exe scripts\\check_env.py")
            print("            (double-clicking a .bat starts a fresh window that")
            print("             does not inherit it)")
            print("         3. Re-check that the SDK and torch wheels share a build date:")
            print("              .venv\\Scripts\\python.exe scripts\\install_rocm_torch.py --list")

    # Raw facts worth having in any bug report, whether or not a device showed up.
    hidden = {v: os.environ[v] for v in
              ('HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES',
               'GPU_DEVICE_ORDINAL', 'HSA_OVERRIDE_GFX_VERSION', 'HSA_ENABLE_SDMA')
              if v in os.environ}
    if hidden:
        for name, value in hidden.items():
            marker = WARN if name != 'HSA_OVERRIDE_GFX_VERSION' else OK
            print(f"{marker} {name}={value}")
        if 'HSA_OVERRIDE_GFX_VERSION' in hidden and available and count:
            print("         The GPU is being reported as a different target. Fine if it")
            print("         works, but if you get wrong output rather than errors, unset it.")
    elif not (available and count):
        print("         (no HIP/ROCR visibility variables are set)")

    return torch


def check_backends():
    section("Backends")
    from pixal3d.compat import probe

    info = probe.summary()
    print(f"{OK} attention: {info['attn_backend']}")
    print(f"{OK} sparse convolution: {info['sparse_conv_backend']}")

    for name, why in (
        ('flash_attn', 'faster attention'),
        ('flex_gemm', 'faster sparse convolution'),
        ('cumesh', 'GPU mesh cleanup, remeshing and UV unwrap'),
        ('o_voxel', 'GPU mesh extraction and GLB export'),
        ('nvdiffrast', 'GPU texture baking'),
    ):
        if info[name]:
            print(f"{OK} {name} available ({why})")
        else:
            print(f"{WARN} {name} missing - using the pure-PyTorch fallback ({why})")
    return info


def check_fallback_deps():
    section("Fallback dependencies")
    from pixal3d.compat import mesh_ops

    missing = mesh_ops.missing_dependencies()
    for pkg in ('trimesh', 'xatlas', 'fast-simplification'):
        if pkg in missing:
            print(f"{BAD} {pkg} missing - GLB export will fail")
        else:
            print(f"{OK} {pkg}")
    # Import name -> what to pass to pip, which differs for one of these.
    for mod, pkg in (('cv2', 'opencv-python-headless'),
                     ('transformers', 'transformers')):
        try:
            __import__(mod)
            print(f"{OK} {mod}")
        except ImportError:
            print(f"{BAD} {mod} missing")
            missing.append(pkg)
    if missing:
        print()
        print("         pip install " + " ".join(missing))

    # MoGe is optional and installs differently, so it is reported separately:
    # it only estimates the camera FOV, which --fov supplies by hand, and its
    # declared dependencies include flex-gemm, which cannot build here.
    try:
        __import__('moge')
        print(f"{OK} moge (camera FOV estimation)")
    except ImportError:
        print(f"{WARN} moge missing - FOV must be passed with --fov")
        print("         pip install --no-deps \\")
        print("           git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a")
    return missing


def gpu_self_test(torch):
    """Exercise the fallbacks on real device tensors, cheaply."""
    section("GPU self-test")
    if torch is None or not torch.cuda.is_available():
        print(f"{WARN} skipped (no GPU)")
        return False

    device = 'cuda'
    try:
        a = torch.randn(512, 512, device=device, dtype=torch.float16)
        assert bool((a @ a).float().isfinite().all())
        print(f"{OK} fp16 matmul")
    except Exception as exc:
        print(f"{BAD} fp16 matmul: {exc}")
        return False

    try:
        b = torch.randn(256, 256, device=device, dtype=torch.bfloat16)
        assert bool((b @ b).float().isfinite().all())
        print(f"{OK} bf16 matmul (the flow models are trained in bfloat16)")
    except Exception as exc:
        print(f"{BAD} bf16 matmul: {exc}")
        print("         The DiT checkpoints are bfloat16; this must work.")
        return False

    # Which of the two is actually fast decides --dtype. RDNA1/2 have packed
    # fp16 but no bf16 instructions, so bf16 there is emulated through fp32.
    try:
        import time
        rates = {}
        for name, dtype in (('fp16', torch.float16), ('bf16', torch.bfloat16)):
            x = torch.randn(4096, 4096, device=device, dtype=dtype)
            for _ in range(3):
                x @ x
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(5):
                x @ x
            torch.cuda.synchronize()
            rates[name] = 2 * 4096 ** 3 * 5 / (time.perf_counter() - started) / 1e12
            del x
        ratio = rates['fp16'] / rates['bf16'] if rates['bf16'] else 0
        print(f"{OK} matmul throughput: fp16 {rates['fp16']:.1f} TFLOP/s, "
              f"bf16 {rates['bf16']:.1f} TFLOP/s")
        if ratio >= 1.25:
            print(f"         fp16 is {ratio:.1f}x faster here - add --dtype float16")
    except Exception as exc:
        print(f"{WARN} could not time matmul throughput: {exc}")

    try:
        from pixal3d import runtime
        backends = runtime.sdpa_backends()
        if backends['flash'] or backends['mem_efficient']:
            names = [n for n in ('flash', 'mem_efficient') if backends[n]]
            print(f"{OK} fused attention available ({', '.join(names)})")
        else:
            print(f"{WARN} no fused attention kernel - torch will use the math path")
            print("         On ROCm this means AOTriton has no kernels for this GPU")
            print("         (RDNA2 among them). Attention becomes memory-bound; the")
            print("         chunked fallback below keeps it from blowing up VRAM.")
    except Exception as exc:
        print(f"{WARN} could not probe SDPA backends: {exc}")

    try:
        from pixal3d.compat.attention import chunked_sdpa
        q = torch.randn(1, 8, 4096, 64, device=device, dtype=torch.float16)
        out = chunked_sdpa(q, q, q)
        assert out.shape == q.shape and bool(out.isfinite().all())
        print(f"{OK} chunked attention (4096 tokens)")
    except Exception as exc:
        print(f"{BAD} chunked attention: {exc}")
        return False

    try:
        os.environ.setdefault('SPARSE_CONV_BACKEND', 'torch')
        from pixal3d.modules.sparse import SparseTensor
        from pixal3d.modules.sparse.conv import SparseConv3d
        n = 20000
        coords = torch.cat([
            torch.zeros(n, 1, dtype=torch.int32, device=device),
            torch.randint(0, 64, (n, 3), dtype=torch.int32, device=device),
        ], dim=1).unique(dim=0)
        x = SparseTensor(feats=torch.randn(coords.shape[0], 64, device=device), coords=coords)
        y = SparseConv3d(64, 64, 3).to(device)(x)
        assert bool(y.feats.isfinite().all())
        print(f"{OK} sparse convolution ({coords.shape[0]} voxels)")
    except Exception as exc:
        print(f"{BAD} sparse convolution: {exc}")
        return False

    try:
        from pixal3d.compat.uv_raster import uv_rasterize
        uvs = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], device=device)
        faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int32, device=device)
        verts = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]], device=device)
        ids, _ = uv_rasterize(uvs, faces, verts, 512)
        coverage = float((ids > 0).float().mean())
        assert coverage > 0.99, f"coverage {coverage:.2%}"
        print(f"{OK} UV rasteriser (coverage {coverage:.1%})")
    except Exception as exc:
        print(f"{BAD} UV rasteriser: {exc}")
        return False

    free, total = torch.cuda.mem_get_info()
    print(f"{OK} {free / 1024 ** 3:.1f} GB of {total / 1024 ** 3:.1f} GB free")
    return True


def main():
    parser = argparse.ArgumentParser(description="Diagnose a Pixal3D install")
    parser.add_argument("--no-gpu-test", action="store_true", help="Skip the device self-test")
    args = parser.parse_args()

    print(f"Pixal3D environment check - {platform.platform()}")
    print(f"Python {sys.version.split()[0]} at {sys.executable}")

    torch = check_torch()
    check_backends()
    missing = check_fallback_deps()

    passed = True
    if not args.no_gpu_test:
        passed = gpu_self_test(torch)

    section("Recommended settings")
    try:
        from pixal3d import runtime
        preset = runtime.pick_preset('auto')
        print(f"  preset          {preset.name}")
        print(f"  resolution      {preset.resolution}")
        print(f"  texture size    {preset.texture_size}")
        print(f"  decimation      {preset.decimation_target} faces")
        print(f"  low_vram        {preset.low_vram}")
        print(f"  block offload   {preset.block_offload}")
        print(f"  cfg batching    {preset.cfg_batch}")
        dtype = runtime.resolve_dtype('auto')
        if dtype is not None:
            print(f"  dtype           {str(dtype).replace('torch.', '')} "
                  f"(this GPU has no bfloat16 hardware)")
        print()
        flags = f"--vram {preset.name}"
        print(f"  python inference.py --image <img> --output out.glb {flags}")
        print(f"  python scripts/benchmark.py     # measure this GPU before tuning further")
    except Exception as exc:
        print(f"{WARN} could not compute a preset: {exc}")

    print()
    if missing or not passed:
        print("Some checks failed - see above.")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
