"""
Environment doctor for the ROCm / Windows port.

Run this first when something does not work — it reports what torch was built
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

    if not torch.cuda.is_available():
        print(f"{BAD} no GPU visible to torch")
        if kind == 'directml':
            print("         DirectML works for many image models but not for this one:")
            print("         it is a separate backend, and the sparse ops here need HIP.")
            print("         Make a second venv and run scripts/install_rocm_torch.py.")
        elif platform.system() == 'Windows' and hip:
            print("         Check that the AMD driver is 26.1.1 or newer and that the")
            print("         ROCm SDK wheels match the torch wheel's build date:")
            print("           python scripts\\install_rocm_torch.py --list")
            print("         If the card is gfx1031/gfx1032 and the wheels only carry")
            print("         gfx1030 code objects, try:")
            print("           set HSA_OVERRIDE_GFX_VERSION=10.3.0")
        return torch

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        reported = getattr(props, 'gcnArchName', None)
        arch = probe.gpu_arch() if i == 0 else None
        arch = reported.split(':')[0] if reported else (arch or f"sm_{props.major}{props.minor}")
        inferred = "" if reported else "  (inferred from the device name)"
        print(f"{OK} device {i}: {props.name} [{arch}]{inferred} "
              f"{props.total_memory / 1024 ** 3:.1f} GB")
    override = os.environ.get('HSA_OVERRIDE_GFX_VERSION')
    if override:
        print(f"{WARN} HSA_OVERRIDE_GFX_VERSION={override} is set - the GPU is being")
        print("         reported as a different target. Fine if it works, but if you")
        print("         get wrong results rather than errors, unset it first.")
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
            print(f"{WARN} {name} missing — using the pure-PyTorch fallback ({why})")
    return info


def check_fallback_deps():
    section("Fallback dependencies")
    from pixal3d.compat import mesh_ops

    missing = mesh_ops.missing_dependencies()
    for pkg in ('trimesh', 'xatlas', 'fast-simplification'):
        if pkg in missing:
            print(f"{BAD} {pkg} missing — GLB export will fail")
        else:
            print(f"{OK} {pkg}")
    # Import name -> what to pass to pip, which differs for two of these.
    for mod, pkg in (('cv2', 'opencv-python-headless'),
                     ('transformers', 'transformers'),
                     ('moge', 'git+https://github.com/microsoft/MoGe.git')):
        try:
            __import__(mod)
            print(f"{OK} {mod}")
        except ImportError:
            print(f"{BAD} {mod} missing")
            missing.append(pkg)
    if missing:
        print()
        print("         pip install " + " ".join(missing))
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
            print(f"         fp16 is {ratio:.1f}x faster here — add --dtype float16")
    except Exception as exc:
        print(f"{WARN} could not time matmul throughput: {exc}")

    try:
        from pixal3d import runtime
        backends = runtime.sdpa_backends()
        if backends['flash'] or backends['mem_efficient']:
            names = [n for n in ('flash', 'mem_efficient') if backends[n]]
            print(f"{OK} fused attention available ({', '.join(names)})")
        else:
            print(f"{WARN} no fused attention kernel — torch will use the math path")
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

    print(f"Pixal3D environment check — {platform.platform()}")
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
        print("Some checks failed — see above.")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
