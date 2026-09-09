"""
Measure what this GPU is actually good at, so the tuning advice is checked
rather than assumed.

Three things decide how long a Pixal3D run takes:

* **GEMM throughput per dtype.** The DiT torsos are ~90% matmul. RDNA2 has
  packed fp16 but no bf16 instructions, so bf16 there is emulated and should
  measure roughly half of fp16. On RDNA3/4 and on CUDA the two should be level,
  and ``--dtype float16`` buys nothing.
* **Whether SDPA has a fused kernel.** Without flash or memory-efficient
  attention, torch falls back to the math path, which materialises the whole
  N x N score matrix. Long sparse sequences then become memory-bound.
* **Sparse convolution.** Only the VAE decoders use it, but they use it on the
  largest tensors in the pipeline.

    python scripts/benchmark.py
    python scripts/benchmark.py --tokens 16384 --quick
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pixal3d import runtime          # noqa: E402  (must precede the first allocation)

import torch                         # noqa: E402


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timeit(fn, warmup: int = 3, iters: int = 10) -> float:
    """Median-ish seconds per call, after a warmup."""
    for _ in range(warmup):
        fn()
    _sync()
    started = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - started) / iters


def supported_dtypes(device):
    if device == 'cpu':
        # CPU has no half-precision GEMM kernels worth the name; timing them
        # takes minutes and says nothing useful.
        return [('float32', torch.float32)]
    dtypes = [('float32', torch.float32), ('float16', torch.float16)]
    try:
        torch.zeros(8, 8, device=device, dtype=torch.bfloat16) @ \
            torch.zeros(8, 8, device=device, dtype=torch.bfloat16)
        dtypes.append(('bfloat16', torch.bfloat16))
    except Exception as exc:
        print(f"  bfloat16 unusable on this build: {type(exc).__name__}")
    return dtypes


def bench_gemm(device, tokens, iters):
    """A DiT MLP-shaped matmul: [tokens, 1536] x [1536, 8192]."""
    print(f"\nGEMM  [{tokens}, 1536] x [1536, 8192]  (one DiT MLP projection)")
    flops = 2 * tokens * 1536 * 8192
    results = {}
    for name, dtype in supported_dtypes(device):
        a = torch.randn(tokens, 1536, device=device, dtype=dtype)
        b = torch.randn(1536, 8192, device=device, dtype=dtype)
        try:
            seconds = timeit(lambda: a @ b, iters=iters)
        except RuntimeError as exc:
            print(f"  {name:<9} failed: {exc}")
            continue
        results[name] = flops / seconds / 1e12
        print(f"  {name:<9} {seconds * 1e3:8.2f} ms   {results[name]:7.2f} TFLOP/s")
        del a, b
        runtime.free_memory()

    if 'float16' in results and 'bfloat16' in results:
        ratio = results['float16'] / results['bfloat16']
        if ratio >= 1.25:
            print(f"  -> float16 is {ratio:.2f}x faster than bfloat16 here. "
                  f"Use --dtype float16.")
        elif ratio <= 0.8:
            print(f"  -> bfloat16 is {1 / ratio:.2f}x faster. Use --dtype bfloat16.")
        else:
            print(f"  -> float16 and bfloat16 are within {abs(1 - ratio) * 100:.0f}%. "
                  f"Keep --dtype auto; bfloat16 has more exponent range.")
    return results


def bench_attention(device, tokens, iters):
    """
    Self-attention over a sparse-stage-sized sequence: 12 heads x 128 channels.

    Also times the query-chunked wrapper, which trades a little arithmetic for a
    much smaller score matrix — the whole point when there is no fused kernel.
    """
    from pixal3d.compat.attention import chunked_sdpa, query_chunk_size

    heads, dim = 12, 128
    print(f"\nAttention  [1, {heads}, {tokens}, {dim}]  (sparse-stage self-attention)")
    backends = runtime.sdpa_backends()
    kind = ('flash' if backends['flash'] else
            'mem-efficient' if backends['mem_efficient'] else 'math (no fused kernel)')
    print(f"  torch SDPA backend: {kind}")
    print(f"  chunk size: {query_chunk_size(tokens, heads, dim)} queries")

    dtype = torch.float16 if device != 'cpu' else torch.float32
    flops = 4 * tokens * tokens * heads * dim
    for label, fn in (('sdpa', torch.nn.functional.scaled_dot_product_attention),
                      ('chunked_sdpa', chunked_sdpa)):
        q = torch.randn(1, heads, tokens, dim, device=device, dtype=dtype)
        try:
            seconds = timeit(lambda: fn(q, q, q), warmup=2, iters=max(3, iters // 3))
            peak = ''
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                fn(q, q, q)
                _sync()
                peak = f"   peak {torch.cuda.max_memory_allocated() / 2**30:5.2f} GiB"
            print(f"  {label:<13} {seconds * 1e3:8.2f} ms   "
                  f"{flops / seconds / 1e12:7.2f} TFLOP/s{peak}")
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            print(f"  {label:<13} failed: {type(exc).__name__}: {str(exc)[:80]}")
        del q
        runtime.free_memory()


def bench_sparse_conv(device, voxels, iters):
    """A submanifold 3x3x3 conv over a shell of occupied voxels, as the VAE does."""
    from pixal3d.modules.sparse import SparseTensor, SparseConv3d
    from pixal3d.modules.sparse import config as sparse_config

    print(f"\nSparse conv  3x3x3, 128 -> 128 channels, {voxels} voxels "
          f"(backend: {sparse_config.CONV})")
    resolution = 64
    coords = torch.randint(0, resolution, (voxels, 3), device=device)
    coords = torch.unique(coords, dim=0)
    coords = torch.cat([torch.zeros(coords.shape[0], 1, dtype=coords.dtype, device=device),
                        coords], dim=1).int()
    x = SparseTensor(feats=torch.randn(coords.shape[0], 128, device=device), coords=coords)
    conv = SparseConv3d(128, 128, 3).to(device)
    try:
        with torch.no_grad():
            # The first call builds the neighbour map; the cache is what the
            # decoder actually reuses, so time the warm path.
            seconds = timeit(lambda: conv(x), warmup=2, iters=max(3, iters // 3))
        print(f"  {seconds * 1e3:8.2f} ms   "
              f"{coords.shape[0] / seconds / 1e6:7.2f} M voxels/s")
    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        print(f"  failed: {type(exc).__name__}: {str(exc)[:100]}")
    runtime.free_memory()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tokens', type=int, default=8192,
                        help="Sequence length for the GEMM and attention benchmarks.")
    parser.add_argument('--voxels', type=int, default=100000,
                        help="Occupied voxels for the sparse-conv benchmark.")
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--quick', action='store_true', help="Fewer iterations.")
    parser.add_argument('--skip', nargs='*', default=[],
                        choices=['gemm', 'attention', 'conv'], help="Benchmarks to skip.")
    args = parser.parse_args()

    runtime.configure(vram='auto', verbose=False)
    device = runtime.get_device()
    print(runtime.describe())
    if device == 'cpu':
        print("No GPU detected — these numbers say nothing about your card.")
    iters = 3 if args.quick else args.iters

    if 'gemm' not in args.skip:
        bench_gemm(device, args.tokens, iters)
    if 'attention' not in args.skip:
        bench_attention(device, args.tokens, iters)
    if 'conv' not in args.skip:
        bench_sparse_conv(device, args.voxels, iters)

    print("\nWhat to do with this")
    print("  * Pick --dtype from the GEMM ratio above.")
    print("  * If SDPA reports 'math', attention is memory-bound: prefer 1024 over")
    print("    1536, keep --max_num_tokens down, and lower --steps before anything else.")
    print("  * A slow sparse conv shows up in the decode stage; see it with --timing.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
