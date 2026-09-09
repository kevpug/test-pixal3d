"""
Correctness tests for the pure-PyTorch fallbacks.

Each fallback is checked against an independent reference — a dense
``F.conv3d`` for the sparse convolution, a Python loop for the trilinear
sampler, a brute-force dictionary for the dual-grid quad lookup — so a pass
means the fallback computes the same thing as the CUDA extension it replaces,
not merely that it runs.

Runs on the GPU when one is visible, otherwise on the CPU. No pytest needed:

    python tests/test_fallbacks.py
    python tests/test_fallbacks.py --device cpu
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('SPARSE_CONV_BACKEND', 'torch')
os.environ.setdefault('ATTN_BACKEND', 'sdpa')

import torch
import torch.nn.functional as F

from pixal3d.compat.attention import chunked_sdpa, query_chunk_size
from pixal3d.compat.dual_grid import flexible_dual_grid_to_mesh
from pixal3d.compat.grid_sample import grid_sample_3d
from pixal3d.compat.hashgrid import CoordLookup, MISS
from pixal3d.compat.uv_raster import uv_rasterize


FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  {status}  {name}" + (f"   ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def section(title):
    print()
    print(title)


# ---------------------------------------------------------------- hashgrid --
def test_hashgrid(device):
    section("Coordinate lookup (replaces the o_voxel / flex_gemm hash maps)")
    w = h = d = 16
    n = 400
    flat = torch.randperm(w * h * d, device=device)[:n]
    coords = torch.stack([torch.zeros(n, dtype=torch.long, device=device),
                          flat // (h * d), (flat // d) % h, flat % d], dim=1)
    lut = CoordLookup(coords, (w, h, d))
    check("self lookup is the identity", bool((lut.lookup(coords) == torch.arange(n, device=device)).all()))

    present = set(flat.tolist())
    absent_flat = torch.tensor([x for x in range(w * h * d) if x not in present][:50], device=device)
    absent = torch.stack([torch.zeros(len(absent_flat), dtype=torch.long, device=device),
                          absent_flat // (h * d), (absent_flat // d) % h, absent_flat % d], dim=1)
    check("absent coordinates miss", bool((lut.lookup(absent) == MISS).all()))

    oob = torch.tensor([[0, -1, 0, 0], [0, w, 0, 0], [0, 0, h, 0], [0, 0, 0, d], [1, 0, 0, 0]],
                       device=device)
    check("out-of-range coordinates miss", bool((lut.lookup(oob) == MISS).all()))


# ------------------------------------------------------------- grid_sample --
def _reference_trilinear(feats, coords, shape, points):
    """Straight transcription of the CUDA kernel's arithmetic, in Python."""
    w, h, d = shape
    index = {tuple(c): i for i, c in enumerate(coords.tolist())}
    out = []
    for p in points.tolist():
        acc = torch.zeros(feats.shape[1])
        total = 0.0
        for dx in (-0.5, 0.5):
            for dy in (-0.5, 0.5):
                for dz in (-0.5, 0.5):
                    nc = [int(p[0] + dx), int(p[1] + dy), int(p[2] + dz)]
                    weight = 1.0
                    for k in range(3):
                        weight *= max(0.0, 1.0 - abs(nc[k] + 0.5 - p[k]))
                    key = (0, nc[0], nc[1], nc[2])
                    if key in index and 0 <= nc[0] < w and 0 <= nc[1] < h and 0 <= nc[2] < d:
                        acc += weight * feats[index[key]]
                        total += weight
        out.append(acc / max(total, 1e-12))
    return torch.stack(out)


def test_grid_sample(device):
    section("Sparse volume sampling (replaces flex_gemm.grid_sample_3d)")
    w = h = d = 8
    channels = 3
    gx, gy, gz = torch.meshgrid(torch.arange(w), torch.arange(h), torch.arange(d), indexing='ij')
    coords = torch.stack([torch.zeros_like(gx), gx, gy, gz], -1).reshape(-1, 4).to(device)
    feats = torch.randn(w * h * d, channels, device=device)
    shape = torch.Size([1, channels, w, h, d])
    points = (torch.rand(64, 3, device=device) * torch.tensor([w - 1., h - 1., d - 1.], device=device) + 0.5)

    got = grid_sample_3d(feats, coords, shape, points.unsqueeze(0))[0]
    expected = _reference_trilinear(feats.cpu(), coords.cpu(), (w, h, d), points.cpu()).to(device)
    check("trilinear matches the reference", torch.allclose(got, expected, atol=1e-4),
          f"max err {float((got - expected).abs().max()):.2e}")

    keep = torch.rand(w * h * d, device=device) > 0.4
    sparse_coords, sparse_feats = coords[keep], feats[keep]
    got = grid_sample_3d(sparse_feats, sparse_coords, shape, points.unsqueeze(0))[0]
    expected = _reference_trilinear(sparse_feats.cpu(), sparse_coords.cpu(), (w, h, d), points.cpu()).to(device)
    check("trilinear over a sparse volume", torch.allclose(got, expected, atol=1e-4),
          f"max err {float((got - expected).abs().max()):.2e}")

    got = grid_sample_3d(feats, coords, shape, points.unsqueeze(0), mode='nearest')[0]
    expected = torch.stack([feats[int(p[0]) * h * d + int(p[1]) * d + int(p[2])] for p in points])
    check("nearest matches the reference", torch.allclose(got, expected, atol=1e-5))

    import pixal3d.compat.grid_sample as module
    original, module._CHUNK = module._CHUNK, 7
    chunked = grid_sample_3d(sparse_feats, sparse_coords, shape, points.unsqueeze(0))[0]
    module._CHUNK = original
    whole = grid_sample_3d(sparse_feats, sparse_coords, shape, points.unsqueeze(0))[0]
    check("chunking does not change the result", torch.allclose(chunked, whole, atol=1e-5))

    grad_feats = sparse_feats.clone().requires_grad_(True)
    grid_sample_3d(grad_feats, sparse_coords, shape, points.unsqueeze(0)).sum().backward()
    check("gradients flow to the features",
          grad_feats.grad is not None and bool(torch.isfinite(grad_feats.grad).all()))


# ---------------------------------------------------------------- uv_raster --
def test_uv_rasterize(device):
    section("UV rasteriser (replaces nvdiffrast for texture baking)")
    size = 64
    uvs = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], device=device)
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int32, device=device)
    verts = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]], device=device)

    ids, pos = uv_rasterize(uvs, faces, verts, size)
    check("a full-atlas quad covers every texel", bool((ids > 0).all()),
          f"{int((ids > 0).sum())}/{size * size}")

    rows, cols = torch.meshgrid(torch.arange(size, device=device),
                                torch.arange(size, device=device), indexing='ij')
    check("row is V and column is U",
          torch.allclose(pos[..., 0], cols.float() / (size - 1), atol=2e-3)
          and torch.allclose(pos[..., 1], rows.float() / (size - 1), atol=2e-3))

    small_uv = torch.tensor([[0.1, 0.1], [0.3, 0.1], [0.1, 0.3]], device=device)
    small_face = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)
    small_vert = torch.tensor([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.]], device=device)
    ids2, pos2 = uv_rasterize(small_uv, small_face, small_vert, size)
    covered = ids2 > 0
    r, c = covered.nonzero(as_tuple=True)
    inside = (int(r.min()) >= math.floor(0.1 * (size - 1)) and int(r.max()) <= math.ceil(0.3 * (size - 1))
              and int(c.min()) >= math.floor(0.1 * (size - 1)) and int(c.max()) <= math.ceil(0.3 * (size - 1)))
    check("a small triangle stays inside its bounding box", bool(covered.any()) and inside)
    check("face ids are one-based", int(ids2.max()) == 1)
    check("positions are interpolated",
          torch.allclose(pos2[covered][:, 2], torch.ones(int(covered.sum()), device=device), atol=1e-5))

    import pixal3d.compat.uv_raster as module
    original, module._PAIR_BUDGET = module._PAIR_BUDGET, 64
    ids3, pos3 = uv_rasterize(uvs, faces, verts, size)
    module._PAIR_BUDGET = original
    check("chunking does not change the result",
          bool((ids3 > 0).all()) and torch.allclose(pos3, pos, atol=1e-5))

    degenerate, _ = uv_rasterize(torch.full((3, 2), 0.5, device=device),
                                 torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device),
                                 torch.zeros(3, 3, device=device), size)
    check("a degenerate triangle is skipped", int((degenerate > 0).sum()) <= 1)


# ---------------------------------------------------------------- dual grid --
def test_dual_grid(device):
    section("Dual-grid mesh extraction (replaces o_voxel)")
    res = 8
    gx, gy, gz = torch.meshgrid(torch.arange(res), torch.arange(res), torch.arange(res), indexing='ij')
    grid = torch.stack([gx, gy, gz], -1).reshape(-1, 3)
    radius = (grid.float() - (res - 1) / 2).norm(dim=1)
    coords = grid[(radius > 1.5) & (radius < 3.0)].int().to(device)
    n = coords.shape[0]
    dual = torch.rand(n, 3, device=device)
    intersected = torch.rand(n, 3, device=device) > 0.5
    weight = torch.rand(n, 1, device=device) + 0.1

    verts, faces = flexible_dual_grid_to_mesh(
        coords, dual, intersected, weight,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], grid_size=res)
    check("vertex count matches the voxel count", verts.shape == (n, 3))
    check("indices are in range", bool((faces >= 0).all()) and bool((faces < n).all()))

    # Brute-force the quad lookup the fallback does with searchsorted.
    occupied = {tuple(c) for c in coords.tolist()}
    offsets = [[[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
               [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
               [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]]
    expected_quads = 0
    for i, c in enumerate(coords.tolist()):
        for axis in range(3):
            if not bool(intersected[i, axis]):
                continue
            corners = [tuple(c[k] + offsets[axis][j][k] for k in range(3)) for j in range(4)]
            if all(x in occupied for x in corners):
                expected_quads += 1
    check("quad count matches brute force", faces.shape[0] == 2 * expected_quads,
          f"{faces.shape[0]} vs {2 * expected_quads}")

    verts_t, faces_t = flexible_dual_grid_to_mesh(
        coords, dual, intersected, weight,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], grid_size=res, train=True)
    check("training mode adds a centre vertex per quad",
          verts_t.shape[0] == n + expected_quads and faces_t.shape[0] == 4 * expected_quads)


# ---------------------------------------------------------------- attention --
def test_attention(device):
    section("Chunked attention (bounds memory on the SDPA path)")
    b, heads, length, dim = 2, 4, 500, 32
    q, k, v = (torch.randn(b, heads, length, dim, device=device) for _ in range(3))

    os.environ['PIXAL3D_ATTN_CHUNK'] = '64'
    got = chunked_sdpa(q, k, v)
    expected = F.scaled_dot_product_attention(q, k, v)
    check("chunked output matches unchunked", torch.allclose(got, expected, atol=1e-4),
          f"max err {float((got - expected).abs().max()):.2e}")

    mask = torch.rand(b, 1, 1, length, device=device) > 0.3
    check("key masks are honoured",
          torch.allclose(chunked_sdpa(q, k, v, attn_mask=mask),
                         F.scaled_dot_product_attention(q, k, v, attn_mask=mask), atol=1e-4))

    full_mask = torch.rand(b, 1, length, length, device=device) > 0.3
    full_mask[..., 0] = True
    check("query-dimension masks are sliced",
          torch.allclose(chunked_sdpa(q, k, v, attn_mask=full_mask),
                         F.scaled_dot_product_attention(q, k, v, attn_mask=full_mask), atol=1e-4))
    del os.environ['PIXAL3D_ATTN_CHUNK']

    os.environ['PIXAL3D_ATTN_BUDGET'] = str(4 * 500)
    check("the chunk size follows the budget", query_chunk_size(4, 500, 5000) == 128)
    del os.environ['PIXAL3D_ATTN_BUDGET']

    from pixal3d.modules.attention import config as attention_config
    from pixal3d.modules.attention import full_attn
    qkv = torch.randn(2, 128, 3, 4, 32, device=device)
    attention_config.set_backend('sdpa')
    with_sdpa = full_attn.scaled_dot_product_attention(qkv)
    attention_config.set_backend('naive')
    with_naive = full_attn.scaled_dot_product_attention(qkv)
    attention_config.set_backend('sdpa')
    check("dense attention matches the naive implementation",
          torch.allclose(with_sdpa, with_naive, atol=1e-4),
          f"max err {float((with_sdpa - with_naive).abs().max()):.2e}")

    from pixal3d.modules.sparse import SparseTensor
    from pixal3d.modules.sparse.attention.full_attn import sparse_scaled_dot_product_attention
    lengths = [37, 91]
    coords = torch.cat([
        torch.cat([torch.full((n, 1), i, dtype=torch.int32, device=device),
                   torch.stack([torch.arange(n, device=device) % 8,
                                torch.arange(n, device=device) // 8 % 8,
                                torch.arange(n, device=device) // 64], -1).int()], 1)
        for i, n in enumerate(lengths)])
    feats = torch.randn(sum(lengths), 3, 4, 32, device=device)
    out = sparse_scaled_dot_product_attention(SparseTensor(feats=feats, coords=coords)).feats
    reference, offset = [], 0
    for n in lengths:
        qi, ki, vi = feats[offset:offset + n].unbind(dim=1)
        reference.append(F.scaled_dot_product_attention(
            qi.permute(1, 0, 2).unsqueeze(0), ki.permute(1, 0, 2).unsqueeze(0),
            vi.permute(1, 0, 2).unsqueeze(0))[0].permute(1, 0, 2))
        offset += n
    reference = torch.cat(reference, 0)
    check("variable-length sparse attention", torch.allclose(out, reference, atol=1e-4),
          f"max err {float((out - reference).abs().max()):.2e}")


# ------------------------------------------------------------ sparse conv ---
def test_sparse_conv(device):
    section("Sparse convolution (replaces flex_gemm)")
    from pixal3d.modules.sparse import SparseTensor
    from pixal3d.modules.sparse.conv import SparseConv3d

    def case(kernel, dilation, in_ch, out_ch, bias=True, w=10, h=9, d=8, density=0.35):
        gx, gy, gz = torch.meshgrid(torch.arange(w), torch.arange(h), torch.arange(d), indexing='ij')
        grid = torch.stack([gx, gy, gz], -1).reshape(-1, 3)
        keep = torch.rand(grid.shape[0]) < density
        c3 = grid[keep].to(device)
        coords = torch.cat([torch.zeros(c3.shape[0], 1, dtype=torch.int32, device=device),
                            c3.int()], dim=1)
        feats = torch.randn(coords.shape[0], in_ch, device=device)

        conv = SparseConv3d(in_ch, out_ch, kernel, dilation=dilation, bias=bias).to(device)
        got = conv(SparseTensor(feats=feats, coords=coords)).feats

        # Submanifold convolution == dense convolution read at the occupied voxels.
        dense = torch.zeros(1, in_ch, w, h, d, device=device)
        dense[0, :, c3[:, 0], c3[:, 1], c3[:, 2]] = feats.t()
        weight = conv.weight.detach().permute(0, 4, 1, 2, 3).contiguous()
        reference = F.conv3d(dense, weight, conv.bias, padding=dilation * (kernel // 2),
                             dilation=dilation)[0, :, c3[:, 0], c3[:, 1], c3[:, 2]].t()
        return float((got - reference).abs().max().detach())

    for kernel, dilation, in_ch, out_ch in ((3, 1, 8, 16), (3, 2, 6, 6), (1, 1, 5, 7), (5, 1, 4, 3)):
        err = case(kernel, dilation, in_ch, out_ch)
        check(f"kernel {kernel}, dilation {dilation}, {in_ch}->{out_ch}", err < 2e-3,
              f"max err {err:.2e}")
    check("without bias", case(3, 1, 8, 8, bias=False) < 2e-3)

    import pixal3d.modules.sparse.conv.conv_torch as module
    saved = module._BLOCK_ELEMS, module._NEIGHBOR_STEP
    module._BLOCK_ELEMS, module._NEIGHBOR_STEP = 64, 13
    err = case(3, 1, 8, 16)
    module._BLOCK_ELEMS, module._NEIGHBOR_STEP = saved
    check("blocking does not change the result", err < 2e-3, f"max err {err:.2e}")

    check("weight layout matches flex_gemm's checkpoints",
          tuple(SparseConv3d(4, 6, 3).weight.shape) == (6, 3, 3, 3, 4))


# ---------------------------------------------------------------- pipeline --
def test_model_stack(device):
    section("Model stack (the released architectures, in miniature)")
    from pixal3d.models.sc_vaes.fdg_vae import FlexiDualGridVaeDecoder
    from pixal3d.models.sparse_structure_flow import SparseStructureFlowModel
    from pixal3d.models.structured_latent_flow import SLatFlowModel
    from pixal3d.modules.sparse import SparseTensor

    coords3 = torch.randint(0, 16, (400, 3), dtype=torch.int32, device=device).unique(dim=0)
    coords = torch.cat([torch.zeros(coords3.shape[0], 1, dtype=torch.int32, device=device),
                        coords3], dim=1)
    timestep = torch.tensor([0.5], device=device)
    cond_channels = 32

    decoder = FlexiDualGridVaeDecoder(
        resolution=32, model_channels=[32, 16], latent_channels=8, num_blocks=[1, 0],
        block_type=["SparseConvNeXtBlock3d"] * 2, up_block_type=["SparseResBlockC2S3d"],
        block_args=[{}, {}], use_fp16=False).to(device).eval()
    with torch.no_grad():
        decoded = decoder(SparseTensor(feats=torch.randn(coords.shape[0], 8, device=device),
                                       coords=coords))
    mesh = (decoded[0] if isinstance(decoded, tuple) else decoded)[0]
    check("shape decoder produces a mesh",
          mesh.faces.shape[0] > 0 and bool(torch.isfinite(mesh.vertices).all()),
          f"{mesh.vertices.shape[0]} verts, {mesh.faces.shape[0]} faces")

    slat = SLatFlowModel(
        resolution=16, in_channels=8, out_channels=8, model_channels=64,
        cond_channels=cond_channels, num_blocks=2, num_heads=4, mlp_ratio=4, pe_mode="rope",
        share_mod=True, qk_rms_norm=True, qk_rms_norm_cross=True,
        image_attn_mode="proj", dtype="float32").to(device).eval()
    cond = {'global': torch.randn(1, 12, cond_channels, device=device),
            'proj': SparseTensor(feats=torch.randn(coords.shape[0], cond_channels, device=device),
                                 coords=coords)}
    with torch.no_grad():
        out = slat(SparseTensor(feats=torch.randn(coords.shape[0], 8, device=device),
                                coords=coords), timestep, cond)
    check("structured-latent flow model runs", bool(torch.isfinite(out.feats).all()),
          str(tuple(out.feats.shape)))

    structure = SparseStructureFlowModel(
        resolution=8, in_channels=8, out_channels=8, model_channels=64,
        cond_channels=cond_channels, num_blocks=2, num_heads=4, mlp_ratio=4, pe_mode="rope",
        share_mod=True, qk_rms_norm=True, qk_rms_norm_cross=True,
        image_attn_mode="proj").to(device).eval()
    with torch.no_grad():
        out = structure(torch.randn(1, 8, 8, 8, 8, device=device), timestep,
                        {'global': torch.randn(1, 12, cond_channels, device=device),
                         'proj': torch.randn(1, 512, cond_channels, device=device)})
    check("sparse-structure flow model runs", bool(torch.isfinite(out).all()),
          str(tuple(out.shape)))


# -------------------------------------------------------------- glb export --
def test_glb_export(device):
    section("GLB export (replaces o_voxel.postprocess + cumesh + nvdiffrast)")
    from pixal3d.compat import mesh_ops
    missing = mesh_ops.missing_dependencies()
    if missing:
        print(f"  SKIP  needs: {', '.join(missing)}")
        return

    import numpy as np
    import trimesh
    from pixal3d.compat.postprocess import to_glb

    res = 64
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=0.35)
    verts = torch.tensor(np.asarray(sphere.vertices), dtype=torch.float32, device=device)
    faces = torch.tensor(np.asarray(sphere.faces), dtype=torch.int32, device=device)

    aabb = np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
    samples, _ = trimesh.sample.sample_surface(sphere, 200000)
    voxels = np.unique(np.floor((samples - aabb[0]) * res).astype(np.int64), axis=0)
    voxels = voxels[(voxels >= 0).all(1) & (voxels < res).all(1)]
    coords = torch.tensor(voxels, dtype=torch.int32, device=device)
    centres = (coords.float() + 0.5) / res + torch.tensor(aabb[0], dtype=torch.float32, device=device)
    attrs = torch.cat([(centres / 0.35 * 0.5 + 0.5).clamp(0, 1),
                       torch.full((len(coords), 1), 0.25, device=device),
                       torch.full((len(coords), 1), 0.60, device=device),
                       torch.ones(len(coords), 1, device=device)], dim=1)
    layout = {'base_color': slice(0, 3), 'metallic': slice(3, 4),
              'roughness': slice(4, 5), 'alpha': slice(5, 6)}

    glb = to_glb(vertices=verts, faces=faces, attr_volume=attrs, coords=coords,
                 attr_layout=layout, aabb=aabb.tolist(), grid_size=res,
                 decimation_target=3000, texture_size=256, remesh=True)

    check("decimated to the target", 100 < len(glb.faces) <= 3200, f"{len(glb.faces)} faces")
    check("every vertex has a UV",
          glb.visual.uv is not None and len(glb.visual.uv) == len(glb.vertices))
    texture = np.asarray(glb.visual.material.baseColorTexture)
    check("texture is the requested size", texture.shape[:2] == (256, 256), str(texture.shape))
    check("texture carries detail", texture[..., :3].std() > 10,
          f"std {texture[..., :3].std():.1f}")
    dark = float((texture[..., :3].max(axis=-1) < 8).mean())
    check("no unbaked black regions", dark < 0.02, f"{dark:.2%} near-black")
    metallic_roughness = np.asarray(glb.visual.material.metallicRoughnessTexture)
    check("metallic baked into blue",
          abs(float(np.median(metallic_roughness[..., 2])) - 0.25 * 255) < 25)
    check("roughness baked into green",
          abs(float(np.median(metallic_roughness[..., 1])) - 0.60 * 255) < 25)
    radii = np.linalg.norm(glb.vertices, axis=1)
    check("geometry survives the round trip", abs(radii.mean() - 0.35) < 0.02,
          f"mean radius {radii.mean():.4f}")

    # Cleanup is what keeps stray speckles out of the exported mesh. The speck
    # is tiny on purpose: its area has to fall under the component threshold.
    from pixal3d.compat import mesh_ops
    speck = trimesh.creation.icosphere(subdivisions=1, radius=5e-4)
    speck.apply_translation([0.45, 0.0, 0.0])
    dirty = trimesh.util.concatenate([sphere, speck])
    dirty_faces = np.asarray(dirty.faces, dtype=np.int64)
    with_duplicates = np.concatenate([dirty_faces, dirty_faces[:20]])
    _, cleaned = mesh_ops.clean_mesh(np.asarray(dirty.vertices, dtype=np.float32),
                                     with_duplicates)
    check("cleanup removes duplicate faces", len(cleaned) <= len(dirty_faces))
    check("cleanup drops small components", len(cleaned) == len(sphere.faces),
          f"{len(with_duplicates)} -> {len(cleaned)} (body has {len(sphere.faces)})")


# ------------------------------------------------------------------ runtime --
def test_runtime(device):
    section("Runtime (VRAM presets and block offloading)")
    import torch.nn as nn
    from pixal3d import runtime

    names = list(runtime.PRESETS)
    check("presets are ordered largest to smallest",
          [runtime.PRESETS[n].min_vram_gb for n in names]
          == sorted((runtime.PRESETS[n].min_vram_gb for n in names), reverse=True))
    check("an explicit preset is returned as-is", runtime.pick_preset('8gb').name == '8gb')
    check("cfg batching is off below 16 GB",
          not any(runtime.PRESETS[n].cfg_batch for n in ('12gb', '8gb', '6gb')))

    check("explicit dtypes resolve", (runtime.resolve_dtype('float16') is torch.float16
                                      and runtime.resolve_dtype('bfloat16') is torch.bfloat16
                                      and runtime.resolve_dtype('float32') is torch.float32))
    try:
        runtime.resolve_dtype('int4')
        check("an unknown dtype raises", False)
    except ValueError:
        check("an unknown dtype raises", True)
    # 'auto' means float16 only where the GPU has no bfloat16 instructions.
    import pixal3d.compat.probe as probe
    saved_arch = probe.gpu_arch
    try:
        probe.gpu_arch = lambda: 'gfx1031'
        check("auto picks float16 on RDNA2", runtime.resolve_dtype('auto') is torch.float16)
        probe.gpu_arch = lambda: 'gfx1201'
        check("auto keeps the checkpoint dtype elsewhere", runtime.resolve_dtype('auto') is None)
        check("cond float32 means no autocast", runtime.resolve_cond_dtype('float32') is None)
    finally:
        probe.gpu_arch = saved_arch

    class ToyFlow(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Linear(4, 4)])
            self.dtype = torch.float32

        def convert_to(self, dtype):
            self.dtype = dtype
            self.blocks.to(dtype)

    class ToyPipeline:
        def __init__(self):
            self.models = {'shape_slat_flow_model_512': ToyFlow(),
                           'shape_slat_decoder': nn.Linear(4, 4)}

    pipe = ToyPipeline()
    converted = runtime.apply_dtype(pipe, torch.float16, verbose=False)
    check("only the flow models are converted", converted == 1)
    check("the flow torso changed dtype",
          pipe.models['shape_slat_flow_model_512'].blocks[0].weight.dtype == torch.float16)
    check("the decoder is left alone",
          pipe.models['shape_slat_decoder'].weight.dtype == torch.float32)
    check("no dtype requested is a no-op", runtime.apply_dtype(pipe, None) == 0)
    check("auto returns a preset", runtime.pick_preset('auto').name in names)
    try:
        runtime.pick_preset('nonsense')
        check("an unknown preset raises", False)
    except ValueError:
        check("an unknown preset raises", True)

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Linear(4, 4)
            self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
            self.head = nn.Linear(4, 4)

        def forward(self, x):
            x = self.stem(x)
            for block in self.blocks:
                x = block(x)
            return self.head(x)

    model = Toy().to(device).eval()
    sample = torch.randn(2, 4, device=device)
    with torch.no_grad():
        before = model(sample)

    check("offloading engages", runtime.enable_block_offload(model, device=device))
    check("blocks leave the module tree", 'blocks' not in model._modules)
    check("blocks stay reachable", len(model.blocks) == 3)
    with torch.no_grad():
        after = model(sample)
    # The hooks must return None; a pre-hook that returns the module would be
    # taken as replacement inputs and the forward would fail outright.
    check("the forward result is unchanged", torch.allclose(before, after))
    check("enabling twice is a no-op", runtime.enable_block_offload(model, device=device))
    model.to(device)
    check("a later .to() does not re-attach the blocks", 'blocks' not in model._modules)
    with torch.no_grad():
        check("still correct after .to()", torch.allclose(model(sample), before))


# --------------------------------------------------------- guidance batching --
def test_cfg_batch(device):
    section("Classifier-free guidance batching")
    import os
    from pixal3d.modules.sparse import SparseTensor
    from pixal3d.pipelines.samplers.flow_euler import FlowEulerCfgSampler

    sampler = FlowEulerCfgSampler(sigma_min=1e-5)

    class DenseModel:
        """A batch-independent function, so batching must not change results."""
        def __call__(self, x, t, cond, **kwargs):
            g = cond['global']
            return torch.tanh(x * 1.3 + g[:, :, None, None, None] + t[:, None, None, None, None] * 1e-3)

    x_t = torch.randn(2, 4, 3, 3, 3, device=device)
    cond = {'global': torch.randn(2, 4, device=device)}
    neg_cond = {'global': torch.zeros(2, 4, device=device)}

    def run(batched):
        os.environ['PIXAL3D_CFG_BATCH'] = '1' if batched else '0'
        sampler._pixal3d_cfg_cache = None
        return sampler._inference_model(DenseModel(), x_t, 0.4, cond, neg_cond,
                                        guidance_strength=3.0)

    ref, got = run(False), run(True)
    check("dense guidance matches", torch.allclose(ref, got, atol=1e-6),
          f"max diff {(ref - got).abs().max().item():.2e}")

    coords = torch.stack([
        torch.tensor([0, 0, 0, 1, 1, 1, 1], device=device),
        torch.tensor([0, 1, 2, 0, 1, 2, 3], device=device),
        torch.zeros(7, dtype=torch.long, device=device),
        torch.zeros(7, dtype=torch.long, device=device),
    ], dim=1).int()
    sp_x = SparseTensor(feats=torch.randn(7, 4, device=device), coords=coords)
    concat = SparseTensor(feats=torch.randn(7, 2, device=device), coords=coords)

    class SparseModel:
        def __call__(self, x, t, cond, concat_cond=None, **kwargs):
            feats = x.feats
            if concat_cond is not None:
                check_shape.append((feats.shape[0], concat_cond.feats.shape[0]))
                feats = torch.cat([feats, concat_cond.feats], dim=-1)[:, :4]
            # Per-row conditioning, so a mis-aligned batch would show up.
            g = cond['global'][x.coords[:, 0].long()]
            return x.replace(torch.tanh(feats * 1.1 + g))

    check_shape = []
    sp_cond = {'global': torch.randn(2, 4, device=device)}
    sp_neg = {'global': torch.zeros(2, 4, device=device)}

    def run_sparse(batched):
        os.environ['PIXAL3D_CFG_BATCH'] = '1' if batched else '0'
        sampler._pixal3d_cfg_cache = None
        return sampler._inference_model(SparseModel(), sp_x, 0.4, sp_cond, sp_neg,
                                        guidance_strength=3.0, concat_cond=concat)

    ref, got = run_sparse(False), run_sparse(True)
    check("sparse guidance matches", torch.allclose(ref.feats, got.feats, atol=1e-6),
          f"max diff {(ref.feats - got.feats).abs().max().item():.2e}")
    check("sparse rows keep their coordinates", torch.equal(ref.coords, got.coords))
    check("concat_cond is duplicated to match", check_shape[-1] == (14, 14),
          f"{check_shape[-1]}")

    # The batched skeleton is built once per run, so the windowed-attention
    # serialisation cached on it survives across denoising steps.
    os.environ['PIXAL3D_CFG_BATCH'] = '1'
    sampler._pixal3d_cfg_cache = None
    sampler._inference_model(SparseModel(), sp_x, 0.4, sp_cond, sp_neg,
                             guidance_strength=3.0, concat_cond=concat)
    template = sampler._pixal3d_cfg_cache['template']
    template.register_spatial_cache('probe', 'kept')
    stepped = sp_x.replace(sp_x.feats * 0.9)
    sampler._inference_model(SparseModel(), stepped, 0.4, sp_cond, sp_neg,
                             guidance_strength=3.0, concat_cond=concat)
    check("the batched skeleton is reused across steps",
          sampler._pixal3d_cfg_cache['template'] is template)
    check("its spatial cache survives",
          template.get_spatial_cache('probe') == 'kept')
    os.environ['PIXAL3D_CFG_BATCH'] = '0'


# ------------------------------------------------- accelerator probing --
def test_accelerator_probe(device):
    section("Accelerator probing (ROCm vs ZLUDA vs DirectML)")
    from pixal3d.compat import probe

    # An "RX 6800M" is gfx1031 while a plain "RX 6800" is gfx1030; the suffixed
    # entry has to win or the dtype choice silently flips.
    cases = [('AMD Radeon RX 6800M', 'gfx1031'), ('AMD Radeon RX 6800 XT', 'gfx1030'),
             ('AMD Radeon RX 6600M', 'gfx1032'), ('AMD Radeon RX 7900 XTX', 'gfx1100'),
             ('AMD Radeon RX 9070 XT', 'gfx1201')]
    wrong = [(n, probe.arch_from_device_name(n)) for n, want in cases
             if probe.arch_from_device_name(n) != want]
    check("device names map to gfx targets", not wrong, str(wrong))
    check("an NVIDIA name maps to nothing",
          probe.arch_from_device_name('NVIDIA GeForce RTX 4090') is None)

    real_platform = probe.platform
    try:
        for kind in ('directml', 'zluda', 'cpu'):
            probe.platform = lambda k=kind: k
            check(f"{kind} is called out", bool(probe.accelerator_note()))
        probe.platform = lambda: 'rocm'
        check("rocm needs no warning", probe.accelerator_note() is None)
    finally:
        probe.platform = real_platform


# --------------------------------------------------------------- fast init --
def test_fast_init(device):
    section("Checkpoint loading (skipping throwaway random init)")
    import json
    import tempfile
    import torch.nn as nn

    try:
        from safetensors.torch import save_file
    except ImportError:
        check("safetensors installed", False, "skipping")
        return

    import pixal3d.models as models

    class _FastInitToy(nn.Module):
        def __init__(self, n=8):
            super().__init__()
            self.lin = nn.Linear(n, n)
            # A non-persistent buffer is invisible to load_state_dict, so a
            # deterministic fill must survive the fast path.
            self.register_buffer('scale', torch.empty(n), persistent=False)
            nn.init.constant_(self.scale, 3.0)

    models.__dict__['_FastInitToy'] = _FastInitToy
    reference = _FastInitToy()

    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, 'toy')
        json.dump({'name': '_FastInitToy', 'args': {'n': 8}}, open(base + '.json', 'w'))
        save_file({'lin.weight': reference.lin.weight.data,
                   'lin.bias': reference.lin.bias.data}, base + '.safetensors')
        loaded = models.from_pretrained(base)
        check("the checkpoint wins over the skipped init",
              torch.equal(loaded.lin.weight, reference.lin.weight))
        check("non-persistent buffers keep their fill",
              torch.equal(loaded.scale, reference.scale))

        # A partial checkpoint must not leave uninitialised memory behind.
        save_file({'lin.weight': reference.lin.weight.data}, base + '.safetensors')
        partial = models.from_pretrained(base)
        check("a missing key falls back to full construction",
              bool(partial.lin.bias.isfinite().all()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default=None, help="cuda or cpu (default: cuda when available)")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    print(f"Pixal3D fallback tests on {device} (torch {torch.__version__})")

    for test in (test_hashgrid, test_grid_sample, test_uv_rasterize, test_dual_grid,
                 test_attention, test_sparse_conv, test_runtime, test_cfg_batch,
                 test_accelerator_probe, test_fast_init, test_model_stack,
                 test_glb_export):
        test(device)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} failed: {', '.join(FAILURES)}")
        return 1
    print("All tests passed.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
