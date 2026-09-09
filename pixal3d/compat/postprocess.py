"""
Textured-GLB export without ``o_voxel``, ``cumesh`` or ``nvdiffrast``.

Mirrors ``o_voxel.postprocess.to_glb`` step for step — clean, decimate, unwrap,
rasterise the atlas, sample the PBR attribute volume, inpaint, assemble — using
:mod:`pixal3d.compat.mesh_ops` for the mesh work and
:mod:`pixal3d.compat.uv_raster` for the bake.

Two deliberate differences from the native path:

* ``remesh`` is ignored. Narrow-band dual-contour remeshing lives entirely in
  ``cumesh``'s CUDA kernels; the decimation branch is what upstream itself runs
  when ``remesh=False``, so that is what happens here.
* Texel positions are not projected back onto the pre-decimation surface. That
  projection needs ``cumesh``'s BVH; skipping it means attributes are sampled
  at the decimated surface instead, which shifts colours by well under a voxel
  as long as the decimation target is not extreme.
"""

from typing import *
import numpy as np
import torch

from . import mesh_ops
from .grid_sample import grid_sample_3d
from .hashgrid import CoordLookup, MISS
from .uv_raster import uv_rasterize


__all__ = ['to_glb']


def _as_tensor(value, dtype, device):
    if isinstance(value, (int, float)):
        value = [value] * 3
    if isinstance(value, (list, tuple)):
        value = np.array(value)
    if isinstance(value, np.ndarray):
        value = torch.tensor(value, dtype=dtype, device=device)
    return value.to(device=device, dtype=dtype)


def _sample_attrs(
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    grid_size: torch.Tensor,
    positions: torch.Tensor,
    search_radius: int = 2,
) -> torch.Tensor:
    """
    Trilinearly sample the attribute volume, then rescue texels that fell into
    empty space.

    The volume only holds voxels near the surface, so a texel whose 3D position
    drifted outside that shell gets zero weight from every corner and would
    bake as a black speck. Those are re-sampled from the nearest occupied voxel
    within ``search_radius``, which is cheap because there are few of them.
    """
    shape = torch.Size([1, attr_volume.shape[1], *grid_size.tolist()])
    coords4 = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1)
    attrs = grid_sample_3d(attr_volume, coords4, shape, positions.reshape(1, -1, 3))[0]

    empty = ~(attrs.abs().sum(dim=1) > 0)
    num_empty = int(empty.sum().item())
    if num_empty == 0:
        return attrs

    lookup = CoordLookup(coords4, grid_size.tolist())
    todo = empty.nonzero(as_tuple=True)[0]
    base = positions[todo].floor().long()

    # Search shells outward so the first hit is (near enough to) the closest.
    offsets = torch.tensor(
        sorted(
            [(x, y, z)
             for x in range(-search_radius, search_radius + 1)
             for y in range(-search_radius, search_radius + 1)
             for z in range(-search_radius, search_radius + 1)],
            key=lambda o: o[0] ** 2 + o[1] ** 2 + o[2] ** 2,
        ),
        dtype=torch.long, device=positions.device,
    )

    found = torch.full((todo.shape[0],), MISS, dtype=torch.int64, device=positions.device)
    for off in offsets:
        pending = found == MISS
        if not bool(pending.any()):
            break
        idx_pending = pending.nonzero(as_tuple=True)[0]
        query = torch.cat([
            torch.zeros((idx_pending.shape[0], 1), dtype=torch.long, device=positions.device),
            base[idx_pending] + off,
        ], dim=1)
        hit = lookup.lookup(query)
        found[idx_pending] = hit

    rescued = found != MISS
    if bool(rescued.any()):
        attrs[todo[rescued]] = attr_volume[found[rescued]].to(attrs.dtype)
    return attrs


def to_glb(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: Dict[str, slice],
    aabb: Union[list, tuple, np.ndarray, torch.Tensor],
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
    decimation_target: int = 1000000,
    texture_size: int = 2048,
    remesh: bool = False,
    remesh_band: float = 1,
    remesh_project: float = 0.9,
    mesh_cluster_threshold_cone_half_angle_rad: float = np.radians(90.0),
    mesh_cluster_refine_iterations: int = 0,
    mesh_cluster_global_iterations: int = 1,
    mesh_cluster_smooth_strength: float = 1,
    verbose: bool = False,
    use_tqdm: bool = False,
    uv_padding: int = 4,
):
    """
    Build a textured ``trimesh.Trimesh`` from an extracted mesh plus a sparse
    PBR attribute volume. Signature-compatible with ``o_voxel.postprocess.to_glb``.
    """
    import cv2
    import trimesh
    import trimesh.visual
    from PIL import Image

    missing = mesh_ops.missing_dependencies()
    if missing:
        raise ImportError(
            "The GLB fallback needs these packages: " + ", ".join(missing) +
            ". Install them with: pip install " + " ".join(missing)
        )

    device = attr_volume.device
    aabb = _as_tensor(aabb, torch.float32, device).reshape(2, 3)
    if voxel_size is not None:
        voxel_size = _as_tensor(voxel_size, torch.float32, device)
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    else:
        assert grid_size is not None, "Either voxel_size or grid_size must be provided"
        grid_size = _as_tensor(grid_size, torch.int32, device)
        voxel_size = (aabb[1] - aabb[0]) / grid_size

    if remesh and verbose:
        print("[to_glb] remesh=True requires cumesh; using the decimation path instead")

    progress = None
    if use_tqdm:
        from tqdm import tqdm
        progress = tqdm(total=5, desc="Extracting GLB")

    def step(desc: str):
        if progress is not None:
            progress.update(1)
            progress.set_description(desc)

    verts_np = vertices.detach().float().cpu().numpy()
    faces_np = faces.detach().int().cpu().numpy()
    if verbose:
        print(f"[to_glb] input mesh: {len(verts_np)} verts, {len(faces_np)} faces")

    # --- Decimation and cleanup -------------------------------------------
    # Decimate before the trimesh repair passes: connected-component analysis
    # and hole filling are O(faces) on the CPU and the raw dual-grid mesh can
    # carry several million of them.
    if progress is not None:
        progress.set_description("Simplifying mesh")
    verts_np, faces_np = mesh_ops.simplify_mesh(
        verts_np, faces_np, decimation_target * 3, verbose=verbose)
    step("Cleaning mesh")

    verts_np, faces_np = mesh_ops.clean_mesh(verts_np, faces_np, verbose=verbose)
    verts_np, faces_np = mesh_ops.simplify_mesh(
        verts_np, faces_np, decimation_target, verbose=verbose)
    verts_np, faces_np = mesh_ops.clean_mesh(
        verts_np, faces_np, unify_orientation=True, verbose=verbose)
    if len(faces_np) == 0:
        raise RuntimeError(
            "Mesh cleanup left no faces. The sparse structure stage probably produced "
            "nothing usable — try a different seed, or a clearer input image.")
    step("Unwrapping UVs")

    # --- UV atlas ----------------------------------------------------------
    normals_np = mesh_ops.vertex_normals(verts_np, faces_np)
    out_verts, out_faces, out_uvs, vmapping = mesh_ops.uv_unwrap(
        verts_np, faces_np, texture_size=texture_size, padding=uv_padding, verbose=verbose)
    out_normals = normals_np[vmapping]
    step("Baking texture")

    # --- Bake --------------------------------------------------------------
    uvs_t = torch.from_numpy(out_uvs).to(device)
    faces_t = torch.from_numpy(out_faces).to(device)
    verts_t = torch.from_numpy(out_verts).to(device)

    face_ids, pos = uv_rasterize(uvs_t, faces_t, verts_t, texture_size, verbose=verbose)
    mask = face_ids > 0
    if not bool(mask.any()):
        raise RuntimeError("UV rasterisation produced an empty atlas; the unwrap likely failed")

    valid_pos = pos[mask]
    # A collapsed triangle can hand back a non-finite position; those become
    # undefined integers a few lines later, so clamp them into the grid first.
    grid_pos = torch.nan_to_num((valid_pos - aabb[0]) / voxel_size, nan=0.0)
    grid_pos = torch.minimum(torch.clamp_min(grid_pos, 0.0), grid_size.float() - 1)
    sampled = _sample_attrs(attr_volume, coords, grid_size, grid_pos)

    attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1],
                        device=device, dtype=torch.float32)
    attrs[mask] = sampled.float()
    step("Finalising")

    # --- Textures ----------------------------------------------------------
    mask_np = mask.cpu().numpy()
    def channel(name: str) -> np.ndarray:
        return np.clip(attrs[..., attr_layout[name]].cpu().numpy() * 255, 0, 255).astype(np.uint8)

    base_color = channel('base_color')
    metallic = channel('metallic')
    roughness = channel('roughness')
    alpha = channel('alpha')

    # Bleed colour outward so bilinear filtering across chart borders does not
    # sample the empty background.
    mask_inv = (~mask_np).astype(np.uint8)
    base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
    metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(
            np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode='OPAQUE',
        doubleSided=True,
    )

    # glTF is Y-up with V pointing down; the pipeline works Z-up with V up.
    out_verts = out_verts.copy()
    out_normals = out_normals.copy()
    out_verts[:, 1], out_verts[:, 2] = out_verts[:, 2].copy(), -out_verts[:, 1].copy()
    out_normals[:, 1], out_normals[:, 2] = out_normals[:, 2].copy(), -out_normals[:, 1].copy()
    out_uvs = out_uvs.copy()
    out_uvs[:, 1] = 1 - out_uvs[:, 1]

    textured = trimesh.Trimesh(
        vertices=out_verts,
        faces=out_faces,
        vertex_normals=out_normals,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=out_uvs, material=material),
    )

    if progress is not None:
        progress.update(1)
        progress.close()
    return textured
