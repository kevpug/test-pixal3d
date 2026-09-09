"""
Decode shape + PBR latent (.npz) to textured GLB mesh and render the PBR front view.

Usage:
    python data_toolkit/visualize_pbr_latent.py \
        --root datasets/ObjaverseXL_sketchfab \
        --sha256 <SHA256_HASH> \
        --resolution 1024 \
        --view_idx 0
"""

import os
import sys
import json
import shutil
import argparse
import numpy as np
import torch
import cv2
from PIL import Image

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pixal3d.models as models
import pixal3d.modules.sparse as sp
from pixal3d.representations import MeshWithVoxel
from pixal3d.renderers import EnvMap
from pixal3d.utils import render_utils
import o_voxel

PBR_ATTR_LAYOUT = {
    'base_color': slice(0, 3),
    'metallic': slice(3, 4),
    'roughness': slice(4, 5),
    'alpha': slice(5, 6),
}


def load_latent(latent_file):
    """Load a latent .npz file and return a SparseTensor on GPU."""
    data = np.load(latent_file)
    coords = torch.tensor(data['coords']).int()
    feats = torch.tensor(data['feats']).float()
    coords = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1)
    return sp.SparseTensor(feats.cuda(), coords.cuda())


def load_envmaps(device='cuda'):
    """Load HDRI environment maps from assets/."""
    base = os.path.join(os.path.dirname(__file__), '..', 'assets', 'hdri')
    envmaps = {}
    for name in ['forest', 'sunset', 'courtyard']:
        path = os.path.join(base, f'{name}.exr')
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            envmaps[name] = EnvMap(torch.tensor(img, dtype=torch.float32, device=device))
    return envmaps


def main():
    parser = argparse.ArgumentParser(description="Decode shape + PBR latent to textured GLB and render")
    parser.add_argument("--root", type=str, required=True, help="Dataset root")
    parser.add_argument("--sha256", type=str, required=True, help="SHA256 of the asset")
    parser.add_argument("--resolution", type=int, default=1024, help="Decoder resolution")
    parser.add_argument("--view_idx", type=int, default=0, help="View index to decode")
    parser.add_argument("--shape_latent_name", type=str, default="shape_enc_next_dc_f16c32_fp16_1024_view",
                        help="Shape latent directory name under shape_latents/")
    parser.add_argument("--pbr_latent_name", type=str, default="tex_enc_next_dc_f16c32_fp16_1024_view_fix",
                        help="PBR latent directory name under pbr_latents/")
    parser.add_argument("--shape_decoder", type=str, default="microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16",
                        help="Pretrained shape decoder")
    parser.add_argument("--pbr_decoder", type=str, default="microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16",
                        help="Pretrained PBR/texture decoder")
    parser.add_argument("--texture_size", type=int, default=4096, help="GLB texture resolution")
    parser.add_argument("--decimation_target", type=int, default=1000000, help="GLB mesh decimation target")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: <root>/vis_pbr/<sha256>)")
    args = parser.parse_args()

    sha256 = args.sha256
    root = args.root
    view_idx = args.view_idx

    # Paths
    shape_latent_dir = os.path.join(root, "shape_latents", args.shape_latent_name, sha256)
    pbr_latent_dir = os.path.join(root, "pbr_latents", args.pbr_latent_name, sha256)
    shape_file = os.path.join(shape_latent_dir, f"view{view_idx:02d}.npz")
    pbr_file = os.path.join(pbr_latent_dir, f"view{view_idx:02d}.npz")
    renders_dir = os.path.join(root, "renders_cond", sha256)
    output_dir = args.output_dir or os.path.join(root, "vis_pbr", sha256)

    # Validate
    assert os.path.exists(shape_file), f"Shape latent not found: {shape_file}"
    assert os.path.exists(pbr_file), f"PBR latent not found: {pbr_file}"
    print(f"[Input] Shape latent: {shape_file}")
    print(f"[Input] PBR latent:   {pbr_file}")
    if os.path.exists(renders_dir):
        print(f"[Input] Renders:      {renders_dir}")

    # 1. Load latents
    print("[Step 1] Loading latents...")
    shape_slat = load_latent(shape_file)
    pbr_slat = load_latent(pbr_file)
    print(f"  Shape: coords {shape_slat.coords.shape}, feats {shape_slat.feats.shape}")
    print(f"  PBR:   coords {pbr_slat.coords.shape}, feats {pbr_slat.feats.shape}")

    # 2. Load decoders
    print(f"[Step 2] Loading decoders...")
    shape_dec = models.from_pretrained(args.shape_decoder)
    shape_dec.set_resolution(args.resolution)
    shape_dec = shape_dec.cuda().eval()

    pbr_dec = models.from_pretrained(args.pbr_decoder)
    pbr_dec = pbr_dec.cuda().eval()

    # 3. Decode shape → mesh + subs, then PBR → voxel
    print("[Step 3] Decoding shape + PBR latents...")
    with torch.no_grad():
        meshes, subs = shape_dec(shape_slat, return_subs=True)
        vox = pbr_dec(pbr_slat, guide_subs=subs) * 0.5 + 0.5

    mesh = meshes[0]
    mesh.fill_holes()
    mesh_with_voxel = MeshWithVoxel(
        mesh.vertices, mesh.faces,
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1 / args.resolution,
        coords=vox[0].coords[:, 1:],
        attrs=vox[0].feats,
        voxel_shape=torch.Size([*vox[0].shape, *vox[0].spatial_shape]),
        layout=PBR_ATTR_LAYOUT,
    )
    print(f"  Mesh: vertices {mesh.vertices.shape}, faces {mesh.faces.shape}")
    print(f"  Voxel: coords {vox[0].coords.shape}, feats {vox[0].feats.shape}")

    # 4. Export GLB with PBR textures
    print("[Step 4] Extracting textured GLB...")
    os.makedirs(output_dir, exist_ok=True)
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh_with_voxel.vertices,
        faces=mesh_with_voxel.faces,
        attr_volume=mesh_with_voxel.attrs,
        coords=mesh_with_voxel.coords,
        attr_layout=PBR_ATTR_LAYOUT,
        grid_size=args.resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args.decimation_target,
        texture_size=args.texture_size,
        remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True,
    )
    # Apply rotation (same as inference.py)
    rot = np.array([
        [-1,  0,  0,  0],
        [ 0,  0, -1,  0],
        [ 0, -1,  0,  0],
        [ 0,  0,  0,  1],
    ], dtype=np.float64)
    glb.apply_transform(rot)
    glb_path = os.path.join(output_dir, f"pbr_view{view_idx:02d}.glb")
    glb.export(glb_path, extension_webp=True)
    print(f"  GLB saved: {glb_path}")

    # 5. Render PBR front view (proj-aligned, same as app.py)
    print("[Step 5] Rendering PBR front view (proj-aligned)...")
    transforms_file = os.path.join(renders_dir, "transforms.json")
    shape_scale_file = os.path.join(shape_latent_dir, f"view{view_idx:02d}_scale.json")
    envmaps = load_envmaps(device='cuda')
    if os.path.exists(transforms_file) and os.path.exists(shape_scale_file) and envmaps:
        with open(transforms_file) as f:
            transforms = json.load(f)
        with open(shape_scale_file) as f:
            scale_info = json.load(f)
        total_scale = scale_info['total_scale']
        frame_info = transforms['frames'][view_idx]
        camera_angle_x = frame_info['camera_angle_x']
        distance = frame_info['radius']
        near = max(0.01, distance - 2.0)
        far = distance + 10.0
        # Scale mesh by 1/total_scale to match blender normalized space
        scaled_mesh = MeshWithVoxel(
            mesh_with_voxel.vertices / total_scale,
            mesh_with_voxel.faces,
            origin=[x / total_scale for x in mesh_with_voxel.origin],
            voxel_size=mesh_with_voxel.voxel_size / total_scale,
            coords=mesh_with_voxel.coords,
            attrs=mesh_with_voxel.attrs,
            voxel_shape=mesh_with_voxel.voxel_shape,
            layout=PBR_ATTR_LAYOUT,
        )
        print(f"  total_scale={total_scale:.4f}, distance={distance:.4f}, fov={camera_angle_x:.4f}")
        renders = render_utils.render_proj_aligned_video(
            scaled_mesh, camera_angle_x=camera_angle_x, distance=distance,
            resolution=1024, num_frames=1, envmap=envmaps, near=near, far=far,
        )
        for key, frames in renders.items():
            for i, frame in enumerate(frames):
                img = Image.fromarray(frame)
                img_path = os.path.join(output_dir, f"decoded_{key}_view{view_idx:02d}_{i:03d}.png")
                img.save(img_path)
            print(f"  Saved {len(frames)} {key} images")
    else:
        if not os.path.exists(transforms_file):
            print("  No transforms.json found, skipping rendering.")
        if not os.path.exists(shape_scale_file):
            print("  No scale file found, skipping rendering.")
        if not envmaps:
            print("  No HDRI envmaps found, skipping PBR rendering.")

    # Free GPU
    del shape_dec, pbr_dec, shape_slat, pbr_slat, meshes, subs, vox
    torch.cuda.empty_cache()

    # 6. Copy condition renders
    if os.path.exists(renders_dir):
        print("[Step 6] Copying condition renders...")
        for fname in sorted(os.listdir(renders_dir)):
            src = os.path.join(renders_dir, fname)
            dst = os.path.join(output_dir, fname)
            shutil.copy2(src, dst)
            print(f"  {fname}")
    else:
        print("[Step 6] No condition renders found, skipping.")

    # 7. Copy scale info
    for src_dir, prefix in [(shape_latent_dir, "shape"), (pbr_latent_dir, "pbr")]:
        scale_file = os.path.join(src_dir, f"view{view_idx:02d}_scale.json")
        if os.path.exists(scale_file):
            shutil.copy2(scale_file, os.path.join(output_dir, f"{prefix}_view{view_idx:02d}_scale.json"))

    print(f"\n[Done] All outputs in: {output_dir}")
    print(f"  Files: {sorted(os.listdir(output_dir))}")


if __name__ == "__main__":
    main()
