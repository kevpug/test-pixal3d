"""
Decode a view-aligned shape latent (.npz) to GLB mesh and render the front view.

Usage:
    python data_toolkit/visualize_shape_latent.py \
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
import trimesh
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pixal3d.models as models
import pixal3d.modules.sparse as sp
from pixal3d.utils import render_utils


def main():
    parser = argparse.ArgumentParser(description="Decode shape latent to GLB and collect renders")
    parser.add_argument("--root", type=str, required=True, help="Dataset root, e.g. /local-ssd/datasets/ObjaverseXL_sketchfab")
    parser.add_argument("--sha256", type=str, required=True, help="SHA256 of the asset")
    parser.add_argument("--resolution", type=int, default=1024, help="Decoder resolution (must match latent resolution)")
    parser.add_argument("--view_idx", type=int, default=0, help="View index to decode")
    parser.add_argument("--latent_name", type=str, default="shape_enc_next_dc_f16c32_fp16_1024_view",
                        help="Latent directory name under shape_latents/")
    parser.add_argument("--decoder", type=str, default="microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16",
                        help="Pretrained shape decoder path (HuggingFace or local)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: <root>/vis/<sha256>)")
    args = parser.parse_args()

    sha256 = args.sha256
    root = args.root
    view_idx = args.view_idx

    # Paths
    latent_dir = os.path.join(root, "shape_latents", args.latent_name, sha256)
    latent_file = os.path.join(latent_dir, f"view{view_idx:02d}.npz")
    scale_file = os.path.join(latent_dir, f"view{view_idx:02d}_scale.json")
    renders_dir = os.path.join(root, "renders_cond", sha256)
    output_dir = args.output_dir or os.path.join(root, "vis", sha256)

    # Validate
    assert os.path.exists(latent_file), f"Latent file not found: {latent_file}"
    print(f"[Input] Latent: {latent_file}")
    if os.path.exists(scale_file):
        print(f"[Input] Scale:  {scale_file}")
    if os.path.exists(renders_dir):
        print(f"[Input] Renders: {renders_dir}")

    # 1. Load latent
    print("[Step 1] Loading shape latent...")
    data = np.load(latent_file)
    coords = torch.tensor(data['coords']).int()
    feats = torch.tensor(data['feats']).float()
    # Prepend batch dim (0) to coords
    coords = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1)
    slat = sp.SparseTensor(feats.cuda(), coords.cuda())
    print(f"  coords: {coords.shape}, feats: {feats.shape}")

    # 2. Load decoder
    print(f"[Step 2] Loading shape decoder: {args.decoder}")
    decoder = models.from_pretrained(args.decoder)
    decoder.set_resolution(args.resolution)
    decoder = decoder.cuda().eval()

    # 3. Decode
    print("[Step 3] Decoding shape latent → mesh...")
    with torch.no_grad():
        meshes, subs = decoder(slat, return_subs=True)
    mesh = meshes[0]
    print(f"  vertices: {mesh.vertices.shape}, faces: {mesh.faces.shape}")

    # 4. Convert to trimesh and export GLB
    print("[Step 4] Exporting GLB...")
    vertices = mesh.vertices.cpu().numpy()
    faces = mesh.faces.cpu().numpy()

    # Apply coordinate rotation (same as inference.py)
    # Swap axes: x→-x, y→-z, z→-y
    rot = np.array([
        [-1,  0,  0],
        [ 0,  0, -1],
        [ 0, -1,  0],
    ], dtype=np.float64)
    vertices = vertices @ rot.T

    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    os.makedirs(output_dir, exist_ok=True)
    glb_path = os.path.join(output_dir, f"shape_view{view_idx:02d}.glb")
    tri_mesh.export(glb_path)
    print(f"  GLB saved: {glb_path}")

    # 5. Render front view (proj-aligned, same as app.py)
    print("[Step 5] Rendering decoded mesh (proj-aligned front view)...")
    transforms_file = os.path.join(renders_dir, "transforms.json")
    if os.path.exists(transforms_file) and os.path.exists(scale_file):
        with open(transforms_file) as f:
            transforms = json.load(f)
        with open(scale_file) as f:
            scale_info = json.load(f)
        total_scale = scale_info['total_scale']
        frame_info = transforms['frames'][view_idx]
        camera_angle_x = frame_info['camera_angle_x']
        distance = frame_info['radius']
        near = max(0.01, distance - 2.0)
        far = distance + 10.0
        # Scale mesh vertices by 1/total_scale to match blender normalized space
        from pixal3d.representations import Mesh
        scaled_mesh = Mesh(mesh.vertices / total_scale, mesh.faces)
        print(f"  total_scale={total_scale:.4f}, distance={distance:.4f}, fov={camera_angle_x:.4f}")
        renders = render_utils.render_proj_aligned_video(
            scaled_mesh, camera_angle_x=camera_angle_x, distance=distance,
            resolution=1024, num_frames=1, near=near, far=far,
        )
        for key, frames in renders.items():
            for i, frame in enumerate(frames):
                img = Image.fromarray(frame)
                img_path = os.path.join(output_dir, f"decoded_{key}_view{view_idx:02d}_{i:03d}.png")
                img.save(img_path)
            print(f"  Saved {len(frames)} {key} images")
    else:
        print("  No transforms.json or scale file found, skipping rendering.")

    # Free decoder GPU memory
    del decoder, slat, meshes, subs
    torch.cuda.empty_cache()

    # 6. Copy renders
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
    if os.path.exists(scale_file):
        shutil.copy2(scale_file, os.path.join(output_dir, f"view{view_idx:02d}_scale.json"))

    print(f"\n[Done] All outputs in: {output_dir}")
    print(f"  Files: {sorted(os.listdir(output_dir))}")


if __name__ == "__main__":
    main()
