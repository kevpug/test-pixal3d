# Dataset Preparation Toolkit

This toolkit provides a comprehensive pipeline for preparing 3D datasets, including downloading, processing, voxelizing, and latent encoding for SC-VAE and Flow Model training.

This toolkit is built upon and extended from the data processing scripts of [TRELLIS.2](https://github.com/microsoft/TRELLIS2). We gratefully acknowledge the TRELLIS.2 team for open-sourcing their data preparation pipeline, which served as the foundation for this work. Our extensions include view-aligned voxelization and latent encoding for Pixal3D.

### Step 1: Install Dependencies

Initialize the environment and install necessary dependencies:

```bash
. ./data_toolkit/setup.sh
```

### Step 2: Initialize Metadata

Before processing, load the dataset metadata.

```bash
python data_toolkit/build_metadata.py <SUBSET> --root <ROOT> [--source <SOURCE>]
```

**Arguments:**
- `SUBSET`: Target dataset subset. Options: `ObjaverseXL`, `ABO`, `HSSD`, `TexVerse` (Training sets); `SketchfabPicked`, `Toys4k` (Test sets).
- `ROOT`: Root directory to save the data.
- `SOURCE`: Data source (Required if `SUBSET` is `ObjaverseXL`). Options: `sketchfab`, `github`.

**Example:**
Load metadata for `ObjaverseXL` (sketchfab) and save to `datasets/ObjaverseXL_sketchfab`:
```bash
python data_toolkit/build_metadata.py ObjaverseXL --source sketchfab --root datasets/ObjaverseXL_sketchfab
```

### Step 3: Download Data

Download the 3D assets to the local storage.

```bash
python data_toolkit/download.py <SUBSET> --root <ROOT> [--rank <RANK> --world_size <WORLD_SIZE>]
```

**Arguments:**
- `RANK` / `WORLD_SIZE`: Parameters for multi-node distributed downloading.

**Example:**
To download the `ObjaverseXL` subset:

> **Note:** The example below sets a large `WORLD_SIZE` (160,000) for demonstration purposes, meaning only a tiny fraction of the dataset will be downloaded by this single process.

```bash
python data_toolkit/download.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab --world_size 160000
```

*Attention: Some datasets may require an interactive Hugging Face login or manual steps. Please follow any on-screen instructions.*

**Update Metadata:**
After downloading, update the metadata registry:
```bash
python data_toolkit/build_metadata.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab
```

If download records are missing but files already exist locally, use `--from_file` to scan and rebuild:
```bash
python data_toolkit/build_metadata.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab --from_file
```

### Step 4: Process Mesh and PBR Textures

Standardize 3D assets by dumping mesh and PBR textures.
*Note: This process utilizes the CPU.*

```bash
# Dump Meshes
python data_toolkit/dump_mesh.py <SUBSET> --root <ROOT> [--rank <RANK> --world_size <WORLD_SIZE>]

# Dump PBR Textures
python data_toolkit/dump_pbr.py <SUBSET> --root <ROOT> [--rank <RANK> --world_size <WORLD_SIZE>]

# Get statistics of the asset
python data_toolkit/asset_stats.py --root <ROOT> [--rank <RANK> --world_size <WORLD_SIZE>]
```

**Example:**
```bash
python data_toolkit/dump_mesh.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab
python data_toolkit/dump_pbr.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab
python data_toolkit/asset_stats.py --root datasets/ObjaverseXL_sketchfab
```

**Update Metadata:**
```bash
python data_toolkit/build_metadata.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab
```

### Step 5: Render Image Conditions

Render multi-view images for each asset. These are used both as image conditions for the generator and as camera transforms for view-aligned processing in subsequent steps.
*Note: Blender and Pillow will be automatically installed on first run.*

```bash
python data_toolkit/render_cond.py <SUBSET> --root <ROOT> [--num_views <NUM_VIEWS>] [--rank <RANK> --world_size <WORLD_SIZE>]
```

**Arguments:**
- `NUM_VIEWS`: Number of views to render per asset. Default is `2`.

**Example:**
```bash
python data_toolkit/render_cond.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab
```

**Update Metadata:**
```bash
python data_toolkit/build_metadata.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab
```

### Step 6: Convert to View-Aligned O-Voxels

Convert the processed meshes and textures into view-aligned O-Voxels format. Each asset is transformed according to camera views from Step 5, producing per-view voxel representations.
*Note: This process utilizes the CPU.*

```bash
python data_toolkit/dual_grid_view.py <SUBSET> --root <ROOT> [--rank <RANK> --world_size <WORLD_SIZE>] [--resolution <RESOLUTION>] [--view_indices <VIEW_INDICES>]

python data_toolkit/voxelize_pbr_view.py <SUBSET> --root <ROOT> [--rank <RANK> --world_size <WORLD_SIZE>] [--resolution <RESOLUTION>] [--view_indices <VIEW_INDICES>]
```

**Arguments:**
- `RESOLUTION`: Target resolutions for O-Voxels, comma-separated (e.g., `256,512,1024`). Default is `256`.
- `VIEW_INDICES`: Specific view indices to process (e.g., `0,1,2` or `0-5`). Default processes all available views.

**Example:**
Convert `ObjaverseXL` to resolution 256 for views 0-1:
```bash
python data_toolkit/dual_grid_view.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab --resolution 256 --view_indices 0-1
python data_toolkit/voxelize_pbr_view.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab --resolution 256 --view_indices 0-1
```

**Update Metadata:**
```bash
python data_toolkit/build_metadata.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab
```

### At this point, the dataset is ready for SC-VAE Training

### Step 7: Encode View-Aligned Latents

Encode view-aligned sparse structures into latents to train the first-stage generator. Each step produces per-view latent files.

```bash
# 1. Encode Shape Latents (multi-view)
python data_toolkit/encode_shape_latent_view.py --root <ROOT> [--rank <RANK> --world_size <WORLD_SIZE>] [--resolution <RESOLUTION>] [--view_indices <VIEW_INDICES>]

# 2. Encode PBR Latents (view-aligned)
python data_toolkit/encode_pbr_latent_view.py --root <ROOT> [--rank <RANK> --world_size <WORLD_SIZE>] [--resolution <RESOLUTION>] [--view_indices <VIEW_INDICES>]

# 3. Update Metadata (Required before next step)
python data_toolkit/build_metadata.py <SUBSET> --root <ROOT>

# 4. Encode Sparse Structure (SS) Latents (multi-view)
python data_toolkit/encode_ss_latent_view.py --root <ROOT> --shape_latent_name <SHAPE_LATENT_NAME> [--rank <RANK> --world_size <WORLD_SIZE>] [--resolution <SS_RESOLUTION>] [--view_indices <VIEW_INDICES>]
```

**Arguments:**
- `RESOLUTION`: Input O-Voxel resolution. Default is `1024`.
- `SS_RESOLUTION`: Resolution for sparse structures. Default is `64`.
- `SHAPE_LATENT_NAME`: The specific version name of the shape latent (use the `_view` variant name).
- `VIEW_INDICES`: Specific view indices to process (e.g., `0,1,2` or `0-5`).

**Example:**
```bash
python data_toolkit/encode_shape_latent_view.py --root datasets/ObjaverseXL_sketchfab --resolution 512 --view_indices 0-1
python data_toolkit/encode_pbr_latent_view.py --root datasets/ObjaverseXL_sketchfab --resolution 512 --view_indices 0-1
python data_toolkit/encode_shape_latent_view.py --root datasets/ObjaverseXL_sketchfab --resolution 1024 --view_indices 0-1
python data_toolkit/encode_pbr_latent_view.py --root datasets/ObjaverseXL_sketchfab --resolution 1024 --view_indices 0-1

# Update metadata
python data_toolkit/build_metadata.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab

# Encode SS Latents (view-aligned)
python data_toolkit/encode_ss_latent_view.py --root datasets/ObjaverseXL_sketchfab --shape_latent_name shape_enc_next_dc_f16c32_fp16_1024_view --resolution 64 --view_indices 0-1

# Final Metadata Update
python data_toolkit/build_metadata.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab
```

### Step 8: Visualize Decoded Latents (Optional)

Decode latent files back to meshes, export GLB, and render a front-view image for visual inspection.

**Shape Latent Visualization:**
```bash
python data_toolkit/visualize_shape_latent.py \
    --root datasets/ObjaverseXL_sketchfab \
    --sha256 <SHA256_HASH> \
    --resolution 1024 \
    --view_idx 0
```

**PBR Latent Visualization (shape + texture):**
```bash
python data_toolkit/visualize_pbr_latent.py \
    --root datasets/ObjaverseXL_sketchfab \
    --sha256 <SHA256_HASH> \
    --resolution 1024 \
    --view_idx 0
```

Outputs are saved to `<ROOT>/vis/<SHA256>/` (shape) or `<ROOT>/vis_pbr/<SHA256>/` (PBR), including:
- Decoded GLB mesh (with PBR textures for PBR variant)
- Front-view rendered images (normal/depth for shape; shaded/base_color/normal etc. for PBR)
- Copied condition renders and camera transforms from Step 5
