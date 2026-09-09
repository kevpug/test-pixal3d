# Pixal3D nodes for ComfyUI

Four nodes: load the cascade once, sample, export. They share the same
pure-PyTorch fallbacks as the CLI, so they work on ROCm and on Windows without
`flash_attn`, `flex_gemm`, `cumesh`, `o_voxel` or `nvdiffrast`.

## Install

Clone this repository into ComfyUI's `custom_nodes` folder and install the
dependencies **into ComfyUI's own Python**:

```bat
cd ComfyUI\custom_nodes
git clone <your-fork-url> Pixal3D
cd Pixal3D
..\..\..\python_embeded\python.exe -m pip install -r requirements-rocm.txt
..\..\..\python_embeded\python.exe scripts\check_env.py
```

Adjust the path to `python.exe` for your ComfyUI install — portable builds use
`python_embeded\python.exe`, a manual install uses its own venv. ComfyUI must
already be running on a ROCm build of PyTorch; these nodes do not install
torch. If ComfyUI is on DirectML or ZLUDA instead, `check_env.py` will say so.

Restart ComfyUI. The nodes appear under the **Pixal3D** category.

## Nodes

| Node | Purpose |
|---|---|
| **Pixal3D Pipeline Loader** | Loads the weights (~20 GB, downloaded on first use) and picks a VRAM preset. Cached across runs; only one pipeline is held at a time. |
| **Pixal3D Image to 3D** | `IMAGE` → mesh. Background removal and MoGe-2 camera estimation are built in; set `fov_radians` above 0 to skip the estimate. |
| **Pixal3D Multi-View to 3D** | A directory of posed views (`transforms.json` + images) → mesh. |
| **Pixal3D Export GLB** | Mesh → textured `.glb` in ComfyUI's output folder. Re-run this alone to try a different texture size or face budget. |

## Minimal graph

```
Load Image ──▶ Pixal3D Image to 3D ──▶ Pixal3D Export GLB
                      ▲
Pixal3D Pipeline Loader
   (mode: single_view)
```

## Notes

- Split across three nodes deliberately: the loader is the expensive one, so
  changing a seed re-runs only sampling, and changing the texture size re-runs
  only the export.
- Set the loader's `vram` to match your card. On 12 GB use `12gb`; if a run
  dies with an out-of-memory error, drop to `8gb`.
- Switching between `single_view` and `multi_view` frees the other pipeline —
  two 1.3B cascades do not fit together on a consumer GPU.
- Generation takes minutes, not seconds, on the pure-PyTorch fallbacks. Watch
  the ComfyUI console for stage-by-stage progress.
