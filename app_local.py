"""
Local web UI for Pixal3D.

``app.py`` targets Hugging Face Spaces: it imports ``spaces``, serves a custom
frontend and renders previews with nvdiffrast, none of which works on a ROCm
box. This is the local equivalent — single-view and multi-view generation with
the VRAM presets exposed, previewing the result in the browser's own glTF
viewer so no rasteriser is needed.

    python app_local.py
    python app_local.py --vram 8gb --share
"""

import argparse
import os
import sys
import time
import traceback
from typing import Optional

os.environ.setdefault("FLEX_GEMM_AUTOTUNE_CACHE_PATH",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), 'autotune_cache.json'))

from pixal3d import runtime, profiling   # must precede the first CUDA/HIP allocation
from pixal3d.compat import dispatch

import gradio as gr
import numpy as np
import torch
from PIL import Image

import inference as sv
import inference_mv as mv


ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PRESET = None
DTYPE = None
_pipelines = {}


def _get_pipeline(kind: str):
    """Load a pipeline once and keep it; both kinds rarely fit at the same time."""
    if kind in _pipelines:
        return _pipelines[kind]

    # Two 1.3B cascades will not coexist on a small card.
    for other in list(_pipelines):
        if other != kind:
            print(f"[app] Releasing the {other} pipeline to make room")
            _pipelines.pop(other)
            runtime.free_memory()

    device = runtime.get_device()
    with profiling.stage('load: pipeline'):
        if kind == 'single':
            pipeline = sv.init_pipeline(device=device, low_vram=PRESET.low_vram,
                                        block_offload=PRESET.block_offload)
        else:
            pipeline = mv.init_pipeline(device=device, low_vram=PRESET.low_vram,
                                        block_offload=PRESET.block_offload)
        runtime.apply_dtype(pipeline, DTYPE)
    _pipelines[kind] = pipeline
    return pipeline


def _sampler_params(steps, cfg, rescale, rescale_t):
    return {"steps": int(steps), "guidance_strength": float(cfg),
            "guidance_rescale": float(rescale), "rescale_t": float(rescale_t)}


def _export(mesh, pipeline, res, texture_size, decimate, out_path):
    with profiling.stage('export: GLB (decimate, unwrap, bake)'):
        glb = dispatch.to_glb(
            vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
            coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout,
            grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=int(decimate), texture_size=int(texture_size),
            remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True,
        )
    rot = np.array([[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
    glb.apply_transform(rot)
    glb.export(out_path, extension_webp=True)
    return out_path


def generate_single(image, seed, resolution, texture_size, decimate, fov,
                    ss_steps, shape_steps, tex_steps, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Upload an image first.")
    started = time.time()
    try:
        progress(0.02, desc="Loading models")
        pipeline = _get_pipeline('single')
        device = runtime.get_device()

        progress(0.1, desc="Removing background")
        image_pre = pipeline.preprocess_image(Image.fromarray(image) if isinstance(image, np.ndarray) else image)

        tmp_path = os.path.join(OUTPUT_DIR, f"_tmp_{int(time.time() * 1000)}.png")
        image_pre.save(tmp_path)
        try:
            if fov and fov > 0:
                progress(0.15, desc="Using the given field of view")
                grid_point = torch.tensor([-1.0, 0.0, 0.0])
                distance = sv.distance_from_fov(
                    float(fov), grid_point, torch.tensor([0, 511]), 1.0, 512)["distance_from_x"]
                camera_params = {'camera_angle_x': float(fov), 'distance': distance, 'mesh_scale': 1.0}
            else:
                progress(0.15, desc="Estimating the camera (MoGe-2)")
                moge = sv.load_moge_model(device=device)
                camera_params = sv.get_camera_params_wild_moge(tmp_path, moge, device=device)
                moge.cpu()
                del moge
                runtime.free_memory()
        finally:
            os.remove(tmp_path)

        progress(0.25, desc="Generating geometry and texture")
        torch.manual_seed(int(seed))
        mesh_list, (_, _, res) = pipeline.run(
            image_pre,
            camera_params=camera_params,
            seed=int(seed),
            sparse_structure_sampler_params=_sampler_params(ss_steps, 7.5, 0.7, 5.0),
            shape_slat_sampler_params=_sampler_params(shape_steps, 7.5, 0.5, 3.0),
            tex_slat_sampler_params=_sampler_params(tex_steps, 1.0, 0.0, 3.0),
            preprocess_image=False,
            return_latent=True,
            pipeline_type=f"{int(resolution)}_cascade",
            max_num_tokens=PRESET.max_num_tokens,
        )
        mesh = mesh_list[0]
        del mesh_list
        runtime.free_memory()

        progress(0.85, desc="Building the GLB")
        out_path = os.path.join(OUTPUT_DIR, f"pixal3d_{int(time.time())}.glb")
        _export(mesh, pipeline, res, texture_size, decimate, out_path)
        runtime.free_memory()

        elapsed = time.time() - started
        profiling.print_summary()
        profiling.reset()
        return out_path, out_path, f"Done in {elapsed:.0f}s — resolution {res}, texture {int(texture_size)}px"
    except torch.cuda.OutOfMemoryError:
        runtime.free_memory()
        raise gr.Error(
            "Out of VRAM. Restart with a smaller budget (--vram 8gb or --vram 6gb), "
            "or lower the resolution and texture size above.")
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(f"{type(exc).__name__}: {exc}")


def generate_multiview(views_dir, num_views, seed, resolution, texture_size, decimate,
                       ss_steps, shape_steps, tex_steps, progress=gr.Progress()):
    if not views_dir or not os.path.isdir(views_dir):
        raise gr.Error("Enter a directory containing transforms.json and the view images.")
    if not os.path.exists(os.path.join(views_dir, 'transforms.json')):
        raise gr.Error(f"No transforms.json in {views_dir}. See the README for the expected layout.")

    started = time.time()
    try:
        progress(0.02, desc="Loading models")
        pipeline = _get_pipeline('multi')

        progress(0.1, desc="Loading views")
        views = mv.load_views(views_dir, int(num_views) if num_views else None,
                              rembg=mv.make_rembg(pipeline))
        mv.check_main_view(views)

        progress(0.25, desc="Generating geometry and texture")
        torch.manual_seed(int(seed))
        mesh_list, (_, _, res) = pipeline.run_mv(
            views,
            seed=int(seed),
            sparse_structure_sampler_params=_sampler_params(ss_steps, 7.5, 0.7, 5.0),
            shape_slat_sampler_params=_sampler_params(shape_steps, 7.5, 0.5, 3.0),
            tex_slat_sampler_params=_sampler_params(tex_steps, 1.0, 0.0, 3.0),
            return_latent=True,
            pipeline_type=f"{int(resolution)}_cascade",
            max_num_tokens=PRESET.max_num_tokens,
        )
        mesh = mesh_list[0]
        del mesh_list
        runtime.free_memory()

        progress(0.85, desc="Building the GLB")
        out_path = os.path.join(OUTPUT_DIR, f"pixal3d_mv_{int(time.time())}.glb")
        _export(mesh, pipeline, res, texture_size, decimate, out_path)
        runtime.free_memory()

        elapsed = time.time() - started
        profiling.print_summary()
        profiling.reset()
        return out_path, out_path, f"Done in {elapsed:.0f}s — {len(views['view_names'])} views, resolution {res}"
    except torch.cuda.OutOfMemoryError:
        runtime.free_memory()
        raise gr.Error("Out of VRAM. Restart with --vram 8gb or --vram 6gb, or use fewer views.")
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(f"{type(exc).__name__}: {exc}")


def build_ui():
    with gr.Blocks(title="Pixal3D", theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# Pixal3D\n{runtime.describe()}  \n"
                    f"VRAM preset **{PRESET.name}** — resolution {PRESET.resolution}, "
                    f"texture {PRESET.texture_size}px, low_vram={PRESET.low_vram}, "
                    f"block_offload={PRESET.block_offload}")

        def settings_row():
            with gr.Row():
                resolution = gr.Dropdown([1024, 1536], value=PRESET.resolution, label="Resolution")
                texture_size = gr.Dropdown([1024, 2048, 4096], value=PRESET.texture_size, label="Texture size")
                decimate = gr.Slider(50_000, 1_000_000, value=PRESET.decimation_target,
                                     step=50_000, label="Target faces")
            with gr.Row():
                seed = gr.Number(value=42, precision=0, label="Seed")
                ss_steps = gr.Slider(4, 50, value=12, step=1, label="Structure steps")
                shape_steps = gr.Slider(4, 50, value=12, step=1, label="Shape steps")
                tex_steps = gr.Slider(4, 50, value=12, step=1, label="Texture steps")
            return resolution, texture_size, decimate, seed, ss_steps, shape_steps, tex_steps

        with gr.Tab("Single image"):
            with gr.Row():
                with gr.Column():
                    image = gr.Image(label="Input image", type="pil", height=320)
                    fov = gr.Slider(0.0, 1.2, value=0.0, step=0.01,
                                    label="Field of view (radians, 0 = estimate with MoGe-2)")
                    sv_settings = settings_row()
                    sv_button = gr.Button("Generate", variant="primary")
                with gr.Column():
                    sv_model = gr.Model3D(label="Result", height=420)
                    sv_file = gr.File(label="Download GLB")
                    sv_status = gr.Markdown()
            sv_button.click(
                generate_single,
                inputs=[image, sv_settings[3], sv_settings[0], sv_settings[1], sv_settings[2],
                        fov, sv_settings[4], sv_settings[5], sv_settings[6]],
                outputs=[sv_model, sv_file, sv_status],
            )

        with gr.Tab("Multi-view"):
            gr.Markdown(
                "A directory holding `transforms.json` plus the view images. The first frame "
                "is the main view and should be the canonical front view. The shipped "
                "`assets/mv_images/example` is a working four-view orbit you can copy."
            )
            with gr.Row():
                with gr.Column():
                    views_dir = gr.Textbox(
                        value=os.path.join(ROOT, 'assets', 'mv_images', 'example'),
                        label="Views directory")
                    num_views = gr.Number(value=0, precision=0,
                                          label="Views to use (0 = all)")
                    mv_settings = settings_row()
                    mv_button = gr.Button("Generate", variant="primary")
                with gr.Column():
                    mv_model = gr.Model3D(label="Result", height=420)
                    mv_file = gr.File(label="Download GLB")
                    mv_status = gr.Markdown()
            mv_button.click(
                generate_multiview,
                inputs=[views_dir, num_views, mv_settings[3], mv_settings[0], mv_settings[1],
                        mv_settings[2], mv_settings[4], mv_settings[5], mv_settings[6]],
                outputs=[mv_model, mv_file, mv_status],
            )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Pixal3D local web UI")
    parser.add_argument("--vram", default="auto",
                        choices=["auto", "max", "16gb", "12gb", "8gb", "6gb"],
                        help="VRAM budget preset. Default: auto-detect.")
    parser.add_argument("--attn", default=None, help="Attention backend override")
    parser.add_argument("--dtype", default="auto",
                        choices=["auto", "bfloat16", "float16", "float32"],
                        help="Precision for the DiT torsos. 'auto' picks float16 where the GPU has "
                             "no bfloat16 hardware.")
    parser.add_argument("--cond_dtype", default="auto",
                        choices=["auto", "bfloat16", "float16", "float32"],
                        help="Autocast precision for the DINOv3 / NAF conditioning encoders.")
    parser.add_argument("--cfg_batch", dest="cfg_batch", action="store_true", default=None,
                        help="Run both guidance branches in one batched forward.")
    parser.add_argument("--no_cfg_batch", dest="cfg_batch", action="store_false",
                        help="Run the two guidance branches separately.")
    parser.add_argument("--timing", action="store_true",
                        help="Print a per-stage wall-clock breakdown after each generation.")
    parser.add_argument("--conv", default=None, help="Sparse convolution backend override")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--share", action="store_true", help="Expose a public Gradio link")
    args = parser.parse_args()

    global PRESET, DTYPE
    if args.timing:
        profiling.set_enabled(True)
    PRESET = runtime.configure(vram=args.vram, attn_backend=args.attn,
                               conv_backend=args.conv, cfg_batch=args.cfg_batch)
    DTYPE = runtime.resolve_dtype(args.dtype)
    runtime.set_cond_dtype(runtime.resolve_cond_dtype(args.cond_dtype))
    if runtime.get_device() == 'cpu':
        print("[WARN] No GPU detected — run scripts/check_env.py to diagnose.")

    build_ui().launch(server_name=args.host, server_port=args.port,
                      share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
