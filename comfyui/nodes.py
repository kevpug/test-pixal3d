"""
ComfyUI nodes for Pixal3D.

Generation is split across three nodes rather than one so the expensive parts
can be cached independently: the pipeline loader holds ~20 GB of weights and
should run once, sampling is the part you re-run with a new seed, and the GLB
export is cheap to redo at a different texture size or face budget.

The nodes import ``pixal3d`` lazily. ComfyUI imports every custom node pack at
startup, and a missing dependency here must not stop the rest of ComfyUI from
loading.
"""

import os
import sys
import time
from typing import Any, Dict, Optional

# The repo root, so `import pixal3d` works wherever ComfyUI put this pack.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

VRAM_PRESETS = ["auto", "max", "16gb", "12gb", "8gb", "6gb"]
ATTN_BACKENDS = ["auto", "flash_attn", "xformers", "sdpa", "naive"]
CONV_BACKENDS = ["auto", "flex_gemm", "spconv", "torchsparse", "torch"]
DTYPES = ["auto", "bfloat16", "float16", "float32"]
CATEGORY = "Pixal3D"


def _output_directory() -> str:
    try:
        import folder_paths
        return folder_paths.get_output_directory()
    except Exception:
        path = os.path.join(_ROOT, 'outputs')
        os.makedirs(path, exist_ok=True)
        return path


def _to_pil(image):
    """ComfyUI IMAGE ([B, H, W, C] float in 0..1) -> PIL, first frame only."""
    import numpy as np
    from PIL import Image

    array = image[0] if image.ndim == 4 else image
    array = (array.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    if array.shape[-1] == 4:
        return Image.fromarray(array, mode='RGBA')
    return Image.fromarray(array, mode='RGB')


def _sampler(steps, cfg, rescale, rescale_t):
    return {"steps": int(steps), "guidance_strength": float(cfg),
            "guidance_rescale": float(rescale), "rescale_t": float(rescale_t)}


class Pixal3DPipelineLoader:
    """Loads a Pixal3D cascade and keeps it resident between runs."""

    _cache: Dict[tuple, Any] = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["single_view", "multi_view"],),
                "vram": (VRAM_PRESETS, {"default": "auto"}),
            },
            "optional": {
                "model_path": ("STRING", {"default": "TencentARC/Pixal3D"}),
                "attention": (ATTN_BACKENDS, {"default": "auto"}),
                "sparse_conv": (CONV_BACKENDS, {"default": "auto"}),
                "dtype": (DTYPES, {"default": "auto"}),
                "cond_dtype": (DTYPES, {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("PIXAL3D_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, mode, vram, model_path="TencentARC/Pixal3D",
             attention="auto", sparse_conv="auto", dtype="auto", cond_dtype="auto"):
        from pixal3d import runtime

        key = (mode, vram, model_path, attention, sparse_conv, dtype, cond_dtype)
        if key in Pixal3DPipelineLoader._cache:
            return (Pixal3DPipelineLoader._cache[key],)

        # Two cascades will not fit together on a consumer card.
        for other in list(Pixal3DPipelineLoader._cache):
            Pixal3DPipelineLoader._cache.pop(other)
        runtime.free_memory()

        preset = runtime.configure(
            vram=vram,
            attn_backend=None if attention == "auto" else attention,
            conv_backend=None if sparse_conv == "auto" else sparse_conv,
        )
        device = runtime.get_device()
        torch_dtype = runtime.resolve_dtype(dtype)
        runtime.set_cond_dtype(runtime.resolve_dtype(cond_dtype))

        if mode == "single_view":
            import inference as entry
            pipeline = entry.init_pipeline(model_path=model_path, device=device,
                                           low_vram=preset.low_vram,
                                           block_offload=preset.block_offload)
        else:
            import inference_mv as entry
            pipeline = entry.init_pipeline(model_path=model_path, device=device,
                                           low_vram=preset.low_vram,
                                           block_offload=preset.block_offload)

        runtime.apply_dtype(pipeline, torch_dtype)
        bundle = {"pipeline": pipeline, "preset": preset, "mode": mode}
        Pixal3DPipelineLoader._cache[key] = bundle
        return (bundle,)


class Pixal3DImageTo3D:
    """One image to a textured mesh, still in voxel-attribute form."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("PIXAL3D_PIPELINE",),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFF}),
                "resolution": ([1024, 1536], {"default": 1024}),
            },
            "optional": {
                "remove_background": ("BOOLEAN", {"default": True}),
                "fov_radians": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.5, "step": 0.01}),
                "structure_steps": ("INT", {"default": 12, "min": 4, "max": 50}),
                "shape_steps": ("INT", {"default": 12, "min": 4, "max": 50}),
                "texture_steps": ("INT", {"default": 12, "min": 4, "max": 50}),
                "structure_cfg": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 15.0, "step": 0.1}),
                "shape_cfg": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 15.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("PIXAL3D_MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, pipeline, image, seed, resolution, remove_background=True,
                 fov_radians=0.0, structure_steps=12, shape_steps=12, texture_steps=12,
                 structure_cfg=7.5, shape_cfg=7.5):
        import torch
        from pixal3d import runtime
        import inference as sv

        if pipeline["mode"] != "single_view":
            raise ValueError("This node needs a pipeline loaded in single_view mode.")
        pipe, preset = pipeline["pipeline"], pipeline["preset"]
        device = runtime.get_device()

        pil = _to_pil(image)
        image_pre = pipe.preprocess_image(pil) if remove_background else pil.convert('RGB')

        tmp = os.path.join(_output_directory(), f"_pixal3d_tmp_{int(time.time() * 1000)}.png")
        image_pre.save(tmp)
        try:
            if fov_radians > 0:
                distance = sv.distance_from_fov(
                    float(fov_radians), torch.tensor([-1.0, 0.0, 0.0]),
                    torch.tensor([0, 511]), 1.0, 512)["distance_from_x"]
                camera = {'camera_angle_x': float(fov_radians),
                          'distance': distance, 'mesh_scale': 1.0}
            else:
                moge = sv.load_moge_model(device=device)
                camera = sv.get_camera_params_wild_moge(tmp, moge, device=device)
                moge.cpu()
                del moge
                runtime.free_memory()
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

        torch.manual_seed(int(seed))
        mesh_list, (_, _, res) = pipe.run(
            image_pre,
            camera_params=camera,
            seed=int(seed),
            sparse_structure_sampler_params=_sampler(structure_steps, structure_cfg, 0.7, 5.0),
            shape_slat_sampler_params=_sampler(shape_steps, shape_cfg, 0.5, 3.0),
            tex_slat_sampler_params=_sampler(texture_steps, 1.0, 0.0, 3.0),
            preprocess_image=False,
            return_latent=True,
            pipeline_type=f"{int(resolution)}_cascade",
            max_num_tokens=preset.max_num_tokens,
        )
        mesh = mesh_list[0]
        del mesh_list
        runtime.free_memory()
        return ({"mesh": mesh, "resolution": res, "pipeline": pipe},)


class Pixal3DMultiViewTo3D:
    """Several posed views to a textured mesh."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("PIXAL3D_PIPELINE",),
                "views_dir": ("STRING", {"default": "assets/mv_images/example"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFF}),
                "resolution": ([1024, 1536], {"default": 1024}),
            },
            "optional": {
                "num_views": ("INT", {"default": 0, "min": 0, "max": 32}),
                "structure_steps": ("INT", {"default": 12, "min": 4, "max": 50}),
                "shape_steps": ("INT", {"default": 12, "min": 4, "max": 50}),
                "texture_steps": ("INT", {"default": 12, "min": 4, "max": 50}),
            },
        }

    RETURN_TYPES = ("PIXAL3D_MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, pipeline, views_dir, seed, resolution, num_views=0,
                 structure_steps=12, shape_steps=12, texture_steps=12):
        import torch
        from pixal3d import runtime
        import inference_mv as mv

        if pipeline["mode"] != "multi_view":
            raise ValueError("This node needs a pipeline loaded in multi_view mode.")
        pipe, preset = pipeline["pipeline"], pipeline["preset"]

        if not os.path.isabs(views_dir):
            views_dir = os.path.join(_ROOT, views_dir)
        if not os.path.exists(os.path.join(views_dir, 'transforms.json')):
            raise FileNotFoundError(f"No transforms.json in {views_dir}")

        views = mv.load_views(views_dir, int(num_views) or None, rembg=mv.make_rembg(pipe))
        mv.check_main_view(views)

        torch.manual_seed(int(seed))
        mesh_list, (_, _, res) = pipe.run_mv(
            views,
            seed=int(seed),
            sparse_structure_sampler_params=_sampler(structure_steps, 7.5, 0.7, 5.0),
            shape_slat_sampler_params=_sampler(shape_steps, 7.5, 0.5, 3.0),
            tex_slat_sampler_params=_sampler(texture_steps, 1.0, 0.0, 3.0),
            return_latent=True,
            pipeline_type=f"{int(resolution)}_cascade",
            max_num_tokens=preset.max_num_tokens,
        )
        mesh = mesh_list[0]
        del mesh_list
        runtime.free_memory()
        return ({"mesh": mesh, "resolution": res, "pipeline": pipe},)


class Pixal3DExportGLB:
    """Clean, unwrap, bake and write a textured GLB into ComfyUI's output folder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("PIXAL3D_MESH",),
                "filename_prefix": ("STRING", {"default": "pixal3d"}),
                "texture_size": ([512, 1024, 2048, 4096], {"default": 2048}),
                "target_faces": ("INT", {"default": 300000, "min": 10000,
                                         "max": 2000000, "step": 10000}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    FUNCTION = "export"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def export(self, mesh, filename_prefix, texture_size, target_faces):
        import numpy as np
        from pixal3d import runtime
        from pixal3d.compat import dispatch

        payload, pipe, res = mesh["mesh"], mesh["pipeline"], mesh["resolution"]
        glb = dispatch.to_glb(
            vertices=payload.vertices, faces=payload.faces, attr_volume=payload.attrs,
            coords=payload.coords, attr_layout=pipe.pbr_attr_layout,
            grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=int(target_faces), texture_size=int(texture_size),
            remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True,
        )
        rot = np.array([[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
                       dtype=np.float64)
        glb.apply_transform(rot)

        out_dir = _output_directory()
        name = f"{filename_prefix}_{int(time.time())}.glb"
        path = os.path.join(out_dir, name)
        glb.export(path, extension_webp=True)
        runtime.free_memory()
        print(f"[Pixal3D] wrote {path}")

        return {"ui": {"text": [path]}, "result": (path,)}


NODE_CLASS_MAPPINGS = {
    "Pixal3DPipelineLoader": Pixal3DPipelineLoader,
    "Pixal3DImageTo3D": Pixal3DImageTo3D,
    "Pixal3DMultiViewTo3D": Pixal3DMultiViewTo3D,
    "Pixal3DExportGLB": Pixal3DExportGLB,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Pixal3DPipelineLoader": "Pixal3D Pipeline Loader",
    "Pixal3DImageTo3D": "Pixal3D Image to 3D",
    "Pixal3DMultiViewTo3D": "Pixal3D Multi-View to 3D",
    "Pixal3DExportGLB": "Pixal3D Export GLB",
}
