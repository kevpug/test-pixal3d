import os
import argparse
import math
import time
from typing import Optional
import numpy as np
from PIL import Image

os.environ.setdefault("FLEX_GEMM_AUTOTUNE_CACHE_PATH",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), 'autotune_cache.json'))
os.environ.setdefault("FLEX_GEMM_AUTOTUNER_VERBOSE", '1')

# Must precede the first CUDA/HIP allocation: sets the allocator config and
# selects attention / sparse-conv backends that match what is installed.
from pixal3d import runtime, profiling
from pixal3d.compat import dispatch

import torch
import cv2

# ============================================================================
# Constants & Defaults
# ============================================================================

MOGE_MODEL_NAME = "Ruicheng/moge-2-vitl"
MODEL_PATH = "TencentARC/Pixal3D"

IMAGE_COND_CONFIGS = {
    "ss": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 16,
    },
    "shape_512": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 32,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "shape_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "tex_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 1024,
    },
}

# ============================================================================
# Model Loading
# ============================================================================

def build_image_cond_model(config: dict):
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor
    model = DinoV3ProjFeatureExtractor(**config)
    model.eval()
    return model


def load_moge_model(device="cuda", model_name=MOGE_MODEL_NAME):
    from moge.model.v2 import MoGeModel
    moge_model = MoGeModel.from_pretrained(model_name)
    moge_model = moge_model.to(device)
    moge_model.eval()
    return moge_model


def init_pipeline(model_path=MODEL_PATH, device=None, low_vram=False, block_offload=False):
    from pixal3d.pipelines import Pixal3DImageTo3DPipeline

    device = device or runtime.get_device()
    print(f"[Pipeline] Loading from {model_path}...")
    pipeline = Pixal3DImageTo3DPipeline.from_pretrained(model_path)

    print("[ImageCond] Building DinoV3ProjFeatureExtractor models...")
    pipeline.image_cond_model_ss = build_image_cond_model(IMAGE_COND_CONFIGS["ss"])
    pipeline.image_cond_model_shape_512 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_512"])
    pipeline.image_cond_model_shape_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_1024"])
    pipeline.image_cond_model_tex_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["tex_1024"])

    if block_offload:
        # Stream the DiT blocks per forward pass. Caps flow-model residency at
        # one block instead of 2.6 GB, which is what makes an 8 GB card viable.
        offloaded = 0
        for key, model in pipeline.models.items():
            if 'flow_model' in key and runtime.enable_block_offload(model, device):
                offloaded += 1
        print(f"[LowVRAM] Block offloading enabled for {offloaded} flow models.")

    if low_vram:
        # Low-VRAM mode: models stay on CPU, loaded to GPU on-demand per stage.
        # Peak VRAM = one flow model + one DinoV3, not all ~18 GB at once.
        print("[NAF] Pre-downloading NAF upsampler weights (CPU only)...")
        for attr in ['image_cond_model_ss', 'image_cond_model_shape_512',
                     'image_cond_model_shape_1024', 'image_cond_model_tex_1024']:
            m = getattr(pipeline, attr, None)
            if m is not None and getattr(m, 'use_naf_upsample', False):
                m._load_naf()
        pipeline._device = torch.device(device)
        pipeline.low_vram = True
        print("[Pipeline] Low-VRAM mode enabled.")
    else:
        # Standard mode: all models loaded to GPU at once (faster, needs more VRAM).
        pipeline.low_vram = False
        pipeline.to(torch.device(device))
        pipeline.image_cond_model_ss.to(device)
        pipeline.image_cond_model_shape_512.to(device)
        pipeline.image_cond_model_shape_1024.to(device)
        pipeline.image_cond_model_tex_1024.to(device)
        print("[NAF] Pre-loading NAF upsampler model...")
        for attr in ['image_cond_model_ss', 'image_cond_model_shape_512',
                     'image_cond_model_shape_1024', 'image_cond_model_tex_1024']:
            m = getattr(pipeline, attr, None)
            if m is not None and getattr(m, 'use_naf_upsample', False):
                m._load_naf()
        print("[Pipeline] Standard mode (all models on GPU).")

    return pipeline

# ============================================================================
# Camera Estimation
# ============================================================================

def compute_f_pixels(camera_angle_x: float, resolution: int) -> float:
    focal_length = 16.0 / torch.tan(torch.tensor(camera_angle_x / 2.0))
    f_pixels = focal_length * resolution / 32.0
    return float(f_pixels.item())


def distance_from_fov(camera_angle_x, grid_point, target_point, mesh_scale, image_resolution):
    rotation_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    gp = grid_point.to(torch.float32) @ rotation_matrix.T
    gp = gp / mesh_scale / 2
    xw, yw, zw = gp[0].item(), gp[1].item(), gp[2].item()
    xt, yt = float(target_point[0].item()), float(target_point[1].item())
    f_pixels = compute_f_pixels(camera_angle_x, image_resolution)
    x_ndc = xt - image_resolution / 2.0
    y_ndc = -(yt - image_resolution / 2.0)
    distance_x = f_pixels * xw / x_ndc - yw
    return {"distance_from_x": float(distance_x), "f_pixels": float(f_pixels)}


def get_camera_params_wild_moge(image_path, moge_model, device="cuda", mesh_scale=1.0, extend_pixel=0, image_resolution=512):
    pil_image = Image.open(image_path).convert("RGB")
    width, height = pil_image.size
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(device)
    with torch.no_grad():
        output = moge_model.infer(image_tensor)
    intrinsics = output["intrinsics"].squeeze().cpu().numpy()
    fx_normalized = intrinsics[0, 0]
    fx = fx_normalized * width
    camera_angle_x = 2 * math.atan(width / (2 * fx))

    grid_point = torch.tensor([-1.0, 0.0, 0.0])
    distance = distance_from_fov(
        camera_angle_x, grid_point,
        torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
        mesh_scale, image_resolution
    )["distance_from_x"]
    return {'camera_angle_x': camera_angle_x, 'distance': distance, 'mesh_scale': mesh_scale}

# ============================================================================
# Main Inference
# ============================================================================

def run_inference(
    image_path: str,
    output_path: str,
    seed: int = 42,
    ss_guidance_strength: float = 7.5,
    ss_guidance_rescale: float = 0.7,
    ss_sampling_steps: int = 12,
    ss_rescale_t: float = 5.0,
    shape_slat_guidance_strength: float = 7.5,
    shape_slat_guidance_rescale: float = 0.5,
    shape_slat_sampling_steps: int = 12,
    shape_slat_rescale_t: float = 3.0,
    tex_slat_guidance_strength: float = 1.0,
    tex_slat_guidance_rescale: float = 0.0,
    tex_slat_sampling_steps: int = 12,
    tex_slat_rescale_t: float = 3.0,
    mesh_scale: float = 1.0,
    extend_pixel: int = 0,
    image_resolution: int = 512,
    max_num_tokens: int = -1,
    model_path: str = MODEL_PATH,
    manual_fov: float = -1.0,
    low_vram: Optional[bool] = None,
    resolution: int = -1,
    vram: str = 'auto',
    texture_size: int = -1,
    decimation_target: int = -1,
    attn_backend: Optional[str] = None,
    conv_backend: Optional[str] = None,
    block_offload: Optional[bool] = None,
    dtype: str = 'auto',
    cond_dtype: str = 'auto',
    cfg_batch: Optional[bool] = None,
    timing: bool = False,
):
    if timing:
        profiling.set_enabled(True)
    preset = runtime.configure(vram=vram, attn_backend=attn_backend,
                               conv_backend=conv_backend, cfg_batch=cfg_batch)
    device = runtime.get_device()
    if device == 'cpu':
        print("[WARN] No GPU detected — this will be extremely slow. "
              "Run scripts/check_env.py to diagnose the install.")

    # Explicit arguments win over the preset; the preset fills in the rest.
    low_vram = preset.low_vram if low_vram is None else low_vram
    block_offload = preset.block_offload if block_offload is None else block_offload
    resolution = preset.resolution if resolution <= 0 else resolution
    max_num_tokens = preset.max_num_tokens if max_num_tokens <= 0 else max_num_tokens
    texture_size = preset.texture_size if texture_size <= 0 else texture_size
    decimation_target = preset.decimation_target if decimation_target <= 0 else decimation_target

    torch_dtype = runtime.resolve_dtype(dtype)
    runtime.set_cond_dtype(runtime.resolve_cond_dtype(cond_dtype))

    # Load models
    with profiling.stage('load: pipeline'):
        pipeline = init_pipeline(model_path, device=device, low_vram=low_vram,
                                 block_offload=block_offload)
        runtime.apply_dtype(pipeline, torch_dtype)

    # Preprocess image first — rembg loads to GPU for this call, then offloads.
    # MoGe is loaded afterwards so both never occupy VRAM at the same time.
    print(f"[Inference] Processing image: {image_path}")
    img = Image.open(image_path)
    with profiling.stage('preprocess (rembg)'):
        image_preprocessed = pipeline.preprocess_image(img)

    # Save preprocessed image for MoGe
    tmp_path = os.path.join(os.path.dirname(os.path.abspath(output_path)), f"_tmp_preprocessed_{int(time.time()*1000)}.png")
    image_preprocessed.save(tmp_path)

    # Camera estimation
    if manual_fov > 0:
        # Use manually specified FOV (in radians)
        camera_angle_x = float(manual_fov)
        grid_point = torch.tensor([-1.0, 0.0, 0.0])
        distance = distance_from_fov(
            camera_angle_x, grid_point,
            torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
            mesh_scale, image_resolution
        )["distance_from_x"]
        camera_params = {'camera_angle_x': camera_angle_x, 'distance': distance, 'mesh_scale': mesh_scale}
        print(f"[Inference] Using manual FOV: {math.degrees(manual_fov):.2f}° ({manual_fov:.4f} rad), distance={distance:.4f}")
    else:
        print("[MoGe-2] Loading model for camera estimation...")
        with profiling.stage('camera: MoGe-2'):
            moge_model = load_moge_model(device=device)
            print("[Inference] Estimating camera parameters...")
            camera_params = get_camera_params_wild_moge(
                tmp_path, moge_model, device=device,
                mesh_scale=mesh_scale, extend_pixel=extend_pixel,
                image_resolution=image_resolution,
            )
        print(f"  camera_angle_x={camera_params['camera_angle_x']:.4f}, distance={camera_params['distance']:.4f}")
        # MoGe is only needed for camera estimation; free its VRAM for inference.
        moge_model.cpu()
        del moge_model
        runtime.free_memory()
    os.remove(tmp_path)

    # Run pipeline
    print("[Inference] Running 3D generation pipeline...")
    torch.manual_seed(seed)

    ss_sampler_override = {
        "steps": ss_sampling_steps, "guidance_strength": ss_guidance_strength,
        "guidance_rescale": ss_guidance_rescale, "rescale_t": ss_rescale_t,
    }
    shape_sampler_override = {
        "steps": shape_slat_sampling_steps, "guidance_strength": shape_slat_guidance_strength,
        "guidance_rescale": shape_slat_guidance_rescale, "rescale_t": shape_slat_rescale_t,
    }
    tex_sampler_override = {
        "steps": tex_slat_sampling_steps, "guidance_strength": tex_slat_guidance_strength,
        "guidance_rescale": tex_slat_guidance_rescale, "rescale_t": tex_slat_rescale_t,
    }

    pipeline_type = f"{resolution}_cascade"
    print(f"[Inference] Using pipeline_type={pipeline_type}")
    mesh_list, (shape_slat, tex_slat, res) = pipeline.run(
        image_preprocessed,
        camera_params=camera_params,
        seed=seed,
        sparse_structure_sampler_params=ss_sampler_override,
        shape_slat_sampler_params=shape_sampler_override,
        tex_slat_sampler_params=tex_sampler_override,
        preprocess_image=False,
        return_latent=True,
        pipeline_type=pipeline_type,
        max_num_tokens=max_num_tokens,
    )

    mesh = mesh_list[0]

    # Free the sampling-stage weights before the export stage allocates its
    # texture buffers — on a 12 GB card the two do not fit together.
    del mesh_list
    runtime.free_memory()

    # Extract GLB
    print("[Inference] Extracting GLB...")
    with profiling.stage('export: GLB (decimate, unwrap, bake)'):
        glb = dispatch.to_glb(
            vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
            coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout,
            grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation_target, texture_size=texture_size,
            remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True,
        )

    # Apply rotation
    rot = np.array([
        [-1,  0,  0,  0],
        [ 0,  0, -1,  0],
        [ 0, -1,  0,  0],
        [ 0,  0,  0,  1],
    ], dtype=np.float64)
    glb.apply_transform(rot)

    # Export
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    glb.export(output_path, extension_webp=True)
    print(f"[Done] GLB saved to: {output_path}")
    profiling.print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pixal3D Inference: Image to GLB")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="./output.glb", help="Output GLB file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fov", type=float, default=-1.0,
                        help="Manual camera FOV in radians (e.g. 0.2). "
                             "If not set, FOV is auto-estimated via MoGe-2. "
                             "Try 0.2 rad if you notice distortion.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Model path or HuggingFace repo")
    parser.add_argument("--vram", type=str, default="auto",
                        choices=["auto", "max", "16gb", "12gb", "8gb", "6gb"],
                        help="VRAM budget preset. Sets resolution, token cap, texture size, decimation "
                             "and offloading together. Default: auto-detect from the installed GPU.")
    parser.add_argument("--low_vram", dest="low_vram", action="store_true", default=None,
                        help="Force per-stage CPU offloading (overrides the preset).")
    parser.add_argument("--no_low_vram", dest="low_vram", action="store_false",
                        help="Force everything onto the GPU at once (overrides the preset).")
    parser.add_argument("--block_offload", dest="block_offload", action="store_true", default=None,
                        help="Stream DiT blocks between CPU and GPU. Slower, but caps flow-model VRAM "
                             "at one block instead of ~2.6 GB.")
    parser.add_argument("--no_block_offload", dest="block_offload", action="store_false",
                        help="Disable block streaming (overrides the preset).")
    parser.add_argument("--resolution", type=int, default=-1, choices=[-1, 1024, 1536],
                        help="Pipeline resolution. Default: from --vram.")
    parser.add_argument("--max_num_tokens", type=int, default=-1,
                        help="Cap on sparse tokens in the high-res stage. Default: from --vram.")
    parser.add_argument("--texture_size", type=int, default=-1,
                        help="Baked texture resolution. Default: from --vram.")
    parser.add_argument("--decimate", type=int, default=-1,
                        help="Target face count for the exported mesh. Default: from --vram.")
    parser.add_argument("--attn", type=str, default=None,
                        choices=["flash_attn", "flash_attn_3", "xformers", "sdpa", "naive"],
                        help="Attention backend. Default: the fastest one installed.")
    parser.add_argument("--conv", type=str, default=None,
                        choices=["flex_gemm", "spconv", "torchsparse", "torch"],
                        help="Sparse convolution backend. Default: the fastest one installed.")
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "bfloat16", "float16", "float32"],
                        help="Precision for the DiT torsos. 'auto' picks float16 on GPUs with no "
                             "bfloat16 hardware (all of RDNA1/2), where a bf16 matmul is emulated "
                             "through fp32 and runs at roughly half the fp16 rate.")
    parser.add_argument("--cond_dtype", type=str, default="auto",
                        choices=["auto", "bfloat16", "float16", "float32"],
                        help="Autocast precision for the DINOv3 / NAF conditioning encoders, which "
                             "run four times per generation at up to 1024x1024.")
    parser.add_argument("--steps", type=int, default=-1,
                        help="Denoising steps for every stage (default 12 each). "
                             "8 is usually indistinguishable; 6 is visibly rougher.")
    parser.add_argument("--ss_steps", type=int, default=-1, help="Steps for the sparse-structure stage.")
    parser.add_argument("--shape_steps", type=int, default=-1, help="Steps for both shape stages.")
    parser.add_argument("--tex_steps", type=int, default=-1, help="Steps for the texture stage.")
    parser.add_argument("--cfg_batch", dest="cfg_batch", action="store_true", default=None,
                        help="Run both guidance branches in one batched forward. Fewer kernel "
                             "launches, but roughly double the peak activation memory.")
    parser.add_argument("--no_cfg_batch", dest="cfg_batch", action="store_false",
                        help="Run the two guidance branches separately (overrides the preset).")
    parser.add_argument("--timing", action="store_true",
                        help="Print a per-stage wall-clock breakdown at the end.")

    args = parser.parse_args()

    steps = {}
    if args.steps > 0:
        steps = {'ss': args.steps, 'shape': args.steps, 'tex': args.steps}
    if args.ss_steps > 0:
        steps['ss'] = args.ss_steps
    if args.shape_steps > 0:
        steps['shape'] = args.shape_steps
    if args.tex_steps > 0:
        steps['tex'] = args.tex_steps

    run_inference(
        image_path=args.image,
        output_path=args.output,
        seed=args.seed,
        manual_fov=args.fov,
        model_path=args.model_path,
        low_vram=args.low_vram,
        resolution=args.resolution,
        vram=args.vram,
        max_num_tokens=args.max_num_tokens,
        texture_size=args.texture_size,
        decimation_target=args.decimate,
        attn_backend=args.attn,
        conv_backend=args.conv,
        block_offload=args.block_offload,
        dtype=args.dtype,
        cond_dtype=args.cond_dtype,
        cfg_batch=args.cfg_batch,
        timing=args.timing,
        **{f"{k}_sampling_steps": v for k, v in
           {'ss': steps.get('ss'), 'shape_slat': steps.get('shape'),
            'tex_slat': steps.get('tex')}.items() if v},
    )
