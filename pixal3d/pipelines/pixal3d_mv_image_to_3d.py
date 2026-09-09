from typing import *
import torch
import torch.nn as nn
from .pixal3d_image_to_3d import Pixal3DImageTo3DPipeline
from ..modules.sparse import SparseTensor
from .. import runtime


class Pixal3DMVImageTo3DPipeline(Pixal3DImageTo3DPipeline):
    """
    Multi-view variant of the Pixal3D proj pipeline.

    Same cascade as the single-view pipeline (SS -> Shape 512 -> Shape 1024 -> Tex 1024);
    the only difference is the image condition: instead of one image at the canonical
    front view, it takes V views with explicit c2w matrices and lets
    DinoV3ProjMultiViewFeatureExtractor project all of them into the shared 3D grid,
    averaging the per-view features.

    Why the denoisers are unchanged: with multiview_fusion="average" the fused z_proj
    has exactly the single-view shape [B, R^3, C], so every per-block proj_linear /
    proj cross-attn weight stays compatible. View 0 is the main view and its
    calc_mat_0 == F by construction, so V=1 degenerates to the single-view path.

    This class only overrides the two condition builders. `run()` is inherited
    verbatim: there, `image` and `camera_params` are pure pass-through to those two
    builders, so we route the whole multi-view bundle through them (see `run_mv`).
    """

    def _views_of(self, image) -> dict:
        """Unwrap the views bundle that `run()` passes along as its `image` argument."""
        views = image[0] if isinstance(image, (list, tuple)) else image
        if not isinstance(views, dict) or 'images' not in views:
            raise TypeError(
                "Pixal3DMVImageTo3DPipeline expects a views dict (see run_mv), "
                f"got {type(views).__name__}. Use run_mv() instead of run().")
        return views

    @staticmethod
    def _image_for(views: dict, image_cond_model: nn.Module) -> torch.Tensor:
        """
        Pick the pre-decoded [1, V, 3, H, W] batch matching this stage's input size.

        Unlike the single-view extractor (which takes PIL and resizes internally), the
        multi-view one takes tensors, so the caller has to supply the right resolution:
        the SS / shape-512 stages run at 512 and the 1024 stages at 1024.
        """
        size = image_cond_model.image_size
        if size not in views['images']:
            raise KeyError(
                f"No view images at resolution {size}; got {sorted(views['images'])}. "
                f"Build them in load_views().")
        return views['images'][size]

    def _run_extractor(self, image_cond_model: nn.Module, views: dict,
                       camera_angle_x, distance, mesh_scale):
        device = self.device
        image = self._image_for(views, image_cond_model).to(device)
        cam_angle = torch.as_tensor(camera_angle_x, dtype=torch.float32, device=device)
        dist = torch.as_tensor(distance, dtype=torch.float32, device=device)
        transform_matrix = views['transform_matrix'].to(device)
        # mesh_scale is per-object, i.e. [B]; the extractor expands it over views.
        scale = torch.as_tensor(mesh_scale, dtype=torch.float32, device=device).reshape(-1)
        with runtime.cond_autocast():
            return image_cond_model(
                image,
                camera_angle_x=cam_angle,
                distance=dist,
                mesh_scale=scale,
                transform_matrix=transform_matrix,
            )

    @torch.no_grad()
    def get_proj_cond_ss(
        self,
        image,
        camera_angle_x=None,
        distance=None,
        mesh_scale=1.0,
    ) -> dict:
        """Multi-view proj conditioning for the sparse structure stage (dense grid)."""
        views = self._views_of(image)
        image_cond_model = self.image_cond_model_ss
        if self.low_vram:
            image_cond_model.to(self.device)
        z_global, z_proj = self._run_extractor(
            image_cond_model, views, camera_angle_x, distance, mesh_scale)
        if self.low_vram:
            image_cond_model.cpu()
        return {
            'cond': {'global': z_global, 'proj': z_proj},
            'neg_cond': {'global': torch.zeros_like(z_global), 'proj': torch.zeros_like(z_proj)},
        }

    @torch.no_grad()
    def get_proj_cond_shape(
        self,
        image_cond_model: nn.Module,
        image,
        coords: torch.Tensor,
        camera_angle_x=None,
        distance=None,
        mesh_scale=1.0,
        grid_resolution_override: int = None,
    ) -> dict:
        """Multi-view proj conditioning for the shape / texture stages (sparse tokens)."""
        views = self._views_of(image)
        device = self.device
        if self.low_vram:
            image_cond_model.to(device)

        # The HR cascade's grid resolution floats with the token budget (1536/16=96 and
        # down), so it can differ from what this stage was trained at; swap the grid.
        orig_grid_res = image_cond_model.grid_resolution
        override = (grid_resolution_override is not None
                    and grid_resolution_override != orig_grid_res)
        if override:
            image_cond_model.grid_resolution = grid_resolution_override
            image_cond_model.proj_grid = image_cond_model.proj_grid.__class__(
                grid_resolution=grid_resolution_override,
                image_resolution=image_cond_model.proj_grid.image_resolution,
            ).to(device)

        z_global, z_proj = self._run_extractor(
            image_cond_model, views, camera_angle_x, distance, mesh_scale)

        B = z_global.shape[0]
        grid_res = image_cond_model.grid_resolution
        b_idx = coords[:, 0].long()
        x_idx = coords[:, 1].long()
        y_idx = coords[:, 2].long()
        z_idx = coords[:, 3].long()
        z_proj_grid = z_proj.reshape(B, grid_res, grid_res, grid_res, -1)
        z_proj_sparse = z_proj_grid[b_idx, x_idx, y_idx, z_idx]
        z_proj_st = SparseTensor(feats=z_proj_sparse, coords=coords)

        if override:
            image_cond_model.grid_resolution = orig_grid_res
            image_cond_model.proj_grid = image_cond_model.proj_grid.__class__(
                grid_resolution=orig_grid_res,
                image_resolution=image_cond_model.proj_grid.image_resolution,
            ).to(device)

        if self.low_vram:
            image_cond_model.cpu()
        return {
            'cond': {'global': z_global, 'proj': z_proj_st},
            'neg_cond': {
                'global': torch.zeros_like(z_global),
                'proj': SparseTensor(feats=torch.zeros_like(z_proj_sparse), coords=coords),
            },
        }

    @torch.no_grad()
    def run_mv(self, views: dict, **kwargs):
        """
        Run the cascade on a multi-view bundle.

        Args:
            views: as built by inference_mv.load_views():
                images:           {image_size: [1, V, 3, H, W]} alpha-premultiplied
                camera_angle_x:   [1, V] horizontal fov in radians
                camera_distance:  [1, V] camera distance (norm of the c2w translation)
                transform_matrix: [1, V, 4, 4] c2w, index 0 = main view
                mesh_scale:       float
            **kwargs: forwarded to the inherited run() (seed, sampler params,
                pipeline_type, max_num_tokens, return_latent, ...).

        The bundle rides along as run()'s `image` argument and the camera tensors as its
        `camera_params`; run() only forwards both to the overridden condition builders.
        """
        return super().run(
            views,
            camera_params={
                'camera_angle_x': views['camera_angle_x'],
                'distance': views['camera_distance'],
                'mesh_scale': views['mesh_scale'],
            },
            preprocess_image=False,
            **kwargs,
        )
