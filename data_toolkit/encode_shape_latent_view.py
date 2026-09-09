import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
import argparse
import shutil
import torch
import numpy as np
import pandas as pd
import o_voxel
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from utils import parse_view_indices

import pixal3d.models as models
import pixal3d.modules.sparse as sp

torch.set_grad_enabled(False)


def is_valid_sparse_tensor(tensor):
    return torch.isfinite(tensor.feats).all() and torch.isfinite(tensor.coords).all()

def clear_cuda_error():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--dual_grid_root', type=str, default=None,
                        help='Directory containing the dual grids')
    parser.add_argument('--shape_latent_root', type=str, default=None,
                        help='Directory to save the shape latent files')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--resolution', type=int, default=1024,
                        help='Sparse voxel resolution')
    parser.add_argument('--enc_pretrained', type=str, default='microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16',
                        help='Pretrained encoder model')
    parser.add_argument('--model_root', type=str,
                        help='Root directory of models')
    parser.add_argument('--enc_model', type=str,
                        help='Encoder model. if specified, use this model instead of pretrained model')
    parser.add_argument('--ckpt', type=str,
                        help='Checkpoint to load')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--view_indices', type=str, default=None,
                        help='View indices to process, e.g., "0,1,2" or "0-5". None for all views')
    parser.add_argument('--num_views', type=int, default=24,
                        help='Total number of views (used when view_indices is None)')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    opt = parser.parse_args()
    opt = edict(vars(opt))
    opt.dual_grid_root = opt.dual_grid_root or opt.root
    opt.shape_latent_root = opt.shape_latent_root or opt.root

    # Parse view_indices
    view_indices = parse_view_indices(opt.view_indices)
    if view_indices is None:
        view_indices = list(range(opt.num_views))
    
    print(f'View indices to process: {view_indices}')

    if opt.enc_model is None:
        latent_name = f'{opt.enc_pretrained.split("/")[-1]}_{opt.resolution}'
        encoder = models.from_pretrained(opt.enc_pretrained).eval().cuda()
    else:
        latent_name = f'{opt.enc_model.split("/")[-1]}_{opt.ckpt}_{opt.resolution}'
        cfg = edict(json.load(open(os.path.join(opt.model_root, opt.enc_model, 'config.json'), 'r')))
        encoder = getattr(models, cfg.models.encoder.name)(**cfg.models.encoder.args).cuda()
        ckpt_path = os.path.join(opt.model_root, opt.enc_model, 'ckpts', f'encoder_{opt.ckpt}.pt')
        encoder.load_state_dict(torch.load(ckpt_path), strict=False)
        encoder.eval()
        print(f'Loaded model from {ckpt_path}')
    
    # Multi-view latent output directory
    latent_view_name = f'{latent_name}_view'
    os.makedirs(os.path.join(opt.shape_latent_root, 'shape_latents', latent_view_name, 'new_records'), exist_ok=True)
    
    # Get file list
    if not os.path.exists(os.path.join(opt.root, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.root, 'metadata.csv')).set_index('sha256')
    if os.path.exists(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')):
        aesthetic_metadata = pd.read_csv(os.path.join(opt.root, 'aesthetic_scores','metadata.csv')).set_index('sha256')
        metadata = metadata.join(aesthetic_metadata, how='left', rsuffix='_aesthetic')
    
    # Check dual_grid_view metadata
    dual_grid_view_path = os.path.join(opt.dual_grid_root, f'dual_grid_view_{opt.resolution}', 'metadata.csv')
    if os.path.exists(dual_grid_view_path):
        dual_grid_metadata = pd.read_csv(dual_grid_view_path).set_index('sha256')
        metadata = metadata.join(dual_grid_metadata, how='left', rsuffix='_dual_grid')
    
    # Check shape_latent_view metadata (used to skip already completed tasks)
    shape_latent_view_metadata_path = os.path.join(opt.shape_latent_root, 'shape_latents', latent_view_name, 'metadata.csv')
    if os.path.exists(shape_latent_view_metadata_path):
        shape_latent_view_metadata = pd.read_csv(shape_latent_view_metadata_path).set_index('sha256')
        metadata = metadata.join(shape_latent_view_metadata, how='left', rsuffix='_shape_latent_view')
        print(f'Loaded shape_latent_view metadata with {len(shape_latent_view_metadata)} records')
    else:
        print(f'Warning: shape_latent_view metadata not found at {shape_latent_view_metadata_path}')
    
    metadata = metadata.reset_index()
    
    if opt.instances is None:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        
        # Filter to objects that have dual_grid_view data
        # Use first view as indicator
        first_view_col = f'dual_grid_view{view_indices[0]:02d}_converted_{opt.resolution}'
        if first_view_col in metadata.columns:
            metadata = metadata[metadata[first_view_col] == True]
        else:
            print(f'Warning: Column {first_view_col} not found in metadata, will check files directly')
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []
    
    # Build task list: (sha256, view_idx), filter already completed tasks via metadata
    tasks = []
    skipped_count = 0
    
    # Pre-fetch completion status columns for each view
    encoded_cols = {view_idx: f'shape_latent_view{view_idx:02d}_encoded' for view_idx in view_indices}
    
    for _, row in metadata.iterrows():
        sha256 = row['sha256']
        for view_idx in view_indices:
            encoded_col = encoded_cols[view_idx]
            # Check if already marked as completed in metadata
            if encoded_col in metadata.columns and row.get(encoded_col, False) == True:
                skipped_count += 1
                continue
            tasks.append((sha256, view_idx))
    
    print(f'Total tasks: {len(tasks) + skipped_count}, Already done (from metadata): {skipped_count}, To process: {len(tasks)}')
    
    load_queue = Queue(maxsize=32)
    
    with ThreadPoolExecutor(max_workers=32) as loader_executor, \
         ThreadPoolExecutor(max_workers=32) as saver_executor:

        def loader(task):
            sha256, view_idx = task
            try:
                # Check if output file already exists, skip if so
                output_path = os.path.join(
                    opt.shape_latent_root, 
                    'shape_latents', 
                    latent_view_name, 
                    sha256, 
                    f'view{view_idx:02d}.npz'
                )
                if os.path.exists(output_path):
                    load_queue.put((sha256, view_idx, None, None))
                    return
                
                # dual_grid_view path: dual_grid_view_{res}/{sha256}/view{idx:02d}.vxz
                vxz_path = os.path.join(
                    opt.dual_grid_root, 
                    f'dual_grid_view_{opt.resolution}', 
                    sha256, 
                    f'view{view_idx:02d}.vxz'
                )
                
                if not os.path.exists(vxz_path):
                    print(f"[Loader Skip] {sha256}/view{view_idx:02d}: vxz file not found")
                    load_queue.put((sha256, view_idx, None, None))
                    return
                
                coords, attr = o_voxel.io.read_vxz(vxz_path, num_threads=4)
                vertices = sp.SparseTensor(
                    (attr['vertices'] / 255.0).float(),
                    torch.cat([torch.zeros_like(coords[:, 0:1]), coords], dim=-1),
                )
                intersected = vertices.replace(torch.cat([
                    attr['intersected'] % 2,
                    attr['intersected'] // 2 % 2,
                    attr['intersected'] // 4 % 2,
                ], dim=-1).bool())
                load_queue.put((sha256, view_idx, vertices, intersected))
            except Exception as e:
                print(f"[Loader Error] {sha256}/view{view_idx:02d}: {e}")
                load_queue.put((sha256, view_idx, None, None))

        loader_executor.map(loader, tasks)
        
        def saver(sha256, view_idx, pack):
            sha256_dir = os.path.join(opt.shape_latent_root, 'shape_latents', latent_view_name, sha256)
            os.makedirs(sha256_dir, exist_ok=True)
            save_path = os.path.join(sha256_dir, f'view{view_idx:02d}.npz')
            np.savez_compressed(save_path, **pack)
            
            # Copy scale json from dual_grid_view
            src_scale_path = os.path.join(
                opt.dual_grid_root,
                f'dual_grid_view_{opt.resolution}',
                sha256,
                f'view{view_idx:02d}_scale.json'
            )
            dst_scale_path = os.path.join(sha256_dir, f'view{view_idx:02d}_scale.json')
            if os.path.exists(src_scale_path):
                shutil.copy2(src_scale_path, dst_scale_path)
            
            records.append({
                'sha256': sha256,
                f'shape_latent_view{view_idx:02d}_encoded': True,
                f'shape_latent_view{view_idx:02d}_tokens': pack['coords'].shape[0]
            })
            
        for _ in tqdm(range(len(tasks)), desc="Extracting view latents"):
            try:
                sha256, view_idx, vertices, intersected = load_queue.get()
                if vertices is None or intersected is None:
                    continue
                
                num_voxels = vertices.feats.shape[0]

                # NaN/Inf check
                if not (is_valid_sparse_tensor(vertices) and is_valid_sparse_tensor(intersected)):
                    print(f"[Skip] {sha256}/view{view_idx:02d}: NaN/Inf in input")
                    continue

                z = encoder(vertices.cuda(), intersected.cuda())
                torch.cuda.synchronize()

                if not torch.isfinite(z.feats).all():
                    print(f"[Skip] {sha256}/view{view_idx:02d}: Non-finite latent in z.feats")
                    clear_cuda_error()
                    continue

                pack = {
                    'feats': z.feats.cpu().numpy().astype(np.float32),
                    'coords': z.coords[:, 1:].cpu().numpy().astype(np.uint8),
                }
                saver_executor.submit(saver, sha256, view_idx, pack)

            except Exception as e:
                print(f"[Error] {sha256}/view{view_idx:02d} ({num_voxels} voxels): {e}")
                clear_cuda_error()
                continue
            
        saver_executor.shutdown(wait=True)
        
    records = pd.DataFrame.from_records(records)
    records.to_csv(os.path.join(opt.shape_latent_root, 'shape_latents', latent_view_name, 'new_records', f'part_{opt.rank}.csv'), index=False)
