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
    parser.add_argument('--pbr_voxel_root', type=str, default=None,
                        help='Directory containing the pbr voxels')
    parser.add_argument('--pbr_latent_root', type=str, default=None,
                        help='Directory to save the pbr latent files')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--resolution', type=int, default=1024,
                        help='Sparse voxel resolution')
    parser.add_argument('--enc_pretrained', type=str, default='microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16',
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
    opt.pbr_voxel_root = opt.pbr_voxel_root or opt.root
    opt.pbr_latent_root = opt.pbr_latent_root or opt.root

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
    latent_view_name = f'{latent_name}_view_fix'
    os.makedirs(os.path.join(opt.pbr_latent_root, 'pbr_latents', latent_view_name, 'new_records'), exist_ok=True)
    
    # Get file list
    if not os.path.exists(os.path.join(opt.root, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.root, 'metadata.csv')).set_index('sha256')
    if os.path.exists(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')):
        aesthetic_metadata = pd.read_csv(os.path.join(opt.root, 'aesthetic_scores','metadata.csv')).set_index('sha256')
        metadata = metadata.join(aesthetic_metadata, how='left', rsuffix='_aesthetic')
    
    # Check pbr_voxels_view_fix metadata
    pbr_voxel_view_path = os.path.join(opt.pbr_voxel_root, f'pbr_voxels_view_fix_{opt.resolution}', 'metadata.csv')
    if os.path.exists(pbr_voxel_view_path):
        pbr_voxel_metadata = pd.read_csv(pbr_voxel_view_path).set_index('sha256')
        metadata = metadata.join(pbr_voxel_metadata, how='left', rsuffix='_pbr_voxel')
    
    # Check pbr_latent_view metadata (used to skip already completed tasks)
    pbr_latent_view_metadata_path = os.path.join(opt.pbr_latent_root, 'pbr_latents', latent_view_name, 'metadata.csv')
    if os.path.exists(pbr_latent_view_metadata_path):
        pbr_latent_view_metadata = pd.read_csv(pbr_latent_view_metadata_path).set_index('sha256')
        metadata = metadata.join(pbr_latent_view_metadata, how='left', rsuffix='_pbr_latent_view')
        print(f'Loaded pbr_latent_view metadata with {len(pbr_latent_view_metadata)} records')
    else:
        print(f'Warning: pbr_latent_view metadata not found at {pbr_latent_view_metadata_path}')
    
    metadata = metadata.reset_index()
    
    if opt.instances is None:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        
        # Filter to objects that have pbr_voxels_view_fix data
        # Use first view as indicator
        first_view_col = f'pbr_voxelized_view_fix{view_indices[0]:02d}_{opt.resolution}'
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

    records = []
    
    # Build task list: (sha256, view_idx), filter already completed tasks via metadata
    all_tasks = []
    skipped_count = 0
    
    # Pre-fetch completion status columns for each view
    encoded_cols = {view_idx: f'pbr_latent_view{view_idx:02d}_encoded' for view_idx in view_indices}
    
    for _, row in metadata.iterrows():
        sha256 = row['sha256']
        for view_idx in view_indices:
            encoded_col = encoded_cols[view_idx]
            # Check if already marked as completed in metadata
            if encoded_col in metadata.columns and row.get(encoded_col, False) == True:
                skipped_count += 1
                continue
            all_tasks.append((sha256, view_idx))
    
    # Split tasks by rank after filtering completed ones
    start = len(all_tasks) * opt.rank // opt.world_size
    end = len(all_tasks) * (opt.rank + 1) // opt.world_size
    tasks = all_tasks[start:end]
    
    print(f'Total tasks: {len(all_tasks) + skipped_count}, Already done (from metadata): {skipped_count}, To process (all ranks): {len(all_tasks)}, This rank: {len(tasks)}')
    
    load_queue = Queue(maxsize=32)
    
    with ThreadPoolExecutor(max_workers=32) as loader_executor, \
         ThreadPoolExecutor(max_workers=32) as saver_executor:

        def loader(task):
            sha256, view_idx = task
            try:
                # Check if output file already exists, skip if so (but still record)
                output_path = os.path.join(
                    opt.pbr_latent_root, 
                    'pbr_latents', 
                    latent_view_name, 
                    sha256, 
                    f'view{view_idx:02d}.npz'
                )
                if os.path.exists(output_path):
                    try:
                        data = np.load(output_path)
                        num_tokens = data['coords'].shape[0]
                    except Exception:
                        num_tokens = -1
                    records.append({
                        'sha256': sha256,
                        f'pbr_latent_view{view_idx:02d}_encoded': True,
                        f'pbr_latent_view{view_idx:02d}_tokens': num_tokens,
                    })
                    load_queue.put((sha256, view_idx, None))
                    return
                
                # pbr_voxels_view_fix path: pbr_voxels_view_fix_{res}/{sha256}/view{idx:02d}.vxz
                vxz_path = os.path.join(
                    opt.pbr_voxel_root, 
                    f'pbr_voxels_view_fix_{opt.resolution}', 
                    sha256, 
                    f'view{view_idx:02d}.vxz'
                )
                
                if not os.path.exists(vxz_path):
                    print(f"[Loader Skip] {sha256}/view{view_idx:02d}: vxz file not found")
                    load_queue.put((sha256, view_idx, None))
                    return
                
                attrs = ['base_color', 'metallic', 'roughness', 'alpha']
                coords, attr = o_voxel.io.read_vxz(vxz_path, num_threads=4)
                feats = torch.concat([attr[k] for k in attrs], dim=-1) / 255.0 * 2 - 1
                x = sp.SparseTensor(
                    feats.float(),
                    torch.cat([torch.zeros_like(coords[:, 0:1]), coords], dim=-1),
                )
                load_queue.put((sha256, view_idx, x))
            except Exception as e:
                print(f"[Loader Error] {sha256}/view{view_idx:02d}: {e}")
                load_queue.put((sha256, view_idx, None))

        loader_executor.map(loader, tasks)
        
        def saver(sha256, view_idx, pack):
            sha256_dir = os.path.join(opt.pbr_latent_root, 'pbr_latents', latent_view_name, sha256)
            os.makedirs(sha256_dir, exist_ok=True)
            save_path = os.path.join(sha256_dir, f'view{view_idx:02d}.npz')
            np.savez_compressed(save_path, **pack)
            
            # Copy scale json from pbr_voxels_view_fix
            src_scale_path = os.path.join(
                opt.pbr_voxel_root,
                f'pbr_voxels_view_fix_{opt.resolution}',
                sha256,
                f'view{view_idx:02d}_scale.json'
            )
            dst_scale_path = os.path.join(sha256_dir, f'view{view_idx:02d}_scale.json')
            if os.path.exists(src_scale_path):
                shutil.copy2(src_scale_path, dst_scale_path)
            
            records.append({
                'sha256': sha256,
                f'pbr_latent_view{view_idx:02d}_encoded': True,
                f'pbr_latent_view{view_idx:02d}_tokens': pack['coords'].shape[0]
            })
            
        for _ in tqdm(range(len(tasks)), desc=f"Extracting {os.path.basename(opt.root)} PBR view latents (res={opt.resolution})"):
            try:
                sha256, view_idx, voxels = load_queue.get()
                if voxels is None:
                    continue
                
                num_voxels = voxels.feats.shape[0]

                # NaN/Inf check
                if not is_valid_sparse_tensor(voxels):
                    print(f"[Skip] {sha256}/view{view_idx:02d}: NaN/Inf in input")
                    continue

                z = encoder(voxels.cuda())
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
    records.to_csv(os.path.join(opt.pbr_latent_root, 'pbr_latents', latent_view_name, 'new_records', f'part_{opt.rank}.csv'), index=False)
