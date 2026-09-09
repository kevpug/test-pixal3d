"""
Build metadata.csv for TexVerse dataset from metadata.json.

Output format matches ABO metadata.csv:
    sha256, file_identifier, local_path

Usage:
    python data_toolkit/build_texverse_metadata.py \
        --metadata_json /path/to/TexVerse/metadata.json \
        --data_root /path/to/TexVerse \
        --output /path/to/TexVerse/metadata.csv \
        --max_workers 32
"""

import os
import json
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd


def get_file_hash(file: str) -> str:
    sha256 = hashlib.sha256()
    with open(file, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256.update(byte_block)
    return sha256.hexdigest()


def process_one(uid, glb_paths, data_root):
    """Find the first existing GLB file for this uid and compute its sha256."""
    for rel_path in glb_paths:
        full_path = os.path.join(data_root, rel_path)
        if os.path.exists(full_path):
            sha256 = get_file_hash(full_path)
            return {
                'sha256': sha256,
                'file_identifier': uid,
                'local_path': rel_path,
            }
    return None


def main():
    parser = argparse.ArgumentParser(description='Build metadata.csv for TexVerse')
    parser.add_argument('--metadata_json', type=str, required=True,
                        help='Path to TexVerse metadata.json')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of TexVerse dataset (where glbs/ is)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for metadata.csv')
    parser.add_argument('--max_workers', type=int, default=16,
                        help='Number of parallel workers')
    args = parser.parse_args()

    with open(args.metadata_json, 'r') as f:
        metadata = json.load(f)

    print(f'Total entries in metadata.json: {len(metadata)}')

    # Load existing metadata.csv and skip already processed entries
    existing_uids = set()
    existing_records = []
    if os.path.exists(args.output):
        existing_df = pd.read_csv(args.output)
        existing_uids = set(existing_df['file_identifier'].values)
        existing_records = existing_df.to_dict('records')
        print(f'Found existing metadata.csv with {len(existing_uids)} entries, skipping them')

    # Filter out already processed uids
    to_process = {uid: info for uid, info in metadata.items() if uid not in existing_uids}
    print(f'New entries to process: {len(to_process)}')

    if len(to_process) == 0:
        print('Nothing to do, all entries already exist.')
        return

    new_records = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_one, uid, info['glb_paths'], args.data_root): uid
            for uid, info in to_process.items()
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc='Building metadata'):
            try:
                result = future.result()
                if result is not None:
                    new_records.append(result)
            except Exception as e:
                uid = futures[future]
                print(f'Error processing {uid}: {e}')

    all_records = existing_records + new_records
    df = pd.DataFrame.from_records(all_records, columns=['sha256', 'file_identifier', 'local_path'])
    df.to_csv(args.output, index=False)
    print(f'Added {len(new_records)} new entries, total {len(df)} entries in {args.output}')


if __name__ == '__main__':
    main()
