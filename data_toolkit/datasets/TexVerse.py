import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd


def add_args(parser: argparse.ArgumentParser):
    pass


def get_metadata(**kwargs):
    raise NotImplementedError("TexVerse metadata should be built from metadata.json using build_texverse_metadata.py")


def download(metadata, root, **kwargs):
    raise NotImplementedError("TexVerse GLB files are already available locally. No download needed.")


def _process_instance(args):
    """Worker function for ProcessPoolExecutor (must be top-level for pickling)"""
    metadatum, output_dir, func = args
    try:
        local_path = metadatum['local_path']
        sha256 = metadatum['sha256']
        file = os.path.join(output_dir, local_path)
        record = func(file, sha256)
        return record
    except Exception as e:
        print(f"Error processing object {metadatum.get('sha256', '?')}: {e}")
        return None


def foreach_instance(metadata, output_dir, func, max_workers=None, desc='Processing objects', timeout=None) -> pd.DataFrame:
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
    from tqdm import tqdm

    metadata = metadata.to_dict('records')

    max_workers = max_workers or os.cpu_count()
    records = []
    timeout_count = 0

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_instance, (m, output_dir, func)): m['sha256']
                for m in metadata
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                sha256 = futures[future]
                try:
                    r = future.result(timeout=timeout)
                    if r is not None:
                        records.append(r)
                except TimeoutError:
                    timeout_count += 1
                    print(f"Timeout processing object {sha256} (>{timeout}s)")
                    records.append({'sha256': sha256, 'error': f'Timeout (>{timeout}s)'})
                except Exception as e:
                    print(f"Error processing object {sha256}: {e}")
    except Exception as e:
        print(f"Error happened during processing: {e}")

    if timeout_count > 0:
        print(f"Total timeout: {timeout_count} objects")

    return pd.DataFrame.from_records(records)
