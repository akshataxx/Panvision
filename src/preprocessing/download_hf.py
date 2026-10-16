"""Download selected original CT volumes from the PANORAMA community mirror."""

import argparse
import json
import shutil
from pathlib import Path

from src.preprocessing.storage import data_root


REPO = 'MedOtter/PANORAMA'


def select_images(siblings, count=5, offset=0, studies=None):
    images = sorted((f for f in siblings
                     if f.rfilename.startswith('images/')
                     and f.rfilename.endswith('_0000.nii.gz')),
                    key=lambda f: f.rfilename)
    if studies:
        wanted = set(studies)
        selected = [f for f in images if Path(f.rfilename).name.removesuffix('_0000.nii.gz') in wanted]
        found = {Path(f.rfilename).name.removesuffix('_0000.nii.gz') for f in selected}
        if wanted != found:
            raise ValueError(f'Studies not found: {sorted(wanted - found)}')
        return selected
    if count < 1 or offset < 0:
        raise ValueError('count must be positive and offset nonnegative')
    selected = images[offset:offset + count]
    if not selected:
        raise ValueError('No images selected; offset exceeds available studies')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--studies', nargs='+', help='Explicit study IDs instead of a range')
    parser.add_argument('--revision', help='Exact mirror commit to use; saved after first download')
    parser.add_argument('--download', action='store_true', help='Actually transfer scans; default only lists them')
    args = parser.parse_args()
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        parser.exit(1, 'Install the downloader first: python -m pip install huggingface_hub\n')
    root = data_root()
    source = root / 'manifests/huggingface_source.json'
    try:
        previous = json.loads(source.read_text()) if source.exists() else None
        revision = args.revision or (previous['revision'] if previous else 'main')
        if previous and revision != previous['revision']:
            raise ValueError('Revision differs from existing downloads. Use a different data root for a new revision.')
        info = HfApi().dataset_info(REPO, revision=revision, files_metadata=True)
        selected = select_images(info.siblings, args.count, args.offset, args.studies)
        if any(f.size is None for f in selected):
            raise ValueError('Server did not return file sizes; cannot check storage safely')
        total = sum(f.size for f in selected)
        probe = root
        while not probe.exists():
            probe = probe.parent
        free = shutil.disk_usage(probe).free
        print(f'Mirror: {REPO}\nRevision: {info.sha}')
        print(f'{len(selected)} scans, {total / 1e9:.2f} GB; local free space {free / 1e9:.1f} GB')
        for file in selected:
            print(f'  {file.rfilename}: {file.size / 1e6:.1f} MB')
        if not args.download:
            print('Preview only. Add --download to transfer these scans.')
            return
        # Conservatively reserve the full transfer size, even when some files exist.
        if free < total + 5 * 1024**3:
            raise ValueError('Not enough local free space, including a 5 GiB reserve')
        source.parent.mkdir(parents=True, exist_ok=True)
        record = previous or {'repo_id': REPO, 'revision': info.sha, 'files': {}}
        source.write_text(json.dumps(record, indent=2) + '\n')
        for file in selected:
            path = hf_hub_download(repo_id=REPO, repo_type='dataset', revision=info.sha,
                                   filename=file.rfilename, local_dir=str(root / 'raw'))
            record['files'][file.rfilename] = {'size_bytes': file.size}
            source.write_text(json.dumps(record, indent=2) + '\n')
            print(f'Downloaded: {path}')
        print('Scans downloaded. Keep official annotations in raw/annotations, then build and validate the manifest.')
        print('This is an exploratory subset; do not establish the final shared split from the first five scans.')
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
