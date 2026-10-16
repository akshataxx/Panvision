"""Plan or run the full cohort, one study at a time, with resumable outputs."""

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import SimpleITK as sitk

from src.preprocessing.build_manifest import build_manifest
from src.preprocessing.create_splits import clinical_cohort
from src.preprocessing.download_hf import REPO, select_images
from src.preprocessing.pipeline import run as prepare, validate_splits
from src.preprocessing.storage import data_root


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def cohort_files(cohort, siblings):
    files = select_images(siblings, count=len(siblings))
    indexed = {}
    for file in files:
        study = Path(file.rfilename).name.removesuffix('_0000.nii.gz')
        if study in indexed:
            raise ValueError(f'Duplicate mirror CT: {study}')
        indexed[study] = file
    missing = set(cohort.study_id) - set(indexed)
    if missing:
        raise ValueError(f'{len(missing)} clinical studies missing in mirror: {sorted(missing)[:5]}')
    return [(study, indexed[study]) for study in sorted(cohort.study_id)]


def completed(path, signature):
    receipt = path / 'complete.json'
    if not receipt.exists():
        return False
    data = json.loads(receipt.read_text())
    if data['signature'] != signature:
        raise ValueError(f'Output configuration/source differs: {path}')
    for relative, size in data['outputs'].items():
        target = path / relative
        if not target.is_file() or target.stat().st_size != size:
            raise ValueError(f'Completed output is missing or changed: {target}')
    return True


def process_study(root, study, file, revision, signature, split_path, reserve, download):
    destination = root / 'processed/full' / study
    if completed(destination, signature):
        return 'complete'
    if destination.exists():
        raise ValueError(f'Incomplete final directory needs review: {destination}')
    raw = root / 'raw'
    if file.size is None:
        raise ValueError('Missing remote CT size')
    if shutil.disk_usage(root).free < file.size + reserve:
        raise OSError('LOW_DISK: free local space before downloading another scan')
    download(repo_id=REPO, repo_type='dataset', revision=revision,
             filename=file.rfilename, local_dir=str(raw))
    frame, _ = build_manifest(raw / 'images', raw / 'annotations',
                              raw / 'annotations/clinical_information.xlsx', studies=[study])
    # Estimate uncompressed output size after canonical axis permutation.
    image = sitk.DICOMOrient(sitk.ReadImage(frame.iloc[0].image_path), 'LPS')
    dimensions = [max(2, int(math.ceil((n - 1) * old / new)) + 1)
                  for n, old, new in zip(image.GetSize(), image.GetSpacing(), (1.5, 1.5, 3.0))]
    del image
    required = math.prod(dimensions) * 8 + reserve
    if shutil.disk_usage(root).free < required:
        raise OSError('LOW_DISK: insufficient space for processed output plus reserve')
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Each retry uses a new staging directory. Old partial data is retained for review.
    staging = Path(tempfile.mkdtemp(prefix=f'.partial-{study}-', dir=destination.parent))
    manifest = staging / 'input.csv'
    frame.to_csv(manifest, index=False)
    prepare(SimpleNamespace(command='prepare', manifest=manifest, splits=split_path,
                            output=staging / 'volumes', seed=42,
                            spacing=[1.5, 1.5, 3.0], window=[-100, 300]))
    # Portable output references: these remain valid on teammates' machines.
    output_manifest = staging / 'volumes/manifest.csv'
    result = pd.read_csv(output_manifest, dtype=str, keep_default_na=False)
    for column in ('processed_image', 'processed_labels'):
        result[column] = result[column].map(
            lambda value: str(Path(value).relative_to(staging / 'volumes')) if value else '')
    result['output_paths_relative_to'] = 'manifest_directory'
    for column in ('image_path', 'mask_path'):
        result[column] = result[column].map(lambda value: str(Path(value).relative_to(root)) if value else '')
    result['raw_paths_relative_to'] = 'PanVisionData_root'
    result.to_csv(output_manifest, index=False)
    # Remove stale staging paths in provenance (source settings remain in the receipt).
    config_file = staging / 'volumes/config.json'
    config = json.loads(config_file.read_text())
    config['input_manifest'] = '../input.csv'
    config['splits'] = '../splits.csv'
    save_json(config_file, config)
    shutil.copyfile(split_path, staging / 'splits.csv')
    frame['image_path'] = frame.image_path.map(lambda value: str(Path(value).relative_to(root)))
    frame['mask_path'] = frame.mask_path.map(lambda value: str(Path(value).relative_to(root)) if value else '')
    # input.csv is provenance relative to dataset root, not a directly runnable local manifest.
    frame.to_csv(manifest, index=False)
    outputs = {str(p.relative_to(staging)): p.stat().st_size
               for p in staging.rglob('*') if p.is_file()}
    save_json(staging / 'complete.json', {'study_id': study, 'signature': signature,
                                         'revision': revision, 'outputs': outputs,
                                         'input_paths_relative_to': 'PanVisionData_root'})
    staging.rename(destination)
    return 'complete'


def run(args, api, download):
    root = Path(args.data_root).resolve()
    if not root.is_dir():
        raise ValueError('Configured data root does not exist')
    clinical = root / 'raw/annotations/clinical_information.xlsx'
    cohort = clinical_cohort(clinical)
    split_path = Path(args.splits).resolve()
    splits = pd.read_csv(split_path, dtype=str, keep_default_na=False)
    validate_splits(cohort, splits)
    source = root / 'manifests/huggingface_source.json'
    previous = json.loads(source.read_text()) if source.exists() else None
    if previous and previous.get('repo_id') != REPO:
        raise ValueError('Recorded source repository differs from the expected mirror')
    info = api.dataset_info(REPO, revision=previous['revision'] if previous else 'main', files_metadata=True)
    files = cohort_files(cohort, info.siblings)
    signature = hashlib.sha256(json.dumps({
        'revision': info.sha, 'clinical': hashlib.sha256(clinical.read_bytes()).hexdigest(),
        'splits': hashlib.sha256(split_path.read_bytes()).hexdigest(),
        'spacing': [1.5, 1.5, 3.0], 'window': [-100, 300], 'pipeline_version': 1,
        'annotations': sorted((str(p.relative_to(root / 'raw/annotations')), p.stat().st_size)
                              for p in (root / 'raw/annotations').rglob('*.nii.gz')),
    }, sort_keys=True).encode()).hexdigest()
    pending = [(s, f) for s, f in files if not completed(root / 'processed/full' / s, signature)]
    print(f'Cohort: {len(files)} studies; complete: {len(files) - len(pending)}; pending: {len(pending)}')
    print(f'Revision: {info.sha}\nOutput: {root / "processed/full"}')
    print(f'Local free space: {shutil.disk_usage(root).free / 1024**3:.1f} GiB; reserve: {args.reserve_gib} GiB')
    if not args.run:
        print('Plan only. Add --run to process; default stops after one batch.')
        return 0
    source.parent.mkdir(parents=True, exist_ok=True)
    save_json(source, previous or {'repo_id': REPO, 'revision': info.sha, 'files': {}})
    status_path = root / 'manifests/full_progress.json'
    status = {'signature': signature, 'total': len(files), 'completed_before_run': len(files)-len(pending),
              'attempts': {}, 'state': 'running'}
    limit = len(pending) if args.max_batches == 0 else args.batch_size * args.max_batches
    failures = 0
    for study, file in pending[:limit]:
        try:
            process_study(root, study, file, info.sha, signature, split_path,
                          args.reserve_gib * 1024**3, download)
            status['attempts'][study] = {'status': 'complete'}
            failures = 0
            print(f'Completed {study}')
        except Exception as exc:
            status['attempts'][study] = {'status': 'failed', 'error': str(exc)}
            failures += 1
            print(f'Failed {study}: {exc}')
            with (source.parent / 'full_errors.jsonl').open('a') as log:
                log.write(json.dumps({'study_id': study, 'signature': signature, 'error': str(exc)}) + '\n')
            if 'LOW_DISK:' in str(exc) or failures >= 3:
                status['state'] = 'needs_attention'
                save_json(status_path, status)
                print('Stopped. Review errors or free synced local copies, then rerun the same command.')
                return 1
        save_json(status_path, status)
    status['state'] = 'run_finished'
    status['remaining'] = len(files) - sum(completed(root / 'processed/full' / s, signature) for s, _ in files)
    save_json(status_path, status)
    print(f'Remaining studies: {status["remaining"]}. Progress: {status_path}')
    return int(any(v['status'] == 'failed' for v in status['attempts'].values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default=str(data_root()))
    parser.add_argument('--splits', default='data/splits.csv')
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--max-batches', type=int, default=1, help='0 = continue until finished or stopped')
    parser.add_argument('--reserve-gib', type=float, default=10)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if (args.batch_size < 1 or args.max_batches < 0 or
            not math.isfinite(args.reserve_gib) or args.reserve_gib < 5):
        parser.error('batch-size must be positive, max-batches nonnegative, reserve-gib at least 5')
    from huggingface_hub import HfApi, hf_hub_download
    try:
        # Local advisory lock avoids two processes using this root on one machine.
        # OneDrive does not provide a distributed lock: assign one team writer.
        import fcntl
        lock_id = hashlib.sha256(str(Path(args.data_root).resolve()).encode()).hexdigest()
        with open(Path(tempfile.gettempdir()) / f'panvision-{lock_id}.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            code = run(args, HfApi(), hf_hub_download)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    raise SystemExit(code)


if __name__ == '__main__':
    main()
