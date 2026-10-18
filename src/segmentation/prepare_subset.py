"""Preprocess a balanced PANORAMA subset on Colab (CPU is enough), reusing src.preprocessing unchanged.

Each selected study is downloaded to fast local disk, processed with src.preprocessing.full_dataset.process_study
(same HU window, spacing, orientation and completion receipt as the team's full run), copied to
<shared>/processed/full/<study_id>, and its raw CT is deleted. Re-running skips studies already complete in the
shared folder, so the team's full run will also recognise these outputs as complete.

    python -m src.segmentation.prepare_subset --shared /content/data --train 300 --val 80 --test 80 --dry-run
    python -m src.segmentation.prepare_subset --shared /content/data --train 300 --val 80 --test 80

Selection per split: half PDAC with a *manual* lesion mask (needed for lesion supervision/evaluation), half
non-PDAC, sampled with a fixed seed.
"""

import argparse
import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from src.preprocessing.create_splits import clinical_cohort
from src.preprocessing.full_dataset import cohort_files, completed, process_study
from src.preprocessing.pipeline import validate_splits
from src.preprocessing.download_hf import REPO


def select(cohort, splits, manual_studies, counts, seed=42):
    frame = cohort.merge(splits[['patient_id', 'split']], on='patient_id', validate='many_to_one')
    frame['manual'] = frame.study_id.isin(manual_studies)
    picks = []
    for split, count in counts.items():
        part = frame[frame.split == split]
        pdac = part[(part.label == '1') & part.manual]
        non_pdac = part[part.label == '0']
        n_pdac = min(count // 2, len(pdac))
        picks += [pdac.sample(n_pdac, random_state=seed),
                  non_pdac.sample(min(count - n_pdac, len(non_pdac)), random_state=seed)]
    return pd.concat(picks).sort_values('study_id').reset_index(drop=True)


def signature_for(root, clinical, split_path, revision):
    """Identical to the signature in src.preprocessing.full_dataset.run, so receipts are interchangeable."""
    return hashlib.sha256(json.dumps({
        'revision': revision, 'clinical': hashlib.sha256(clinical.read_bytes()).hexdigest(),
        'splits': hashlib.sha256(split_path.read_bytes()).hexdigest(),
        'spacing': [1.5, 1.5, 3.0], 'window': [-100, 300], 'pipeline_version': 1,
        'annotations': sorted((str(p.relative_to(root / 'raw/annotations')), p.stat().st_size)
                              for p in (root / 'raw/annotations').rglob('*.nii.gz')),
    }, sort_keys=True).encode()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--shared', required=True, help='shared PanVisionData folder (contains raw/annotations)')
    parser.add_argument('--work', default='/content/pv', help='fast local scratch root')
    parser.add_argument('--splits', default='data/splits.csv')
    parser.add_argument('--train', type=int, default=300)
    parser.add_argument('--val', type=int, default=80)
    parser.add_argument('--test', type=int, default=80)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)

    shared, work = Path(args.shared), Path(args.work)
    annotations = shared / 'raw/annotations'
    clinical = annotations / 'clinical_information.xlsx'
    missing = [p.name for p in (clinical, annotations / 'manual_labels', annotations / 'automatic_labels')
               if not p.exists()]
    if missing:
        listing = sorted(p.name + ('/' if p.is_dir() else '') for p in annotations.iterdir()) if annotations.exists() else []
        raise SystemExit(f'{annotations} is missing {missing}. It contains: {listing}. '
                         'Expected the official DIAGNijmegen/panorama_labels layout.')
    split_path = Path(args.splits).resolve()
    cohort = clinical_cohort(clinical)
    splits = pd.read_csv(split_path, dtype=str, keep_default_na=False)
    validate_splits(cohort, splits)
    manual = {p.name.removesuffix('.nii.gz') for p in (annotations / 'manual_labels').glob('*.nii.gz')}
    chosen = select(cohort, splits, manual, {'train': args.train, 'val': args.val, 'test': args.test}, args.seed)
    print(chosen.groupby(['split', 'label']).size().unstack(fill_value=0).rename(columns={'0': 'non-PDAC', '1': 'PDAC'}))

    # Local root mirrors the shared layout. Annotations (~80 MB) are copied, not linked: the preprocessing code
    # resolves mask paths and stores them relative to the root, which fails across a symlink.
    (work / 'raw').mkdir(parents=True, exist_ok=True)
    if not (work / 'raw/annotations/clinical_information.xlsx').exists():
        print('Copying annotations to local disk...', flush=True)
        shutil.copytree(annotations, work / 'raw/annotations', dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('.git'))
    from huggingface_hub import HfApi, hf_hub_download
    source = shared / 'manifests/huggingface_source.json'
    previous = json.loads(source.read_text()) if source.exists() else None
    info = HfApi().dataset_info(REPO, revision=previous['revision'] if previous else 'main', files_metadata=True)
    files = dict(cohort_files(cohort, info.siblings))
    print('Hashing annotation listing for the completion signature (one-off, can take a minute)...', flush=True)
    signature = signature_for(work, work / 'raw/annotations/clinical_information.xlsx', split_path, info.sha)
    target_root = shared / 'processed/full'
    pending = [s for s in chosen.study_id if not completed(target_root / s, signature)]
    print(f'Selected {len(chosen)}; already complete in shared folder: {len(chosen) - len(pending)}; '
          f'pending: {len(pending)}; mirror revision {info.sha[:10]}', flush=True)
    if args.dry_run:
        return

    def one(study):
        final = target_root / study
        if final.exists():
            raise ValueError(f'Incomplete folder in shared storage needs review: {final}')
        process_study(work, study, files[study], info.sha, signature, split_path, 5 * 1024 ** 3, hf_hub_download)
        local = work / 'processed/full' / study
        staging = target_root / f'.partial-{study}'
        shutil.rmtree(staging, ignore_errors=True)
        shutil.copytree(local, staging)
        staging.rename(final)
        shutil.rmtree(local)
        for raw in (work / 'raw/images').rglob(f'{study}_0000.nii.gz'):
            raw.unlink()
        return study

    target_root.mkdir(parents=True, exist_ok=True)
    start, done, failed = time.time(), 0, []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(one, study): study for study in pending}
        for future in as_completed(futures):
            study = futures[future]
            try:
                future.result()
                done += 1
                rate = (time.time() - start) / done
                print(f'[{done}/{len(pending)}] {study}  ~{rate * (len(pending) - done) / 60:.0f} min left', flush=True)
            except Exception as exc:  # keep going; failures are listed at the end
                failed.append(study)
                print(f'FAILED {study}: {exc}', flush=True)
    print(f'Done: {done} processed, {len(failed)} failed {failed[:10]}. Re-run the same command to retry failures.')


if __name__ == '__main__':
    main()
