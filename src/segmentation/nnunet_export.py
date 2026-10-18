"""Bridge to nnU-Net v2 for the P3 pancreas candidate, using the same patients and evaluation as P1/P2.

    python -m src.segmentation.nnunet_export install-trainer
    python -m src.segmentation.nnunet_export dataset --raw $nnUNet_raw
    nnUNetv2_plan_and_preprocess -d 501 -c 2d -np 2
    python -m src.segmentation.nnunet_export splits --raw $nnUNet_raw --preprocessed $nnUNet_preprocessed
    nnUNetv2_train 501 2d 0 -tr nnUNetTrainer_PanVision --c
    python -m src.segmentation.nnunet_export inputs --out /content/nnunet/inputs
    nnUNetv2_predict -i /content/nnunet/inputs -o <predictions> -d 501 -c 2d -f 0 -tr nnUNetTrainer_PanVision -chk checkpoint_best.pth
    python -m src.segmentation.evaluate pancreas --predictions <predictions> --name P3_nnunet_2d --split val --max-studies 100

nnU-Net trains on the same train subset and validates on the same val subset as the U-Net runs (same sampling
seed), receives the same preprocessed volumes, and its masks are scored by our evaluate.py, so the comparison
differs only in the method.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from src.segmentation.data import find_studies, pancreas_rows, pancreas_target, subsample

DATASET = 'Dataset501_PanVisionPancreas'

TRAINER = '''import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_PanVision(nnUNetTrainer):
    """PanVision P3: 50 epochs, checkpoint every 2 epochs (free Colab disconnects), no mirroring
    (abdominal anatomy is not left-right symmetric; matches the U-Net candidates)."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 50
        self.save_every = 2

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \\
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, None
'''


def install_trainer():
    import nnunetv2
    target = Path(nnunetv2.__file__).parent / 'training/nnUNetTrainer/variants/nnUNetTrainer_PanVision.py'
    target.write_text(TRAINER)
    print(f'Installed {target}')


def _copy_image(row, folder):
    destination = folder / f'{row.study_id}_0000.nii.gz'
    if not destination.exists():
        shutil.copyfile(row.processed_image, destination)


def dataset(args):
    studies = find_studies(args.data_root, args.splits)
    train = subsample(pancreas_rows(studies, 'train'), args.max_train, args.seed)
    val = subsample(pancreas_rows(studies, 'val'), args.max_val, args.seed)
    root = Path(args.raw) / DATASET
    (root / 'imagesTr').mkdir(parents=True, exist_ok=True)
    (root / 'labelsTr').mkdir(parents=True, exist_ok=True)
    rows = pd.concat([train, val])
    for number, row in enumerate(rows.itertuples(index=False), 1):
        _copy_image(row, root / 'imagesTr')
        label_path = root / 'labelsTr' / f'{row.study_id}.nii.gz'
        if not label_path.exists():
            labels = sitk.ReadImage(row.processed_labels)
            binary = sitk.GetImageFromArray(pancreas_target(sitk.GetArrayFromImage(labels)).astype(np.uint8))
            binary.CopyInformation(labels)
            sitk.WriteImage(binary, str(label_path))
        if number % 25 == 0:
            print(f'{number}/{len(rows)} cases exported', flush=True)
    (root / 'dataset.json').write_text(json.dumps({
        'channel_names': {'0': 'CT'}, 'labels': {'background': 0, 'pancreas': 1},
        'numTraining': len(rows), 'file_ending': '.nii.gz',
        'description': 'PanVision pancreas (labels 1, 4, 5) on preprocessed PANORAMA volumes'}, indent=2))
    (root / 'panvision_split.json').write_text(json.dumps(
        [{'train': sorted(train.study_id), 'val': sorted(val.study_id)}], indent=2))
    print(f'Exported {len(train)} train + {len(val)} val cases to {root}')


def splits(args):
    """Write our patient-level split as nnU-Net fold 0 (run after plan_and_preprocess)."""
    source = Path(args.raw) / DATASET / 'panvision_split.json'
    target = Path(args.preprocessed) / DATASET / 'splits_final.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    split = json.loads(source.read_text())[0]
    print(f'Fold 0: {len(split["train"])} train / {len(split["val"])} val -> {target}')


def inputs(args):
    """Images nnU-Net must predict so evaluate.py can score it on exactly the cases the U-Nets are scored on."""
    from src.segmentation.evaluate import evaluation_rows
    studies = find_studies(args.data_root, args.splits)
    rows = pd.concat([evaluation_rows(studies, 'pancreas', 'val', args.max_val_studies),
                      evaluation_rows(studies, 'lesion', 'val'),
                      evaluation_rows(studies, 'pancreas', 'test'),
                      evaluation_rows(studies, 'lesion', 'test', max_negatives=0)]).drop_duplicates('study_id')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for row in rows.itertuples(index=False):
        _copy_image(row, out)
    print(f'{len(rows)} val/test images ready in {out}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('install-trainer')
    for name in ('dataset', 'splits', 'inputs'):
        p = sub.add_parser(name)
        p.add_argument('--data-root', default=None)
        p.add_argument('--splits', default='data/splits.csv')
        p.add_argument('--seed', type=int, default=42)
        if name in ('dataset', 'splits'):
            p.add_argument('--raw', required=True)
        if name == 'dataset':  # defaults match src.segmentation.train
            p.add_argument('--max-train', type=int, default=250)
            p.add_argument('--max-val', type=int, default=30)
        if name == 'splits':
            p.add_argument('--preprocessed', required=True)
        if name == 'inputs':
            p.add_argument('--out', required=True)
            p.add_argument('--max-val-studies', type=int, default=100, help='match evaluate --max-studies')
    args = parser.parse_args(argv)
    if args.command == 'install-trainer':
        install_trainer()
    else:
        {'dataset': dataset, 'splits': splits, 'inputs': inputs}[args.command](args)


if __name__ == '__main__':
    main()
