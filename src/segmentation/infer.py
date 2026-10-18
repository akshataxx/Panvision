"""Inference wrappers for the cascade (PancreasSegmentor -> ROI -> LesionSegmentor) and batch export.

Inputs are preprocessed volumes (image.nii.gz from src/preprocessing). Masks are returned in the same grid.

Batch export of the selected models for teammates:
    python -m src.segmentation.infer --pancreas-checkpoint <p.pt> --lesion-checkpoint <l.pt>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from monai.inferers import sliding_window_inference

from src.preprocessing.storage import data_root
from src.segmentation.data import context_stack, find_studies, iter_cases
from src.segmentation.measure import lesion_measurements
from src.segmentation.models import build_model
from src.segmentation.roi import crop, largest_component, make_roi, paste, remove_small


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_checkpoint(path, device=None):
    device = device or get_device()
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = build_model(checkpoint['model'], checkpoint['context'])
    model.load_state_dict(checkpoint['state_dict'])
    return model.to(device).eval(), checkpoint


@torch.no_grad()
def predict_probs(model, volume, context, patch, device, batch=16):
    """Slice-wise 2.5D foreground probabilities for a (Z, Y, X) volume."""
    model.eval()
    probs = np.zeros(volume.shape, dtype=np.float32)
    for start in range(0, volume.shape[0], batch):
        zs = range(start, min(volume.shape[0], start + batch))
        x = torch.from_numpy(np.stack([context_stack(volume, z, context) for z in zs]).astype(np.float32)).to(device)
        with torch.autocast(device.type, enabled=device.type == 'cuda'):
            logits = sliding_window_inference(x, (patch, patch), sw_batch_size=batch, predictor=model, overlap=0.25)
        probs[start:start + len(zs)] = torch.sigmoid(logits.float())[:, 0].cpu().numpy()
    return probs


class _Segmentor:
    def __init__(self, checkpoint, device=None, threshold=0.5):
        self.device = device or get_device()
        if isinstance(checkpoint, (str, Path)):
            self.model, meta = load_checkpoint(checkpoint, self.device)
        else:  # (model, meta) during training-time validation
            self.model, meta = checkpoint
        self.context, self.patch, self.threshold = meta['context'], meta['patch'], threshold
        self.roi_mode = meta.get('roi', 'pancreas')


class PancreasSegmentor(_Segmentor):
    def predict(self, image):
        """Binary pancreas mask (largest connected component) for a preprocessed (Z, Y, X) volume."""
        probs = predict_probs(self.model, image, self.context, self.patch, self.device)
        return largest_component(probs >= self.threshold)

    def predict_case(self, case):
        return self.predict(case['image'])


class MaskFolder:
    """Pancreas masks predicted elsewhere (nnU-Net), read from <folder>/<study_id>.nii.gz.

    The largest component is kept, the same post-processing as PancreasSegmentor, so candidates compare fairly.
    """

    def __init__(self, folder):
        self.folder = Path(folder)

    def predict_case(self, case):
        path = self.folder / f'{case["study_id"]}.nii.gz'
        if not path.exists():
            raise FileNotFoundError(f'No prediction for {case["study_id"]} in {self.folder}')
        mask = sitk.GetArrayFromImage(sitk.ReadImage(str(path))) > 0
        if mask.shape != case['image'].shape:
            raise ValueError(f'{path} does not match the preprocessed image grid')
        return largest_component(mask)


def pancreas_source(checkpoint=None, predictions=None):
    if bool(checkpoint) == bool(predictions):
        raise ValueError('Give exactly one of a pancreas checkpoint or a pancreas predictions folder')
    return PancreasSegmentor(checkpoint) if checkpoint else MaskFolder(predictions)


class LesionSegmentor(_Segmentor):
    def __init__(self, checkpoint, device=None, threshold=0.5, min_voxels=10, margin_mm=10.0):
        super().__init__(checkpoint, device, threshold)
        self.min_voxels, self.margin_mm = min_voxels, margin_mm

    def predict(self, image, pancreas, spacing_xyz):
        """(lesion mask, ROI or None). Only the pancreas ROI is searched; no pancreas means no lesion.

        The no-ROI baseline (roi='none') searches every full slice and ignores the pancreas mask.
        """
        if self.roi_mode == 'none':
            probs = predict_probs(self.model, image, self.context, self.patch, self.device)
            return remove_small(probs >= self.threshold, self.min_voxels), None
        roi = make_roi(largest_component(pancreas), spacing_xyz, self.patch, self.margin_mm)
        if roi is None:
            return np.zeros(image.shape, dtype=bool), None
        probs = predict_probs(self.model, crop(image, roi), self.context, self.patch, self.device)
        mask = remove_small(probs >= self.threshold, self.min_voxels)
        return paste(mask, roi, image.shape).astype(bool), roi


def write_mask(mask, reference, path):
    image = sitk.GetImageFromArray(mask.astype(np.uint8))
    image.CopyInformation(reference)
    sitk.WriteImage(image, str(path))


def export(args):
    studies = find_studies(args.data_root, args.splits)
    rows = studies[studies.split.isin(args.split)]
    pancreas = pancreas_source(args.pancreas_checkpoint, args.pancreas_predictions)
    lesion = LesionSegmentor(args.lesion_checkpoint)
    out = Path(args.out)
    records = []
    for row, case in zip(rows.itertuples(index=False), iter_cases(rows)):
        reference = sitk.ReadImage(row.processed_image)
        pancreas_mask = pancreas.predict_case(case)
        lesion_mask, roi = lesion.predict(case['image'], pancreas_mask, case['spacing'])
        folder = out / row.study_id
        folder.mkdir(parents=True, exist_ok=True)
        write_mask(pancreas_mask, reference, folder / 'pancreas.nii.gz')
        write_mask(lesion_mask, reference, folder / 'lesion.nii.gz')
        (folder / 'roi.json').write_text(json.dumps(roi.to_dict() if roi else None) + '\n')
        record = {'study_id': row.study_id, 'patient_id': row.patient_id, 'split': row.split, 'label': row.label,
                  'pancreas_found': bool(pancreas_mask.any())}
        record.update(lesion_measurements(lesion_mask, pancreas_mask, case['spacing']))
        records.append(record)
        print(f'{row.study_id}: pancreas={record["pancreas_found"]} lesion={record["lesion_detected"]}', flush=True)
    pd.DataFrame(records).to_csv(out / 'measurements.csv', index=False)
    print(f'Wrote {len(records)} studies to {out}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pancreas-checkpoint', default=None)
    parser.add_argument('--pancreas-predictions', default=None, help='folder of nnU-Net pancreas masks instead')
    parser.add_argument('--lesion-checkpoint', required=True)
    parser.add_argument('--split', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--data-root', default=None)
    parser.add_argument('--splits', default='data/splits.csv')
    parser.add_argument('--out', default=None)
    args = parser.parse_args(argv)
    args.out = args.out or str(data_root() if args.data_root is None else Path(args.data_root)) + '/outputs/segmentation'
    export(args)


if __name__ == '__main__':
    main()
