"""Feed the saved 2D crops to a PyTorch model. Shared by everyone training a classifier.

Read the crops made by make_crops.py and hand them over one picture at a time.
Every picture comes with the patient's label (1 = PDAC) and a patient number, so the
scores of one patient's slices can be averaged afterwards.

mode="train": annotation-assisted PDAC crops may use known tumour slices. Predicted-
              pancreas crops always use all selected pancreas slices and never use
              lesion metadata. Non-PDAC patients use all saved pancreas slices.
mode="eval":  every patient contributes all of their saved pancreas slices, because at
              test time we do not know where the tumour is.

Use it like this:
    train = CropDataset("outputs/crops", split="train", mode="train", augment=True)
    val   = CropDataset("outputs/crops", split="val",   mode="eval")
    loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.classification.crop_policy import select_rows

# Standard values the ImageNet-trained ResNet-50 expects (same as the explainability code).
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_cache(cache_dir, split, mode, drop_unlocalised_pdac=False):
    """Read the saved crops for one split. Returns plain numpy arrays (no torch)."""
    images, labels, patient_idx, study_ids = [], [], [], []
    patient_ids, patient_labels = [], []
    for path in sorted(Path(cache_dir).glob("*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        with np.load(path) as f:
            if split != "all" and str(f["split"]) != split:
                continue
            label = int(f["label"])
            crop_source = str(f.get("crop_source", "annotation_assisted"))
            rows = select_rows(f["has_lesion"], label, mode, drop_unlocalised_pdac,
                               crop_source=crop_source)
            if len(rows) == 0:
                continue
            pid = str(f["patient_id"])
            if pid not in patient_ids:
                patient_ids.append(pid)
                patient_labels.append(label)
            images.append(f["crops"][rows])
            labels += [label] * len(rows)
            patient_idx += [patient_ids.index(pid)] * len(rows)
            study_ids += [str(f["study_id"])] * len(rows)
    if not images:
        raise FileNotFoundError(f"No crops found for split '{split}' in {cache_dir}")
    return {
        "images": np.concatenate(images),
        "labels": np.array(labels),
        "patient_idx": np.array(patient_idx),
        "study_ids": np.array(study_ids),
        "patient_ids": np.array(patient_ids),
        "patient_labels": np.array(patient_labels),
    }


class CropDataset(Dataset):
    def __init__(self, cache_dir, split, mode="train", augment=False, drop_unlocalised_pdac=False):
        data = load_cache(cache_dir, split, mode, drop_unlocalised_pdac)
        self.images = data["images"]  # (N, 224, 224) whole numbers 0-255, kept in memory
        self.labels = data["labels"]
        self.patient_idx = data["patient_idx"]  # which patient each picture belongs to
        self.study_ids = data["study_ids"]
        self.patient_ids = data["patient_ids"]
        self.patient_labels = data["patient_labels"]
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        x = torch.from_numpy(self.images[i]).float() / 255.0  # (224, 224), values 0-1
        if self.augment:
            x = self._jitter(x)
        x = x.unsqueeze(0).repeat(3, 1, 1)  # models expect 3 colour channels
        x = (x - MEAN) / STD
        return x, int(self.labels[i]), int(self.patient_idx[i])

    @staticmethod
    def _jitter(x, max_shift=8):
        """Small random brightness/contrast change and a few pixels of shift.

        No flips or rotations: left/right and up/down mean something anatomically.
        """
        contrast = 0.9 + 0.2 * torch.rand(1)
        brightness = -0.05 + 0.1 * torch.rand(1)
        x = ((x - 0.5) * contrast + 0.5 + brightness).clamp(0, 1)
        padded = torch.nn.functional.pad(
            x[None, None], (max_shift,) * 4, mode="replicate")[0, 0]
        dy, dx = torch.randint(0, 2 * max_shift + 1, (2,)).tolist()
        return padded[dy:dy + x.shape[0], dx:dx + x.shape[1]]

    def pos_weight(self):
        """How much more to weight PDAC pictures so the 30% minority is not ignored."""
        positives = float((self.labels == 1).sum())
        return float((len(self.labels) - positives) / max(positives, 1.0))
