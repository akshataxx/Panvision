"""Count what is in the crop folder: studies, patients and pictures per split and label.

Run from the repo root:
    python src/classification/crops_summary.py outputs/crops
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "outputs/crops"
    studies = defaultdict(int)
    patients = defaultdict(set)
    pictures = defaultdict(int)
    lesion_pictures = defaultdict(int)
    for path in sorted(Path(folder).glob("*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        with np.load(path) as f:
            key = (str(f["split"]), int(f["label"]))
            studies[key] += 1
            patients[key].add(str(f["patient_id"]))
            pictures[key] += len(f["crops"])
            lesion_pictures[key] += int(np.sum(f["has_lesion"]))
    print(f"{'split':6s} {'PDAC?':6s} {'patients':>9s} {'studies':>8s} {'pictures':>9s} {'tumour pics':>12s}")
    for key in sorted(studies):
        split, label = key
        print(f"{split:6s} {'yes' if label else 'no':6s} {len(patients[key]):9d} "
              f"{studies[key]:8d} {pictures[key]:9d} {lesion_pictures[key]:12d}")
    for split in ("train", "val", "test"):
        n_pdac = len(patients.get((split, 1), ()))
        n_all = n_pdac + len(patients.get((split, 0), ()))
        if n_all:
            print(f"{split}: {n_all} patients, {n_pdac} PDAC ({n_pdac / n_all:.0%})")


if __name__ == "__main__":
    main()
