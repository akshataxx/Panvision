"""Crop every study ONCE and save the 2D pancreas crops to small files.

Why: reading 3D scans is slow. Doing it once here means training (and Miti's training
too) just reads ready-made 224 x 224 pictures, which is fast and works the same on a
laptop or on Colab.

How each slice is cropped: a FIXED physical window (default 144 mm, the same on every
slice and every patient), centred on the pancreas, then resized to 224 x 224. A fixed
window keeps the scale constant, so the picture's zoom cannot hint at the label.
Tiny stray fragments of the automatic pancreas mask are ignored when finding the centre.

For each study this saves one file, <cache>/<study_id>.npz, holding:
  crops        (n, 224, 224) greyscale pictures, whole numbers 0-255
  slice_idx    which slice of the scan each picture came from
  has_lesion   True for slices that contain tumour (from the lesion mask)
  lesion_available   did this study have a lesion mask at all?
  label (1 = PDAC, 0 = not), patient_id, split, study_id, window_mm

Which slices are saved:
  * slices where the pancreas is clearly visible (up to --max-pancreas, evenly spread)
  * slices that contain lesion (up to --max-lesion, evenly spread), when a mask exists

Run from the repo root:
    python src/classification/make_crops.py "C:\\path\\to\\studies" --out outputs/crops

Test it on your 21 sample studies first, then run it on the full data folder.
Safe to stop and restart: finished studies are skipped. If you change the window or
any other setting, use a NEW --out folder, or old and new crops get mixed.
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image
from scipy import ndimage

# Same file-name patterns as src/explainability/view_scan.py.
PATTERNS = {
    "image": ["image*.nii*", "images*.nii*"],
    "pancreas": ["pancreas*.nii*"],
    "lesion": ["lesion*.nii*", "lesions*.nii*"],
}


def find_file(folder, patterns):
    """Return the first file under `folder` that matches one of the patterns."""
    for pattern in patterns:
        hits = sorted(Path(folder).rglob(pattern))
        if hits:
            return hits[0]
    return None


def load_array(path):
    """Read a NIfTI file. Returns (array ordered slice/row/column, in-plane pixel size in mm)."""
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img), float(img.GetSpacing()[0])


def main_pieces(mask2d, keep_fraction=0.2):
    """Drop tiny stray fragments: keep pieces at least 20% as big as the largest piece."""
    labelled, n = ndimage.label(mask2d)
    if n <= 1:
        return mask2d
    sizes = np.array(ndimage.sum(mask2d, labelled, range(1, n + 1)))
    keep = np.where(sizes >= keep_fraction * sizes.max())[0] + 1
    return np.isin(labelled, keep)


def window_box(mask2d, window_px):
    """Fixed-size square box centred on the pancreas.

    Returns (top, bottom, left, right, too_wide); too_wide is True when the pancreas
    on this slice is bigger than the window, so part of it is cut off.
    """
    rows, cols = np.where(main_pieces(mask2d))
    r_mid = (rows.min() + rows.max() + 1) // 2
    c_mid = (cols.min() + cols.max() + 1) // 2
    side = int(min(window_px, *mask2d.shape))  # never bigger than the image
    top = int(np.clip(r_mid - side // 2, 0, mask2d.shape[0] - side))
    left = int(np.clip(c_mid - side // 2, 0, mask2d.shape[1] - side))
    too_wide = max(rows.max() - rows.min() + 1, cols.max() - cols.min() + 1) > side
    return top, top + side, left, left + side, bool(too_wide)


def evenly_spaced(indices, max_n):
    """Keep at most max_n of the indices, spread evenly from first to last."""
    indices = np.asarray(indices, dtype=int)
    if len(indices) <= max_n:
        return indices
    pick = np.linspace(0, len(indices) - 1, max_n).round().astype(int)
    return indices[pick]


def choose_slices(pancreas, lesion, max_pancreas, max_lesion, min_lesion_px):
    """Return (slice numbers to save, boolean 'has lesion' for each of them)."""
    pancreas_area = pancreas.sum(axis=(1, 2))
    wanted = np.array([], dtype=int)
    if pancreas_area.max() > 0:
        clear = np.where(pancreas_area >= max(50, 0.25 * pancreas_area.max()))[0]
        wanted = evenly_spaced(clear, max_pancreas)
    lesion_area = np.zeros(len(pancreas), dtype=int)
    if lesion is not None:
        lesion_area = lesion.sum(axis=(1, 2))
        with_lesion = np.where(lesion_area >= min_lesion_px)[0]
        wanted = np.union1d(wanted, evenly_spaced(with_lesion, max_lesion))
    wanted = np.sort(wanted).astype(int)
    return wanted, lesion_area[wanted] >= min_lesion_px


def crop_slice(ct2d, box_mask, window_px):
    """Crop one slice with the fixed window and resize to 224 x 224 (whole numbers 0-255)."""
    r0, r1, c0, c1, too_wide = window_box(box_mask, window_px)
    crop = np.clip(ct2d[r0:r1, c0:c1], 0, 1)
    small = Image.fromarray((crop * 255).astype(np.uint8)).resize((224, 224))
    return np.asarray(small, dtype=np.uint8), too_wide


def main():
    parser = argparse.ArgumentParser(description="Save 2D pancreas crops for training")
    parser.add_argument("studies", help="folder holding one sub-folder per study")
    parser.add_argument("--splits", default="data/splits.csv")
    parser.add_argument("--out", default="outputs/crops")
    parser.add_argument("--only-split", default="all", choices=["all", "train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None, help="stop after this many studies")
    parser.add_argument("--max-pancreas", type=int, default=24)
    parser.add_argument("--max-lesion", type=int, default=24)
    parser.add_argument("--min-lesion-px", type=int, default=10)
    parser.add_argument("--window-mm", type=float, default=144.0,
                        help="side of the square crop in millimetres (same for every slice)")
    args = parser.parse_args()

    info = {}  # patient_id -> (split, label) from the team's shared split file
    with open(args.splits, newline="") as f:
        for row in csv.DictReader(f):
            info[row["patient_id"]] = (row["split"], int(row["label"]))

    os.makedirs(args.out, exist_ok=True)
    done = skipped = 0
    n_slices = n_too_wide = 0
    pdac_no_lesion = []
    for folder in sorted(p for p in Path(args.studies).iterdir() if p.is_dir()):
        if folder.name.startswith("."):
            continue  # leftover ".partial-..." folders from interrupted preprocessing
        patient_id = folder.name.split("_")[0]
        if patient_id not in info or not (folder / "complete.json").exists():
            print(f"SKIP {folder.name}: not in splits.csv or not finished")
            skipped += 1
            continue
        split, label = info[patient_id]
        if args.only_split != "all" and split != args.only_split:
            continue
        target = Path(args.out) / f"{folder.name}.npz"
        if target.exists():
            continue  # already done on an earlier run
        if args.limit is not None and done >= args.limit:
            break

        image_path = find_file(folder, PATTERNS["image"])
        pancreas_path = find_file(folder, PATTERNS["pancreas"])
        lesion_path = find_file(folder, PATTERNS["lesion"])
        if image_path is None or pancreas_path is None:
            print(f"SKIP {folder.name}: no image or no pancreas mask")
            skipped += 1
            continue
        ct, pixel_mm = load_array(image_path)
        pancreas = load_array(pancreas_path)[0] > 0
        lesion = load_array(lesion_path)[0] > 0 if lesion_path is not None else None
        if pancreas.shape != ct.shape or (lesion is not None and lesion.shape != ct.shape):
            print(f"SKIP {folder.name}: mask shape does not match the CT")
            skipped += 1
            continue
        window_px = int(round(args.window_mm / pixel_mm))  # e.g. 144 mm / 1.5 mm = 96 pixels

        slices, has_lesion = choose_slices(
            pancreas, lesion, args.max_pancreas, args.max_lesion, args.min_lesion_px)
        crops, kept, kept_lesion = [], [], []
        for z, hl in zip(slices, has_lesion):
            # Centre on the pancreas; if this slice has none, fall back to the lesion.
            box_mask = pancreas[z] if pancreas[z].any() else (lesion[z] if lesion is not None else None)
            if box_mask is None or not box_mask.any():
                continue
            small, too_wide = crop_slice(ct[z], box_mask, window_px)
            crops.append(small)
            kept.append(z)
            kept_lesion.append(hl)
            n_slices += 1
            n_too_wide += too_wide
        if not crops:
            print(f"SKIP {folder.name}: no usable slices")
            skipped += 1
            continue

        lesion_ok = lesion is not None and bool(np.any(kept_lesion))
        if label == 1 and not lesion_ok:
            pdac_no_lesion.append(folder.name)
        tmp = Path(args.out) / f"{folder.name}.tmp.npz"
        np.savez_compressed(
            tmp, crops=np.stack(crops), slice_idx=np.array(kept),
            has_lesion=np.array(kept_lesion), lesion_available=lesion_ok,
            label=label, patient_id=patient_id, split=split, study_id=folder.name,
            window_mm=args.window_mm)
        os.replace(tmp, target)  # only appears under its real name once fully written
        done += 1
        print(f"{folder.name}  {split:5s} PDAC={label}  slices={len(crops)}  "
              f"with lesion={int(np.sum(kept_lesion))}")

    print(f"\nSaved {done} studies to {args.out} ({skipped} skipped).")
    if n_slices:
        print(f"Window {args.window_mm:.0f} mm: the pancreas was wider than the window on "
              f"{n_too_wide} of {n_slices} saved slices ({n_too_wide / n_slices:.1%}).")
    if pdac_no_lesion:
        print(f"{len(pdac_no_lesion)} PDAC studies had no usable lesion slices: {pdac_no_lesion[:10]}")
        print("For these, training falls back to ALL pancreas slices (a noisier label).")


if __name__ == "__main__":
    main()