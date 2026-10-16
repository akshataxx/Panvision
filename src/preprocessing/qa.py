"""Render original CT/mask overlays without creating a patient split."""

import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from src.preprocessing.pipeline import load_manifest, read_pair
from src.preprocessing.storage import data_root, manifest_path


def select_slices(labels, depth):
    if labels is None:
        return sorted(set([depth // 4, depth // 2, 3 * depth // 4]))
    pancreas = np.flatnonzero((labels == 4).sum(axis=(1, 2)))
    candidates = [int(pancreas[i]) for i in (0, len(pancreas) // 2, -1)] if len(pancreas) else [depth // 2]
    lesion_area = (labels == 1).sum(axis=(1, 2))
    if lesion_area.max() > 0:
        candidates[1 if len(candidates) > 1 else 0] = int(lesion_area.argmax())
    return sorted(set(candidates))


def render(image, labels, slices):
    panels = []
    for z in slices:
        gray = ((np.clip(image[z], -100, 300) + 100) / 400 * 255).astype(np.uint8)
        raw = np.repeat(gray[..., None], 3, axis=2)
        overlay = raw.copy()
        if labels is not None:
            for value, color in ((4, [0, 255, 80]), (1, [255, 50, 50])):
                region = labels[z] == value
                overlay[region] = (0.55 * raw[region] + 0.45 * np.array(color)).astype(np.uint8)
        panels.append(np.concatenate([raw, overlay], axis=1))
    return np.concatenate(panels, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(manifest_path()))
    parser.add_argument('--output', default=str(data_root() / 'qa'))
    args = parser.parse_args()
    try:
        frame = load_manifest(args.manifest)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        sections, summary = [], []
        for row in frame.itertuples(index=False):
            image, mask = read_pair(row)
            image = sitk.DICOMOrient(image, 'LPS')
            mask = sitk.DICOMOrient(mask, 'LPS') if mask is not None else None
            array = sitk.GetArrayFromImage(image)
            labels = sitk.GetArrayFromImage(mask) if mask is not None else None
            slices = select_slices(labels, len(array))
            montage = render(array, labels, slices)
            filename = f'{row.study_id}.png'
            sitk.WriteImage(sitk.GetImageFromArray(montage, isVector=True), str(output / filename))
            lesion = int((labels == 1).sum()) if labels is not None else None
            pancreas = int((labels == 4).sum()) if labels is not None else None
            note = 'Review selected slices and full volume before accepting annotations.'
            if row.label == '1' and lesion == 0:
                note = 'REVIEW: PDAC case has no lesion voxels in this annotation.'
            if labels is None:
                note = 'No annotation available; overlay panel duplicates CT.'
            summary.append(dict(study_id=row.study_id, label=row.label,
                                annotation_tier=row.annotation_tier,
                                pancreas_voxels=pancreas, lesion_voxels=lesion,
                                selected_z=','.join(map(str, slices)), review_note=note))
            caption = (f'{row.study_id} | PDAC={row.label} | lesion tier={row.annotation_tier} | '
                       f'rows: z={slices} (zero-based, after LPS orientation). {note}')
            sections.append(f'<section><h2>{html.escape(caption)}</h2>'
                            f'<img src="{filename}" alt="CT and mask overlay"></section>')
            print(f'{row.study_id}: slices {slices}; pancreas={pancreas}, lesion={lesion}')
        pd.DataFrame(summary).to_csv(output / 'summary.csv', index=False)
        document = ('<!doctype html><html><meta charset="utf-8"><title>PanVision mask QA</title>'
                    '<style>body{font-family:system-ui;background:#111;color:#eee;margin:24px}'
                    'h2{font-size:16px}img{max-width:100%;width:1000px}section{margin-bottom:40px}</style>'
                    '<h1>Original CT and annotation review</h1>'
                    '<p>Left: CT. Right: overlay. Green: pancreas (4). Red: PDAC lesion (1). '
                    'HU display window: [-100,300]. Panels use voxel pixel aspect, not measurement scale. '
                    'LPS axis permutation retains any obliquity. Selected slices are a spot check, '
                    'not a complete annotation quality assessment.</p>' + ''.join(sections) + '</html>')
        (output / 'index.html').write_text(document)
        print(f'Open {output / "index.html"}')
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
