# segmentation
Two-stage cascade on preprocessed volumes (`src/preprocessing`): Stage 1 pancreas U-Net → pancreas ROI
(largest component, 10 mm margin, 160×160 in-plane) → Stage 2 lesion U-Net inside the ROI → measurements
(volume, max axial diameter, voxel count, head/body/tail) using physical voxel spacing.

- Pancreas target = labels {1, 4, 5} (whole gland incl. lesion and duct). Mostly PANORAMA automatic labels,
  so pancreas Dice is agreement with an automatic reference.
- Lesion supervision and evaluation use **manual** lesion masks only; non-PDAC studies are negatives.
- Pancreas candidates: P1 2D U-Net (baseline), P2 2.5D U-Net, P3 nnU-Net v2 `2d` (50 epochs, same patients,
  scored by the same `evaluate.py`).
- Lesion candidates: L1 U-Net vs L2 Attention U-Net on the pancreas ROI (Dice + Focal loss); L0 full-slice U-Net
  without ROI (shows what the cascade adds); L3 loss ablation (best of L1/L2 with Dice + CE).
- Selection on val Dice (lesion: end-to-end, recall breaks ties); test once. Paired Wilcoxon + bootstrap 95% CIs.
- No flip augmentation (pancreas is not left-right symmetric), also disabled for nnU-Net.

Train/evaluate on Colab: `notebooks/colab_segmentation.ipynb` (P1, P2, L0-L3, selection, test, export) and
`notebooks/colab_nnunet.ipynb` (P3). P3 pancreas masks for every scan, split across Colab accounts:
`notebooks/colab_pancreas_all.ipynb` (`src/segmentation/nnunet_predict.py`). Outputs in the shared data folder:
`models/segmentation/<run>/best.pt`, `outputs/segmentation/<study_id>/{pancreas,lesion}.nii.gz`, `roi.json`,
`outputs/segmentation/measurements.csv`, `outputs/segmentation/results/*.csv`.

For other modules:
```python
from src.segmentation.infer import PancreasSegmentor, LesionSegmentor
pancreas = PancreasSegmentor('.../P2_unet_25d/best.pt').predict(image)           # image: (Z, Y, X) preprocessed
lesion, roi = LesionSegmentor('.../L2_attention_unet_roi/best.pt').predict(image, pancreas, spacing_xyz)
```

Tests: `python -m unittest discover -s src/segmentation/tests -t .`

## Exploratory lesion postprocessing tuning

After the original test evaluation, the selected L2 model showed a high non-PDAC false-positive rate.
`colab_segmentation.ipynb` cell 20 sweeps probability thresholds and minimum connected-component sizes on
**validation cases only**, running the network once per study. The default (0.5, 10 voxels) reproduces the
original postprocessing. The table reports positive-case Dice, IoU, recall and detection rate, alongside the
negative-case false-positive rate. Raising a threshold or size can remove false positives but also small real lesions.

```bash
python -m src.segmentation.tune \
  --checkpoint /content/data/models/segmentation/L2_attention_unet_roi/best.pt \
  --pancreas-predictions /content/data/outputs/segmentation/nnunet/predictions
```

The per-case and summary CSVs go to `outputs/segmentation/results/`. If a validation-chosen setting warrants an
exploratory export, pass `--lesion-threshold` and `--lesion-min-voxels` to `src.segmentation.infer`, and choose a
separate `--out` folder to preserve the original masks. The original test has already been viewed; any subsequent
test result for a changed setting is exploratory rather than a new untouched holdout result.
