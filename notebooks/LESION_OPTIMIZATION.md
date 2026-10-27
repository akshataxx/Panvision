# Isolated lesion experiments

Open `colab_lesion_optimization.ipynb` to train L4 (negative sampling), L5 (strong CT intensity augmentation),
or L6 (lower OneCycle peak learning rate). The notebook uses `src.segmentation.train_experiments` and
`src.segmentation.experimental_data`; the original training module and `colab_segmentation.ipynb` are unchanged.

Each experiment has a separate run directory under `models/segmentation`, so the saved L2 checkpoint is preserved.
The notebook compares all candidates with L2 on the same validation studies and reports positive-case Dice,
recall, detection, and non-PDAC false-positive rate. Because the original test set has already been viewed,
use validation results to choose among these experiments.

The final notebook section also tests a **no-training L2 + L5 ensemble**. It averages the saved models' voxel
probabilities and compares L2, L5, and the ensemble on identical validation scans and postprocessing settings.
It records Dice, recall, detection, false-positive rate, and mean absolute volume and axial-diameter errors.
The source checkpoints and original result files are not overwritten.

## Pancreas model used for validation

P3 nnU-Net has the highest saved pancreas validation Dice (0.819 over 80 cases, versus 0.783 for P2).
The experiment notebook uses P3's saved `outputs/segmentation/nnunet/predictions` masks for every lesion
comparison and writes result files with a `_P3` suffix, preserving earlier results. P3's trained checkpoint is
`models/segmentation/P3_nnunet_2d/results/Dataset501_PanVisionPancreas/nnUNetTrainer_PanVision__nnUNetPlans__2d/fold_0/checkpoint_best.pth`;
nnU-Net also needs the surrounding plans and dataset files when predicting new scans.
