# classification
Custom CNN baseline and ResNet-50, trained on the same patient-level splits. Includes slice-to-patient score aggregation.
Save the trained model so that `src/explainability` can load it (see TEAM_GUIDE.md).

## Files
- `make_crops.py`: crops every study once (a fixed 144 mm window around the pancreas, resized to 224 x 224) and saves small files; PDAC studies also keep their tumour slices
- `dataset.py`: feeds the saved crops to a model; PDAC patients use tumour slices for training, all patients use all slices for evaluation
- `resnet_model.py`: the ResNet-50 classifier (2048 -> 512 -> 2 scores), plus saving and loading
- `train_resnet.py`: trains ResNet-50 and keeps the best model by validation AUC
- `evaluation.py`: shared scoring; averages slice scores into one score per patient, then reports AUC, sensitivity, specificity
- `view_crops.py`: draws the crops of one study, to check the cropping by eye
- `crops_summary.py`: counts patients, studies and pictures per split and label

## How to run
1. `python src/classification/make_crops.py <folder of processed studies> --out outputs/crops`
2. `python src/classification/crops_summary.py outputs/crops` (check the counts)
3. `python src/classification/train_resnet.py --crops outputs/crops` (needs a GPU, e.g. Colab)

Choose settings on the validation patients; look at the test patients once, at the end.
