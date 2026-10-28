# Classification

Patient-level PDAC classification using shared crop data and fixed patient-level
train, validation, and test splits. The package contains ResNet-50, SimpleCNN,
and DenseNet-121 workflows plus common evaluation utilities.

## Main files

- `make_crops.py`: create annotation-assisted pancreas crops.
- `make_predicted_crops.py`: create deployment-style crops from predicted pancreas masks without reading lesion masks.
- `dataset.py`: load crop files for training and evaluation.
- `resnet_model.py`, `cnn_model.py`, `densenet_model.py`: model definitions.
- `train_resnet.py`: ResNet-50 training.
- `train_cnn.py`, `train_densenet.py`: SimpleCNN and DenseNet-121 entry points.
- `evaluate_model.py`: ResNet evaluation.
- `evaluate_baseline.py`: CNN/DenseNet validation and aggregation comparison.
- `export_embeddings.py`: export frozen DenseNet embeddings for KNN retrieval.
- `evaluation.py`, `experiment_utils.py`: shared patient aggregation and metrics.

## Setup

```bash
pip install -r src/classification/requirements.txt
```

## Crop generation

Annotation-assisted crops:

```bash
python src/classification/make_crops.py <processed/full> --out outputs/crops
```

Predicted-pancreas crops:

```bash
python -m src.classification.make_predicted_crops <processed/full> \
  --predictions <segmentation-output> \
  --prediction-id <checkpoint-or-run> \
  --out <predicted-crops>
```

Predictions must be stored as `<study_id>/pancreas.nii.gz`. Missing, empty, or
off-grid predictions are reported and skipped. Use a separate crop directory for
each segmentation run. The segmenter must be frozen and trained only on training
patients before its predictions are used for validation or test crops.

## Training

```bash
python -m src.classification.train_resnet --crops /path/to/crops
python -m src.classification.train_cnn --crops /path/to/crops --out outputs/cnn_run
python -m src.classification.train_densenet --crops /path/to/crops --out outputs/densenet_run
```

CNN and DenseNet runs save `best.pt`, `latest.pt`, `history.json`, configuration,
and a crop audit. DenseNet uses ImageNet initialization. Both models expose a
512-dimensional embedding for retrieval and a Grad-CAM layer.

## Evaluation

```bash
python -m src.classification.evaluate_baseline --model densenet \
  --checkpoint outputs/densenet_run/best.pt \
  --crops /path/to/crops \
  --out outputs/densenet_validation
```

Evaluation compares patient-level mean, max, top-k, and top-fraction aggregation
and writes predictions, metrics, bootstrap intervals, and checkpoint hashes.
Select aggregation and threshold on validation patients, freeze them, then use
the held-out test patients once.

## Embedding export

After selecting and freezing the DenseNet checkpoint, export embeddings using
predicted-pancreas crops:

```bash
python -m src.classification.export_embeddings \
  --checkpoint outputs/densenet_run/best.pt \
  --crops /path/to/predicted-crops \
  --out outputs/densenet_run
```

`patient_embeddings_train.npz` is the only KNN reference bank. Validation and
test banks are query-only and remain separate. The exporter also writes study
embeddings, CSV manifests, checkpoint provenance, and an embedding audit.

## Tests

```bash
python -m unittest discover -s src/classification/tests
```

Do not commit crop data, model weights, predictions, database credentials, or
patient data.
