# CNN and DenseNet-121 research

Use the same crop cache and `data/splits.csv`. These are annotation-assisted
baselines: the current crop generator uses reference lesion information even
for held-out crops. Keep test patients out of model/aggregation selection.

```bash
pip install -r research/classification/requirements.txt
python -m research.classification.train_cnn --crops /path/to/crops --out outputs/cnn_run1
python -m research.classification.train_densenet --crops /path/to/crops --out outputs/densenet_run1
```

Defaults: 8 epochs, batch 16, AdamW at 1e-4, mean patient aggregation. For a smoke
check add `--epochs 1 --max-batches 3` and use a separate output directory.
CNN trains from scratch; DenseNet uses ImageNet initialization. Both expose
`embedding()` (512 features), `forward()` (2 logits), and `gradcam_layer`.

Each new run saves `best.pt`, `latest.pt`, `history.json`, config and crop audit.
Existing run directories are not overwritten. `latest.pt` includes optimizer
state but automatic resume is not implemented. Put `--out` under mounted Drive
in Colab for persistence. Scripts accept both module and direct-file invocation.

```bash
python -m research.classification.evaluate_baseline --model cnn \
  --checkpoint outputs/cnn_run1/best.pt --crops /path/to/crops --out outputs/cnn_val1
python -m research.classification.evaluate_baseline --model densenet \
  --checkpoint outputs/densenet_run1/best.pt --crops /path/to/crops --out outputs/densenet_val1
```

Evaluation generates fresh validation predictions and compares mean, max,
top-k and top-fraction aggregation. Saves patient/slice scores, AUC, bootstrap
intervals, average precision, sensitivity, specificity, accuracy, F1 and counts.
Checkpoint hashes prevent ambiguity about which model produced a report.
Validation-selected thresholds and aggregation must be frozen before final
held-out testing. Bootstrap intervals do not account for model selection.
Notebook checkpoints from the corresponding CNN/DenseNet models are supported.

```bash
python -m unittest discover -s research/classification/tests
```

Do not commit crops, model weights, predictions or credentials.

Uses the shared loaders and scoring functions in `src/classification/`.
