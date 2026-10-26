# explainability
Grad-CAM heat maps and Top-3 similar-case retrieval, for whichever classifier the team picked.

## Interface with the classifier
- Input: a trained classifier saved by `src/classification` (weights plus notes) and the shared crops
  made by `src/classification/make_crops.py` (fixed 144 mm window, 224 x 224).
- The classifier must offer `embedding(x)` (512 numbers), `forward(x)` (two scores) and `gradcam_layer`
       (last convolutional block). `model_loader.py` loads ResNet-50, DenseNet-121 or the small CNN from the
     checkpoint itself and is the one place to add a new network.
- Retrieval database = training patients only. A query patient never retrieves themselves.

## Files
- `model_loader.py`: loads the classifier; turns saved crops into model input exactly as in training
- `gradcam.py`: heat map generation, and keeping the heat inside a pancreas mask
- `gradcam_patient.py`: Grad-CAM on the most suspicious slices of one study, with reference tumour slices marked
- `slice_localisation.py`: are the model's most suspicious slices the tumour slices? (compared with chance)
- `build_embeddings.py`: one 512-number fingerprint per study, from the shared crops, saved to `outputs/embeddings.npz`
- `retrieval.py`: embedding index, Top-3 search, Precision@3 checks (numpy only)
- `evaluate_retrieval.py`: Precision@3 for validation (or test) patients against the training database
- `explain.py`: one function the dashboard can call: score, key slices with heat maps, three similar cases
- Earlier prototypes with the random stand-in model, kept for reference: `standin_model.py`,
  `crop_gradcam_scan.py`, `retrieval_demo.py`, `demo_gradcam.py`, `view_scan.py`

## How to run (from the repo root, after the classifier is trained)
1. `python src/explainability/build_embeddings.py --model models/classifier.pt --crops outputs/crops`
2. `python src/explainability/evaluate_retrieval.py --split val`
3. `python src/explainability/gradcam_patient.py <study_id> --model models/classifier.pt --crops outputs/crops`
4. `python src/explainability/slice_localisation.py --split val`

Choose settings on validation patients; look at the test patients once, at the end.
