# explainability
Grad-CAM heatmaps and Top-3 similar-case retrieval.

## Interface with the classifier
- Input: a trained PyTorch model saved by the classification owner, plus preprocessed ROI images.
- Grad-CAM needs the name of the last convolutional layer.
- Retrieval uses the 512-d penultimate-layer vector as the case embedding.
- Retrieval database = training cases only. A query case must never retrieve itself.

## Files
- `gradcam.py`: heatmap generation, and keeping the heat inside the pancreas mask
- `retrieval.py`: embedding index, Top-3 search, Precision@3 check (numpy only)
- `standin_model.py`: temporary stand-in classifier (random head), used until the real one exists
- `view_scan.py`: look at one processed CT study with its masks
- `crop_gradcam_scan.py`: crop the pancreas from one slice, run Grad-CAM, keep heat inside the pancreas
- `build_embeddings.py`: one 512-number fingerprint per study, saved to `outputs/embeddings.npz`
- `retrieval_demo.py`: Top-3 example, same-patient check, Precision@3 vs random baseline
- `demo_gradcam.py`: Grad-CAM on an ordinary photo (a first test of the library)

## Status
Everything above runs on the stand-in model, so the results are not medically meaningful.
When the real classifier exists, swap it in for `StandInClassifier` and rerun.
