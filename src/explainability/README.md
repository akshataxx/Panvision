# explainability
Grad-CAM heatmaps and Top-3 similar-case retrieval.

## Interface with the classifier
- Input: a trained PyTorch model saved by the classification owner, plus preprocessed ROI images.
- Grad-CAM needs the name of the last convolutional layer.
- Retrieval uses the 512-d penultimate-layer vector as the case embedding.
- Retrieval database = training cases only. A query case must never retrieve itself.

## Files
- `gradcam.py`: heatmap generation
- `retrieval.py`: embedding index and Top-3 search
