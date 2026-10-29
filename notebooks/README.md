# Notebooks

Exploration and Colab workflows. Reusable pipeline code belongs in `src/`.

Classification notebooks are under `notebooks/classification/`:

- `Resnet50_trainer.ipynb`: ResNet-50 training and patient-level evaluation.
- `PanVision_SimpleCNN.ipynb`: SimpleCNN baseline training and evaluation.
- `PanVision_Generate_Embeddings.ipynb`: use the frozen DenseNet checkpoint to
  export train-only KNN references and separate validation/test query embeddings.

DenseNet is maintained as reusable Python code in `src/classification/` rather
than as a training notebook.

Explainability notebooks (in `notebooks/`):

- `explainability_resnet.ipynb`: builds the 512-number fingerprints for every study with the trained ResNet-50, scores the Top-3 similar-case retrieval and the tumour-slice check on the validation patients, and draws Grad-CAM pictures for one PDAC and one healthy study. Needs a T4 GPU. Runs in about 15 minutes.
- `explainability_densenet.ipynb`: the same explainability checks run with the DenseNet-121 classifier (fingerprints, Top-3 retrieval on validation patients, tumour-slice check). Needs a T4 GPU. Runs in about 10 minutes.

The notebooks use data mounted from Google Drive. They must not contain access
tokens, database URLs, passwords, model weights, crops, or patient data.