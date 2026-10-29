# Notebooks

Exploration and Colab workflows. Reusable pipeline code belongs in `src/`.

Classification notebooks are under `notebooks/classification/`:

- `Resnet50_trainer.ipynb`: ResNet-50 training and patient-level evaluation.
- `PanVision_SimpleCNN.ipynb`: SimpleCNN baseline training and evaluation.
- `PanVision_Generate_Embeddings.ipynb`: use the frozen DenseNet checkpoint to
  export train-only KNN references and separate validation/test query embeddings.

DenseNet is maintained as reusable Python code in `src/classification/` rather
than as a training notebook.

The notebooks use data mounted from Google Drive. They must not contain access
tokens, database URLs, passwords, model weights, crops, or patient data.
