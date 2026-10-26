# notebooks
Exploration notebooks. Name them `initials_topic.ipynb` to avoid clashes.

- `Resnet50_trainer.ipynb`: trains the ResNet-50 classifier on Colab (T4 GPU), picks how slice scores become a patient score using validation, and scores the test patients once. Needs `crops.zip` and `classification.zip` in the PanVision folder on Google Drive.

- `explainability_resnet.ipynb`: builds the 512-number fingerprints for every study with the trained ResNet-50, scores the Top-3 similar-case retrieval and the tumour-slice check on the validation patients, and draws Grad-CAM pictures for one PDAC and one healthy study. Needs `crops.zip`, `code.zip` (the `src` and `research` folders) and `classifier.pt` in the PanVision folder on Google Drive, and a T4 GPU. Runs in about 15 minutes.

- `explainability_densenet.ipynb`: the same explainability checks as the ResNet notebook, run with Miti's DenseNet-121: builds the 512-number fingerprints for every study, scores the Top-3 similar-case retrieval on the validation patients, and runs the tumour-slice check. Needs `crops.zip`, `code.zip` (the `src` and `research` folders) and the DenseNet checkpoint (`densenet_best.pt`) in the PanVision folder on Google Drive, and a T4 GPU. Runs in about 10 minutes.