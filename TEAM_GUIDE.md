# Team guide

## Who owns what
| Area | Owner |
|---|---|
| Preprocessing, segmentation | (name) |
| Classification | (name) |
| Explainability (Grad-CAM, retrieval), repo | (name) |
| Report, dashboard | (shared) |

## Working rules
- Work only inside your own folder. If you need to change someone else's file, tell them first.
- Upload small changes often rather than one big change at the end.
- Never upload CT data or model weights (`.nii.gz`, `.mha`, `.pt`, `.pth`). The `.gitignore` blocks most of them; share these through a shared drive instead.
- Write a short message when you save a change, e.g. "Add Grad-CAM function".
- Use the same patient-level split file for every model. Do not make your own.

## Shared agreements (fill these in)
- Split file location: `data/splits.csv`, columns: `patient_id, split, label`
- Saved classifier file: `models/classifier.pt` containing the full model
- Embedding size: 512
