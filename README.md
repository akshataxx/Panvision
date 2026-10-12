# PanVision

Explainable AI-assisted pancreatic CT analysis (NUS ISS GC, Pattern Recognition Systems practice project, Group 3).

PanVision is an **academic research prototype**. It is not a clinical tool and gives no diagnosis or treatment recommendation.

## What it does
1. Preprocesses PANORAMA portal-venous CT volumes (HU windowing, resampling, patient-level split).
2. Segments the pancreas, then lesions, with 2D U-Nets and computes lesion volume and diameter.
3. Classifies PDAC vs non-PDAC (custom CNN vs ResNet-50).
4. Explains predictions with Grad-CAM and retrieves the Top-3 similar reference cases.
5. Produces a template-based summary in a Streamlit dashboard.

## Repository layout
| Folder | Purpose |
|---|---|
| `data/` | Local data only. Not uploaded to GitHub. |
| `src/preprocessing/` | CT loading, windowing, resampling, patient-level split |
| `src/segmentation/` | Pancreas and lesion U-Nets, measurements |
| `src/classification/` | Custom CNN and ResNet-50 classifiers |
| `src/explainability/` | Grad-CAM and similar-case retrieval |
| `src/report/` | Template-based summary text |
| `app/` | Streamlit dashboard |
| `notebooks/` | Experiments and exploration |
| `docs/` | Proposal, slides, report |

## Setup (Windows)
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Team rules
See `TEAM_GUIDE.md`.
