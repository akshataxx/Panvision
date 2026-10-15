# PanVision team data access

Code and the agreed patient split live on GitHub. Large CT files, masks, and
processed outputs live in the team's shared **OneDrive / PanVisionData** folder.

**Shared folder link: NOT ADDED YET.** Miti must add the OneDrive sharing link here
and grant teammates access. A GitHub repository URL does not grant OneDrive access.
Use the team folder with access restricted to the intended collaborators.

## Access the data

1. Request the PanVisionData OneDrive invitation from Miti if the link is absent
   or your account cannot access the folder.
2. Sign in with the account granted access and open the shared folder.
3. Sync it to your computer or download only the processed studies needed for your
   task. Make those files locally available before running a model against them.
4. Use the existing `splits.csv` from this GitHub repository for every model.
   Do not make a new train/validation/test split or split by slice.

The full cohort contains 2,238 studies from 2,224 unique patients. Processing and
uploading are incremental: the shared folder may not yet contain every study.
`manifests/full_progress.json` reports the latest processing run. A completion
receipt indicates processing finished locally, not that OneDrive upload finished.

## Folder layout

```text
PanVisionData/
├── raw/images/                       Original CT volumes (downloaded as needed)
├── raw/annotations/                  Official masks and clinical metadata
├── manifests/                        Download provenance, progress and failures
├── qa/                               Visual review images and HTML report
└── processed/full/<study_id>/
    ├── complete.json                 Completed-study receipt
    ├── input.csv                     Raw input provenance
    ├── splits.csv                    Shared split snapshot
    └── volumes/
        ├── manifest.csv              Diagnosis, split, provenance and output paths
        ├── config.json               HU window, spacing and orientation
        ├── audit.json                Original geometry
        └── <train|val|test>/<study_id>/
            ├── image.nii.gz          Normalized CT
            ├── labels.nii.gz         Combined structure labels, if available
            ├── pancreas.nii.gz       Binary label 4, if annotation available
            └── lesion.nii.gz         Binary label 1, if annotation available
```

Use **processed/full/** for shared model inputs. Earlier five-study trials may
also exist directly under `processed/train`, `processed/val`, and `processed/test`;
do not load both layouts together because that duplicates those studies.
Keep `.nii.gz` files compressed; SimpleITK reads them directly. Missing annotation
files are not equivalent to an empty or negative mask. Preserve manual versus
automatic annotation provenance when selecting supervision and evaluation cases.

## Connect your local checkout

Install preprocessing dependencies as described in
[the preprocessing guide](../src/preprocessing/README.md). Create
`data/storage.json` in your Git checkout with **your own** local synced path:

```json
{
  "data_root": "/your/local/OneDrive/PanVisionData"
}
```

On Windows, a forward-slash path such as `C:/Users/yourname/OneDrive/PanVisionData`
also works. This setting is ignored by Git; never commit another teammate's local
path. `PANVISION_DATA_DIR` can override the setting.

Full-run output paths in `volumes/manifest.csv` are relative to that manifest's
directory. For example, load a CT path in Python as:

```python
from pathlib import Path
import pandas as pd
from src.preprocessing.storage import data_root

manifest = (data_root() / "processed/full/100000_00001/volumes/manifest.csv")
cases = pd.read_csv(manifest, dtype=str, keep_default_na=False)
image_path = manifest.parent / cases.iloc[0]["processed_image"]
```

Collect manifests from all needed completed studies, without including older trial
outputs. Training loaders are not yet supplied. Raw paths in these full-run
manifests are relative to PanVisionData, as their columns declare; the earlier
exploration manifest has machine-specific absolute paths and should be regenerated
locally if needed. No raw CT download is needed merely to use processed model inputs.

## Team ownership and local disk space

Miti runs the full-dataset writer; teammates consume completed outputs. Avoid
running another full-dataset writer against the same shared folder. Do not rename,
edit, or delete shared study files while processing is running.

Cloud capacity does not increase local disk capacity. After OneDrive confirms
uploads are complete, use its **Free Up Space** action for completed files you
do not currently need locally. Deleting a synced file deletes the shared copy too.

## Source and Git rules

- Official dataset: https://zenodo.org/records/13715870 (batch 1; four batches total).
- Selected-scan mirror: https://huggingface.co/datasets/MedOtter/PANORAMA.
- Official annotations: https://github.com/DIAGNijmegen/panorama_labels.
- Follow source attribution and CC BY-NC 4.0 terms.

`data/splits.csv` and this README are tracked by Git. Machine-local settings,
scans, manifests, and outputs under `data/` remain ignored. Do not upload scans,
masks, credentials, or private clinical metadata to the GitHub repository.
