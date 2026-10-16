# PANORAMA preparation in VS Code

Teammates accessing already processed files should start with
[the shared OneDrive access guide](../../data/README.md). The instructions below
are for the person downloading and preparing the dataset.

Run commands in **Terminal > New Terminal** from the PanVision folder on `data`.


## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r src/preprocessing/requirements.txt
```

Install Microsoft's Python extension if needed. Use **Cmd+Shift+P > Python:
Select Interpreter** and choose `.venv/bin/python`. Torch is not required yet.

## Download

### Recommended for this workspace: original volumes from Hugging Face

Community mirror: https://huggingface.co/datasets/MedOtter/PANORAMA
Use individual original NIfTI volumes, not preview images or overlays.

```bash
python -m pip install -r src/preprocessing/requirements.txt
python -m src.preprocessing.download_hf --count 5
python -m src.preprocessing.download_hf --count 5 --download
```

The first command previews network file sizes without downloading scans. The
second downloads five scans to the configured `<root>/raw/images` directory.
The first actual download records the exact mirror commit in
`<root>/manifests/huggingface_source.json`; later downloads reuse that revision.
Downloaded file lists and sizes are retained there. The official annotation clone
already belongs in `<root>/raw/annotations` and is reused. Record its Git commit
too (`git -C <root>/raw/annotations rev-parse HEAD`) for experiment provenance.
Validate case coverage, CT geometry and label consistency before training; a
community mirror is not an independent verification of equivalence to Zenodo.

For the next five scans use `--offset 5 --count 5`; for chosen cases use
`--studies 100000_00001 100002_00001`. Selection is sorted by filename and is for
pipeline exploration, not a representative training sample. Do not create a final
patient split using just this subset. Agree the full cohort and split before
model experiments. After downloading a subset, `build_manifest` and `validate`
can check it without creating splits. The SDK handles file transfers; OneDrive
still needs time to upload completed files. Avoid concurrent downloaders writing
the same shared root. Cloud quota does not replace local disk space.

### Alternative: official Zenodo archives

- Images: https://zenodo.org/records/10998332
- Annotations and clinical metadata: https://github.com/DIAGNijmegen/panorama_labels
- Challenge information: https://panorama.grand-challenge.org/

The verified Zenodo record lists `batch_1.zip`, **48.24 GB compressed**, and a
README. This record is a subset, not the full cohort. Locate remaining batches
through the challenge website. Allow additional space for extraction
and processed volumes; an external disk works with absolute manifest paths.
The record has a CC BY-NC 4.0 license. Retain attribution and source information.

Download from the page or use a resumable terminal download:

```bash
mkdir -p data/raw/images
curl -L --fail --retry 3 -C - \
  'https://zenodo.org/api/records/10998332/files/batch_1.zip/content' \
  -o data/raw/batch_1.zip
md5 data/raw/batch_1.zip
```

Expected MD5: `565c98f4ca5df5dc56540fd6b90376ab`. Extract after it matches:

```bash
unzip data/raw/batch_1.zip -d data/raw/images
git clone https://github.com/DIAGNijmegen/panorama_labels.git data/raw/annotations
```

Inspect `clinical_information.xlsx`, `manual_labels/` and `automatic_labels/`.
PANORAMA combined annotation values are lesion=1 and pancreas parenchyma=4.
Manual lesion provenance does not imply manual pancreas annotations.

## Manifest

After downloading images and annotations, build the manifest automatically:

```bash
python -m src.preprocessing.build_manifest
```

This joins CT filenames to the verified `PANORAMA_study_id` and
`PANORAMA_patient_id` spreadsheet columns, maps PDAC/non-PDAC, prefers available
manual annotations, and reports undownloaded studies. Review `data/manifest.csv`
in VS Code. Create the final shared split from the full clinical metadata;
the CT volumes do not all need to be downloaded first.
For a custom layout pass `--images`, `--annotations`, and `--clinical` paths.
Alternatively copy `manifest.example.csv` and enter one row per CT study:

| Column | Meaning |
| --- | --- |
| study_id | Unique study ID, letters/numbers/underscore/hyphen |
| patient_id | Source patient ID shared by all their studies |
| label | 1=PDAC, 0=non-PDAC, from clinical metadata |
| image_path | CT NIfTI or MHA path relative to CSV, or absolute |
| mask_path | Combined annotation path; blank if unavailable |
| annotation_tier | `manual`, `automatic`, `none` for lesion provenance |
| external_source | Source institution/dataset where known |

The example contains headers only. Preserve leading zeros in IDs. Do not infer
diagnosis from mask presence or assume study ID equals patient ID. The adapter
uses the official `_0000.nii.gz` CT filename convention and fails on unknown IDs.
If structures live in separate files, inspect and combine on the same physical
grid before using this pipeline. MSD label values differ from PANORAMA.

## Validate and prepare

```bash
python -m src.preprocessing.pipeline validate
python -m src.preprocessing.pipeline prepare
```

Or select **Terminal > Run Task > PanVision: validate data**, then
**PanVision: preprocess data**.

Validation checks 3D CT integrity, finite intensities, label values, and matching
image/mask size, origin, spacing and direction. Geometry mismatches fail instead
of silently resampling a potentially misregistered annotation. Visual overlays
are still required on real scans; these checks cannot verify annotation quality
or portal-venous acquisition phase.

First create the deterministic, stratified **70/15/15 patient split** from the
full clinical metadata, independent of which CTs are on disk:

```bash
python -m src.preprocessing.create_splits
```

The ratio is an initial project choice, not prescribed by the documents. Existing
`data/splits.csv` is validated and reused. Preparation requires this split and
allows a downloaded subset of its patients. Unknown patients and changed labels
fail instead of rewriting assignments. All models use this shared split.
The companion `data/splits.metadata.json` records source/split checksums and the
seed; it stays local under the current ignore rules. Share the CSV through Git.

For later batches, pass `--studies` to `build_manifest` using the study IDs selected
by the downloader and a fresh `--output <root>/manifests/batch_0002.csv` path.
Then run `pipeline prepare --manifest <that-file> --output <root>/processed/batch_0002`.
Use a separate output directory per batch to retain every batch's manifest and
configuration. Angle-bracket paths above are placeholders to replace. Never
recreate splits per batch. Outputs use train/val/test subdirectories within each
batch; downstream loaders must collect manifests from all completed batches.

Defaults: LPS orientation, XYZ spacing **1.5 × 1.5 × 3 mm**, HU window
**[-100, 300]**, then [0,1] normalization. These are configurable experimental
starting choices. CT uses linear interpolation, masks use nearest neighbour.
Physical geometry is retained in NIfTI files. SimpleITK arrays have Z,Y,X order.
DICOMOrient flips/permutes axes; oblique inputs retain direction cosines and need
review before assuming array slices are anatomical axial planes.

```bash
python -m src.preprocessing.pipeline prepare \
  --spacing 1 1 3 --window -100 300 --output data/processed_experiment2
```

Outputs include original geometry in `audit.json`, parameters in `config.json`,
an output `manifest.csv`, and `train/<study>/image.nii.gz`, `labels.nii.gz`,
`pancreas.nii.gz`, `lesion.nii.gz` (likewise for val/test), under `data/processed`.
Missing masks produce no mask outputs and must never become background-only
supervision. Automatic lesions remain distinct from manual evaluation labels.
Empty lesions in PDAC cases need review before lesion training. Pancreas labels
are automatic and need QA. Use physical spacing for future measurements.

Existing study outputs are not overwritten. After an interrupted run use a fresh
output directory. Inputs must already be in HU; raw DICOM series ingestion is
not implemented. Preserve original scans for QA and measurement checks.

## Tests and next steps

```bash
python -m unittest discover -s src/preprocessing/tests -v
```

Synthetic tests exercise grouping, split reuse, geometry, label interpolation,
windowing, missing annotations and an end-to-end run. Real scans still require
overlays and class/source distribution review after download. Then add 2D slice
indexing and train-only augmentation in model data loaders. Keep full physical
volumes now rather than resizing away measurement geometry. Use predicted pancreas
ROIs for end-to-end evaluation; document ground-truth ROI experiments separately.

## Save code to your branch

```bash
git add src/preprocessing .vscode
git commit -m "Add patient-level PANORAMA preprocessing pipeline"
git push
```

Raw scans and processed outputs are ignored. `data/splits.csv` is the existing
explicit exception for team sharing; review it before committing. Do not commit
private clinical metadata.

## Visual mask QA

After manifest creation and validation, run:

```bash
python -m src.preprocessing.qa
```

Open `<root>/qa/index.html` in your browser. Each study has unmodified CT and
overlay panels (green pancreas, red lesion), selected slice indices, and a CSV
summary. This never creates splits or processed training data. Review masks
through the full volume in a medical image viewer before accepting annotations.
Use `--output outputs/qa` to keep a local review outside the OneDrive folder.

## Full dataset runner

Once the initial scans pass QA and `data/splits.csv` exists, use:

```bash
# Metadata-only coverage and progress check; no scans are downloaded.
python -m src.preprocessing.full_dataset

# Process the next five unfinished studies.
python -m src.preprocessing.full_dataset --run

# Continue through the full cohort until complete, interrupted, low on disk,
# or three consecutive failures occur.
python -m src.preprocessing.full_dataset --run --max-batches 0
```

Re-running skips completed studies and retries unfinished studies. The runner
checks clinical cohort membership against the mirror, pins the mirror revision,
and validates the full patient split before downloading. Raw images go into the
configured OneDrive `raw/images` folder; existing official annotations are reused.

Each study is prepared in a temporary `.partial-...` folder. After all files are
written, the directory is renamed to `processed/full/<study_id>/`, containing a
`complete.json` receipt, input provenance, a split snapshot, and `volumes/` outputs.
The full runner keeps outputs separate from earlier manual preparation runs; it
does not automatically adopt outputs lacking a completion receipt. This may make
another processed copy of the first five studies. Output manifests use relative
paths (as declared in their columns) so they work on teammates' machines.

`manifests/full_progress.json` records the latest run and remaining count;
`full_errors.jsonl` retains failures across runs. A study is counted complete only
with a matching source/configuration signature and the expected local file sizes.
This is not a cryptographic integrity check of every processed voxel or a check
that OneDrive has uploaded the output. The signature includes the clinical/split
hashes, mirror revision, preprocessing settings, and annotation filenames/sizes.
Changes to annotation contents without size changes require a fresh run root.

The default disk reserve is **10 GiB**. Disk estimates cover selected downloads
and uncompressed processed output, but other processes and OneDrive activity can
also consume space. The runner stops rather than deleting or evicting any files.
It cannot query your cloud quota, guarantee upload completion, or reclaim disk.
After stopping, confirm OneDrive syncing has finished, use **Free Up Space** on
completed raw and processed files, and rerun. Do not delete synced files to free
local storage. Do not evict or edit files while the runner is processing them.

One machine must own writes to a shared root. A local lock prevents duplicate
processes on that machine; it is not a distributed lock across teammates' Macs.
Interrupted `.partial-...` directories remain for review and are never counted
complete. A retry starts a fresh staging directory. Review old partial data before
manually removing it because OneDrive deletions propagate to the cloud.

Successful processing does not certify annotation quality: review original and
resampled overlays across the cohort, especially automatic masks and small lesions.
No training or raw-file removal is performed by this command.
