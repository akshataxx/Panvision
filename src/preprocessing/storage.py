"""Machine-local storage settings; personal paths stay out of Git."""

import json
import os
from pathlib import Path


def data_root():
    override = os.environ.get('PANVISION_DATA_DIR')
    if override:
        return Path(override).expanduser().resolve()
    project = Path(__file__).resolve().parents[2]
    config = project / 'data/storage.json'
    if config.exists():
        value = json.loads(config.read_text())['data_root']
        return Path(value).expanduser().resolve()
    return project / 'data'


def manifest_path():
    root = data_root()
    project = Path(__file__).resolve().parents[2]
    return root / 'manifest.csv' if root == project / 'data' else root / 'manifests/manifest.csv'
