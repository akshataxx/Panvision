import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.preprocessing.create_splits import clinical_cohort


class ClinicalSplitTests(unittest.TestCase):
    def test_repeated_patient_is_preserved_and_conflicts_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'clinical.xlsx'
            frame = pd.DataFrame([
                {'PANORAMA_patient_id': '001', 'PANORAMA_study_id': '001_1', 'label': 'PDAC'},
                {'PANORAMA_patient_id': '001', 'PANORAMA_study_id': '001_2', 'label': 'PDAC'},
            ])
            frame.to_excel(path, index=False)
            cohort = clinical_cohort(path)
            self.assertEqual(cohort.patient_id.nunique(), 1)
            self.assertEqual(len(cohort), 2)
            self.assertEqual(cohort.iloc[0].patient_id, '001')
            frame.loc[1, 'label'] = 'non-PDAC'
            frame.to_excel(path, index=False)
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                clinical_cohort(path)
