import unittest

import pandas as pd

from src.segmentation.prepare_subset import select


class SelectTests(unittest.TestCase):
    def test_balanced_manual_pdac_and_split_respected(self):
        rows = []
        for i in range(40):
            rows.append({'patient_id': f'p{i}', 'study_id': f's{i}', 'label': '1' if i % 2 else '0'})
        cohort = pd.DataFrame(rows)
        splits = pd.DataFrame({'patient_id': cohort.patient_id,
                               'split': ['train'] * 30 + ['val'] * 10, 'label': cohort.label})
        manual = {f's{i}' for i in range(1, 40, 4)}  # only some PDAC studies have manual lesion masks
        chosen = select(cohort, splits, manual, {'train': 8, 'val': 4})
        pdac = chosen[chosen.label == '1']
        self.assertTrue(pdac.study_id.isin(manual).all())
        train = chosen[chosen.split == 'train']
        self.assertEqual(len(train), 8)
        self.assertEqual((train.label == '1').sum(), 4)
        self.assertEqual(set(chosen.split), {'train', 'val'})
        self.assertTrue(chosen.equals(select(cohort, splits, manual, {'train': 8, 'val': 4})))  # deterministic


if __name__ == '__main__':
    unittest.main()
