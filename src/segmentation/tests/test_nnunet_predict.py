import unittest

import pandas as pd

from src.segmentation.nnunet_predict import shard_rows


class ShardTests(unittest.TestCase):
    def test_shards_are_disjoint_and_cover_every_study(self):
        studies = pd.DataFrame({'study_id': [f'{n:03d}_1' for n in range(10)][::-1]})
        shards = [set(shard_rows(studies, i, 3).study_id) for i in range(3)]
        self.assertEqual(set.union(*shards), set(studies.study_id))
        self.assertEqual(sum(map(len, shards)), len(studies))

    def test_done_studies_are_skipped(self):
        studies = pd.DataFrame({'study_id': ['a', 'b', 'c', 'd']})
        self.assertEqual(shard_rows(studies, 0, 2, done={'a'}).study_id.tolist(), ['c'])


if __name__ == '__main__':
    unittest.main()
