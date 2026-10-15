import unittest
from types import SimpleNamespace

from src.preprocessing.download_hf import select_images


class DownloadSelectionTests(unittest.TestCase):
    def test_selects_original_volumes_and_ranges(self):
        files = [SimpleNamespace(rfilename=name) for name in (
            'images/b_0000.nii.gz', 'previews/a.png', 'images/a_0000.nii.gz',
            'labels/manual/a.nii.gz')]
        self.assertEqual(select_images(files, count=1)[0].rfilename, 'images/a_0000.nii.gz')
        self.assertEqual(select_images(files, count=1, offset=1)[0].rfilename, 'images/b_0000.nii.gz')
        self.assertEqual(select_images(files, studies=['b'])[0].rfilename, 'images/b_0000.nii.gz')
        with self.assertRaises(ValueError):
            select_images(files, studies=['missing'])
        with self.assertRaises(ValueError):
            select_images(files, offset=10)
        with self.assertRaises(ValueError):
            select_images(files, count=0)
