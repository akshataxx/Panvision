import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.knn.retrieval import build_index, load_reference_bank, top_k
from src.knn.neon_embeddings import (
    _load_metadata, embedding_rows, load_query_patient, upload_reference_bank,
    vector_literal,
)


class _Result:
    def fetchone(self):
        return {'count': 1}


class _Cursor:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def executemany(self, statement, rows):
        self.connection.uploaded_rows = list(rows)


class _Psycopg3Connection:
    """Minimal Psycopg 3 seam: bulk execution is exposed by its cursor."""

    def __init__(self):
        self.uploaded_rows = []

    def cursor(self):
        return _Cursor(self)

    def execute(self, *_args, **_kwargs):
        return _Result()


class RetrievalTests(unittest.TestCase):
    def test_top_k_uses_cosine_similarity_and_excludes_query_patient(self):
        index = build_index(
            np.asarray([[1, 0], [0.8, 0.2], [0, 1]], dtype=np.float32),
            ['query', 'near', 'far'],
            [1, 1, 0],
        )

        matches = top_k(index, np.asarray([1, 0], dtype=np.float32), k=2,
                        exclude_patient_id='query')

        self.assertEqual([match['patient_id'] for match in matches], ['near', 'far'])
        self.assertEqual([match['label'] for match in matches], [1, 0])
        self.assertAlmostEqual(matches[0]['similarity'], 0.9701425, places=6)

    def test_reference_bank_rejects_non_training_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bank.npz'
            np.savez_compressed(
                path,
                embeddings=np.ones((2, 512), dtype=np.float32),
                ids=np.asarray(['p1', 'p2']),
                labels=np.asarray([0, 1]),
                split=np.asarray(['train', 'val']),
                entity=np.asarray('patient'),
                knn_reference=np.asarray(True),
                dimensions=np.asarray(512),
            )

            with self.assertRaisesRegex(ValueError, 'training patients only'):
                load_reference_bank(path)

    def test_reference_bank_loads_legacy_export_with_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'legacy.npz'
            embeddings = np.zeros((2, 512), dtype=np.float32)
            embeddings[:, 0] = 1
            np.savez_compressed(
                path, embeddings=embeddings,
                patient_ids=np.asarray(['p1', 'p2']),
                labels=np.asarray([0, 1]), splits=np.asarray(['train', 'train']),
                checkpoint_sha256=np.asarray('abc123'),
                aggregation=np.asarray('mean'),
            )

            index = load_reference_bank(path)

        self.assertEqual(index['patient_ids'].tolist(), ['p1', 'p2'])
        self.assertEqual(index['checkpoint_sha256'], 'abc123')

    def test_embedding_rows_include_checkpoint_and_train_split(self):
        index = build_index(
            np.asarray([[3, 4], [0, 2]], dtype=np.float32),
            ['p1', 'p2'], [1, 0],
        )
        index['split'] = np.asarray(['train', 'train'])

        rows = list(embedding_rows(index, 'dense-v1', 'abc123'))

        self.assertEqual(rows[0][:5], ('dense-v1', 'abc123', 'p1', 1, 'train'))
        self.assertEqual(vector_literal(np.asarray([0.6, 0.8], dtype=np.float32)), '[0.6,0.8]')
        self.assertEqual(len(rows), 2)

    def test_embedding_rows_reject_checkpoint_mismatch(self):
        index = build_index(
            np.asarray([[1] + [0] * 511], dtype=np.float32),
            ['p1'], [1],
        )
        index['split'] = np.asarray(['train'])
        index['checkpoint_sha256'] = 'bank-checkpoint'

        with self.assertRaisesRegex(ValueError, 'checkpoint SHA-256 does not match'):
            list(embedding_rows(index, 'dense-v1', 'metadata-checkpoint'))

    def test_metadata_allows_explicit_model_key_for_legacy_export(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'metadata.json'
            path.write_text('{"dimensions":512,"checkpoint_sha256":"abc123"}')

            model, checkpoint = _load_metadata(path, model_key='densenet121_512')

        self.assertEqual(model, 'densenet121_512')
        self.assertEqual(checkpoint, 'abc123')

    def test_upload_uses_psycopg3_cursor_for_executemany(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'legacy.npz'
            embedding = np.zeros((1, 512), dtype=np.float32)
            embedding[0, 0] = 1
            np.savez_compressed(
                path, embeddings=embedding, patient_ids=np.asarray(['p1']),
                labels=np.asarray([1]), splits=np.asarray(['train']),
                checkpoint_sha256=np.asarray('abc123'),
            )
            connection = _Psycopg3Connection()

            count = upload_reference_bank(connection, path, 'dense-v1', 'abc123')

        self.assertEqual(count, 1)
        self.assertEqual(len(connection.uploaded_rows), 1)

    def test_load_query_patient_supports_legacy_bank(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'queries.npz'
            embeddings = np.zeros((2, 512), dtype=np.float32)
            embeddings[0, 0] = 1
            embeddings[1, 1] = 1
            np.savez_compressed(
                path, embeddings=embeddings,
                patient_ids=np.asarray(['p1', 'p2']),
                labels=np.asarray([0, 1]), splits=np.asarray(['val', 'val']),
                checkpoint_sha256=np.asarray('abc123'),
            )

            query, details = load_query_patient(path, 'p2', 'abc123')

        self.assertEqual(query.shape, (512,))
        self.assertEqual(float(query[1]), 1.0)
        self.assertEqual(details, {'patient_id': 'p2', 'label': 1, 'split': 'val'})


if __name__ == '__main__':
    unittest.main()
