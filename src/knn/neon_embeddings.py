"""Import train-only DenseNet embeddings into Neon and run cosine KNN queries."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from src.knn.retrieval import load_reference_bank


TABLE_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS case_embeddings (
    model_key TEXT NOT NULL,
    checkpoint_sha256 TEXT NOT NULL,
    patient_id TEXT NOT NULL,
    label SMALLINT NOT NULL CHECK (label IN (0, 1)),
    split TEXT NOT NULL CHECK (split = 'train'),
    embedding VECTOR(512) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (model_key, checkpoint_sha256, patient_id)
);
CREATE INDEX IF NOT EXISTS case_embeddings_model
    ON case_embeddings(model_key, checkpoint_sha256);
"""


def vector_literal(vector):
    values = np.asarray(vector, dtype=np.float32)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Vector must be finite and one-dimensional')
    return '[' + ','.join(
        np.format_float_positional(value, unique=True, trim='-') for value in values
    ) + ']'


def embedding_rows(index, model_key, checkpoint_sha256):
    """Yield rows for a model-versioned, train-only reference bank."""
    if not model_key.strip() or not checkpoint_sha256.strip():
        raise ValueError('Model key and checkpoint SHA-256 are required')
    bank_checkpoint = str(index.get('checkpoint_sha256') or '').strip()
    if bank_checkpoint and bank_checkpoint != checkpoint_sha256:
        raise ValueError('Embedding bank checkpoint SHA-256 does not match metadata')
    splits = np.asarray(index.get('split', []), dtype=str)
    if len(splits) != len(index['patient_ids']) or set(splits.tolist()) != {'train'}:
        raise ValueError('Only training patients may be uploaded as KNN references')
    for patient_id, label, split, embedding in zip(
            index['patient_ids'], index['labels'], splits, index['embeddings']):
        yield (model_key, checkpoint_sha256, str(patient_id), int(label), str(split),
               vector_literal(embedding))


def database_url():
    url = os.environ.get('PANVISION_DATABASE_URL', '').strip()
    if not url.startswith(('postgresql://', 'postgres://')):
        raise RuntimeError('Set PANVISION_DATABASE_URL to the Neon PostgreSQL connection URL')
    return url


def initialize(connection):
    connection.execute(TABLE_SQL)


def upload_reference_bank(connection, bank_path, model_key, checkpoint_sha256):
    """Create the pgvector table and idempotently upsert one reference bank."""
    index = load_reference_bank(bank_path)
    rows = list(embedding_rows(index, model_key, checkpoint_sha256))
    initialize(connection)
    with connection.cursor() as cursor:
        cursor.executemany(
            """INSERT INTO case_embeddings
               (model_key,checkpoint_sha256,patient_id,label,split,embedding)
               VALUES (%s,%s,%s,%s,%s,%s::vector)
               ON CONFLICT (model_key,checkpoint_sha256,patient_id) DO UPDATE SET
                 label=excluded.label, split=excluded.split,
                 embedding=excluded.embedding, updated_at=CURRENT_TIMESTAMP""",
            rows,
        )
    count = connection.execute(
        """SELECT COUNT(*) AS count FROM case_embeddings
           WHERE model_key=%s AND checkpoint_sha256=%s""",
        (model_key, checkpoint_sha256),
    ).fetchone()
    return int(count['count'] if isinstance(count, dict) else count[0])


def query_neon(connection, query_embedding, model_key, checkpoint_sha256,
               k=3, exclude_patient_id=None):
    """Return exact cosine neighbors from one checkpoint's training references."""
    if k < 1:
        raise ValueError('k must be positive')
    query = np.asarray(query_embedding, dtype=np.float32)
    if query.shape != (512,) or not np.isfinite(query).all() or np.linalg.norm(query) <= 0:
        raise ValueError('Query embedding must be a finite, non-zero 512-vector')
    exclusion = '' if exclude_patient_id is None else 'AND patient_id <> %s'
    params = [vector_literal(query), model_key, checkpoint_sha256]
    if exclude_patient_id is not None:
        params.append(str(exclude_patient_id))
    params.append(k)
    rows = connection.execute(
        f"""SELECT patient_id,label,1-(embedding <=> %s::vector) AS similarity
            FROM case_embeddings
            WHERE model_key=%s AND checkpoint_sha256=%s {exclusion}
            ORDER BY embedding <=> %s::vector
            LIMIT %s""",
        tuple(params[:-1] + [params[0], params[-1]]),
    ).fetchall()
    return [dict(row) for row in rows]


def load_query_patient(bank_path, patient_id, checkpoint_sha256):
    """Read one patient embedding from a current or legacy exported bank."""
    with np.load(Path(bank_path), allow_pickle=False) as bank:
        id_key = 'ids' if 'ids' in bank.files else 'patient_ids'
        split_key = 'split' if 'split' in bank.files else 'splits'
        required = {'embeddings', 'labels', id_key, split_key}
        missing = required.difference(bank.files)
        if missing:
            raise ValueError(f'Query bank is missing: {", ".join(sorted(missing))}')
        bank_checkpoint = (str(np.asarray(bank['checkpoint_sha256']).item())
                           if 'checkpoint_sha256' in bank.files else '')
        if bank_checkpoint and bank_checkpoint != checkpoint_sha256:
            raise ValueError('Query bank checkpoint SHA-256 does not match metadata')
        ids = np.asarray(bank[id_key], dtype=str)
        matches = np.flatnonzero(ids == str(patient_id))
        if len(matches) != 1:
            raise ValueError(f'Patient {patient_id} was not found exactly once in the query bank')
        row = int(matches[0])
        embedding = np.asarray(bank['embeddings'][row], dtype=np.float32)
        details = {
            'patient_id': str(ids[row]),
            'label': int(np.asarray(bank['labels'])[row]),
            'split': str(np.asarray(bank[split_key], dtype=str)[row]),
        }
    if embedding.shape != (512,) or not np.isfinite(embedding).all():
        raise ValueError('Query bank must contain finite 512-dimensional embeddings')
    return embedding, details


def _load_metadata(path, model_key=None):
    metadata = json.loads(Path(path).read_text())
    if int(metadata.get('dimensions', 0)) != 512:
        raise ValueError('Embedding metadata must declare 512 dimensions')
    model_key = str(model_key or metadata.get('model', '')).strip()
    checkpoint = str(metadata.get('checkpoint_sha256', '')).strip()
    if not model_key or not checkpoint:
        raise ValueError('Embedding metadata is missing model or checkpoint SHA-256')
    return model_key, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    subparsers.add_parser('init', help='create or validate the pgvector table')
    importer = subparsers.add_parser('import', help='upload a train reference bank')
    importer.add_argument('--bank', required=True)
    importer.add_argument('--metadata', required=True)
    importer.add_argument('--model-key', help='required for legacy metadata without a model field')
    query = subparsers.add_parser('query', help='retrieve similar training cases for one patient')
    query.add_argument('--bank', required=True, help='patient embedding bank containing the query')
    query.add_argument('--patient-id', required=True)
    query.add_argument('--metadata', required=True)
    query.add_argument('--model-key', help='required for legacy metadata without a model field')
    query.add_argument('-k', type=int, default=3)
    args = parser.parse_args()

    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(database_url(), row_factory=dict_row, connect_timeout=10) as connection:
        if args.command == 'init':
            initialize(connection)
            print('Neon case_embeddings table is ready.')
        elif args.command == 'import':
            model_key, checkpoint = _load_metadata(args.metadata, args.model_key)
            count = upload_reference_bank(connection, args.bank, model_key, checkpoint)
            print(f'Uploaded {count} train reference embeddings for {model_key} ({checkpoint[:12]}...).')
        else:
            model_key, checkpoint = _load_metadata(args.metadata, args.model_key)
            embedding, details = load_query_patient(args.bank, args.patient_id, checkpoint)
            matches = query_neon(
                connection, embedding, model_key, checkpoint, k=args.k,
                exclude_patient_id=args.patient_id,
            )
            print(json.dumps({'query': details, 'neighbors': matches}, indent=2, default=float))


if __name__ == '__main__':
    main()
