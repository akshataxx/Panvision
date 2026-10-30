"""Score the Top-3 similar-case retrieval on patients the database has never seen.

The database holds TRAINING patients only. The queries are the patients of another split
(validation while you are developing; test once, at the very end). For every query the
3 most similar training patients are found, and Precision@3 says how many of them have
the same label as the query. The random baseline shows what blind picking would give.

Needs outputs/embeddings.npz from build_embeddings.py. Uses only numpy.

Run from the repo root:
    python src/explainability/evaluate_retrieval.py --split val
"""
import argparse

import numpy as np

from retrieval_eval import aggregate_by_patient, build_index, precision_at_k_queries


def main():
    parser = argparse.ArgumentParser(description="Precision@k of similar-case retrieval")
    parser.add_argument("--embeddings", default="outputs/embeddings.npz")
    parser.add_argument("--split", default="val", choices=["val", "test"],
                        help="which patients act as queries (use test only once, at the end)")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    data = np.load(args.embeddings)
    patient_ids, labels = data["patient_ids"], data["labels"]
    splits, emb = data["splits"], data["embeddings"]

    train = splits == "train"
    db_emb, db_ids, db_labels = aggregate_by_patient(emb[train], patient_ids[train], labels[train])
    index = build_index(db_emb, db_ids, db_labels)

    query = splits == args.split
    q_emb, q_ids, q_labels = aggregate_by_patient(emb[query], patient_ids[query], labels[query])

    # Safety check: no query patient may be in the database.
    overlap = set(q_ids.tolist()) & set(db_ids.tolist())
    if overlap:
        raise SystemExit(f"{len(overlap)} query patients are also in the database: split problem!")

    r = precision_at_k_queries(index, q_emb, q_labels, args.k)
    print(f"Database: {len(db_ids)} training patients ({int(db_labels.sum())} PDAC)")
    print(f"Queries : {r['n_queries']} {args.split} patients ({int(q_labels.sum())} PDAC)\n")
    print(f"Precision@{args.k}, all queries : {r['precision']:.2f}  (random baseline {r['random_baseline']:.2f})")
    if r["pdac_precision"] is not None:
        print(f"Precision@{args.k}, PDAC queries: {r['pdac_precision']:.2f}  "
              f"(random baseline {r['pdac_random_baseline']:.2f})")
    print(f"At least one of the {args.k} has the same label: {r['at_least_one_same']:.0%} of queries")
    print("\nA useful model must beat the random baseline, especially for PDAC queries.")


if __name__ == "__main__":
    main()
