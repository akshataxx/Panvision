"""Show Top-3 similar cases and score the retrieval (Precision@3).

Reads outputs/embeddings.npz made by build_embeddings.py. Needs no torch.

Run from the repo root:
    python src/explainability/retrieval_demo.py

With the STAND-IN model the scores below are meaningless; the checks that matter are
that the code runs, a patient never finds themselves, and the numbers can be compared
with the random baseline.
"""
import argparse

import numpy as np

from retrieval import aggregate_by_patient, build_index, precision_at_k, top_k


def main():
    parser = argparse.ArgumentParser(description="Top-3 retrieval demo and evaluation")
    parser.add_argument("--embeddings", default="outputs/embeddings.npz")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    data = np.load(args.embeddings)
    study_ids, patient_ids = data["study_ids"], data["patient_ids"]
    labels, splits, emb = data["labels"], data["splits"], data["embeddings"]

    # The database may hold TRAINING patients only.
    keep = splits == "train"
    print(f"{keep.sum()} of {len(splits)} studies are from the train split and go in the database.")
    p_emb, p_ids, p_labels = aggregate_by_patient(emb[keep], patient_ids[keep], labels[keep])
    index = build_index(p_emb, p_ids, p_labels)
    print(f"Database: {len(p_ids)} patients ({int(p_labels.sum())} PDAC)\n")

    # --- Example: Top-3 for one PDAC patient -------------------------------------
    first_pdac = p_ids[p_labels == 1][0]
    print(f"Example query: patient {first_pdac} (PDAC), excluded from their own search")
    for pid, label, sim in top_k(index, p_emb[list(p_ids).index(first_pdac)], args.k, first_pdac):
        print(f"   {pid}  {'PDAC' if label else 'non-PDAC':8s}  similarity {sim:.3f}")

    # --- Check: a study never retrieves another study of the SAME patient ----------
    repeat = [p for p in set(patient_ids[keep].tolist()) if (patient_ids[keep] == p).sum() > 1]
    for pid in sorted(repeat):
        study_vec = emb[keep][patient_ids[keep] == pid][0]  # fingerprint of just one study
        found = top_k(index, study_vec, args.k, exclude_patient_id=pid)
        ok = all(f[0] != pid for f in found)
        print(f"\nSame-patient check, patient {pid} has several studies: "
              f"{'PASS' if ok else 'FAIL'} (own patient not in Top-{args.k})")

    # --- Score: leave-one-patient-out Precision@3 ----------------------------------
    r = precision_at_k(index, args.k)
    print(f"\nPrecision@{args.k} over {r['n_queries']} queries")
    print(f"   all patients : {r['precision']:.2f}   (random baseline {r['random_baseline']:.2f})")
    if r["pdac_precision"] is not None:
        print(f"   PDAC queries : {r['pdac_precision']:.2f}   (random baseline {r['pdac_random_baseline']:.2f})")
    print("A useful model must beat the random baseline. The stand-in model should not.")


if __name__ == "__main__":
    main()
