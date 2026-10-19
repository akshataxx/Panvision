"""Train the ResNet-50 PDAC classifier on the saved crops.

Needs outputs/crops made by make_crops.py. A GPU (for example on Colab) makes it fast;
on a laptop CPU it is far too slow except for a tiny smoke test.

Smoke test on your laptop (checks that everything runs; the numbers mean nothing):
    python src/classification/train_resnet.py --crops outputs/crops_test4 ^
        --val-split train --epochs 1 --max-batches 3 --out outputs/smoke.pt

Real run (on Colab):
    python src/classification/train_resnet.py --crops outputs/crops --epochs 8

What happens each epoch: train on the TRAIN patients, then score the VALIDATION patients
(one averaged score per patient) and keep the best model so far.
"""
import argparse
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import CropDataset
from evaluation import evaluate_patients, format_metrics, patient_scores, pick_threshold, predict_probs
from resnet_model import ResNet50Classifier, save_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train the ResNet-50 classifier")
    parser.add_argument("--crops", default="outputs/crops")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None, help="smoke test: stop each epoch early")
    parser.add_argument("--out", default="models/classifier.pt")
    args = parser.parse_args()

    # Make runs repeatable.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}" + ("  (very slow! use a GPU for a real run)" if device == "cpu" else ""))

    train_ds = CropDataset(args.crops, args.train_split, mode="train", augment=True)
    val_ds = CropDataset(args.crops, args.val_split, mode="eval")
    pos_weight = train_ds.pos_weight()
    print(f"Train: {len(train_ds)} pictures from {len(train_ds.patient_ids)} patients "
          f"({int(train_ds.patient_labels.sum())} PDAC). PDAC pictures weighted x{pos_weight:.1f}.")
    print(f"Validation: {len(val_ds)} pictures from {len(val_ds.patient_ids)} patients "
          f"({int(val_ds.patient_labels.sum())} PDAC).")

    model = ResNet50Classifier(pretrained=True).to(device)
    # Weighted loss: a mistake on a PDAC picture costs more, because PDAC pictures are rarer.
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True)

    best_auc = -1.0
    for epoch in range(1, args.epochs + 1):
        # --- learn from the training pictures ---
        model.train()
        total, seen = 0.0, 0
        for b, (x, y, _) in enumerate(loader):
            if args.max_batches is not None and b >= args.max_batches:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
            seen += len(y)

        # --- check on the validation patients ---
        probs, patient = predict_probs(model, val_ds, device, workers=0)
        metrics = evaluate_patients(probs, patient, val_ds.patient_labels, threshold=0.5)
        print(f"Epoch {epoch}/{args.epochs}  train loss {total / max(seen, 1):.4f}  |  "
              f"validation: {format_metrics(metrics)}")

        if metrics["auc"] > best_auc:  # keep the best model so far
            best_auc = metrics["auc"]
            scores, labels = patient_scores(probs, patient, val_ds.patient_labels)
            threshold = pick_threshold(scores, labels)
            save_checkpoint(model, args.out, epoch=epoch, val_auc=float(best_auc),
                            threshold=float(threshold), seed=int(args.seed))
            print(f"   saved {args.out} (best so far, suggested threshold {threshold:.2f})")

    print(f"\nDone. Best validation AUC {best_auc:.3f}.")


if __name__ == "__main__":
    main()
