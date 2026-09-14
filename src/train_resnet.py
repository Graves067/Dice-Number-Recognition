"""
Trains the ResNet die-face classifier on the labeled dataset built by
label_tool.py (src/Dataset/1..6/).

Usage:
    python train_resnet.py
    python train_resnet.py --epochs 30 --batch-size 16 --lr 0.001

Saves the best checkpoint (by val accuracy) to src/models/dice_resnet_best.pt.
If there isn't enough data yet for a val split, trains on the full set and
saves the final-epoch checkpoint instead — this is meant to be re-run as
the dataset grows, not a one-shot.
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim

from dice_dataset import get_dataloaders, CLASSES
from resnet_model import build_model

CHECKPOINT_DIR = r"src\models"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "dice_resnet_best.pt")


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size, val_split=args.val_split
    )

    model = build_model(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)

        if val_loader is not None:
            val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"train loss {train_loss:.4f} acc {train_acc:.1%} | "
                  f"val loss {val_loss:.4f} acc {val_acc:.1%}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({"model_state": model.state_dict(), "classes": CLASSES},
                           CHECKPOINT_PATH)
                print(f"  -> new best (val acc {val_acc:.1%}), saved to {CHECKPOINT_PATH}")
        else:
            # Not enough data for a val split yet — just track train acc
            # and keep the final checkpoint.
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"train loss {train_loss:.4f} acc {train_acc:.1%}  "
                  f"(no val split — dataset too small)")
            torch.save({"model_state": model.state_dict(), "classes": CLASSES},
                       CHECKPOINT_PATH)

    print(f"\nDone. Checkpoint saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()