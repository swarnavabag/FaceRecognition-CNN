from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from src.shared.config import (
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    CHECKPOINT_PATH,
    BEST_MODEL_PATH
)

from src.reco.dataset import create_dataloaders
from src.reco.model import FaceRecognitionResNet
from src.reco.losses import ArcMarginProduct


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# METRICS
# ============================================================

def calculate_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(model, arcface, loader, criterion, optimizer, device):
    model.train()
    arcface.train()

    running_loss = 0.0
    running_acc = 0.0

    progress_bar = tqdm(loader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        embeddings = model(images)
        logits = arcface(embeddings, labels)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_acc = calculate_accuracy(logits, labels)

        running_loss += loss.item()
        running_acc += batch_acc

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{batch_acc:.4f}"
        })

    epoch_loss = running_loss / len(loader)
    epoch_acc = running_acc / len(loader)

    return epoch_loss, epoch_acc


# ============================================================
# CHECKPOINTING
# ============================================================

def save_checkpoint(model, arcface, optimizer, scheduler, epoch):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "arcface_state_dict": arcface.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict()
    }

    torch.save(checkpoint, CHECKPOINT_PATH)


def load_checkpoint(model, arcface, optimizer, scheduler, device):
    if not CHECKPOINT_PATH.exists():
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    arcface.load_state_dict(checkpoint["arcface_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1

    print(f"Resumed from checkpoint at epoch {start_epoch}")

    return start_epoch


# ============================================================
# TRAIN LOOP
# ============================================================

def train_model(
    model,
    arcface,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs=EPOCHS,
    start_epoch=0
):
    best_acc = 0.0

    history = {
        "train_loss": [],
        "train_acc": []
    }

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print("-" * 40)

        train_loss, train_acc = train_one_epoch(
            model,
            arcface,
            train_loader,
            criterion,
            optimizer,
            device
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Acc : {train_acc:.4f}")

        if train_acc > best_acc:
            best_acc = train_acc

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "arcface_state_dict": arcface.state_dict()
                },
                BEST_MODEL_PATH
            )

            print("Saved best model.")

        save_checkpoint(
            model,
            arcface,
            optimizer,
            scheduler,
            epoch
        )

    return history


# ============================================================
# MAIN
# ============================================================

def main():
    train_loader, _, class_to_idx = create_dataloaders()

    num_classes = len(class_to_idx)

    model = FaceRecognitionResNet().to(device)

    arcface = ArcMarginProduct(
        in_features=512,
        out_features=num_classes,
        s=30.0,
        m=0.3
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(arcface.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    start_epoch = 0

    if CHECKPOINT_PATH.exists():
        choice = input(
            f"Checkpoint found at {CHECKPOINT_PATH}. Resume training? (y/n): "
        ).strip().lower()

        if choice == "y":
            start_epoch = load_checkpoint(
                model,
                arcface,
                optimizer,
                scheduler,
                device
            )

    train_model(
        model,
        arcface,
        train_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        epochs=EPOCHS,
        start_epoch=start_epoch
    )


if __name__ == "__main__":
    main()