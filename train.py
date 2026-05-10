from collections import Counter

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import get_loaders
from model import get_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 25
SAVE_PATH = "emotion_detector.pth"


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0

    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()

    sample_count = len(loader.dataset)
    return total_loss / sample_count, correct / sample_count


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0

    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()

    sample_count = len(loader.dataset)
    return total_loss / sample_count, correct / sample_count


def compute_class_weights(train_dataset):
    targets = train_dataset.targets
    counts = Counter(targets)
    total = len(targets)
    class_count = len(counts)
    weights = [total / (class_count * counts[index]) for index in range(class_count)]
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def plot_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"], label="Val")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"], label="Val")
    ax2.set_title("Accuracy")
    ax2.axhline(0.60, color="red", linestyle="--", label="Target 60%")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()


def run_training():
    train_loader, val_loader, _, classes = get_loaders(batch_size=BATCH_SIZE)
    model = get_model(num_classes=len(classes), freeze_backbone=True).to(DEVICE)

    class_weights = compute_class_weights(train_loader.dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    print("Phase 1: training head only...")
    optimizer = Adam(model.fc.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    for epoch in range(PHASE1_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Ep {epoch + 1}/{PHASE1_EPOCHS} | loss {train_loss:.4f} | val_acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  -> Saved best model ({val_acc:.4f})")

    print("\nPhase 2: fine-tuning all layers...")
    for param in model.parameters():
        param.requires_grad = True

    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    for epoch in range(PHASE2_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Ep {epoch + 1}/{PHASE2_EPOCHS} | loss {train_loss:.4f} | val_acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  -> Saved best model ({val_acc:.4f})")

    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")
    print(f"Model saved to {SAVE_PATH}")
    plot_curves(history)


if __name__ == "__main__":
    run_training()