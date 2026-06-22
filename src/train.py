import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(model, loader: DataLoader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_loader(model, loader: DataLoader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)
    return total_loss / total, correct / total


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer,
    scheduler=None,
    num_epochs: int = 20,
    device: str = "cpu",
    save_path: Path = None,
):
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = evaluate_loader(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch:02d}/{num_epochs}]  "
            f"train loss: {tr_loss:.4f}  train acc: {tr_acc:.4f}  "
            f"val loss: {vl_loss:.4f}  val acc: {vl_acc:.4f}  "
            f"({elapsed:.1f}s)"
        )

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            if save_path is not None:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"  -> Best model saved to {save_path}  (val_acc={best_val_acc:.4f})")

    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")
    return history
