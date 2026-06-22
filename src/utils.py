import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_training_curves(history: dict, title: str = "Training Curves", save_path: Path = None):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} - Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"],   label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{title} - Accuracy")
    axes[1].legend()

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Training curves saved to {save_path}")
    plt.show()


def save_metrics_to_csv(model_name: str, metrics: dict, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"model": model_name, **metrics}
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = df[df["model"] != model_name]  # overwrite existing row for this model
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}")
