"""Results analysis and visualization utilities."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def visualize_predictions(
    images: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    output_dir: str,
    num_samples: int = 10,
    class_colors: Optional[np.ndarray] = None,
) -> None:
    """Visualize segmentation predictions.

    Args:
        images: Input images [B, 3, H, W].
        predictions: Model predictions [B, C, H, W].
        targets: Ground truth masks [B, H, W].
        output_dir: Directory to save visualizations.
        num_samples: Number of samples to visualize.
        class_colors: Color map for classes [num_classes, 3].
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get predicted classes
    pred_classes = predictions.argmax(dim=1).cpu().numpy()
    targets_np = targets.cpu().numpy()
    images_np = images.cpu().numpy()

    # Create default color map if not provided
    if class_colors is None:
        num_classes = predictions.shape[1]
        class_colors = plt.cm.get_cmap("tab20")(np.linspace(0, 1, num_classes))[:, :3]
        class_colors = (class_colors * 255).astype(np.uint8)

    num_samples = min(num_samples, images.shape[0])

    for i in range(num_samples):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Denormalize image
        img = images_np[i].transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        # Convert predictions and targets to color
        pred_color = class_colors[pred_classes[i]]
        target_color = class_colors[targets_np[i]]

        # Plot
        axes[0].imshow(img)
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        axes[1].imshow(target_color)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(pred_color)
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(output_path / f"sample_{i:03d}.png", dpi=150, bbox_inches="tight")
        plt.close()

    logger.info(f"Saved {num_samples} visualization samples to {output_dir}")


def save_predictions(
    predictions: torch.Tensor,
    output_dir: str,
    prefix: str = "pred",
) -> None:
    """Save prediction masks as PNG files.

    Args:
        predictions: Model predictions [B, C, H, W].
        output_dir: Directory to save predictions.
        prefix: Filename prefix.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pred_classes = predictions.argmax(dim=1).cpu().numpy()

    for i, pred in enumerate(pred_classes):
        img = Image.fromarray(pred.astype(np.uint8))
        img.save(output_path / f"{prefix}_{i:05d}.png")

    logger.info(f"Saved {len(pred_classes)} prediction masks to {output_dir}")


def save_metrics_json(
    metrics: Dict[str, Any],
    output_path: str,
) -> None:
    """Save metrics to JSON file.

    Args:
        metrics: Dictionary of metric names and values.
        output_path: Path to save JSON file.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved metrics to {output_path}")


def plot_training_curves(
    history: Dict[str, List[float]],
    output_path: str,
) -> None:
    """Plot training curves.

    Args:
        history: Training history with loss and metric values.
        output_path: Path to save plot.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot losses
    if "train_loss" in history:
        axes[0].plot(history["train_loss"], label="Train Loss")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Plot metrics
    if "val_miou" in history:
        axes[1].plot(history["val_miou"], label="Val mIoU")
    if "val_pixel_acc" in history:
        axes[1].plot(history["val_pixel_acc"], label="Val Pixel Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric Value")
    axes[1].set_title("Validation Metrics")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved training curves to {output_path}")


def create_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> np.ndarray:
    """Create confusion matrix for segmentation.

    Args:
        predictions: Model predictions [B, C, H, W].
        targets: Ground truth [B, H, W].
        num_classes: Number of classes.
        ignore_index: Index to ignore.

    Returns:
        Confusion matrix [num_classes, num_classes].
    """
    pred_classes = predictions.argmax(dim=1).flatten()
    targets_flat = targets.flatten()

    # Mask out ignore index
    mask = targets_flat != ignore_index
    pred_classes = pred_classes[mask]
    targets_flat = targets_flat[mask]

    # Compute confusion matrix
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for t, p in zip(targets_flat, pred_classes):
        confusion[t.long(), p.long()] += 1

    return confusion.cpu().numpy()
