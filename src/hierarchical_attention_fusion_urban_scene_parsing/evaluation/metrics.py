"""Evaluation metrics for semantic segmentation."""

import torch
import torch.nn.functional as F
from typing import Optional


class MeanIoU:
    """Mean Intersection over Union metric."""

    def __init__(self, num_classes: int, ignore_index: int = 255):
        """Initialize mIoU metric.

        Args:
            num_classes: Number of classes.
            ignore_index: Index to ignore in computation.
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        """Reset metric state."""
        self.intersection = torch.zeros(self.num_classes)
        self.union = torch.zeros(self.num_classes)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metric with new predictions.

        Args:
            predictions: Model predictions [B, C, H, W] or [B, H, W].
            targets: Ground truth [B, H, W].
        """
        if predictions.dim() == 4:
            predictions = predictions.argmax(dim=1)

        predictions = predictions.flatten()
        targets = targets.flatten()

        # Mask out ignore index
        mask = targets != self.ignore_index
        predictions = predictions[mask]
        targets = targets[mask]

        # Compute intersection and union for each class
        for cls in range(self.num_classes):
            pred_mask = predictions == cls
            target_mask = targets == cls

            intersection = (pred_mask & target_mask).sum().item()
            union = (pred_mask | target_mask).sum().item()

            self.intersection[cls] += intersection
            self.union[cls] += union

    def compute(self) -> float:
        """Compute mIoU metric.

        Returns:
            Mean IoU value.
        """
        # Avoid division by zero
        valid_classes = self.union > 0
        iou_per_class = self.intersection[valid_classes] / self.union[valid_classes]

        return iou_per_class.mean().item()

    def compute_per_class(self) -> torch.Tensor:
        """Compute per-class IoU.

        Returns:
            IoU for each class.
        """
        valid_classes = self.union > 0
        iou_per_class = torch.zeros(self.num_classes)
        iou_per_class[valid_classes] = (
            self.intersection[valid_classes] / self.union[valid_classes]
        )

        return iou_per_class


class PixelAccuracy:
    """Pixel-wise accuracy metric."""

    def __init__(self, ignore_index: int = 255):
        """Initialize pixel accuracy metric.

        Args:
            ignore_index: Index to ignore in computation.
        """
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        """Reset metric state."""
        self.correct = 0
        self.total = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metric with new predictions.

        Args:
            predictions: Model predictions [B, C, H, W] or [B, H, W].
            targets: Ground truth [B, H, W].
        """
        if predictions.dim() == 4:
            predictions = predictions.argmax(dim=1)

        predictions = predictions.flatten()
        targets = targets.flatten()

        # Mask out ignore index
        mask = targets != self.ignore_index
        predictions = predictions[mask]
        targets = targets[mask]

        self.correct += (predictions == targets).sum().item()
        self.total += targets.numel()

    def compute(self) -> float:
        """Compute pixel accuracy.

        Returns:
            Pixel accuracy value.
        """
        if self.total == 0:
            return 0.0
        return self.correct / self.total


class BoundaryF1Score:
    """Boundary F1 score metric.

    Evaluates segmentation quality at object boundaries.
    """

    def __init__(self, threshold: float = 0.01, ignore_index: int = 255):
        """Initialize boundary F1 metric.

        Args:
            threshold: Distance threshold for boundary matching.
            ignore_index: Index to ignore in computation.
        """
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        """Reset metric state."""
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0

    def _compute_boundaries(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute boundary pixels using Sobel filter.

        Args:
            mask: Segmentation mask [H, W].

        Returns:
            Boundary map [H, W].
        """
        mask_float = mask.unsqueeze(0).unsqueeze(0).float()

        # Sobel filters
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32, device=mask.device).unsqueeze(0).unsqueeze(0)

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32, device=mask.device).unsqueeze(0).unsqueeze(0)

        edge_x = F.conv2d(mask_float, sobel_x, padding=1)
        edge_y = F.conv2d(mask_float, sobel_y, padding=1)

        edges = torch.sqrt(edge_x ** 2 + edge_y ** 2).squeeze()
        boundaries = (edges > 0).float()

        return boundaries

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metric with new predictions.

        Args:
            predictions: Model predictions [B, C, H, W] or [B, H, W].
            targets: Ground truth [B, H, W].
        """
        if predictions.dim() == 4:
            predictions = predictions.argmax(dim=1)

        batch_size = predictions.shape[0]

        for i in range(batch_size):
            pred = predictions[i]
            target = targets[i]

            # Skip if all ignore index
            if (target == self.ignore_index).all():
                continue

            # Compute boundaries
            pred_boundaries = self._compute_boundaries(pred)
            target_boundaries = self._compute_boundaries(target)

            # Simple boundary matching (can be improved with distance transform)
            pred_boundary_pixels = pred_boundaries > 0
            target_boundary_pixels = target_boundaries > 0

            tp = (pred_boundary_pixels & target_boundary_pixels).sum().item()
            fp = (pred_boundary_pixels & ~target_boundary_pixels).sum().item()
            fn = (~pred_boundary_pixels & target_boundary_pixels).sum().item()

            self.true_positives += tp
            self.false_positives += fp
            self.false_negatives += fn

    def compute(self) -> float:
        """Compute boundary F1 score.

        Returns:
            Boundary F1 score value.
        """
        precision = (
            self.true_positives / (self.true_positives + self.false_positives)
            if (self.true_positives + self.false_positives) > 0
            else 0.0
        )
        recall = (
            self.true_positives / (self.true_positives + self.false_negatives)
            if (self.true_positives + self.false_negatives) > 0
            else 0.0
        )

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1
