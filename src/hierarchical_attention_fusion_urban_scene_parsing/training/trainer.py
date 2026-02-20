"""Training loop with learning rate scheduling and early stopping."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..evaluation.metrics import BoundaryF1Score, MeanIoU, PixelAccuracy
from ..models.components import ProgressiveBoundaryLoss

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "max",
    ):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as improvement.
            mode: 'max' to maximize metric, 'min' to minimize.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False

    def step(self, metric_value: float) -> bool:
        """Update early stopping state.

        Args:
            metric_value: Current validation metric value.

        Returns:
            True if training should stop.
        """
        if self.best_value is None:
            self.best_value = metric_value
            return False

        if self.mode == "max":
            improved = metric_value > (self.best_value + self.min_delta)
        else:
            improved = metric_value < (self.best_value - self.min_delta)

        if improved:
            self.best_value = metric_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class Trainer:
    """Trainer for semantic segmentation models."""

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
    ):
        """Initialize trainer.

        Args:
            model: Segmentation model.
            config: Training configuration.
            device: Device to train on.
        """
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Get configurations
        train_config = config.get("training", {})
        loss_config = config.get("loss", {})
        es_config = config.get("early_stopping", {})

        # Setup optimizer
        optimizer_name = train_config.get("optimizer", "adamw").lower()
        lr = train_config.get("learning_rate", 0.001)
        weight_decay = train_config.get("weight_decay", 0.0001)

        if optimizer_name == "adamw":
            self.optimizer = AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "sgd":
            self.optimizer = SGD(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Setup learning rate scheduler
        scheduler_name = train_config.get("scheduler", "cosine").lower()
        epochs = train_config.get("epochs", 100)

        if scheduler_name == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=train_config.get("min_lr", 1e-6),
            )
        elif scheduler_name == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1,
            )
        elif scheduler_name == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=5,
            )
        else:
            self.scheduler = None

        # Setup loss function
        use_boundary = config.get("model", {}).get("use_boundary_refinement", True)

        if use_boundary:
            self.criterion = ProgressiveBoundaryLoss(
                ce_weight=loss_config.get("ce_weight", 1.0),
                boundary_weight_max=loss_config.get("boundary_weight", 0.4),
                schedule=loss_config.get("boundary_schedule", "progressive"),
            ).to(device)
        else:
            # Simple cross-entropy loss
            class SimpleLoss(nn.Module):
                def forward(self, pred, target):
                    loss = nn.functional.cross_entropy(
                        pred, target, ignore_index=255
                    )
                    return {"total": loss, "ce": loss, "boundary": torch.tensor(0.0)}

            self.criterion = SimpleLoss().to(device)

        # Setup early stopping
        if es_config.get("enabled", True):
            self.early_stopping = EarlyStopping(
                patience=es_config.get("patience", 15),
                min_delta=es_config.get("min_delta", 0.0001),
                mode=es_config.get("mode", "max"),
            )
        else:
            self.early_stopping = None

        # Training settings
        self.epochs = epochs
        self.gradient_clip_norm = train_config.get("gradient_clip_norm", 1.0)
        self.mixed_precision = train_config.get("mixed_precision", True)
        self.accumulation_steps = train_config.get("accumulation_steps", 1)
        self.log_interval = config.get("experiment", {}).get("log_interval", 10)

        # Setup mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None

        # Metrics
        num_classes = config.get("model", {}).get("num_classes", 19)
        self.train_metrics = {
            "miou": MeanIoU(num_classes),
            "pixel_acc": PixelAccuracy(),
        }
        self.val_metrics = {
            "miou": MeanIoU(num_classes),
            "pixel_acc": PixelAccuracy(),
            "boundary_f1": BoundaryF1Score(),
        }

        # History
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_miou": [],
            "val_pixel_acc": [],
            "val_boundary_f1": [],
        }

        self.best_miou = 0.0

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader.
            epoch: Current epoch number.

        Returns:
            Dictionary of training metrics.
        """
        self.model.train()

        # Update loss criterion epoch
        if hasattr(self.criterion, "set_epoch"):
            self.criterion.set_epoch(epoch, self.epochs)

        # Reset metrics
        for metric in self.train_metrics.values():
            metric.reset()

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs}")

        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            # Forward pass with mixed precision
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss_dict = self.criterion(outputs, masks)
                    loss = loss_dict["total"] / self.accumulation_steps

                # Backward pass
                self.scaler.scale(loss).backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    # Gradient clipping
                    if self.gradient_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clip_norm,
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, masks)
                loss = loss_dict["total"] / self.accumulation_steps

                loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    if self.gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clip_norm,
                        )

                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Update metrics
            with torch.no_grad():
                for metric in self.train_metrics.values():
                    metric.update(outputs, masks)

            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1

            # Update progress bar
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item() * self.accumulation_steps:.4f}",
                    "avg_loss": f"{total_loss / num_batches:.4f}",
                })

        # Compute metrics
        metrics = {
            "loss": total_loss / num_batches,
            "miou": self.train_metrics["miou"].compute(),
            "pixel_acc": self.train_metrics["pixel_acc"].compute(),
        }

        return metrics

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model.

        Args:
            val_loader: Validation data loader.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()

        # Reset metrics
        for metric in self.val_metrics.values():
            metric.reset()

        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(val_loader, desc="Validation"):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss_dict = self.criterion(outputs, masks)
            loss = loss_dict["total"]

            # Update metrics
            for metric in self.val_metrics.values():
                metric.update(outputs, masks)

            total_loss += loss.item()
            num_batches += 1

        # Compute metrics
        metrics = {
            "loss": total_loss / num_batches,
            "miou": self.val_metrics["miou"].compute(),
            "pixel_acc": self.val_metrics["pixel_acc"].compute(),
            "boundary_f1": self.val_metrics["boundary_f1"].compute(),
        }

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: str,
    ) -> Dict[str, List[float]]:
        """Full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            save_dir: Directory to save checkpoints.

        Returns:
            Training history.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        logger.info("Starting training...")

        for epoch in range(1, self.epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.validate(val_loader)

            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["miou"])
                else:
                    self.scheduler.step()

            # Log metrics
            logger.info(
                f"Epoch {epoch}/{self.epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val mIoU: {val_metrics['miou']:.4f}, "
                f"Val Pixel Acc: {val_metrics['pixel_acc']:.4f}, "
                f"Val Boundary F1: {val_metrics['boundary_f1']:.4f}"
            )

            # Save history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_miou"].append(val_metrics["miou"])
            self.history["val_pixel_acc"].append(val_metrics["pixel_acc"])
            self.history["val_boundary_f1"].append(val_metrics["boundary_f1"])

            # Save best model
            if val_metrics["miou"] > self.best_miou:
                self.best_miou = val_metrics["miou"]
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_miou": self.best_miou,
                    "config": self.config,
                }
                torch.save(checkpoint, save_path / "best_model.pth")
                logger.info(f"Saved best model with mIoU: {self.best_miou:.4f}")

            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping.step(val_metrics["miou"]):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        logger.info("Training completed!")
        return self.history
