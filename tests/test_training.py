"""Tests for training functionality."""

import pytest
import torch
from pathlib import Path

from hierarchical_attention_fusion_urban_scene_parsing.training.trainer import (
    Trainer,
    EarlyStopping,
)
from hierarchical_attention_fusion_urban_scene_parsing.models.model import (
    HierarchicalAttentionSegmentationModel,
)
from hierarchical_attention_fusion_urban_scene_parsing.data.loader import (
    create_dataloaders,
)
from hierarchical_attention_fusion_urban_scene_parsing.evaluation.metrics import (
    MeanIoU,
    PixelAccuracy,
    BoundaryF1Score,
)


class TestEarlyStopping:
    """Test early stopping functionality."""

    def test_early_stopping_max_mode(self):
        """Test early stopping in maximize mode."""
        early_stop = EarlyStopping(patience=3, mode="max")

        # First epoch - best so far
        assert not early_stop.step(0.5)
        assert early_stop.counter == 0

        # Improvement
        assert not early_stop.step(0.6)
        assert early_stop.counter == 0

        # No improvement
        assert not early_stop.step(0.55)
        assert early_stop.counter == 1

        # Still no improvement
        assert not early_stop.step(0.58)
        assert early_stop.counter == 2

        # Still no improvement - should stop
        assert early_stop.step(0.59)
        assert early_stop.should_stop

    def test_early_stopping_min_mode(self):
        """Test early stopping in minimize mode."""
        early_stop = EarlyStopping(patience=2, mode="min")

        assert not early_stop.step(1.0)
        assert not early_stop.step(0.5)  # Improvement
        assert not early_stop.step(0.6)  # No improvement
        assert early_stop.step(0.7)  # Should stop


class TestMetrics:
    """Test evaluation metrics."""

    def test_mean_iou(self):
        """Test mIoU metric."""
        metric = MeanIoU(num_classes=5)

        # Perfect prediction
        predictions = torch.tensor([
            [0, 1, 2],
            [3, 4, 0],
        ])
        targets = predictions.clone()

        metric.update(predictions, targets)
        miou = metric.compute()
        assert miou == 1.0

    def test_mean_iou_with_logits(self):
        """Test mIoU with logit inputs."""
        metric = MeanIoU(num_classes=3)

        # Create logits that clearly predict class 0, 1, 2
        logits = torch.zeros(1, 3, 2, 2)
        logits[0, 0, 0, 0] = 10  # Class 0
        logits[0, 1, 0, 1] = 10  # Class 1
        logits[0, 2, 1, 0] = 10  # Class 2
        logits[0, 0, 1, 1] = 10  # Class 0

        targets = torch.tensor([[
            [0, 1],
            [2, 0],
        ]])

        metric.update(logits, targets)
        miou = metric.compute()
        assert miou == 1.0

    def test_pixel_accuracy(self):
        """Test pixel accuracy metric."""
        metric = PixelAccuracy()

        predictions = torch.tensor([
            [0, 1, 2],
            [1, 1, 2],
        ])
        targets = torch.tensor([
            [0, 1, 1],  # 2/3 correct in first row
            [1, 1, 2],  # 3/3 correct in second row
        ])

        metric.update(predictions, targets)
        accuracy = metric.compute()
        expected = 5.0 / 6.0  # 5 correct out of 6
        assert abs(accuracy - expected) < 0.01

    def test_boundary_f1(self):
        """Test boundary F1 metric."""
        metric = BoundaryF1Score()

        # Create predictions with boundaries
        predictions = torch.zeros(1, 8, 8)
        predictions[0, 3:5, 3:5] = 1  # Small square

        # Perfect match
        targets = predictions.clone()

        metric.update(predictions, targets)
        f1 = metric.compute()
        # F1 should be high for perfect match
        assert f1 > 0.5


class TestTrainer:
    """Test trainer functionality."""

    def test_trainer_creation(self, sample_config, device):
        """Test trainer initialization."""
        model = HierarchicalAttentionSegmentationModel(
            backbone="resnet50",
            num_classes=19,
            pretrained=False,
        )

        trainer = Trainer(model, sample_config, device)
        assert trainer is not None
        assert trainer.optimizer is not None
        assert trainer.criterion is not None

    def test_train_epoch(self, sample_config, device):
        """Test single training epoch."""
        # Use smaller model for faster testing
        sample_config["model"]["backbone"] = "resnet18"
        sample_config["model"]["pretrained"] = False
        sample_config["data"]["batch_size"] = 2
        sample_config["training"]["mixed_precision"] = False

        model = HierarchicalAttentionSegmentationModel(
            backbone="resnet18",
            num_classes=19,
            pretrained=False,
        )

        trainer = Trainer(model, sample_config, device)
        train_loader, _ = create_dataloaders(sample_config)

        # Train for one epoch (just first few batches)
        metrics = trainer.train_epoch(train_loader, epoch=1)

        assert "loss" in metrics
        assert "miou" in metrics
        assert "pixel_acc" in metrics
        assert metrics["loss"] >= 0

    def test_validate(self, sample_config, device):
        """Test validation."""
        sample_config["model"]["backbone"] = "resnet18"
        sample_config["model"]["pretrained"] = False
        sample_config["data"]["batch_size"] = 2

        model = HierarchicalAttentionSegmentationModel(
            backbone="resnet18",
            num_classes=19,
            pretrained=False,
        )

        trainer = Trainer(model, sample_config, device)
        _, val_loader = create_dataloaders(sample_config)

        metrics = trainer.validate(val_loader)

        assert "loss" in metrics
        assert "miou" in metrics
        assert "pixel_acc" in metrics
        assert "boundary_f1" in metrics

    def test_checkpoint_saving(self, sample_config, device, tmp_path):
        """Test model checkpoint saving."""
        sample_config["model"]["backbone"] = "resnet18"
        sample_config["model"]["pretrained"] = False
        sample_config["training"]["epochs"] = 1
        sample_config["data"]["batch_size"] = 2

        model = HierarchicalAttentionSegmentationModel(
            backbone="resnet18",
            num_classes=19,
            pretrained=False,
        )

        trainer = Trainer(model, sample_config, device)
        train_loader, val_loader = create_dataloaders(sample_config)

        save_dir = tmp_path / "checkpoints"
        history = trainer.train(train_loader, val_loader, str(save_dir))

        # Check that checkpoint was saved
        checkpoint_path = save_dir / "best_model.pth"
        assert checkpoint_path.exists()

        # Check that we can load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "best_miou" in checkpoint

    def test_optimizer_types(self, sample_config, device):
        """Test different optimizer types."""
        model = HierarchicalAttentionSegmentationModel(
            backbone="resnet18",
            num_classes=19,
            pretrained=False,
        )

        # Test AdamW
        sample_config["training"]["optimizer"] = "adamw"
        trainer = Trainer(model, sample_config, device)
        assert trainer.optimizer is not None

        # Test SGD
        sample_config["training"]["optimizer"] = "sgd"
        trainer = Trainer(model, sample_config, device)
        assert trainer.optimizer is not None

    def test_scheduler_types(self, sample_config, device):
        """Test different scheduler types."""
        model = HierarchicalAttentionSegmentationModel(
            backbone="resnet18",
            num_classes=19,
            pretrained=False,
        )

        # Test cosine
        sample_config["training"]["scheduler"] = "cosine"
        trainer = Trainer(model, sample_config, device)
        assert trainer.scheduler is not None

        # Test step
        sample_config["training"]["scheduler"] = "step"
        trainer = Trainer(model, sample_config, device)
        assert trainer.scheduler is not None

        # Test plateau
        sample_config["training"]["scheduler"] = "plateau"
        trainer = Trainer(model, sample_config, device)
        assert trainer.scheduler is not None
