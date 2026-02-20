"""Tests for model architecture and components."""

import pytest
import torch
import torch.nn as nn

from hierarchical_attention_fusion_urban_scene_parsing.models.model import (
    HierarchicalAttentionSegmentationModel,
)
from hierarchical_attention_fusion_urban_scene_parsing.models.components import (
    HierarchicalAttentionFusion,
    BoundaryRefinementLoss,
    ProgressiveBoundaryLoss,
)


class TestHierarchicalAttentionFusion:
    """Test hierarchical attention fusion module."""

    def test_module_creation(self):
        """Test module initialization."""
        module = HierarchicalAttentionFusion(
            in_channels_list=[256, 512, 1024, 2048],
            hidden_dim=256,
            num_classes=19,
        )
        assert module is not None

    def test_forward_pass(self):
        """Test forward pass."""
        module = HierarchicalAttentionFusion(
            in_channels_list=[256, 512, 1024, 2048],
            hidden_dim=256,
            num_classes=19,
        )

        # Create dummy features
        features = [
            torch.randn(2, 256, 64, 128),
            torch.randn(2, 512, 32, 64),
            torch.randn(2, 1024, 16, 32),
            torch.randn(2, 2048, 8, 16),
        ]

        output = module(features, target_size=(64, 128))
        assert output.shape == (2, 19, 64, 128)


class TestBoundaryRefinementLoss:
    """Test boundary refinement loss."""

    def test_loss_creation(self):
        """Test loss initialization."""
        loss = BoundaryRefinementLoss(kernel_size=5)
        assert loss is not None

    def test_boundary_computation(self):
        """Test boundary detection."""
        loss = BoundaryRefinementLoss()

        # Create simple mask with clear boundaries
        mask = torch.zeros(2, 32, 32)
        mask[:, 10:20, 10:20] = 1

        boundaries = loss.compute_boundaries(mask)
        assert boundaries.shape == mask.shape
        assert boundaries.max() <= 1.0
        assert boundaries.min() >= 0.0

    def test_loss_forward(self):
        """Test loss computation."""
        loss = BoundaryRefinementLoss()

        predictions = torch.randn(2, 19, 32, 32)
        targets = torch.randint(0, 19, (2, 32, 32))

        loss_value = loss(predictions, targets)
        assert isinstance(loss_value, torch.Tensor)
        assert loss_value.ndim == 0  # Scalar
        assert loss_value.item() >= 0


class TestProgressiveBoundaryLoss:
    """Test progressive boundary loss."""

    def test_loss_creation(self):
        """Test loss initialization."""
        loss = ProgressiveBoundaryLoss(
            ce_weight=1.0,
            boundary_weight_max=0.4,
            schedule="progressive",
        )
        assert loss is not None

    def test_weight_scheduling(self):
        """Test progressive weight scheduling."""
        loss = ProgressiveBoundaryLoss(
            boundary_weight_max=1.0,
            schedule="progressive",
        )

        # At epoch 0, weight should be 0
        loss.set_epoch(0, 100)
        assert loss.get_boundary_weight() == 0.0

        # At epoch 50, weight should be 0.5
        loss.set_epoch(50, 100)
        assert abs(loss.get_boundary_weight() - 0.5) < 0.01

        # At epoch 100, weight should be 1.0
        loss.set_epoch(100, 100)
        assert abs(loss.get_boundary_weight() - 1.0) < 0.01

    def test_constant_schedule(self):
        """Test constant weight schedule."""
        loss = ProgressiveBoundaryLoss(
            boundary_weight_max=0.4,
            schedule="constant",
        )

        loss.set_epoch(0, 100)
        assert loss.get_boundary_weight() == 0.4

        loss.set_epoch(50, 100)
        assert loss.get_boundary_weight() == 0.4

    def test_loss_forward(self):
        """Test loss computation."""
        loss = ProgressiveBoundaryLoss()
        loss.set_epoch(10, 100)

        predictions = torch.randn(2, 19, 32, 32)
        targets = torch.randint(0, 19, (2, 32, 32))

        loss_dict = loss(predictions, targets)
        assert "total" in loss_dict
        assert "ce" in loss_dict
        assert "boundary" in loss_dict
        assert all(isinstance(v, torch.Tensor) for v in loss_dict.values())


class TestHierarchicalAttentionSegmentationModel:
    """Test main segmentation model."""

    def test_model_creation_with_attention(self):
        """Test model with hierarchical attention."""
        model = HierarchicalAttentionSegmentationModel(
            backbone="resnet50",
            num_classes=19,
            pretrained=False,
            use_hierarchical_attention=True,
        )
        assert model is not None

    def test_model_creation_without_attention(self):
        """Test baseline model without attention."""
        model = HierarchicalAttentionSegmentationModel(
            backbone="resnet50",
            num_classes=19,
            pretrained=False,
            use_hierarchical_attention=False,
        )
        assert model is not None

    def test_forward_pass(self):
        """Test model forward pass."""
        model = HierarchicalAttentionSegmentationModel(
            backbone="resnet50",
            num_classes=19,
            pretrained=False,
        )
        model.eval()

        input_tensor = torch.randn(2, 3, 256, 512)
        with torch.no_grad():
            output = model(input_tensor)

        assert output.shape == (2, 19, 256, 512)

    def test_model_output_range(self):
        """Test that model outputs reasonable values."""
        model = HierarchicalAttentionSegmentationModel(
            backbone="resnet50",
            num_classes=19,
            pretrained=False,
        )
        model.eval()

        input_tensor = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            output = model(input_tensor)

        # Check that logits are not NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_get_config(self):
        """Test model configuration retrieval."""
        model = HierarchicalAttentionSegmentationModel(
            backbone="resnet50",
            num_classes=19,
            pretrained=False,
            use_hierarchical_attention=True,
        )

        config = model.get_config()
        assert "num_classes" in config
        assert config["num_classes"] == 19
        assert "use_hierarchical_attention" in config
        assert config["use_hierarchical_attention"] is True

    def test_different_backbones(self):
        """Test model with different backbones."""
        backbones = ["resnet18", "resnet34", "resnet50"]

        for backbone in backbones:
            model = HierarchicalAttentionSegmentationModel(
                backbone=backbone,
                num_classes=19,
                pretrained=False,
            )
            assert model is not None

            input_tensor = torch.randn(1, 3, 256, 256)
            with torch.no_grad():
                output = model(input_tensor)
            assert output.shape == (1, 19, 256, 256)

    def test_model_trainable(self):
        """Test that model parameters are trainable."""
        model = HierarchicalAttentionSegmentationModel(
            backbone="resnet50",
            num_classes=19,
            pretrained=False,
        )

        # Check that at least some parameters require gradients
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        assert trainable_params > 0

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = HierarchicalAttentionSegmentationModel(
            backbone="resnet50",
            num_classes=19,
            pretrained=False,
        )

        input_tensor = torch.randn(1, 3, 128, 128)
        target = torch.randint(0, 19, (1, 128, 128))

        output = model(input_tensor)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()

        # Check that gradients exist
        has_gradients = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_gradients
