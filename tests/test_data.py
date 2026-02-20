"""Tests for data loading and preprocessing."""

import pytest
import torch
import numpy as np

from hierarchical_attention_fusion_urban_scene_parsing.data.loader import (
    SyntheticUrbanDataset,
    get_dataset,
    create_dataloaders,
)
from hierarchical_attention_fusion_urban_scene_parsing.data.preprocessing import (
    get_train_transforms,
    get_val_transforms,
    denormalize_image,
)


class TestPreprocessing:
    """Test preprocessing functions."""

    def test_get_train_transforms(self):
        """Test training transforms creation."""
        transform = get_train_transforms(crop_size=[256, 256])
        assert transform is not None

        # Test transform on sample
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        mask = np.random.randint(0, 19, (512, 512), dtype=np.uint8)

        result = transform(image=image, mask=mask)
        assert "image" in result
        assert "mask" in result
        assert result["image"].shape[1:] == (256, 256)

    def test_get_val_transforms(self):
        """Test validation transforms creation."""
        transform = get_val_transforms(image_size=[256, 512])
        assert transform is not None

        # Test transform on sample
        image = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
        mask = np.random.randint(0, 19, (512, 1024), dtype=np.uint8)

        result = transform(image=image, mask=mask)
        assert result["image"].shape == (3, 256, 512)

    def test_denormalize_image(self):
        """Test image denormalization."""
        # Normalized image
        image = torch.randn(3, 256, 256)

        # Denormalize
        denorm = denormalize_image(image)

        assert denorm.shape == image.shape
        assert isinstance(denorm, torch.Tensor)


class TestSyntheticUrbanDataset:
    """Test synthetic urban dataset."""

    def test_dataset_creation(self):
        """Test dataset initialization."""
        dataset = SyntheticUrbanDataset(
            num_samples=10,
            image_size=(256, 512),
            num_classes=19,
        )
        assert len(dataset) == 10

    def test_dataset_getitem(self):
        """Test getting items from dataset."""
        transform = get_train_transforms(crop_size=[256, 256])
        dataset = SyntheticUrbanDataset(
            num_samples=5,
            image_size=(512, 1024),
            num_classes=19,
            transform=transform,
        )

        sample = dataset[0]
        assert "image" in sample
        assert "mask" in sample
        assert sample["image"].shape[0] == 3
        assert sample["mask"].ndim == 2

    def test_structured_mask_generation(self):
        """Test that masks have spatial structure."""
        dataset = SyntheticUrbanDataset(
            num_samples=5,
            image_size=(256, 512),
            num_classes=19,
        )

        mask = dataset._generate_structured_mask(0)
        assert mask.shape == (256, 512)
        assert mask.min() >= 0
        assert mask.max() < 19


class TestDataLoaders:
    """Test data loader functions."""

    def test_get_dataset(self, sample_config):
        """Test dataset retrieval."""
        dataset = get_dataset("cityscapes", "train", sample_config)
        assert dataset is not None
        assert len(dataset) > 0

    def test_create_dataloaders(self, sample_config):
        """Test dataloader creation."""
        train_loader, val_loader = create_dataloaders(sample_config)

        assert train_loader is not None
        assert val_loader is not None
        assert len(train_loader) > 0
        assert len(val_loader) > 0

        # Test batch from train loader
        batch = next(iter(train_loader))
        assert "image" in batch
        assert "mask" in batch
        assert batch["image"].shape[0] == sample_config["data"]["batch_size"]

    def test_dataloader_determinism(self, sample_config):
        """Test that dataset with same seed produces similar data."""
        from hierarchical_attention_fusion_urban_scene_parsing.utils.config import (
            set_random_seed,
        )

        # Use validation dataset which has no random augmentations
        set_random_seed(42)
        dataset1 = get_dataset("cityscapes", "val", sample_config)
        sample1 = dataset1[0]

        set_random_seed(42)
        dataset2 = get_dataset("cityscapes", "val", sample_config)
        sample2 = dataset2[0]

        # Validation dataset should be deterministic
        assert torch.allclose(sample1["image"], sample2["image"])
        assert torch.allclose(sample1["mask"].float(), sample2["mask"].float())
