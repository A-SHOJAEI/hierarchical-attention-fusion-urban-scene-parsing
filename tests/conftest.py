"""Pytest configuration and fixtures."""

import pytest
import torch
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "model": {
            "backbone": "resnet50",
            "num_classes": 19,
            "pretrained": False,
            "fusion_stages": [1, 2, 3, 4],
            "attention_hidden_dim": 256,
            "use_hierarchical_attention": True,
            "use_boundary_refinement": True,
        },
        "data": {
            "dataset": "cityscapes",
            "image_size": [512, 1024],
            "crop_size": [256, 256],
            "batch_size": 2,
            "num_workers": 0,
            "augmentation": {
                "horizontal_flip": 0.5,
                "scale_range": [0.5, 2.0],
                "rotation_limit": 10,
                "brightness_limit": 0.2,
                "contrast_limit": 0.2,
            },
        },
        "training": {
            "epochs": 5,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "warmup_epochs": 1,
            "min_lr": 0.000001,
            "gradient_clip_norm": 1.0,
            "mixed_precision": False,
            "accumulation_steps": 1,
        },
        "early_stopping": {
            "enabled": False,
            "patience": 3,
            "min_delta": 0.0001,
            "monitor": "val_miou",
            "mode": "max",
        },
        "loss": {
            "ce_weight": 1.0,
            "boundary_weight": 0.4,
            "boundary_schedule": "progressive",
        },
        "experiment": {
            "name": "test_experiment",
            "seed": 42,
            "save_dir": "test_checkpoints",
            "log_interval": 1,
        },
    }


@pytest.fixture
def sample_image():
    """Sample image tensor for testing."""
    return torch.randn(2, 3, 256, 256)


@pytest.fixture
def sample_mask():
    """Sample segmentation mask for testing."""
    return torch.randint(0, 19, (2, 256, 256))


@pytest.fixture
def sample_batch(sample_image, sample_mask):
    """Sample batch for testing."""
    return {
        "image": sample_image,
        "mask": sample_mask,
    }


@pytest.fixture
def device():
    """Device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def temp_config_file(tmp_path, sample_config):
    """Temporary config file for testing."""
    import yaml

    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(sample_config, f)
    return config_file
