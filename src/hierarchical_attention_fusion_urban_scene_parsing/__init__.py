"""Hierarchical Attention Fusion for Urban Scene Parsing.

Multi-scale semantic segmentation with adaptive feature aggregation.
"""

__version__ = "0.1.0"
__author__ = "Alireza Shojaei"

from .models.model import HierarchicalAttentionSegmentationModel
from .training.trainer import Trainer

__all__ = [
    "HierarchicalAttentionSegmentationModel",
    "Trainer",
]
