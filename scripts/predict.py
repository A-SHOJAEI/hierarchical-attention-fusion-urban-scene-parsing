#!/usr/bin/env python
"""Prediction script for hierarchical attention fusion segmentation."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root and src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

from hierarchical_attention_fusion_urban_scene_parsing.models.model import (
    HierarchicalAttentionSegmentationModel,
)
from hierarchical_attention_fusion_urban_scene_parsing.utils.config import load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run inference with hierarchical attention fusion model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image or directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions",
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization with input and prediction",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID",
    )
    return parser.parse_args()


def load_model(
    checkpoint_path: str,
    config: dict,
    device: torch.device,
) -> HierarchicalAttentionSegmentationModel:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        config: Model configuration.
        device: Device to load model on.

    Returns:
        Loaded model.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_config = config.get("model", {})
    model = HierarchicalAttentionSegmentationModel(
        backbone=model_config.get("backbone", "resnet50"),
        num_classes=model_config.get("num_classes", 19),
        pretrained=False,
        fusion_stages=model_config.get("fusion_stages", [1, 2, 3, 4]),
        attention_hidden_dim=model_config.get("attention_hidden_dim", 256),
        use_hierarchical_attention=model_config.get(
            "use_hierarchical_attention", True
        ),
        use_boundary_refinement=model_config.get(
            "use_boundary_refinement", True
        ),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded model from {checkpoint_path}")
    return model


def get_transform(image_size: list) -> A.Compose:
    """Get inference transforms.

    Args:
        image_size: Target image size [height, width].

    Returns:
        Albumentations transform.
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def predict_image(
    model: HierarchicalAttentionSegmentationModel,
    image_path: str,
    transform: A.Compose,
    device: torch.device,
) -> tuple:
    """Run prediction on a single image.

    Args:
        model: Segmentation model.
        image_path: Path to input image.
        transform: Image transforms.
        device: Device to run on.

    Returns:
        Tuple of (original_image, prediction_mask, confidence_scores).
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]

    # Apply transforms
    transformed = transform(image=image)
    input_tensor = transformed["image"].unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, prediction = probabilities.max(dim=1)

    # Resize back to original size
    prediction = prediction.squeeze(0).cpu().numpy()
    confidence = confidence.squeeze(0).cpu().numpy()

    prediction = cv2.resize(
        prediction.astype(np.uint8),
        (original_size[1], original_size[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    confidence = cv2.resize(
        confidence,
        (original_size[1], original_size[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    return image, prediction, confidence


def visualize_prediction(
    image: np.ndarray,
    prediction: np.ndarray,
    confidence: np.ndarray,
    output_path: str,
    num_classes: int = 19,
) -> None:
    """Visualize prediction results.

    Args:
        image: Original image.
        prediction: Predicted segmentation mask.
        confidence: Confidence scores.
        output_path: Path to save visualization.
        num_classes: Number of classes.
    """
    # Create color map
    cmap = plt.cm.get_cmap("tab20", num_classes)
    colors = cmap(np.linspace(0, 1, num_classes))[:, :3]
    colors = (colors * 255).astype(np.uint8)

    # Color prediction
    pred_color = colors[prediction]

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image)
    axes[0].set_title("Input Image", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(pred_color)
    axes[1].set_title("Segmentation Prediction", fontsize=14)
    axes[1].axis("off")

    im = axes[2].imshow(confidence, cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title("Confidence Map", fontsize=14)
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved visualization to {output_path}")


def main() -> None:
    """Main prediction function."""
    args = parse_args()

    # Load configuration
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    if args.config is not None:
        config = load_config(args.config)
    elif "config" in checkpoint:
        config = checkpoint["config"]
    else:
        raise ValueError("No configuration found. Provide --config.")

    # Setup device
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    # Load model
    logger.info("Loading model...")
    model = load_model(args.checkpoint, config, device)

    # Get transform
    image_size = config.get("data", {}).get("image_size", [512, 1024])
    transform = get_transform(image_size)

    # Setup output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if input is file or directory
    input_path = Path(args.input)

    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    else:
        raise ValueError(f"Invalid input path: {args.input}")

    logger.info(f"Found {len(image_files)} images to process")

    # Process each image
    num_classes = config.get("model", {}).get("num_classes", 19)

    for img_path in image_files:
        logger.info(f"Processing {img_path.name}...")

        try:
            # Run prediction
            image, prediction, confidence = predict_image(
                model, str(img_path), transform, device
            )

            # Save prediction mask
            pred_output = output_path / f"{img_path.stem}_pred.png"
            Image.fromarray(prediction.astype(np.uint8)).save(pred_output)

            # Save confidence map
            conf_output = output_path / f"{img_path.stem}_conf.npy"
            np.save(conf_output, confidence)

            # Generate visualization if requested
            if args.visualize:
                vis_output = output_path / f"{img_path.stem}_vis.png"
                visualize_prediction(
                    image, prediction, confidence, str(vis_output), num_classes
                )

            # Print statistics
            unique_classes = np.unique(prediction)
            mean_conf = confidence.mean()
            logger.info(f"  Predicted classes: {unique_classes.tolist()}")
            logger.info(f"  Mean confidence: {mean_conf:.4f}")
            logger.info(f"  Saved to {pred_output}")

        except Exception as e:
            logger.error(f"Failed to process {img_path.name}: {e}")
            continue

    logger.info(f"\nAll predictions saved to {output_path}")


if __name__ == "__main__":
    main()
