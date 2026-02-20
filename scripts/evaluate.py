#!/usr/bin/env python
"""Evaluation script for hierarchical attention fusion segmentation."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root and src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from tqdm import tqdm

from hierarchical_attention_fusion_urban_scene_parsing.data.loader import create_dataloaders
from hierarchical_attention_fusion_urban_scene_parsing.evaluation.analysis import (
    create_confusion_matrix,
    save_metrics_json,
    visualize_predictions,
)
from hierarchical_attention_fusion_urban_scene_parsing.evaluation.metrics import (
    BoundaryF1Score,
    MeanIoU,
    PixelAccuracy,
)
from hierarchical_attention_fusion_urban_scene_parsing.models.model import (
    HierarchicalAttentionSegmentationModel,
)
from hierarchical_attention_fusion_urban_scene_parsing.utils.config import (
    load_config,
    set_random_seed,
)

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
        description="Evaluate hierarchical attention fusion segmentation model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (if not in checkpoint)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization samples",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID",
    )
    return parser.parse_args()


def load_model_from_checkpoint(
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

    # Create model
    model_config = config.get("model", {})
    model = HierarchicalAttentionSegmentationModel(
        backbone=model_config.get("backbone", "resnet50"),
        num_classes=model_config.get("num_classes", 19),
        pretrained=False,  # Don't load pretrained, use checkpoint
        fusion_stages=model_config.get("fusion_stages", [1, 2, 3, 4]),
        attention_hidden_dim=model_config.get("attention_hidden_dim", 256),
        use_hierarchical_attention=model_config.get(
            "use_hierarchical_attention", True
        ),
        use_boundary_refinement=model_config.get(
            "use_boundary_refinement", True
        ),
    )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"Checkpoint best mIoU: {checkpoint.get('best_miou', 'unknown')}")

    return model


@torch.no_grad()
def evaluate_model(
    model: HierarchicalAttentionSegmentationModel,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    visualize: bool = False,
    output_dir: str = "results/evaluation",
) -> dict:
    """Evaluate model on validation set.

    Args:
        model: Model to evaluate.
        val_loader: Validation data loader.
        device: Device to run on.
        num_classes: Number of classes.
        visualize: Whether to generate visualizations.
        output_dir: Output directory for results.

    Returns:
        Dictionary of evaluation metrics.
    """
    # Initialize metrics
    metrics = {
        "miou": MeanIoU(num_classes),
        "pixel_acc": PixelAccuracy(),
        "boundary_f1": BoundaryF1Score(),
    }

    # Storage for visualizations
    vis_images = []
    vis_preds = []
    vis_targets = []

    logger.info("Running evaluation...")
    for batch in tqdm(val_loader, desc="Evaluating"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        # Forward pass
        outputs = model(images)

        # Update metrics
        for metric in metrics.values():
            metric.update(outputs, masks)

        # Store samples for visualization
        if visualize and len(vis_images) < 10:
            vis_images.append(images.cpu())
            vis_preds.append(outputs.cpu())
            vis_targets.append(masks.cpu())

    # Compute final metrics
    results = {
        "miou": metrics["miou"].compute(),
        "pixel_accuracy": metrics["pixel_acc"].compute(),
        "boundary_f1": metrics["boundary_f1"].compute(),
    }

    # Per-class IoU
    per_class_iou = metrics["miou"].compute_per_class()
    results["per_class_iou"] = per_class_iou.tolist()

    # Generate visualizations
    if visualize:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        all_images = torch.cat(vis_images, dim=0)
        all_preds = torch.cat(vis_preds, dim=0)
        all_targets = torch.cat(vis_targets, dim=0)

        visualize_predictions(
            all_images,
            all_preds,
            all_targets,
            str(output_path / "visualizations"),
            num_samples=10,
        )
        logger.info(f"Saved visualizations to {output_path / 'visualizations'}")

    return results


def main() -> None:
    """Main evaluation function."""
    args = parse_args()

    # Load configuration
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    if args.config is not None:
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
    elif "config" in checkpoint:
        logger.info("Loading configuration from checkpoint")
        config = checkpoint["config"]
    else:
        raise ValueError(
            "No configuration found. Provide --config or use checkpoint with config."
        )

    # Set random seed
    seed = config.get("experiment", {}).get("seed", 42)
    set_random_seed(seed)

    # Setup device
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    # Create dataloaders (validation only)
    logger.info("Creating dataloaders...")
    _, val_loader = create_dataloaders(config)
    logger.info(f"Validation batches: {len(val_loader)}")

    # Load model
    logger.info("Loading model...")
    model = load_model_from_checkpoint(args.checkpoint, config, device)

    # Evaluate
    num_classes = config.get("model", {}).get("num_classes", 19)
    results = evaluate_model(
        model,
        val_loader,
        device,
        num_classes,
        visualize=args.visualize,
        output_dir=args.output,
    )

    # Print results
    logger.info("\n" + "=" * 50)
    logger.info("Evaluation Results")
    logger.info("=" * 50)
    logger.info(f"Mean IoU:          {results['miou']:.4f}")
    logger.info(f"Pixel Accuracy:    {results['pixel_accuracy']:.4f}")
    logger.info(f"Boundary F1:       {results['boundary_f1']:.4f}")
    logger.info("=" * 50)

    # Print per-class results
    logger.info("\nPer-Class IoU:")
    for cls_idx, iou in enumerate(results["per_class_iou"]):
        logger.info(f"  Class {cls_idx:2d}: {iou:.4f}")

    # Save results
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    save_metrics_json(results, output_path / "evaluation_results.json")
    logger.info(f"\nResults saved to {output_path / 'evaluation_results.json'}")

    # Also save as CSV for easy viewing
    import pandas as pd

    summary_df = pd.DataFrame([{
        "Mean_IoU": results["miou"],
        "Pixel_Accuracy": results["pixel_accuracy"],
        "Boundary_F1": results["boundary_f1"],
    }])
    summary_df.to_csv(output_path / "summary.csv", index=False)

    per_class_df = pd.DataFrame({
        "Class": range(len(results["per_class_iou"])),
        "IoU": results["per_class_iou"],
    })
    per_class_df.to_csv(output_path / "per_class_iou.csv", index=False)

    logger.info(f"CSV results saved to {output_path}")


if __name__ == "__main__":
    main()
