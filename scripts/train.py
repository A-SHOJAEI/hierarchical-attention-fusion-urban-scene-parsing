#!/usr/bin/env python
"""Training script for hierarchical attention fusion segmentation."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root and src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from hierarchical_attention_fusion_urban_scene_parsing.data.loader import create_dataloaders
from hierarchical_attention_fusion_urban_scene_parsing.evaluation.analysis import (
    plot_training_curves,
    save_metrics_json,
)
from hierarchical_attention_fusion_urban_scene_parsing.models.model import (
    HierarchicalAttentionSegmentationModel,
)
from hierarchical_attention_fusion_urban_scene_parsing.training.trainer import Trainer
from hierarchical_attention_fusion_urban_scene_parsing.utils.config import (
    load_config,
    save_config,
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
        description="Train hierarchical attention fusion segmentation model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID",
    )
    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Set random seed
    seed = config.get("experiment", {}).get("seed", 42)
    set_random_seed(seed)
    logger.info(f"Set random seed to {seed}")

    # Setup device
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    logger.info(
        f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
    )

    # Create model
    logger.info("Creating model...")
    model_config = config.get("model", {})
    model = HierarchicalAttentionSegmentationModel(
        backbone=model_config.get("backbone", "resnet50"),
        num_classes=model_config.get("num_classes", 19),
        pretrained=model_config.get("pretrained", True),
        fusion_stages=model_config.get("fusion_stages", [1, 2, 3, 4]),
        attention_hidden_dim=model_config.get("attention_hidden_dim", 256),
        use_hierarchical_attention=model_config.get(
            "use_hierarchical_attention", True
        ),
        use_boundary_refinement=model_config.get(
            "use_boundary_refinement", True
        ),
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model parameters: {num_params:,} total, {num_trainable:,} trainable"
    )

    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume is not None:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        logger.info(f"Resumed from epoch {start_epoch - 1}")

    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(model, config, device)

    # Setup save directory
    exp_name = config.get("experiment", {}).get("name", "experiment")
    save_dir = Path(config.get("experiment", {}).get("save_dir", "checkpoints"))
    save_dir = save_dir / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    save_config(config, save_dir / "config.yaml")
    logger.info(f"Saved configuration to {save_dir / 'config.yaml'}")

    # Initialize MLflow (optional)
    try:
        import mlflow

        mlflow.set_experiment(exp_name)
        mlflow.start_run()
        mlflow.log_params({
            "learning_rate": config.get("training", {}).get("learning_rate"),
            "batch_size": config.get("data", {}).get("batch_size"),
            "optimizer": config.get("training", {}).get("optimizer"),
            "backbone": model_config.get("backbone"),
            "use_hierarchical_attention": model_config.get(
                "use_hierarchical_attention"
            ),
        })
        mlflow_enabled = True
        logger.info("MLflow tracking enabled")
    except Exception as e:
        logger.warning(f"MLflow not available: {e}")
        mlflow_enabled = False

    try:
        # Train model
        logger.info("Starting training...")
        history = trainer.train(train_loader, val_loader, str(save_dir))

        # Save training history
        results_dir = Path("results") / exp_name
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        final_metrics = {
            "best_val_miou": trainer.best_miou,
            "final_val_miou": history["val_miou"][-1],
            "final_val_pixel_acc": history["val_pixel_acc"][-1],
            "final_val_boundary_f1": history["val_boundary_f1"][-1],
        }
        save_metrics_json(final_metrics, results_dir / "final_metrics.json")

        # Plot training curves
        plot_training_curves(history, results_dir / "training_curves.png")

        # Log final metrics to MLflow
        if mlflow_enabled:
            mlflow.log_metrics(final_metrics)
            mlflow.log_artifact(str(results_dir / "training_curves.png"))
            mlflow.log_artifact(str(save_dir / "config.yaml"))

        logger.info(f"Training complete! Best mIoU: {trainer.best_miou:.4f}")
        logger.info(f"Results saved to {results_dir}")
        logger.info(f"Model checkpoint saved to {save_dir}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    finally:
        if mlflow_enabled:
            mlflow.end_run()


if __name__ == "__main__":
    main()
