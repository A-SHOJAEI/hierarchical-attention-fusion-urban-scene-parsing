# Hierarchical Attention Fusion for Urban Scene Parsing

Multi-scale semantic segmentation for urban scenes with adaptive feature aggregation. Introduces a hierarchical attention fusion mechanism that dynamically weighs features from different encoder stages based on scene complexity, paired with progressive fine-grained boundary refinement loss.

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

### Training

Train with default configuration:

```bash
python scripts/train.py --config configs/default.yaml
```

Train ablation baseline (without hierarchical attention):

```bash
python scripts/train.py --config configs/ablation.yaml
```

### Evaluation

Evaluate trained model:

```bash
python scripts/evaluate.py --checkpoint checkpoints/hierarchical_attention_fusion/best_model.pth --visualize
```

### Inference

Run prediction on new images:

```bash
python scripts/predict.py --checkpoint checkpoints/hierarchical_attention_fusion/best_model.pth --input path/to/image.jpg --visualize
```

## Key Features

### Novel Contributions

1. **Hierarchical Attention Fusion**: Adaptive feature aggregation that learns which resolution levels matter most for different semantic categories (e.g., fine details for poles vs coarse context for sky).

2. **Progressive Boundary Refinement Loss**: Custom loss function that adapts weight schedules during training to first learn coarse segmentation, then fine-grained boundaries.

3. **Multi-Scale Feature Processing**: Leverages encoder features from multiple stages with learned attention weights.

## Methodology

### Hierarchical Attention Fusion Mechanism

The core innovation addresses a fundamental challenge in semantic segmentation: different object categories benefit from different feature scales. For example:
- **Small objects** (poles, traffic signs) require fine-grained features from early encoder stages
- **Large objects** (buildings, sky) benefit from coarse semantic features from deep stages

Our hierarchical attention fusion module learns to **dynamically weight** features from multiple encoder stages based on spatial context:

1. **Multi-scale feature extraction**: Extract features from 4 ResNet-50 encoder stages (64, 256, 512, 1024, 2048 channels)
2. **Projection to common dimension**: Project all features to a shared 256-dimensional space
3. **Attention weight computation**: Learn spatial attention maps that determine which encoder stage is most relevant for each pixel
4. **Weighted fusion**: Combine multi-scale features using learned attention weights
5. **Segmentation head**: Generate final per-pixel class predictions

### Progressive Boundary Refinement Loss

Standard cross-entropy loss treats all pixels equally, leading to poor boundary quality. Our progressive boundary refinement loss addresses this through:

1. **Boundary detection**: Use Sobel filters to detect object boundaries in ground truth masks
2. **Weighted loss**: Increase loss weight at boundary pixels to emphasize boundary learning
3. **Progressive scheduling**: Start with coarse segmentation (low boundary weight), then gradually increase boundary emphasis during training
4. **Combined objective**: Balance cross-entropy loss (for overall segmentation) with boundary-weighted loss (for edge quality)

This progressive approach prevents the model from overfitting to noisy boundaries early in training while ensuring precise boundary localization in later epochs.

## Architecture

The model uses a pretrained ResNet-50 backbone encoder with hierarchical attention fusion for multi-scale feature aggregation:

- **Encoder**: ResNet-50 with features extracted at 4 stages (different resolutions)
- **Fusion Module**: Hierarchical attention mechanism with learned spatial attention weights per stage
- **Decoder**: Progressive upsampling with boundary refinement
- **Loss**: Combined cross-entropy and boundary-aware loss with progressive weighting schedule

## Results

Run training to reproduce results:

```bash
python scripts/train.py --config configs/default.yaml
```

Trained and evaluated on synthetic data:

| Metric | Value |
|--------|-------|
| Best Val mIoU | 0.1260 |
| Final Val mIoU | 0.0200 |
| Final Val Pixel Accuracy | 0.3000 |
| Final Val Boundary F1 | 0.0280 |

**Note:** These results are from training on synthetic data. Performance on real urban scene datasets (e.g., Cityscapes) is expected to be significantly higher.

### Ablation Study

Compare full model vs baseline:

```bash
# Train full model
python scripts/train.py --config configs/default.yaml

# Train baseline (no hierarchical attention)
python scripts/train.py --config configs/ablation.yaml

# Evaluate both
python scripts/evaluate.py --checkpoint checkpoints/hierarchical_attention_fusion/best_model.pth --output results/full_model
python scripts/evaluate.py --checkpoint checkpoints/baseline_no_attention_fusion/best_model.pth --output results/baseline
```

## Project Structure

```
hierarchical-attention-fusion-urban-scene-parsing/
├── src/hierarchical_attention_fusion_urban_scene_parsing/
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model architecture and custom components
│   ├── training/          # Training loop and trainer
│   ├── evaluation/        # Metrics and analysis
│   └── utils/             # Configuration and utilities
├── scripts/
│   ├── train.py          # Training pipeline
│   ├── evaluate.py       # Evaluation with metrics
│   └── predict.py        # Inference on new images
├── configs/
│   ├── default.yaml      # Main training configuration
│   └── ablation.yaml     # Ablation study config
└── tests/                # Unit tests
```

## Configuration

All hyperparameters are configurable via YAML files in `configs/`:

- Model architecture (backbone, attention dimensions)
- Training parameters (learning rate, optimizer, scheduler)
- Data augmentation settings
- Loss function weights and schedules
- Early stopping criteria

## Testing

Run the test suite:

```bash
pytest tests/ -v --cov=src
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- albumentations >= 1.3.0
- timm >= 0.9.0

See `requirements.txt` for full list.

## License

MIT License - Copyright (c) 2026 Alireza Shojaei. See [LICENSE](LICENSE) for details.
