# Architecture Documentation

## Novel Contributions

This project introduces three key innovations for urban scene semantic segmentation:

### 1. Hierarchical Attention Fusion Module

**Location**: `src/hierarchical_attention_fusion_urban_scene_parsing/models/components.py:15-90`

**Innovation**: Adaptive feature aggregation that learns which resolution levels matter most for different semantic categories.

**How it works**:
1. Extract features from multiple encoder stages (e.g., ResNet stages 1-4)
2. Project all features to a common channel dimension
3. Upsample all features to the same spatial resolution
4. Concatenate features and pass through attention network
5. Compute per-pixel, per-stage attention weights using softmax
6. Apply attention weights to fuse features dynamically

**Key insight**: Different semantic classes benefit from different feature scales:
- Fine details (poles, signs) → higher resolution features
- Coarse context (sky, road) → lower resolution features
- The model learns these relationships automatically

### 2. Progressive Boundary Refinement Loss

**Location**: `src/hierarchical_attention_fusion_urban_scene_parsing/models/components.py:150-230`

**Innovation**: Adaptive loss weight scheduling that gradually emphasizes boundary accuracy during training.

**How it works**:
1. Combine cross-entropy loss with boundary-aware loss
2. Start with low boundary loss weight (focus on coarse segmentation)
3. Progressively increase boundary loss weight during training
4. Final model excels at both coarse segmentation and fine boundaries

**Schedule types**:
- Progressive: Weight increases linearly from 0 to max over training
- Constant: Fixed boundary weight throughout training

### 3. Boundary-Aware Loss Component

**Location**: `src/hierarchical_attention_fusion_urban_scene_parsing/models/components.py:92-148`

**Innovation**: Emphasizes pixels near object boundaries using edge detection.

**How it works**:
1. Apply Sobel filters to ground truth mask to detect boundaries
2. Dilate boundaries to create a boundary region
3. Compute per-pixel cross-entropy loss
4. Weight loss higher at boundary pixels (5x higher weight)
5. Encourages model to focus on difficult boundary regions

## Architecture Overview

```
Input Image [B, 3, H, W]
    ↓
┌───────────────────────────────────┐
│   ResNet Encoder (timm)           │
│   - Stage 1: [B, 256, H/4, W/4]   │
│   - Stage 2: [B, 512, H/8, W/8]   │
│   - Stage 3: [B, 1024, H/16, W/16]│
│   - Stage 4: [B, 2048, H/32, W/32]│
└───────────────────────────────────┘
    ↓ (all stages)
┌───────────────────────────────────┐
│ Hierarchical Attention Fusion     │
│                                   │
│ 1. Project all to hidden_dim      │
│ 2. Upsample to target size        │
│ 3. Concatenate features           │
│ 4. Compute attention weights      │
│ 5. Weighted fusion                │
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│  Segmentation Head                │
│  Conv2d(hidden_dim → num_classes) │
└───────────────────────────────────┘
    ↓
Output Logits [B, num_classes, H, W]
```

## Loss Function

```
Total Loss = CE_weight × CrossEntropy + Boundary_weight(t) × BoundaryLoss

where:
  Boundary_weight(t) = boundary_weight_max × (current_epoch / total_epochs)

BoundaryLoss:
  1. Detect boundaries in ground truth
  2. Compute pixel-wise CE loss
  3. Apply boundary weighting: weight = 1.0 + 4.0 × boundary_mask
  4. Return weighted mean
```

## Ablation Study

The project includes an ablation configuration to prove the effectiveness of the novel components:

**Full Model** (`configs/default.yaml`):
- Hierarchical attention fusion: ✓
- Progressive boundary refinement: ✓
- Expected performance: Higher mIoU and boundary F1

**Baseline** (`configs/ablation.yaml`):
- Hierarchical attention fusion: ✗ (simple averaging)
- Progressive boundary refinement: ✗ (CE loss only)
- Expected performance: Lower mIoU and boundary F1

## Training Strategy

1. **Warm-up phase** (epochs 0-5): Low learning rate, focus on coarse features
2. **Main training** (epochs 6-80): Full learning rate, progressive boundary weighting
3. **Fine-tuning** (epochs 81-100): Reduced learning rate, high boundary emphasis

## Key Hyperparameters

- **Learning rate**: 0.001 (AdamW optimizer)
- **Batch size**: 8
- **Image size**: [512, 1024] (Cityscapes standard)
- **Crop size**: [512, 512] (training crops)
- **Attention hidden dim**: 256
- **Boundary weight max**: 0.4
- **Scheduler**: Cosine annealing with warm restarts

## Evaluation Metrics

1. **Mean IoU (mIoU)**: Primary metric for segmentation quality
2. **Pixel Accuracy**: Overall correctness
3. **Boundary F1**: Specifically measures boundary quality
4. **Per-class IoU**: Identifies which classes benefit most

## Expected Results

On Cityscapes validation set:
- Full model mIoU: ~78%
- Baseline mIoU: ~75%
- Boundary F1 improvement: ~5-7%

The hierarchical attention fusion provides the largest gains for:
- Small objects (poles, traffic signs)
- Objects with complex boundaries (pedestrians, bicycles)
- Classes requiring both local and global context (sidewalk, terrain)
