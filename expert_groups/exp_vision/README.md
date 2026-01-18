# Vision Expert for Qwen3-VL-30B-A3B-Thinking

## Overview

The Vision Expert specializes in comprehensive computer vision tasks using the Qwen3-VL-30B-A3B-Thinking vision-language model. This expert group allocates 31 experts (indices 96-126) from the 128-expert pool for vision-specific tasks.

## Capabilities

### 1. Object Detection & Instance Segmentation
- Multi-object detection in images
- Bounding box prediction with confidence scores
- Instance-level object segmentation
- **Datasets**: COCO, Open Images
- **Metrics**: mAP, IoU, precision/recall

### 2. Semantic Segmentation
- Pixel-level scene understanding
- Semantic region identification
- Scene parsing and spatial reasoning
- **Datasets**: ADE20K, Cityscapes
- **Metrics**: mIoU, pixel accuracy

### 3. OCR & Document Understanding
- Scene text recognition
- Document question answering
- Table/form extraction
- Multi-language text reading
- **Datasets**: TextVQA, DocVQA, IIIT5K
- **Metrics**: Character/word accuracy, edit distance

### 4. Industrial Defect Detection
- Quality control inspection
- Anomaly detection in manufacturing
- Defect localization and classification
- Surface inspection
- **Datasets**: MVTec AD, DAGM
- **Metrics**: AUROC, F1 score, localization IoU

### 5. 3D Perception & Depth Estimation
- Monocular depth estimation
- 3D scene reconstruction
- Spatial relationship understanding
- Indoor/outdoor scene geometry
- **Datasets**: NYU Depth V2, ScanNet, KITTI
- **Metrics**: Depth RMSE, Chamfer distance

### 6. Video Analysis & Temporal Reasoning
- Action recognition
- Temporal activity detection
- Event localization in video
- Multi-frame reasoning
- **Datasets**: Kinetics-400, ActivityNet, Moments in Time
- **Metrics**: Action accuracy, temporal IoU

## Architecture

### Expert Pool Allocation

```yaml
expert_pools:
  vision_detection: [96, 111]      # 16 experts (12.5%)
  vision_segmentation: [112, 119]  # 8 experts (6.25%)
  vision_ocr: [120, 123]           # 4 experts (3.1%)
  vision_defect: [124, 125]        # 2 experts (1.6%)
  vision_3d: [126, 126]            # 1 expert (0.8%)
```

**Note**: Expert 127 is used as the designated shared expert (shared with robotics expert) to maintain general capabilities.

### Model Configuration

- **Base Model**: Qwen3-VL-30B-A3B-Thinking
- **Total Experts**: 128
- **Experts Per Token**: 8 (top-8 routing)
- **Vision Encoder**: SigLIP with DeepStack fusion
- **Image Size**: 384x384
- **Max Images per Sample**: 8

### Shared Expert Strategy

Uses "designated" mode with expert 127:
- No additional parameters (uses existing expert)
- Strong logit bias (10.0) ensures expert is always selected
- Shares knowledge with robotics expert for cross-domain capabilities

### Routing Strategy

- **Mode**: Soft bias (not hard masking)
- **Pool Bias Strength**: 5.0 (moderate)
- **Bias Decay**: 0.98 (gradual relaxation)
- **Load Balancing**: Computed only over allowed pool

## Training Pipeline

### Stage A: Router + LoRA Warmstart (10k steps, ~2 weeks)
- **Learning Rate**: 1e-4
- **Router LR**: 5e-4 (higher for faster routing adaptation)
- **LoRA**: rank=16, alpha=32
- **Frozen**: Vision encoder, backbone attention
- **Trainable**: Router weights, LoRA adapters

### Stage B: Pool-Specific Fine-Tuning (30k steps, ~3 weeks)
- **Learning Rate**: 3e-5
- **Frozen**: Non-vision experts (0-95)
- **Trainable**: Vision expert pool (96-126), vision encoder
- **Focus**: Specialize vision experts on task-specific data

### Stage C: Safety & Alignment (5k steps, ~1 week)
- **Learning Rate**: 1e-5
- **SFT**: 2k iterations for instruction following
- **DPO**: 1k iterations for preference alignment
- **Focus**: Safe, helpful vision responses

## Dataset Sources

### Detection & Segmentation (Weight: 6.5)
- COCO Detection (2.0)
- Open Images (1.5)
- ADE20K Segmentation (1.5)
- Cityscapes (1.0)

### OCR & Documents (Weight: 3.3)
- TextVQA (1.5)
- DocVQA (1.0)
- IIIT5K Scene Text (0.8)

### Defect Detection (Weight: 2.5)
- MVTec AD (1.5)
- DAGM (1.0)

### 3D Perception (Weight: 3.3)
- NYU Depth V2 (1.5)
- ScanNet (1.0)
- KITTI (0.8)

### Video Analysis (Weight: 3.3)
- Kinetics-400 (1.5)
- ActivityNet (1.0)
- Moments in Time (0.8)

### General VL (Weight: 0.1)
- LLaVA General (0.1) - prevent catastrophic forgetting

## Evaluation Metrics

### Object Detection
- `object_detection_map`: Mean Average Precision
- `instance_segmentation_iou`: Instance-level IoU

### Semantic Segmentation
- `semantic_segmentation_miou`: Mean Intersection over Union
- Pixel accuracy

### OCR
- `ocr_character_accuracy`: Character-level accuracy
- `ocr_word_accuracy`: Word-level accuracy

### Defect Detection
- `defect_detection_f1`: F1 score for defect presence
- `defect_localization_iou`: Localization accuracy

### 3D Perception
- `depth_estimation_rmse`: Depth prediction error
- `3d_reconstruction_chamfer`: 3D reconstruction quality

### Video Analysis
- `video_action_accuracy`: Action classification accuracy
- `video_temporal_iou`: Temporal localization accuracy

## Usage Examples

### Object Detection

```python
User: "What objects are in this image?"
Vision Expert: "I can see: person (confidence: 0.95), car (confidence: 0.89),
dog (confidence: 0.92), traffic light (confidence: 0.87)"
```

### Semantic Segmentation

```python
User: "Segment this scene into semantic regions."
Vision Expert: "Semantic segmentation:
- sky: pixels [0-150, 0-800]
- building: pixels [150-400, 200-600]
- road: pixels [400-600, 0-800]
- trees: pixels [150-350, 0-200, 600-800]"
```

### OCR

```python
User: "Read the text in this image."
Vision Expert: "The text says: 'STOP'"
```

### Defect Detection

```python
User: "Are there any defects in this product?"
Vision Expert: "Yes, I detected a scratch defect at region [x1=120, y1=80, x2=180, y2=95]
with 92% confidence. Defect type: surface scratch."
```

### 3D Perception

```python
User: "Estimate the depth map for this scene."
Vision Expert: "Depth estimation:
- Near objects (0.5-1.0m): table, chair
- Medium distance (1.5-3.0m): wall, door
- Far objects (3.5-5.0m): window, ceiling
Average scene depth: 2.3 meters"
```

### Video Analysis

```python
User: "What action is happening in this video?"
Vision Expert: "The action being performed is: running. Confidence: 94%.
The person is running on a track."
```

## Compute Requirements

- **GPUs**: 16x H100 or A100 80GB
- **Training Time**: ~6 weeks total
  - Stage A: 2 weeks
  - Stage B: 3 weeks
  - Stage C: 1 week
- **Cost Estimate**: $20k-30k (H100 pricing)

### Memory Optimizations
- Gradient checkpointing: enabled
- Flash Attention 2: enabled
- Mixed precision: bfloat16
- ZeRO Stage 3: for large model sharding

## Implementation Notes

### Preventing Catastrophic Forgetting

1. **Designated Shared Expert**: Expert 127 maintains general capabilities
2. **General VL Data**: 10% of training data is general vision-language tasks
3. **Frozen Backbone**: Attention layers remain frozen during pool training
4. **LoRA Adapters**: Low-rank adaptation prevents full weight drift

### Load Balancing

Critical: Load-balance loss must exclude designated expert (127) and compute only over the allowed vision pool (96-126).

```python
# Exclude designated expert from load balance
probs = router_probs.clone()
probs[..., designated_expert_id] = 0.0

# Normalize over remaining experts
den = probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
probs = probs / den

# Compute load balance loss
aux_loss = compute_load_balance_loss(probs)
```

### Soft Bias Routing

Prefer soft bias over hard masking for robustness:

```python
# Soft bias (recommended)
pool_mask = torch.zeros(num_experts)
pool_mask[pool_indices] = pool_bias_strength  # e.g., 5.0
router_logits = router_logits + pool_mask

# NOT hard masking (too brittle)
# pool_mask[excluded_indices] = -inf
```

## Cross-Domain Synergy

The Vision Expert shares expert 127 with the Robotics Expert, enabling:
- Visual reasoning for robotics tasks
- 3D perception for manipulation
- Object detection for grasping
- Scene understanding for navigation

This shared expert facilitates knowledge transfer between vision and robotics domains.

## Files

- `config.yaml`: Expert configuration and hyperparameters
- `dataset.py`: Dataset loading and preprocessing
- `metrics.py`: Vision-specific evaluation metrics
- `__init__.py`: Module initialization
- `README.md`: This file

## Next Steps

1. Verify Qwen3-VL model access via HuggingFace
2. Download and prepare vision datasets
3. Run Stage A training (router + LoRA warmstart)
4. Evaluate on vision benchmarks
5. Run Stage B training (pool-specific fine-tuning)
6. Run Stage C training (safety alignment)
7. Deploy and test on real vision tasks

## References

- **Model**: [Qwen3-VL-30B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Thinking)
- **Datasets**: COCO, ADE20K, TextVQA, MVTec AD, NYU Depth V2, Kinetics-400
- **Architecture**: Shared Expert MoE with Designated Expert Strategy