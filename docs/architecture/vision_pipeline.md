# Vision Pipeline Architecture

## Overview

The Vision expert group extends the text-only MoE system to support **multimodal inputs** (images + text), enabling tasks like visual question answering (VQA), image captioning, and visual reasoning.

## Architecture

### Model Structure

```
Qwen3-VL-MoE
├── Visual Encoder
│   ├── Vision Transformer (ViT)
│   ├── Image patches → embeddings
│   └── Output: Visual tokens
│
├── Multimodal Fusion
│   └── Concatenate [visual_tokens, text_tokens]
│
└── Language Model (MoE)
    └── Process combined sequence
        ├── Attention over visual + text
        └── MoE experts (Vision group: experts 2,3)
```

### Vision Expert Specialization

**Vision Group (ID: 3)**
- **Assigned Experts:** [2, 3] per layer
- **Total Experts:** 96 (2 experts × 48 layers)
- **Specialization:** Visual understanding, VQA, image-text alignment

**Datasets:**
1. **TextVQA:** Text in images, reasoning
2. **VQAv2:** General visual questions
3. **Refocused COCO Captions:** Descriptive image understanding
4. **Visual Reasoning (CLEVR):** Compositional understanding

## Data Pipeline

### Image Preprocessing

```python
from transformers import Qwen3VLProcessor

processor = Qwen3VLProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")

# Process image + text
inputs = processor(
    images=image,  # PIL Image
    text="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
         "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
         "What is in this image?<|im_end|>\n"
         "<|im_start|>assistant\n",
    return_tensors="pt",
    padding=True,
)

# Returns:
# - pixel_values: [batch, channels, height, width]
# - input_ids: [batch, seq_len] (includes special vision tokens)
# - attention_mask: [batch, seq_len]
# - image_grid_thw: [batch, 3] (temporal, height, width)
```

**Special Tokens:**
- `<|vision_start|>`: Marks beginning of visual content
- `<|image_pad|>`: Placeholder for image patch embeddings
- `<|vision_end|>`: Marks end of visual content

### Vision Dataset Structure

```python
class VisionMoEDataset(Dataset):
    """Multimodal dataset for Vision expert training."""
    
    def __init__(self, config, processor):
        self.datasets = [
            ("textvqa", self.load_textvqa()),
            ("vqav2", self.load_vqav2()),
            ("coco_captions", self.load_coco()),
            ("clevr", self.load_clevr()),
        ]
        self.processor = processor
        
    def __getitem__(self, idx):
        # Sample from dataset mixture
        dataset_name, dataset = random.choice(self.datasets)
        item = dataset[idx % len(dataset)]
        
        # Process
        inputs = self.processor(
            images=item["image"],
            text=self.format_prompt(item),
            return_tensors="pt",
        )
        
        return {
            "pixel_values": inputs["pixel_values"],
            "input_ids": inputs["input_ids"],
            "labels": self.create_labels(inputs["input_ids"], item["answer"]),
            "dataset_source": dataset_name,
        }
```

## Vision Encoder Details

### Image Embedding Process

```
Input Image (e.g., 512×512 RGB)
    ↓
Patch Extraction (16×16 patches)
    ↓
Linear Projection → Embeddings
    ↓
Add Position Embeddings
    ↓
Vision Transformer Layers
    ↓
Visual Token Sequence [1, num_patches, hidden_dim]
    ↓
Concatenate with Text Tokens
    ↓
Feed to Language Model (MoE)
```

**Example:**
```python
# Image: 512×512
# Patch size: 16×16
# Num patches: (512/16)² = 1024 patches

# Visual sequence length: 1024 tokens
# Text sequence length: 256 tokens
# Total sequence: 1280 tokens
```

### Vision Transformer Configuration

```python
vision_config = {
    "hidden_size": 1280,
    "num_hidden_layers": 32,
    "num_attention_heads": 16,
    "patch_size": 14,
    "image_size": 448,
    "intermediate_size": 5120,
    "projection_dim": 2048,  # Project to language model dim
}
```

## Training Strategy

### Vision-Specific Loss

```python
def compute_loss(outputs, labels, pixel_values):
    # Standard language modeling loss
    lm_loss = F.cross_entropy(
        outputs.logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=-100,
    )
    
    # Auxiliary loss for vision-text alignment (optional)
    if config.use_vision_alignment_loss:
        vision_embeds = outputs.vision_hidden_states[-1]
        text_embeds = outputs.last_hidden_state
        alignment_loss = contrastive_loss(vision_embeds, text_embeds)
        
        total_loss = lm_loss + 0.1 * alignment_loss
    else:
        total_loss = lm_loss
    
    return total_loss
```

### Vision Expert Routing

The router learns to activate Vision experts (2, 3) for image-related tokens:

```python
# Token: "What" → Experts [0, 2] (generic + vision)
# Token: "color" → Experts [2, 3] (vision + vision)
# Token: "is" → Experts [0, 1] (generic)
# Token: "the" → Experts [1, 2] (generic + vision)
# Token: "car" → Experts [2, 3] (vision + vision)

# Vision experts activate more for:
# - Tokens near <|image_pad|>
# - Visual attribute words (color, shape, object)
# - Spatial reasoning words (left, above, behind)
```

## Validation Metrics

### Vision-Specific Metrics

```python
from mycelia.shared.expert_specific_metrics import (
    compute_vqa_accuracy,
    compute_caption_similarity,
    compute_ocr_accuracy,
)

def evaluate_vision_expert(model, dataloader):
    metrics = {
        "vqa_accuracy": 0.0,
        "caption_bleu": 0.0,
        "caption_cider": 0.0,
        "ocr_accuracy": 0.0,
    }
    
    for batch in dataloader:
        # Generate predictions
        outputs = model.generate(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            max_new_tokens=100,
        )
        
        # Decode
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        references = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
        
        # Compute metrics
        if batch["dataset_source"] == "vqav2":
            metrics["vqa_accuracy"] += compute_vqa_accuracy(predictions, references)
        elif batch["dataset_source"] == "coco_captions":
            bleu, cider = compute_caption_similarity(predictions, references)
            metrics["caption_bleu"] += bleu
            metrics["caption_cider"] += cider
        elif batch["dataset_source"] == "textvqa":
            metrics["ocr_accuracy"] += compute_ocr_accuracy(predictions, references)
    
    return {k: v / len(dataloader) for k, v in metrics.items()}
```

**Metric Definitions:**

- **VQA Accuracy:** Exact match or soft match (for answers like "yes"/"yeah")
- **BLEU Score:** N-gram overlap for captions
- **CIDEr Score:** Consensus-based image description metric
- **OCR Accuracy:** Correct text extraction from images

## Anti-Cheat Mechanisms

### Vision-Specific Anti-Cheat

1. **Dynamic Image Augmentation**
```python
def augment_image(image):
    transforms = random.choice([
        A.ColorJitter(brightness=0.2, contrast=0.2),
        A.GaussianBlur(blur_limit=(3, 7)),
        A.Rotate(limit=15),
        A.RandomCrop(height=420, width=420),
    ])
    return transforms(image=image)["image"]
```

2. **Compositional VQA**
```python
# Require multi-step reasoning
"What color is the car that the person on the left is standing next to?"

# Cannot be solved by:
# - Memorizing image-caption pairs
# - Simple object detection
# - Direct label lookup
```

3. **Novel Image Combinations**
```python
# Validation set: Unseen image + question combinations
# Even if miner memorized training set
```

4. **Vision Token Masking**
```python
# Randomly mask 10-15% of vision tokens during validation
# Forces model to reason from partial information
```

## Memory Optimization

### Vision-Specific Considerations

```
Component                Memory per Batch
──────────────────────────────────────────
Image tensors (batch=2)  0.5 GB
Vision encoder forward   2.0 GB
Language model forward   1.5 GB
Gradients               4.0 GB
──────────────────────────────────────────
Total per batch         8.0 GB

Recommendation: batch_size=1-2 for vision tasks
```

### Gradient Checkpointing

```python
# Enable for vision encoder
model.visual.gradient_checkpointing_enable()

# Trades compute for memory:
# - Memory: 2.0 GB → 0.5 GB
# - Speed: 100% → 80% (20% slower)
```

## Example Usage

### Vision Miner Script

```python
from mycelia.miner.run import run
from mycelia.shared.config import MinerConfig

config = MinerConfig.from_path("config.yaml")
config.task.expert_group_id = 3  # Vision group

# Expert assignment fetched automatically from SN owner
# ✓ Assigned Vision group: experts [2,3] per layer

run(config)
```

### Vision Validation

```python
from mycelia.validator.evaluator import evaluate_model
from mycelia.shared.dataloader import get_dataloader

# Load vision dataloader
dataloader = get_dataloader(
    config,
    expert_group_id=3,  # Vision
    tokenizer=tokenizer,
)

# Evaluate
metrics = evaluate_model(
    step=global_step,
    model=model,
    dataloader=dataloader,
    device="cuda",
    expert_group_id=3,
    tokenizer=tokenizer,
)

print(f"VQA Accuracy: {metrics['vqa_accuracy']:.2%}")
print(f"Caption BLEU: {metrics['caption_bleu']:.2f}")
```

## Common Issues

### Issue 1: Vision Tokens Exceeding Max Sequence Length

**Problem:** Large images generate 1000+ vision tokens, exceeding model's max length (2048)

**Solution:**
```python
# Reduce image resolution
processor = Qwen3VLProcessor.from_pretrained(
    model_path,
    max_pixels=224*224,  # Lower resolution
)
```

### Issue 2: Vision Experts Not Activating

**Problem:** Router routes all tokens to non-vision experts

**Solution:**
```python
# Check router logits during training
if step % 100 == 0:
    router_stats = analyze_routing(model, vision_batch)
    # Vision tokens should activate experts [2, 3] more
```

### Issue 3: Poor VQA Performance

**Problem:** Model generates plausible but incorrect answers

**Solution:**
```python
# Increase vision expert training ratio
config.task.data.vision_ratio = 0.7  # 70% vision, 30% text
```

## References

- **Dataset Docs:** `expert_groups/exp_vision/README.md`
- **Implementation:** `expert_groups/exp_vision/dataset.py`
- **Metrics:** `mycelia/shared/expert_specific_metrics.py`
- **Example Script:** `run_vision_miner.py`

## Next Steps

- See [MoE Architecture](moe_architecture.md) for overall system
- See [ESFT Overview](esft_overview.md) for expert selection
- See [Miner Setup Guide](../guides/miner_setup.md) to train vision experts

