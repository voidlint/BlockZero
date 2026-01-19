# Mixture-of-Experts (MoE) Architecture

## Overview

This subnet implements a **distributed Mixture-of-Experts (MoE)** architecture for decentralized machine learning. The system distributes expert training across multiple miners, coordinates through a subnet owner, and validates quality through a network of validators.

## Core Architecture

### Model Structure

```
Qwen3-VL-MoE (30B parameters)
├── Visual Encoder (standard)
│   └── Processes image inputs
│
└── Language Model (MoE)
    ├── 48 Transformer Layers
    │   ├── Attention (standard)
    │   └── MLP → Sparse MoE Block
    │       ├── Router (selects top-k experts)
    │       └── 4 Experts per layer (192 total experts)
    │    
    │
    └── Output Head
```

**Key Parameters:**
- Total experts: `4 experts/layer × 48 layers = 192 experts`
- Active experts per token: `2` (top-2 routing)
- Expert groups: `4` (Math, Agentic, Planning, Vision)

### Expert Group Distribution

Each expert group trains a subset of the 192 total experts:

| Group | ID | Total Experts | Specialization |
|-------|----|---------------|----------------|
| Math | 0 | 96 (48×2) | Mathematical reasoning |
| Agentic | 1 | 96 (48×2) | Tool use, function calls |
| Planning | 2 | 96 (48×2) | Multi-step planning |
| Vision | 3 | 96 (48×2) | Visual understanding |

**Note:** Expert IDs are remapped per group (`my_expert_id` → `org_expert_id`)

### Sparse MoE Block

```python
class SparseMoeBlock(nn.Module):
    def __init__(self, config, layer_id):
        self.num_experts = 4
        self.top_k = 2  # Activate 2 experts per token
        
        # Router: selects which experts to use
        self.gate = TopKRouter(config, available_experts=[0,1,2,3])
        
        # Experts: specialized MLPs
        self.experts = nn.ModuleDict({
            "0": Qwen3VLMoeTextMLP(config),
            "1": Qwen3VLMoeTextMLP(config),
            "2": Qwen3VLMoeTextMLP(config),
            "3": Qwen3VLMoeTextMLP(config),
        })
        
    def forward(self, hidden_states):
        # 1. Router selects top-k experts
        router_logits, routing_weights, selected_experts = self.gate(hidden_states)
        
        # 2. Route tokens to selected experts
        expert_outputs = []
        for expert_id in selected_experts:
            expert_output = self.experts[str(expert_id)](hidden_states)
            expert_outputs.append(routing_weights[expert_id] * expert_output)
        
        # 3. Add shared expert output
        shared_output = self.shared_expert(hidden_states)
        
        # 4. Combine
        return sum(expert_outputs) + shared_output, router_logits
```

## Distributed Training Architecture

### Network Topology

```
                    ┌─────────────────┐
                    │   Subnet Owner  │
                    │   - Coordinator │
                    │   - Assignment  │
                    │   - Phase Mgmt  │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   ┌────▼─────┐         ┌───▼────┐          ┌────▼─────┐
   │ Validator│         │Validator│          │Validator │
   │    #1    │◄────────┤   #2   │─────────►│    #3    │
   └────┬─────┘         └───┬────┘          └────┬─────┘
        │                   │                     │
        │ Evaluates         │ Evaluates           │ Evaluates
        │                   │                     │
   ┌────▼─────┐         ┌──▼─────┐          ┌────▼─────┐
   │  Miner   │         │ Miner  │          │  Miner   │
   │   #1     │         │  #2    │          │   #3     │
   │ (Math)   │         │(Agentic)│          │(Planning)│
   └──────────┘         └────────┘          └──────────┘
```

### Training Cycle

The subnet operates in synchronized phases:

```
Block:    0────────100───────200───────300───────400───────500
          │         │        │         │         │         │
Phase:    Distribute─┤Train──┤Commit───┤Submit──┤Validate─┤Merge─┤
          │         │        │         │         │         │
Duration: 100 blocks 100 bl  100 bl    100 bl    100 bl    100 bl

Cycle: 600 blocks (~1 hour on Bittensor)
```

**Phase Descriptions:**

1. **Distribute** (100 blocks): Miners download latest global model from validators
2. **Train** (100 blocks): Miners train their assigned experts locally
3. **Commit** (100 blocks): Miners commit model hash; validators commit random seed
4. **Submit** (100 blocks): Miners submit trained experts to validators
5. **Validate** (100 blocks): Validators evaluate all miner submissions
6. **Merge** (100 blocks): Validators aggregate best experts into global model

### Expert Assignment System

**Centralized Control:** The subnet owner assigns which experts each miner trains.

```python
# Subnet Owner
assignment_manager = ExpertAssignmentManager(config)
assignment_manager.auto_assign_miners(
    miner_hotkeys=["5Abc...", "5Def...", "5Ghi..."],
    strategy="round_robin",
)

# Assignment Result:
# Miner 5Abc... → Math group, experts [0,1] in layers [0,1,2,...,47]
# Miner 5Def... → Agentic group, experts [2,3] in layers [0,1,2,...,47]
# Miner 5Ghi... → Planning group, experts [0,1] in layers [0,1,2,...,47]
```

**Miner:** Fetches assignment from subnet owner API at startup

```python
# Automatic fetch via ExpertManager
expert_manager = ExpertManager(config)
# → GET http://sn_owner:7000/get-expert-assignment?miner=5Abc...&group=0
# ✓ Received assignment: 96 experts across 48 layers
```

**Validator:** Verifies miners trained only assigned experts

```python
# During evaluation
is_valid, reason = verify_miner_expert_assignment(
    miner_hotkey="5Abc...",
    expert_group_id=0,
    checkpoint_path="miner_submission.pt",
)
# ✓ Verified: Miner trained experts [0,1] only
```

## Memory Efficiency

### Quantization Strategy

**Base Model:** 4-bit quantized (frozen, inference only)
**Experts:** FP16 (trainable)

```python
# Load base model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    ),
    device_map="auto",
)

# Replace assigned experts with FP16 trainable versions
for layer_idx, layer in enumerate(model.language_model.layers):
    for expert_id in assigned_expert_ids:
        # Dequantize → FP16 → trainable
        layer.mlp.experts[expert_id] = create_fp16_expert(...)
```

### Expert-Specific Loading

Miners only load their assigned experts:

```python
# Expert Group: Math (experts 0,1 per layer)
moe_block.experts = nn.ModuleDict({
    "0": Qwen3VLMoeTextMLP(config),  # Load
    "1": Qwen3VLMoeTextMLP(config),  # Load
    # "2": NOT LOADED (not assigned)
    # "3": NOT LOADED (not assigned)
})

# Memory saved: 50% per layer (2/4 experts)
```

## Router Mechanism

### Top-K Expert Selection

For each token, the router selects the top-k most relevant experts:

```python
def forward(self, hidden_states):
    # hidden_states: [batch, seq_len, hidden_dim]
    
    # 1. Compute routing scores
    router_logits = self.weight(hidden_states)  # [batch*seq, num_experts]
    
    # 2. Mask unavailable experts (if partial loading)
    router_logits = mask_unavailable_experts(router_logits)
    
    # 3. Select top-k
    routing_weights = F.softmax(router_logits, dim=-1)
    routing_weights, selected_experts = torch.topk(
        routing_weights, 
        k=self.top_k,  # 2
        dim=-1
    )
    
    # 4. Normalize
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    
    return router_logits, routing_weights, selected_experts
```

**Load Balancing:** Auxiliary loss encourages even expert usage

```python
# Auxiliary loss (added to main loss)
aux_loss = load_balance_loss(router_logits, selected_experts)

# Prevents: One expert dominating all routing decisions
# Encourages: Even distribution across experts
```

## Model Checkpointing

### Checkpoint Structure

```python
checkpoint = {
    # Metadata
    "model_meta": {
        "global_ver": 42,           # Global training step
        "block": 12345,             # Bittensor block number
        "expert_group_id": 0,       # Math group
        "model_hash": "abc123...",  # For verification
    },
    
    # Model weights (only trained experts)
    "model_state_dict": {
        "language_model.layers.0.mlp.experts.0.gate_proj.weight": ...,
        "language_model.layers.0.mlp.experts.1.gate_proj.weight": ...,
        # ... 96 experts (2 per layer × 48 layers)
    },
    
    # Training state
    "optimizer_state_dict": {...},
    "scheduler_state_dict": {...},
    "scaler_state_dict": {...},
    
    # Dataloader state (for resumption)
    "dataloader_state_dict": {...},
}
```

### Distributed Checkpoint Syncing

```
Miner                    Validator
──────                   ─────────
  │                          │
  │  1. Train experts        │
  │                          │
  │  2. Save checkpoint      │
  │     ckpt/model.pt        │
  │                          │
  │  3. Submit               │
  │─────────────────────────>│
  │  POST /submit-checkpoint │
  │                          │
  │                          │  4. Evaluate
  │                          │     - Load checkpoint
  │                          │     - Compute metrics
  │                          │     - Score miner
  │                          │
  │                          │  5. Aggregate
  │                          │     - Merge best experts
  │                          │     - Update global model
  │                          │
  │  6. Download new model   │
  │<─────────────────────────│
  │  GET /get-checkpoint     │
  │                          │
```

## Key Design Decisions

### Why Qwen3-VL-MoE?

1. **Multimodal:** Supports both text and vision inputs
2. **Sparse MoE:** Efficient scaling (activate 2/4 experts per token)
3. **Proven architecture:** Based on Qwen3 research
4. **Pre-trained:** Strong base performance before fine-tuning

### Why 4 Expert Groups?

1. **Task diversity:** Math, Agentic, Planning, Vision cover major domains
2. **Network size:** 4 groups = reasonable number of miners (4-16 miners)
3. **Specialization:** Each group focuses on one domain for better quality
4. **Validation:** Distinct evaluation metrics per group

### Why Top-2 Routing?

1. **Efficiency:** Activates 50% of experts (vs 100% dense model)
2. **Specialization:** Forces experts to specialize on specific patterns
3. **Redundancy:** 2 experts provide robustness vs 1
4. **Research:** Optimal balance per Mixtral/Qwen papers

## References

- **Model:** [Qwen/Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct)
- **MoE Paper:** [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- **ESFT Paper:** [Let the Expert Stick to His Last](https://arxiv.org/abs/2407.01906)
- **Implementation:** `mycelia/shared/modeling/custom_qwen3_vl_moe.py`

## Next Steps

- See [Vision Pipeline](vision_pipeline.md) for multimodal details
- See [ESFT Overview](esft_overview.md) for expert selection
- See [Miner Setup Guide](../guides/miner_setup.md) to start training

