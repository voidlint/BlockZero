"""
Expert Profiling System

Tests individual experts in each MoE layer to understand their specialization.
Bypasses the router to force-route specific inputs to individual experts.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

from mycelia.shared.app_logging import structlog
from mycelia.shared.expert_specific_metrics import (
    get_expert_metrics_computer,
    ExpertGroup,
)
from mycelia.shared.domain_evaluators import DOMAIN_EVALUATORS

logger = structlog.get_logger(__name__)


@dataclass
class ExpertProfile:
    """Profile of a single expert's capabilities across multiple domains."""

    layer_id: int
    expert_id: int

    # Core domain scores (0-1, higher is better)
    math_score: float = 0.0
    agentic_score: float = 0.0
    planning_score: float = 0.0

    # Extended domain scores
    code_score: float = 0.0
    science_score: float = 0.0
    reasoning_score: float = 0.0
    vision_score: float = 0.0
    robotics_score: float = 0.0
    multilingual_score: float = 0.0
    creative_score: float = 0.0
    factual_score: float = 0.0
    conversation_score: float = 0.0

    # Performance metrics
    avg_loss: float = 0.0
    perplexity: float = 0.0

    # Utilization statistics
    token_count: int = 0
    routing_weight_avg: float = 0.0

    # Specialization classification
    specialization: str = "unknown"
    # Possible values: "math", "agentic", "planning", "code", "science", "reasoning",
    # "vision", "robotics", "multilingual", "creative", "factual", "conversation",
    # "generalist", "weak"

    def __post_init__(self):
        """Classify specialization based on scores."""
        self.specialization = self._classify_specialization()

    def _classify_specialization(self) -> str:
        """Determine expert specialization from domain scores."""
        scores = {
            "math": self.math_score,
            "agentic": self.agentic_score,
            "planning": self.planning_score,
            "code": self.code_score,
            "science": self.science_score,
            "reasoning": self.reasoning_score,
            "vision": self.vision_score,
            "robotics": self.robotics_score,
            "multilingual": self.multilingual_score,
            "creative": self.creative_score,
            "factual": self.factual_score,
            "conversation": self.conversation_score,
        }

        # Filter out zero scores (domains not tested)
        tested_scores = {k: v for k, v in scores.items() if v > 0.0}

        if not tested_scores:
            return "unknown"

        max_domain = max(tested_scores, key=tested_scores.get)
        max_score = tested_scores[max_domain]

        # Weak expert: all scores below threshold
        if max_score < 0.5:
            return "weak"

        # Specialist: one domain significantly higher than others
        other_scores = [s for d, s in tested_scores.items() if d != max_domain]
        if other_scores and max_score - max(other_scores) > 0.15:
            return max_domain

        # Generalist: all scores relatively balanced and above threshold
        if len(tested_scores) >= 3:
            if min(tested_scores.values()) > 0.6 and max_score - min(tested_scores.values()) < 0.2:
                return "generalist"

        # Default to dominant domain
        return max_domain

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        result = {
            "layer_id": self.layer_id,
            "expert_id": self.expert_id,
            "specialization": self.specialization,
            "avg_loss": round(self.avg_loss, 3),
            "perplexity": round(self.perplexity, 3),
            "token_count": self.token_count,
            "routing_weight_avg": round(self.routing_weight_avg, 3),
        }

        # Add all domain scores (only include non-zero to save space)
        domain_scores = {
            "math_score": self.math_score,
            "agentic_score": self.agentic_score,
            "planning_score": self.planning_score,
            "code_score": self.code_score,
            "science_score": self.science_score,
            "reasoning_score": self.reasoning_score,
            "vision_score": self.vision_score,
            "robotics_score": self.robotics_score,
            "multilingual_score": self.multilingual_score,
            "creative_score": self.creative_score,
            "factual_score": self.factual_score,
            "conversation_score": self.conversation_score,
        }

        for key, value in domain_scores.items():
            if value > 0.0:  # Only include tested domains
                result[key] = round(value, 3)

        return result


@dataclass
class LayerProfile:
    """Profile of all experts in a single layer."""

    layer_id: int
    expert_profiles: Dict[int, ExpertProfile] = field(default_factory=dict)

    def add_expert(self, profile: ExpertProfile):
        """Add an expert profile to this layer."""
        self.expert_profiles[profile.expert_id] = profile

    def get_specialists(self, domain: str) -> List[ExpertProfile]:
        """Get experts specialized in a domain."""
        return [
            ep for ep in self.expert_profiles.values()
            if ep.specialization == domain
        ]

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for this layer."""
        if not self.expert_profiles:
            return {}

        specializations = defaultdict(int)
        for ep in self.expert_profiles.values():
            specializations[ep.specialization] += 1

        return {
            "layer_id": self.layer_id,
            "num_experts": len(self.expert_profiles),
            "specialization_distribution": dict(specializations),
            "avg_math_score": np.mean([ep.math_score for ep in self.expert_profiles.values()]),
            "avg_agentic_score": np.mean([ep.agentic_score for ep in self.expert_profiles.values()]),
            "avg_planning_score": np.mean([ep.planning_score for ep in self.expert_profiles.values()]),
        }


class ExpertProfiler:
    """
    Profiles individual experts by forcing routing to specific experts.

    This tool bypasses the model's router to test each expert independently
    on all available domains: Math, Agentic, Planning, Code, Science, Reasoning,
    Vision, Robotics, Multilingual, Creative, Factual, and Conversation.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: torch.device,
        num_samples_per_domain: int = 50,
    ):
        """
        Initialize the expert profiler.

        Args:
            model: MoE model to profile
            tokenizer: Tokenizer for encoding/decoding
            device: Device to run on
            num_samples_per_domain: Number of test samples per domain
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_samples_per_domain = num_samples_per_domain

        # Detect MoE layers
        self.moe_layers = self._detect_moe_layers()
        logger.info(f"Detected {len(self.moe_layers)} MoE layers", layers=self.moe_layers)

        # Initialize core metrics computers (using factory function)
        self.math_computer = get_expert_metrics_computer(0)  # Math group
        self.agentic_computer = get_expert_metrics_computer(1)  # Agentic group
        self.planning_computer = get_expert_metrics_computer(2)  # Planning group

        # Initialize extended domain evaluators
        self.domain_evaluators = DOMAIN_EVALUATORS

    def _detect_moe_layers(self) -> List[int]:
        """Detect which layers have MoE blocks."""
        moe_layers = []

        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            for layer_idx, layer in enumerate(self.model.model.layers):
                # Check if this layer has MoE (SparseMoeBlock)
                if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                    moe_layers.append(layer_idx)

        return moe_layers

    def _get_expert_count(self, layer_id: int) -> int:
        """Get number of experts in a layer."""
        layer = self.model.model.layers[layer_id]
        if hasattr(layer.mlp, 'experts'):
            return len(layer.mlp.experts)
        return 0

    def profile_all_experts(
        self,
        math_dataloader: Optional[Any] = None,
        agentic_dataloader: Optional[Any] = None,
        planning_dataloader: Optional[Any] = None,
    ) -> Dict[int, LayerProfile]:
        """
        Profile all experts across all MoE layers.

        Args:
            math_dataloader: DataLoader for math tasks
            agentic_dataloader: DataLoader for agentic tasks
            planning_dataloader: DataLoader for planning tasks

        Returns:
            Dictionary mapping layer_id to LayerProfile
        """
        logger.info("Starting expert profiling", num_layers=len(self.moe_layers))

        layer_profiles = {}

        for layer_id in self.moe_layers:
            logger.info(f"Profiling layer {layer_id}")
            layer_profile = self.profile_layer(
                layer_id=layer_id,
                math_dataloader=math_dataloader,
                agentic_dataloader=agentic_dataloader,
                planning_dataloader=planning_dataloader,
            )
            layer_profiles[layer_id] = layer_profile

            # Log summary
            summary = layer_profile.get_summary()
            logger.info("Layer profile complete", **summary)

        return layer_profiles

    def profile_layer(
        self,
        layer_id: int,
        math_dataloader: Optional[Any] = None,
        agentic_dataloader: Optional[Any] = None,
        planning_dataloader: Optional[Any] = None,
    ) -> LayerProfile:
        """
        Profile all experts in a single layer.

        Args:
            layer_id: Layer to profile
            math_dataloader: DataLoader for math tasks
            agentic_dataloader: DataLoader for agentic tasks
            planning_dataloader: DataLoader for planning tasks

        Returns:
            LayerProfile with all expert profiles
        """
        layer_profile = LayerProfile(layer_id=layer_id)
        num_experts = self._get_expert_count(layer_id)

        for expert_id in range(num_experts):
            logger.info(f"Profiling expert {expert_id} in layer {layer_id}")

            expert_profile = self.profile_expert(
                layer_id=layer_id,
                expert_id=expert_id,
                math_dataloader=math_dataloader,
                agentic_dataloader=agentic_dataloader,
                planning_dataloader=planning_dataloader,
            )

            layer_profile.add_expert(expert_profile)

        return layer_profile

    def profile_expert(
        self,
        layer_id: int,
        expert_id: int,
        domain_dataloaders: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ExpertProfile:
        """
        Profile a single expert on all domains.

        Args:
            layer_id: Layer containing the expert
            expert_id: Expert to profile
            domain_dataloaders: Dict mapping domain names to dataloaders
                                e.g., {"math": math_loader, "code": code_loader, ...}
            **kwargs: Backward compatibility - accepts math_dataloader, agentic_dataloader, planning_dataloader

        Returns:
            ExpertProfile with domain scores
        """
        profile = ExpertProfile(layer_id=layer_id, expert_id=expert_id)

        # Backward compatibility: handle old-style arguments
        if domain_dataloaders is None:
            domain_dataloaders = {}
            if 'math_dataloader' in kwargs and kwargs['math_dataloader'] is not None:
                domain_dataloaders['math'] = kwargs['math_dataloader']
            if 'agentic_dataloader' in kwargs and kwargs['agentic_dataloader'] is not None:
                domain_dataloaders['agentic'] = kwargs['agentic_dataloader']
            if 'planning_dataloader' in kwargs and kwargs['planning_dataloader'] is not None:
                domain_dataloaders['planning'] = kwargs['planning_dataloader']

        # Evaluate on all provided domains
        for domain_name, dataloader in domain_dataloaders.items():
            if dataloader is None:
                continue

            try:
                results = self._evaluate_expert_on_domain(
                    layer_id=layer_id,
                    expert_id=expert_id,
                    dataloader=dataloader,
                    domain=domain_name,
                )

                # Map results to profile attributes
                score_attr = f"{domain_name}_score"
                if hasattr(profile, score_attr):
                    setattr(profile, score_attr, results["domain_score"])
                else:
                    logger.warning(f"Unknown domain: {domain_name}, skipping")

            except Exception as e:
                logger.warning(f"Error profiling domain {domain_name}: {e}")
                continue

        # Recompute specialization after all scores are set
        profile.specialization = profile._classify_specialization()

        return profile

    def _evaluate_expert_on_domain(
        self,
        layer_id: int,
        expert_id: int,
        dataloader: Any,
        domain: str,
    ) -> Dict[str, float]:
        """
        Evaluate a specific expert on a specific domain.

        This method temporarily modifies the model to force-route to the target expert.

        Args:
            layer_id: Layer containing the expert
            expert_id: Expert to evaluate
            dataloader: Data for this domain
            domain: Domain name ("math", "agentic", "planning")

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()

        total_loss = 0.0
        num_batches = 0
        predictions = []
        references = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= self.num_samples_per_domain:
                    break

                # Move batch to device
                device_batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass with forced routing
                try:
                    outputs = self._forward_with_forced_expert(
                        batch=device_batch,
                        layer_id=layer_id,
                        expert_id=expert_id,
                    )

                    if hasattr(outputs, 'loss') and outputs.loss is not None:
                        total_loss += outputs.loss.item()
                        num_batches += 1

                    # Collect predictions if possible
                    if hasattr(outputs, 'logits'):
                        pred_ids = torch.argmax(outputs.logits, dim=-1)
                        pred_text = self.tokenizer.decode(pred_ids[0], skip_special_tokens=True)
                        ref_text = self.tokenizer.decode(device_batch['input_ids'][0], skip_special_tokens=True)

                        predictions.append(pred_text)
                        references.append(ref_text)

                except Exception as e:
                    logger.warning(f"Error evaluating expert {expert_id}: {e}")
                    continue

        # Compute domain-specific score
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

        # Use domain-specific metrics if predictions available
        domain_score = 0.5  # Default fallback

        if predictions and references:
            # Core domains with specialized computers
            if domain == "math":
                try:
                    metrics = self.math_computer.compute_metrics(
                        predictions=predictions,
                        references=references,
                        base_metrics={"val_loss": avg_loss},
                    )
                    domain_score = metrics.get("numerical_accuracy", 0.0)
                    logger.info(f"Math evaluation complete", num_predictions=len(predictions), score=domain_score)
                except Exception as e:
                    logger.warning(f"Math evaluation failed: {e}")
                    domain_score = 0.0

            elif domain == "agentic":
                metrics = self.agentic_computer.compute_metrics(
                    predictions=predictions,
                    references=references,
                    base_metrics={"val_loss": avg_loss},
                )
                domain_score = metrics.get("tool_selection_accuracy", 0.0)

            elif domain == "planning":
                metrics = self.planning_computer.compute_metrics(
                    predictions=predictions,
                    references=references,
                    base_metrics={"val_loss": avg_loss},
                )
                domain_score = metrics.get("uncertainty_calibration", 0.0)

            # Extended domains with pattern-based evaluators
            elif domain in self.domain_evaluators:
                evaluator = self.domain_evaluators[domain]
                domain_score = evaluator.evaluate(predictions, references)

            else:
                # Unknown domain: use loss-based fallback
                logger.warning(f"Unknown domain {domain}, using loss-based scoring")
                domain_score = max(0.0, 1.0 - (avg_loss / 10.0))
        else:
            # Fallback: use loss-based score
            domain_score = max(0.0, 1.0 - (avg_loss / 10.0))

        return {
            "domain_score": domain_score,
            "avg_loss": avg_loss,
            "num_samples": num_batches,
        }

    def _forward_with_forced_expert(
        self,
        batch: Dict[str, torch.Tensor],
        layer_id: int,
        expert_id: int,
    ) -> Any:
        """
        Forward pass with forced routing to a specific expert.

        This is done by temporarily replacing the router's forward method
        to always select the target expert, and freezing all other model
        parameters to isolate the expert's behavior.

        Args:
            batch: Input batch
            layer_id: Layer to force routing in
            expert_id: Expert to route to

        Returns:
            Model outputs
        """
        layer = self.model.model.layers[layer_id]
        original_gate_forward = layer.mlp.gate.forward

        # Detect router architecture by checking what the original gate returns
        # Try a dummy forward to see the return format
        dummy_input = torch.zeros((1, self.model.config.hidden_size), device=self.device, dtype=next(layer.mlp.gate.parameters()).dtype)
        with torch.no_grad():
            try:
                original_output = original_gate_forward(dummy_input)
                router_returns_tuple = isinstance(original_output, tuple)
            except:
                # If dummy forward fails, assume it returns tuple (custom router)
                router_returns_tuple = True

        def forced_router(hidden_states):
            """Router that always selects the target expert."""
            # Handle different router architectures
            if hasattr(layer.mlp.gate, 'num_experts'):
                # Custom router with num_experts attribute
                num_experts = layer.mlp.gate.num_experts
            elif hasattr(layer.mlp.gate, 'out_features'):
                # nn.Linear router - out_features is number of experts
                num_experts = layer.mlp.gate.out_features
            elif hasattr(layer.mlp, 'num_experts'):
                # MoE layer has num_experts
                num_experts = layer.mlp.num_experts
            else:
                # Fallback: count experts
                num_experts = len(layer.mlp.experts)

            batch_size = hidden_states.shape[0]

            # Create router logits that heavily favor the target expert
            router_logits = torch.full(
                (batch_size, num_experts),
                -1000.0,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            router_logits[:, expert_id] = 1000.0  # Extreme preference

            # If router returns just logits (original architecture), return logits only
            if not router_returns_tuple:
                return router_logits

            # Otherwise, return full tuple (custom architecture)
            # Determine top_k
            if hasattr(layer.mlp.gate, 'top_k'):
                top_k = layer.mlp.gate.top_k
            elif hasattr(layer.mlp, 'top_k'):
                top_k = layer.mlp.top_k
            else:
                top_k = 1  # Default to top-1 routing

            # Create routing weights (all weight to target expert)
            routing_weights = torch.zeros(
                (batch_size, top_k),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            routing_weights[:, 0] = 1.0  # All weight to first (and only) selected expert

            # Selected experts (all pointing to target expert)
            selected_experts = torch.full(
                (batch_size, top_k),
                expert_id,
                device=hidden_states.device,
                dtype=torch.long,
            )

            return router_logits, routing_weights, selected_experts

        # Save original requires_grad states for all parameters
        original_requires_grad = {}
        for name, param in self.model.named_parameters():
            original_requires_grad[name] = param.requires_grad

        try:
            # Freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False

            # Unfreeze only the target expert's parameters
            # Expert modules are typically named like: model.layers.{layer_id}.mlp.experts.{expert_id}.*
            target_expert_prefix = f"model.layers.{layer_id}.mlp.experts.{expert_id}"
            for name, param in self.model.named_parameters():
                if target_expert_prefix in name:
                    param.requires_grad = True

            # Temporarily replace router
            layer.mlp.gate.forward = forced_router

            # Forward pass
            outputs = self.model(**batch)

            return outputs

        finally:
            # Restore original router
            layer.mlp.gate.forward = original_gate_forward

            # Restore original requires_grad states
            for name, param in self.model.named_parameters():
                param.requires_grad = original_requires_grad[name]


def print_skill_matrix(layer_profiles: Dict[int, LayerProfile], top_k: int = 5):
    """
    Print a skill matrix showing expert specializations.

    Args:
        layer_profiles: Dictionary of layer profiles
        top_k: Number of top experts to show per domain
    """
    print("\n" + "="*80)
    print("EXPERT SKILL MATRIX")
    print("="*80)

    for layer_id in sorted(layer_profiles.keys()):
        layer_profile = layer_profiles[layer_id]
        print(f"\n--- Layer {layer_id} ---")

        summary = layer_profile.get_summary()
        print(f"Total experts: {summary['num_experts']}")
        print(f"Specialization distribution: {summary['specialization_distribution']}")

        # Print top experts per domain
        for domain in ["math", "agentic", "planning"]:
            score_attr = f"{domain}_score"
            experts = sorted(
                layer_profile.expert_profiles.values(),
                key=lambda ep: getattr(ep, score_attr),
                reverse=True,
            )[:top_k]

            print(f"\nTop {top_k} {domain.capitalize()} experts:")
            for rank, expert in enumerate(experts, 1):
                print(
                    f"  {rank}. Expert {expert.expert_id}: "
                    f"{getattr(expert, score_attr):.3f} "
                    f"({expert.specialization})"
                )

    print("\n" + "="*80)


def export_profiles_to_json(layer_profiles: Dict[int, LayerProfile], filepath: str):
    """
    Export expert profiles to JSON file.

    Args:
        layer_profiles: Dictionary of layer profiles
        filepath: Output file path
    """
    import json

    data = {}
    for layer_id, layer_profile in layer_profiles.items():
        data[str(layer_id)] = {
            "summary": layer_profile.get_summary(),
            "experts": {
                str(expert_id): profile.to_dict()
                for expert_id, profile in layer_profile.expert_profiles.items()
            },
        }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Expert profiles exported to {filepath}")
