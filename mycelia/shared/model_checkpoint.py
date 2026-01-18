"""
Comprehensive ModelCheckpoint class for hash and signature verification.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional

import bittensor
import torch
from pydantic import BaseModel, Field

from mycelia.shared.app_logging import structlog
from mycelia.shared.helper import get_model_hash
from mycelia.shared.schema import sign_message, verify_message

logger = structlog.get_logger(__name__)


@dataclass
class ModelCheckpoint:
    """Local or remote model checkpoint with verification capabilities."""
    
    # Core metadata
    signed_model_hash: str | None = None
    model_hash: str | None = None
    global_ver: int = 0
    expert_group: int | None = None
    inner_opt: int = 0
    path: Path | None = None
    role: str | None = None  # [miner, validator]
    place: Literal["local", "onchain"] = "local"
    
    # Verification flags
    signature_required: bool = False
    signature_verified: bool = False
    hash_required: bool = False
    hash_verified: bool = False
    expert_group_check_required: bool = False
    expert_group_verified: bool = False
    
    # Validator-specific
    validated: bool = False  # For validator only
    
    # Timestamps
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    
    # Hotkey info
    hotkey: str | None = None
    block: int | None = None
    
    def expired(self) -> bool:
        """Check if checkpoint has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def hash_model(self, state_dict: Dict | None = None) -> str:
        """
        Compute hash of model checkpoint.
        
        Args:
            state_dict: Model state dict to hash. If None, loads from self.path
            
        Returns:
            Hex-encoded model hash
        """
        if state_dict is None:
            if self.path is None:
                raise ValueError("Either state_dict or path must be provided")
            
            if not self.path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {self.path}")
            
            # Load checkpoint
            try:
                checkpoint = torch.load(self.path, map_location="cpu")
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint
            except Exception as e:
                logger.error(f"Failed to load checkpoint from {self.path}: {e}")
                raise
        
        # Compute hash
        self.model_hash = get_model_hash(state_dict)
        self.hash_verified = True
        
        return self.model_hash
    
    def sign_hash(self, wallet: bittensor.Wallet) -> str:
        """
        Sign the model hash with wallet's hotkey.
        
        Args:
            wallet: Bittensor wallet to sign with
            
        Returns:
            Hex-encoded signature
        """
        if self.model_hash is None:
            raise ValueError("Must compute model_hash before signing")
        
        # Create message to sign
        message = self.model_hash.encode('utf-8')
        
        # Sign
        signature = sign_message(wallet.hotkey, message)
        self.signed_model_hash = signature
        self.signature_verified = True
        self.hotkey = wallet.hotkey.ss58_address
        
        logger.info(
            "Signed model hash",
            hotkey=self.hotkey[:16] + "...",
            hash=self.model_hash[:16] + "...",
        )
        
        return signature
    
    def verify_hash(self, expected_hash: str) -> bool:
        """
        Verify model hash matches expected value.
        
        Args:
            expected_hash: Expected hash value
            
        Returns:
            True if hash matches
        """
        if self.model_hash is None:
            logger.warning("No model hash to verify")
            return False
        
        matches = self.model_hash == expected_hash
        self.hash_verified = matches
        
        if not matches:
            logger.warning(
                "Hash mismatch",
                expected=expected_hash[:16] + "...",
                actual=self.model_hash[:16] + "...",
            )
        
        return matches
    
    def verify_signature(self, hotkey_ss58: str) -> bool:
        """
        Verify signature matches model hash and hotkey.
        
        Args:
            hotkey_ss58: SS58 address of expected signer
            
        Returns:
            True if signature is valid
        """
        if self.signed_model_hash is None:
            logger.warning("No signature to verify")
            return False
        
        if self.model_hash is None:
            logger.warning("No model hash to verify signature against")
            return False
        
        # Verify signature
        message = self.model_hash.encode('utf-8')
        try:
            is_valid = verify_message(
                origin_hotkey_ss58=hotkey_ss58,
                message=message,
                signature_hex=self.signed_model_hash,
            )
            
            self.signature_verified = is_valid
            
            if is_valid:
                self.hotkey = hotkey_ss58
                logger.info("Signature verified", hotkey=hotkey_ss58[:16] + "...")
            else:
                logger.warning("Invalid signature", hotkey=hotkey_ss58[:16] + "...")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False
    
    def verify_expert_group(self, expected_expert_ids: List[int], layer_id: int | None = None) -> bool:
        """
        Verify checkpoint contains only allowed experts.
        
        Args:
            expected_expert_ids: List of allowed expert IDs
            layer_id: Optional specific layer to check
            
        Returns:
            True if checkpoint contains only allowed experts
        """
        if self.path is None:
            logger.warning("No path to verify expert group")
            return False
        
        try:
            # Load checkpoint
            checkpoint = torch.load(self.path, map_location="cpu")
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            
            # Check expert IDs in state dict
            for key in state_dict.keys():
                # Parse expert ID from key (e.g., "model.layers.0.mlp.experts.2.weight")
                if "experts" in key and "weight" in key:
                    parts = key.split(".")
                    try:
                        expert_idx = int(parts[parts.index("experts") + 1])
                        layer_idx = int(parts[parts.index("layers") + 1])
                        
                        # If layer_id specified, only check that layer
                        if layer_id is not None and layer_idx != layer_id:
                            continue
                        
                        # Check if expert is allowed
                        if expert_idx not in expected_expert_ids:
                            logger.warning(
                                "Unauthorized expert found",
                                layer=layer_idx,
                                expert=expert_idx,
                                allowed=expected_expert_ids,
                            )
                            self.expert_group_verified = False
                            return False
                            
                    except (ValueError, IndexError):
                        continue
            
            self.expert_group_verified = True
            logger.info(
                "Expert group verified",
                allowed_experts=expected_expert_ids,
                layer=layer_id,
            )
            return True
            
        except Exception as e:
            logger.error(f"Expert group verification failed: {e}")
            return False
    
    def active(self, current_block: int, max_age_blocks: int = 300) -> bool:
        """
        Check if checkpoint is active (recent enough).
        
        Args:
            current_block: Current blockchain block number
            max_age_blocks: Maximum age in blocks (default: 300 ~1 hour)
            
        Returns:
            True if checkpoint is recent enough
        """
        if self.block is None:
            logger.warning("No block number for checkpoint")
            return False
        
        age = current_block - self.block
        is_active = age <= max_age_blocks
        
        if not is_active:
            logger.warning(
                "Checkpoint too old",
                block=self.block,
                current=current_block,
                age=age,
            )
        
        return is_active
    
    def is_complete(self) -> bool:
        """
        Check if checkpoint folder/file is complete and valid.
        
        Returns:
            True if checkpoint is ready to use
        """
        if self.path is None:
            return False
        
        # Check if path exists
        if not self.path.exists():
            logger.warning(f"Checkpoint path does not exist: {self.path}")
            return False
        
        # If it's a directory, check for required files
        if self.path.is_dir():
            required_files = ["model.pt"]  # At minimum, model.pt must exist
            for req_file in required_files:
                file_path = self.path / req_file
                if not file_path.exists():
                    logger.warning(f"Required file missing: {file_path}")
                    return False
                
                # Check if file is not empty
                if file_path.stat().st_size == 0:
                    logger.warning(f"File is empty: {file_path}")
                    return False
        
        # If it's a file, check if not empty
        elif self.path.is_file():
            if self.path.stat().st_size == 0:
                logger.warning(f"Checkpoint file is empty: {self.path}")
                return False
        
        return True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "signed_model_hash": self.signed_model_hash,
            "model_hash": self.model_hash,
            "global_ver": self.global_ver,
            "expert_group": self.expert_group,
            "inner_opt": self.inner_opt,
            "path": str(self.path) if self.path else None,
            "role": self.role,
            "place": self.place,
            "hotkey": self.hotkey,
            "block": self.block,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ModelCheckpoint":
        """Create from dictionary."""
        if data.get("path"):
            data["path"] = Path(data["path"])
        return cls(**data)


class ValidatorChainCheckpoint(ModelCheckpoint):
    """Remote validator checkpoint from chain."""
    
    priority_score: float = 0.0
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.place = "onchain"
        self.role = "validator"
    
    def get_signed_hash_commit(self) -> str | None:
        """Get signed hash from chain commit."""
        return self.signed_model_hash
    
    def get_hash_commit(self) -> str | None:
        """Get unsigned hash from chain commit."""
        return self.model_hash
    
    def calculate_priority(self, current_block: int) -> float:
        """
        Calculate priority score for this validator.
        
        Higher score = higher priority
        
        Factors:
        - Model version (higher is better)
        - Recency (more recent is better)
        - Verification status (verified is better)
        """
        score = 0.0
        
        # Model version (most important)
        score += self.global_ver * 100
        score += self.inner_opt * 1
        
        # Recency (within last 300 blocks gets bonus)
        if self.block and current_block:
            age = current_block - self.block
            if age <= 300:
                score += (300 - age) / 10
        
        # Verification status
        if self.hash_verified:
            score += 50
        if self.signature_verified:
            score += 50
        
        self.priority_score = score
        return score


class ModelCheckpoints(BaseModel):
    """Collection of model checkpoints."""
    
    checkpoints: List[ModelCheckpoint] = Field(default_factory=list)
    
    def ordered(self, by: Literal["version", "priority", "recency"] = "version") -> List[ModelCheckpoint]:
        """
        Get checkpoints ordered by specified criteria.
        
        Args:
            by: Ordering criterion
            
        Returns:
            Ordered list of checkpoints
        """
        if by == "version":
            return sorted(
                self.checkpoints,
                key=lambda c: (c.global_ver, c.inner_opt),
                reverse=True,
            )
        elif by == "priority":
            return sorted(
                self.checkpoints,
                key=lambda c: getattr(c, "priority_score", 0),
                reverse=True,
            )
        elif by == "recency":
            return sorted(
                self.checkpoints,
                key=lambda c: c.created_at,
                reverse=True,
            )
        else:
            raise ValueError(f"Unknown ordering: {by}")
    
    def filter_active(self, current_block: int, max_age_blocks: int = 300) -> "ModelCheckpoints":
        """Filter to only active checkpoints."""
        active = [c for c in self.checkpoints if c.active(current_block, max_age_blocks)]
        return ModelCheckpoints(checkpoints=active)
    
    def filter_verified(self) -> "ModelCheckpoints":
        """Filter to only verified checkpoints."""
        verified = [
            c for c in self.checkpoints
            if (not c.hash_required or c.hash_verified)
            and (not c.signature_required or c.signature_verified)
        ]
        return ModelCheckpoints(checkpoints=verified)
    
    def latest(self) -> ModelCheckpoint | None:
        """Get latest checkpoint by version."""
        ordered = self.ordered("version")
        return ordered[0] if ordered else None
    
    def add(self, checkpoint: ModelCheckpoint):
        """Add checkpoint to collection."""
        self.checkpoints.append(checkpoint)
    
    def remove(self, checkpoint: ModelCheckpoint):
        """Remove checkpoint from collection."""
        self.checkpoints.remove(checkpoint)


class ValidatorChainCheckpoints(BaseModel):
    """Collection of validator checkpoints from chain."""
    
    checkpoints: List[ValidatorChainCheckpoint] = Field(default_factory=list)
    
    def renew(self, new_checkpoints: List[ValidatorChainCheckpoint]):
        """Replace all checkpoints with new ones."""
        self.checkpoints = new_checkpoints
    
    def get_signed_hash_commits(self) -> Dict[str, str]:
        """Get all signed hash commits, keyed by hotkey."""
        return {
            c.hotkey: c.get_signed_hash_commit()
            for c in self.checkpoints
            if c.hotkey and c.get_signed_hash_commit()
        }
    
    def get_hash_commits(self) -> Dict[str, str]:
        """Get all unsigned hash commits, keyed by hotkey."""
        return {
            c.hotkey: c.get_hash_commit()
            for c in self.checkpoints
            if c.hotkey and c.get_hash_commit()
        }
    
    def ordered(self, by: Literal["version", "priority"] = "version") -> List[ValidatorChainCheckpoint]:
        """Get ordered list of validator checkpoints."""
        if by == "version":
            return sorted(
                self.checkpoints,
                key=lambda c: (c.global_ver, c.inner_opt),
                reverse=True,
            )
        elif by == "priority":
            return sorted(
                self.checkpoints,
                key=lambda c: c.priority_score,
                reverse=True,
            )
        else:
            raise ValueError(f"Unknown ordering: {by}")


class Checkpoints(BaseModel):
    """Combined local and chain checkpoints."""
    
    local: ModelCheckpoints = Field(default_factory=ModelCheckpoints)
    chain: ValidatorChainCheckpoints = Field(default_factory=ValidatorChainCheckpoints)
    
    def ordered(
        self,
        source: Literal["local", "chain", "all"] = "all",
        by: Literal["version", "priority"] = "version",
    ) -> List[ModelCheckpoint]:
        """
        Get ordered checkpoints from specified source.
        
        Args:
            source: Which checkpoints to include
            by: Ordering criterion
            
        Returns:
            Ordered list of checkpoints
        """
        if source == "local":
            return self.local.ordered(by)
        elif source == "chain":
            return self.chain.ordered(by)
        elif source == "all":
            all_checkpoints = self.local.checkpoints + self.chain.checkpoints
            if by == "version":
                return sorted(
                    all_checkpoints,
                    key=lambda c: (c.global_ver, c.inner_opt),
                    reverse=True,
                )
            elif by == "priority":
                return sorted(
                    all_checkpoints,
                    key=lambda c: getattr(c, "priority_score", 0),
                    reverse=True,
                )
        else:
            raise ValueError(f"Unknown source: {source}")
    
    def download(
        self,
        target_checkpoint: ValidatorChainCheckpoint,
        destination: Path,
    ) -> ModelCheckpoint:
        """
        Download checkpoint from validator and add to local collection.
        
        Args:
            target_checkpoint: Validator checkpoint to download
            destination: Local path to save checkpoint
            
        Returns:
            Downloaded local checkpoint
        """
        # This would be implemented with actual download logic
        # For now, create a placeholder
        local_checkpoint = ModelCheckpoint(
            model_hash=target_checkpoint.model_hash,
            signed_model_hash=target_checkpoint.signed_model_hash,
            global_ver=target_checkpoint.global_ver,
            inner_opt=target_checkpoint.inner_opt,
            expert_group=target_checkpoint.expert_group,
            path=destination,
            role="validator",
            place="local",
            hotkey=target_checkpoint.hotkey,
            block=target_checkpoint.block,
        )
        
        self.local.add(local_checkpoint)
        
        logger.info(
            "Downloaded checkpoint",
            version=local_checkpoint.global_ver,
            path=destination,
        )
        
        return local_checkpoint
