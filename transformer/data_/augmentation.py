import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
import logging

from transformer.core.config import AugmentationConfig


logger = logging.getLogger(__name__)


@dataclass
class AugmentationResult:
    features: torch.Tensor
    metadata: Dict[str, Any]


class TemporalAugmentor:
    
    def __init__(self, config: AugmentationConfig, seed: Optional[int] = None):

        self.config = config
        self.rng = np.random.default_rng(seed)
        
        logger.debug(
            "Initialized TemporalAugmentor",
            extra={"config": config.__dict__ if hasattr(config, '__dict__') else str(config)}
        )
        
    def __call__(
        self,
        features: torch.Tensor,
        training: bool = True
    ) -> AugmentationResult:

        if not training or not self.config.enabled:
            return AugmentationResult(features=features, metadata={})
            
        metadata = {}
        augmented = features.clone()
        
        # Speed augmentation
        if self.rng.random() < self.config.speed_augment_prob:
            augmented, speed_factor = self._speed_augment(augmented)
            metadata["speed_factor"] = speed_factor
            
        # Temporal cropping
        if self.rng.random() < self.config.temporal_crop_prob:
            augmented, crop_info = self._temporal_crop(augmented)
            metadata["crop"] = crop_info
            
        # Temporal masking
        if self.rng.random() < self.config.temporal_mask_prob:
            augmented, mask_info = self._temporal_mask(augmented)
            metadata["mask"] = mask_info
        
        # Non-linear temporal warping
        if hasattr(self.config, 'temporal_warp_prob') and self.rng.random() < self.config.temporal_warp_prob:
            augmented, warp_info = self._temporal_warp(augmented)
            metadata["temporal_warp"] = warp_info
            
        return AugmentationResult(features=augmented, metadata=metadata)
        
    def _speed_augment(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:

        seq_len = features.shape[0]
        min_speed, max_speed = self.config.speed_range
        speed_factor = self.rng.uniform(min_speed, max_speed)
        
        # Calculate new sequence length
        new_len = int(seq_len / speed_factor)
        new_len = max(self.config.temporal_crop_ratio[0] * seq_len, new_len)
        new_len = int(min(seq_len, new_len))
        
        if new_len == seq_len:
            return features, 1.0
            
        # Linear interpolation indices
        old_indices = torch.linspace(0, seq_len - 1, new_len)
        
        # Interpolate features
        augmented = self._interpolate_features(features, old_indices)
        
        return augmented, speed_factor
        
    def _interpolate_features(
        self,
        features: torch.Tensor,
        indices: torch.Tensor
    ) -> torch.Tensor:

        seq_len = features.shape[0]
        
        # Get floor and ceil indices
        floor_indices = indices.long().clamp(0, seq_len - 1)
        ceil_indices = (floor_indices + 1).clamp(0, seq_len - 1)
        
        # Calculate interpolation weights
        weights = (indices - floor_indices.float()).unsqueeze(-1)
        
        # Linear interpolation
        interpolated = (
            features[floor_indices] * (1 - weights) +
            features[ceil_indices] * weights
        )
        
        return interpolated
        
    def _temporal_crop(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, int]]:

        seq_len = features.shape[0]
        min_ratio, max_ratio = self.config.temporal_crop_ratio
        
        # Random crop ratio
        crop_ratio = self.rng.uniform(min_ratio, max_ratio)
        crop_len = int(seq_len * crop_ratio)
        crop_len = max(1, crop_len)
        
        # Random start position
        max_start = seq_len - crop_len
        start = self.rng.integers(0, max(1, max_start + 1))
        
        cropped = features[start:start + crop_len]
        
        return cropped, {"start": start, "length": crop_len}
        
    def _temporal_mask(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        seq_len = features.shape[0]
        num_mask = int(seq_len * self.config.temporal_mask_ratio)
        num_mask = max(1, min(num_mask, seq_len - 1))
        
        mask_indices = self.rng.choice(seq_len, size=num_mask, replace=False)
        
        masked = features.clone()
        masked[mask_indices] = 0
        
        return masked, {"masked_indices": mask_indices.tolist()}
    
    def _temporal_warp(
        self,
        features: torch.Tensor,
        num_control_points: int = 5
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        seq_len = features.shape[0]
        if seq_len < 4:
            return features, {"warped": False}
        
        # Generate random control points for time warping
        # These define a monotonically increasing mapping from old time -> new time
        control_x = np.sort(self.rng.uniform(0, 1, num_control_points))
        control_x = np.concatenate([[0.0], control_x, [1.0]])
        
        # Random displacements
        noise = self.rng.uniform(-0.1, 0.1, len(control_x))
        noise[0] = 0.0  # Fix start
        noise[-1] = 0.0  # Fix end
        control_y = control_x + noise
        
        # Ensure monotonically increasing
        for i in range(1, len(control_y)):
            control_y[i] = max(control_y[i], control_y[i-1] + 0.01)
        
        # Normalize to [0, 1]
        control_y = (control_y - control_y[0]) / (control_y[-1] - control_y[0])
        
        # Interpolate to get mapping for each frame
        old_times = np.linspace(0, 1, seq_len)
        new_times = np.interp(old_times, control_x, control_y)
        
        # Map new times back to frame indices
        new_indices = torch.tensor(new_times * (seq_len - 1), dtype=torch.float32)
        
        # Interpolate features at new positions
        warped = self._interpolate_features(features, new_indices)
        
        return warped, {"warped": True, "num_control_points": num_control_points}


class PoseAugmentor:
    
    # Joint group boundaries (in keypoint indices, multiply by 3 for feature indices)
    LEFT_HAND  = (0, 21)      # 21 keypoints
    RIGHT_HAND = (21, 42)     # 21 keypoints
    FACE       = (42, 110)    # 68 keypoints
    BODY       = (110, 143)   # 33 keypoints
    
    JOINT_GROUPS = {
        "left_hand":  LEFT_HAND,
        "right_hand": RIGHT_HAND,
        "face":       FACE,
        "body_upper": (110, 123),  # Upper body subset
        "body_lower": (123, 143),  # Lower body subset
    }
    
    def __init__(self, config: AugmentationConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        
    def __call__(
        self,
        pose_features: torch.Tensor,
        training: bool = True
    ) -> AugmentationResult:
        if not training or not self.config.enabled:
            return AugmentationResult(features=pose_features, metadata={})
            
        metadata = {}
        augmented = pose_features.clone()
        
        # Original: Noise injection
        if self.rng.random() < self.config.pose_noise_prob:
            augmented = self._add_noise(augmented)
            metadata["noise_added"] = True
            
        # Original: Keypoint dropout
        if self.rng.random() < self.config.pose_dropout_prob:
            augmented, dropout_info = self._keypoint_dropout(augmented)
            metadata["dropout"] = dropout_info
        
        return AugmentationResult(features=augmented, metadata=metadata)
    
    def _get_base_dim(self, feature_dim: int) -> int:
        """Determine base dimension (before velocity/acceleration)."""
        # 429 = 143 keypoints * 3 coords
        if feature_dim >= 429 * 2:  # has velocity
            return 429
        return feature_dim   
    
    def _add_noise(self, features: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(features) * self.config.pose_noise_std
        return features + noise
        
    def _keypoint_dropout(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        pose_dim = features.shape[-1]
        num_keypoints = pose_dim // 3
        
        num_drop = int(num_keypoints * self.config.pose_dropout_ratio)
        num_drop = max(1, num_drop)
        
        drop_indices = self.rng.choice(num_keypoints, size=num_drop, replace=False)
        
        mask = torch.ones_like(features)
        for idx in drop_indices:
            start = idx * 3
            mask[:, start:start + 3] = 0
            
        return features * mask, {"dropped_keypoints": drop_indices.tolist()}
    

class MixupAugmentor:
    
    def __init__(self, alpha: float = 0.2, seed: Optional[int] = None):

        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        
    def __call__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:

        if not training or self.alpha <= 0:
            return features, labels, labels, 1.0
            
        batch_size = features.shape[0]
        
        # Sample lambda from beta distribution
        lam = self.rng.beta(self.alpha, self.alpha)
        
        # Random permutation for mixing pairs
        indices = torch.randperm(batch_size)
        
        # Mix features
        mixed_features = lam * features + (1 - lam) * features[indices]
        
        return mixed_features, labels, labels[indices], lam


def create_augmentor_pipeline(
    config: AugmentationConfig,
    seed: Optional[int] = None
) -> Dict[str, Any]:

    return {
        "temporal": TemporalAugmentor(config, seed),
        "pose": PoseAugmentor(config, seed),
        "mixup": MixupAugmentor(config.mixup_alpha, seed) if config.mixup_alpha > 0 else None,
    }
