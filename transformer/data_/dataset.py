
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import json
import h5py
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass

from transformer.core.config import Config, DataConfig, AugmentationConfig, FeatureType
from transformer.core.exceptions import DataLoadError
from transformer.data_.augmentation import (
    create_augmentor_pipeline
)
from transformer.data_.sampler import (
    compute_class_weights
)

logger = logging.getLogger(__name__)


@dataclass
class BatchData:
    visual_features: torch.Tensor   
    pose_features: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    lengths: torch.Tensor
    sample_ids: Optional[List[str]] = None
    
    def to(self, device: torch.device) -> "BatchData":
        return BatchData(
            visual_features=self.visual_features.to(device),
            pose_features=self.pose_features.to(device),
            labels=self.labels.to(device),
            attention_mask=self.attention_mask.to(device),
            lengths=self.lengths.to(device),
            sample_ids=self.sample_ids
        )


class SignLanguageDataset(Dataset):
    
    def __init__(
        self,
        data_config: DataConfig,
        augmentation_config: Optional[AugmentationConfig] = None,
        feature_type: FeatureType = FeatureType.POSE,
        split: str = "train",
        seed: Optional[int] = None
    ):
        self.data_config = data_config
        self.feature_type = feature_type
        self.split = split
        self.training = (split == "train")
        
        self.features_dir = Path(data_config.features_dir)
        self._visual_h5 = None
        self._pose_h5 = None
        
        self.samples = self._load_samples()
        self.label_to_idx, self.idx_to_label = self._build_label_mapping()
        
        if augmentation_config and self.training:
            self.augmentors = create_augmentor_pipeline(augmentation_config, seed)
        else:
            self.augmentors = None
        
        logger.info(
            f"SignLanguageDataset ({feature_type.value} mode): "
            f"split={split}, samples={len(self.samples)}, "
            f"classes={len(self.label_to_idx)}"
        )
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        split_file = self.features_dir / f"{self.split}_samples.json"
        if split_file.exists():
            with open(split_file, 'r') as f:
                return json.load(f)
        
        labels_file = Path(self.data_config.labels_file)
        if not labels_file.exists():
            raise DataLoadError(
                f"Labels file not found: {labels_file}",
                recovery_hint="Run preprocessing to generate labels file"
            )
    
    def _build_label_mapping(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        unique_labels = sorted(set(s["label"] for s in self.samples))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        return label_to_idx, idx_to_label
    
    @property
    def visual_h5(self) -> h5py.File:
        if self._visual_h5 is None:
            visual_path = self.features_dir / "visual_features.h5"
            if not visual_path.exists():
                raise DataLoadError(f"Visual features file not found: {visual_path}")
            self._visual_h5 = h5py.File(visual_path, 'r')
        return self._visual_h5
    
    @property
    def pose_h5(self) -> h5py.File:
        if self._pose_h5 is None:
            pose_path = self.features_dir / "pose_features.h5"
            if not pose_path.exists():
                raise DataLoadError(f"Pose features file not found: {pose_path}")
            self._pose_h5 = h5py.File(pose_path, 'r')
        return self._pose_h5
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        sample_id = sample["id"]
        label = self.label_to_idx[sample["label"]]
        
        if self.feature_type == FeatureType.POSE:
            pose_features = self._load_pose_features(sample_id)
            
            # Apply pose augmentation
            if self.augmentors and self.training:
                temporal_result = self.augmentors["temporal"](
                    pose_features, training=True
                )
                pose_features = temporal_result.features
                
                pose_result = self.augmentors["pose"](
                    pose_features, training=True
                )
                pose_features = pose_result.features
            
            original_length = len(pose_features)
            
            pose_features, mask = self._pad_single(
                pose_features, self.data_config.pose_feature_dim
            )
            
            # model ignores these
            visual_features = torch.zeros(
                self.data_config.max_seq_length,
                self.data_config.visual_feature_dim
            )
            
        elif self.feature_type == FeatureType.MULTIMODAL:
            visual_features = self._load_visual_features(sample_id)
            pose_features = self._load_pose_features(sample_id)
            
            min_len = min(len(visual_features), len(pose_features))
            visual_features = visual_features[:min_len]
            pose_features = pose_features[:min_len]
            
            if self.augmentors and self.training:
                visual_result = self.augmentors["temporal"](
                    visual_features, training=True
                )
                visual_features = visual_result.features
                pose_result = self.augmentors["pose"](
                    pose_features, training=True
                )
                pose_features = pose_result.features
                min_len = min(len(visual_features), len(pose_features))
                visual_features = visual_features[:min_len]
                pose_features = pose_features[:min_len]
            
            original_length = len(visual_features)
            visual_features, pose_features, mask = self._pad_features(
                visual_features, pose_features
            )
        else:
            raise ValueError(f"Unsupported feature_type: {self.feature_type}")
        
        return {
            "visual_features": visual_features,
            "pose_features": pose_features,
            "label": torch.tensor(label, dtype=torch.long),
            "attention_mask": mask,
            "length": torch.tensor(original_length, dtype=torch.long),
            "sample_id": sample_id
        }
    
    def _load_visual_features(self, sample_id: str) -> torch.Tensor:
        try:
            features = self.visual_h5[sample_id][:]
            return torch.tensor(features, dtype=torch.float32)
        except KeyError:
            raise DataLoadError(f"Visual features not found: {sample_id}")
    
    def _load_pose_features(self, sample_id: str) -> torch.Tensor:
        try:
            features = self.pose_h5[sample_id][:]
            return torch.tensor(features, dtype=torch.float32)
        except KeyError:
            raise DataLoadError(f"Pose features not found: {sample_id}")
    
    def _pad_single(
        self, features: torch.Tensor, feature_dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad/truncate a single feature tensor."""
        max_len = self.data_config.max_seq_length
        features = features[:max_len]
        seq_len = len(features)
        
        if seq_len < max_len:
            pad_len = max_len - seq_len
            features = torch.cat([
                features,
                torch.zeros(pad_len, features.shape[-1])
            ])
            mask = torch.cat([
                torch.ones(seq_len, dtype=torch.bool),
                torch.zeros(pad_len, dtype=torch.bool)
            ])
        else:
            mask = torch.ones(max_len, dtype=torch.bool)
        
        return features, mask
    
    def _pad_features(self, visual, pose):
        max_len = self.data_config.max_seq_length
        visual = visual[:max_len]
        pose = pose[:max_len]
        seq_len = len(visual)
        
        if seq_len < max_len:
            pad_len = max_len - seq_len
            visual = torch.cat([visual, torch.zeros(pad_len, visual.shape[-1])])
            pose = torch.cat([pose, torch.zeros(pad_len, pose.shape[-1])])
            mask = torch.cat([
                torch.ones(seq_len, dtype=torch.bool),
                torch.zeros(pad_len, dtype=torch.bool)
            ])
        else:
            mask = torch.ones(max_len, dtype=torch.bool)
        
        return visual, pose, mask
    
    def get_labels(self) -> List[int]:
        return [self.label_to_idx[s["label"]] for s in self.samples]
    
    def close(self) -> None:
        if self._visual_h5 is not None:
            self._visual_h5.close()
            self._visual_h5 = None
        if self._pose_h5 is not None:
            self._pose_h5.close()
            self._pose_h5 = None
    
    def __del__(self):
        self.close()


def collate_fn(batch: List[Dict[str, Any]]) -> BatchData:
    return BatchData(
        visual_features=torch.stack([b["visual_features"] for b in batch]),
        pose_features=torch.stack([b["pose_features"] for b in batch]),
        labels=torch.stack([b["label"] for b in batch]),
        attention_mask=torch.stack([b["attention_mask"] for b in batch]),
        lengths=torch.stack([b["length"] for b in batch]),
        sample_ids=[b["sample_id"] for b in batch]
    )


def create_dataloaders(
    config: Config,
    seed: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """Create dataloaders with proper feature_type support."""
    
    feature_type = config.model.feature_type
    
    train_dataset = SignLanguageDataset(
        config.data, config.augmentation,
        feature_type=feature_type, split="train", seed=seed
    )
    val_dataset = SignLanguageDataset(
        config.data, augmentation_config=None,
        feature_type=feature_type, split="val", seed=seed
    )
    test_dataset = SignLanguageDataset(
        config.data, augmentation_config=None,
        feature_type=feature_type, split="test", seed=seed
    )
    
    train_labels = train_dataset.get_labels()
    class_weights = compute_class_weights(
        train_labels, power=config.training.class_weight_power
    )
    
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.training.batch_size,
        shuffle=True, num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory, collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=config.training.eval_batch_size,
        shuffle=False, num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.training.eval_batch_size,
        shuffle=False, num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory, collate_fn=collate_fn
    )
    
    logger.info(
        f"Created dataloaders ({feature_type.value}): "
        f"train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}"
    )
    
    return train_loader, val_loader, test_loader, class_weights
