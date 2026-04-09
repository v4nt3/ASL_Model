from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Any, Dict, Union
from pathlib import Path
import yaml
from enum import Enum


class ConfigurationError(Exception):
    """Configuration validation error."""
    def __init__(self, message: str, details: Dict = None, recovery_hint: str = None):
        self.details = details or {}
        self.recovery_hint = recovery_hint
        super().__init__(message)


class ModelType(str, Enum):
    TRANSFORMER = "transformer"

class FeatureType(str, Enum):
    POSE = "pose"
    MULTIMODAL = "multimodal"
    VISUAL = "visual"


class OptimizerType(str, Enum):
    ADAM = "adam"
    ADAMW = "adamw"

class SchedulerType(str, Enum):
    COSINE = "cosine"
    COSINE_WARMUP = "cosine_warmup"
    PLATEAU = "plateau"


@dataclass
class DataConfig:
    
    data_dir: str = "data/dataset"
    features_dir: str = "data/features"
    labels_file: str = "data/labels.json"
    
    num_classes: int = 2286
    max_seq_length: int = 64
    min_seq_length: int = 8
    
    visual_feature_dim: int = 2048
    pose_feature_dim: int = 858
    
    # Splits
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    num_workers: int = 0
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    normalize_pose: bool = True
    add_velocity: bool = False # It was true in the feature extraction.
    
    def __post_init__(self):
        self._validate()
        
    def _validate(self) -> None:
        if self.num_classes <= 0:
            raise ConfigurationError(
                "num_classes must be positive",
                details={"num_classes": self.num_classes}
            )
        if not (0 < self.train_split < 1):
            raise ConfigurationError(
                "train_split must be between 0 and 1",
                details={"train_split": self.train_split}
            )
        splits_sum = self.train_split + self.val_split + self.test_split
        if abs(splits_sum - 1.0) > 1e-6:
            raise ConfigurationError(
                "Data splits must sum to 1.0",
                details={"sum": splits_sum}
            )
        if self.max_seq_length < self.min_seq_length:
            raise ConfigurationError(
                "max_seq_length must be >= min_seq_length"
            )


@dataclass
class AugmentationConfig:
    enabled: bool = True
    temporal_crop_prob: float = 0.3
    temporal_crop_ratio: Tuple[float, float] = (0.85, 1.0)
    speed_augment_prob: float = 0.3
    speed_range: Tuple[float, float] = (0.85, 1.15)
    temporal_mask_prob: float = 0.15
    temporal_mask_ratio: float = 0.05
    pose_noise_prob: float = 0.2
    pose_noise_std: float = 0.005
    pose_dropout_prob: float = 0.1
    pose_dropout_ratio: float = 0.03
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 0.0
    mix_prob: float = 0.3


@dataclass
class ModelConfig:

    model_type: ModelType = ModelType.TRANSFORMER
    feature_type: FeatureType = FeatureType.POSE
    
    # Architecture
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    ff_dim: int = 2048
    
    # Regularization
    dropout: float = 0.2
    attention_dropout: float = 0.1
    path_dropout: float = 0.1
    
    # Feature processing
    visual_proj_dim: int = 512
    pose_proj_dim: int = 512
    
    use_cross_modal_attention: bool = False
    cross_modal_layers: int = 0
    
    # Classification
    classifier_hidden_dim: int = 1024
    use_pooling: str = "attention"
    
    # Positional encoding
    use_learnable_pos_encoding: bool = True
    max_position_embeddings: int = 128
    
    def __post_init__(self):
        self._validate()
        
    def _validate(self) -> None:
        if self.hidden_dim % self.num_heads != 0:
            raise ConfigurationError(
                "hidden_dim must be divisible by num_heads",
                details={"hidden_dim": self.hidden_dim, "num_heads": self.num_heads}
            )


@dataclass
class TrainingConfig:
    batch_size: int = 128
    eval_batch_size: int = 256
    num_epochs: int = 150
    
    optimizer: OptimizerType = OptimizerType.ADAMW
    learning_rate: float = 5e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.05
    
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    scheduler: SchedulerType = SchedulerType.COSINE_WARMUP
    warmup_epochs: int = 8
    warmup_ratio: float = 0.06
    
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.001
    
    save_top_k: int = 3
    checkpoint_dir: str = "outputs/checkpoints"
    save_every_n_epochs: int = 10
    
    use_amp: bool = True
    amp_dtype: str = "float16"
    
    use_class_weights: bool = True
    class_weight_power: float = 0.5
    
    label_smoothing: float = 0.1
    seed: int = 42
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps



@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    experiment_name: str = "sign_language_pose_only"
    output_dir: str = "outputs"
    log_dir: str = "logs"
    
    
    def validate(self) -> None:
        if self.model.max_position_embeddings < self.data.max_seq_length:
            raise ConfigurationError(
                "max_position_embeddings must be >= max_seq_length"
            )
        # Validate feature_type consistency
        if self.model.feature_type == FeatureType.POSE:
            if self.model.use_cross_modal_attention:
                print("[WARNING] Use_cross_modal_attention=True but feature_type=pose. "
                      "Cross-modal attention will be disabled automatically.")
    
    def to_dict(self) -> Dict[str, Any]:
        def convert_value(value: Any) -> Any:
            if isinstance(value, Enum):
                return value.value
            elif isinstance(value, tuple):
                return list(value)
            return value
        def convert_dict(d: Dict) -> Dict:
            return {k: convert_value(v) for k, v in d.items()}
        return {
            "data": convert_dict(asdict(self.data)),
            "augmentation": convert_dict(asdict(self.augmentation)),
            "model": convert_dict(asdict(self.model)),
            "training": convert_dict(asdict(self.training)),
            "experiment_name": self.experiment_name,
            "output_dir": self.output_dir,
        }
    
    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        path = Path(path)
        if not path.exists():
            raise ConfigurationError(f"Config file not found: {path}")
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        if "model" in data:
            if "model_type" in data["model"]:
                data["model"]["model_type"] = ModelType(data["model"]["model_type"])
            if "feature_type" in data["model"]:
                data["model"]["feature_type"] = FeatureType(data["model"]["feature_type"])
        if "training" in data:
            if "optimizer" in data["training"]:
                data["training"]["optimizer"] = OptimizerType(data["training"]["optimizer"])
            if "scheduler" in data["training"]:
                data["training"]["scheduler"] = SchedulerType(data["training"]["scheduler"])
        if "augmentation" in data:
            for key in ["temporal_crop_ratio", "speed_range"]:
                if key in data["augmentation"] and isinstance(data["augmentation"][key], list):
                    data["augmentation"][key] = tuple(data["augmentation"][key])
        return cls(
            data=DataConfig(**data.get("data", {})),
            augmentation=AugmentationConfig(**data.get("augmentation", {})),
            model=ModelConfig(**data.get("model", {})),
            training=TrainingConfig(**data.get("training", {})),
            experiment_name=data.get("experiment_name", "sign_language_pose_only"),
            output_dir=data.get("output_dir", "outputs"),
        )


def get_pose_only_config() -> Config:
    return Config(
        model=ModelConfig(
            feature_type=FeatureType.POSE,
            hidden_dim=512,
            num_layers=4,
            num_heads=8,
            ff_dim=2048,
            dropout=0.2,
            attention_dropout=0.1,
            use_cross_modal_attention=False, 
        ),
        data=DataConfig(
            pose_feature_dim=858,
            add_velocity=False,
        ),
    )
