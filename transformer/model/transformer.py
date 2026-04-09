import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging
from dataclasses import dataclass

from transformer.model.components import (
    PositionalEncoding,
    FeatureProjection,
    CrossModalAttention,
    TransformerEncoderLayer,
    ClassificationHead,
)
from transformer.core.config import ModelConfig, DataConfig, FeatureType


logger = logging.getLogger(__name__)

@dataclass
class ModelOutput:
    logits: torch.Tensor
    features: Optional[torch.Tensor] = None
    pooled: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None


class SignLanguageTransformer(nn.Module):
    
    def __init__(
        self,
        config: ModelConfig,
        num_classes: int,
        visual_dim: int = 2048,
        pose_dim: int = 858
    ):
        super().__init__()
        
        self.config = config
        self.feature_type = config.feature_type
        self.num_classes = num_classes
        self.d_model = config.hidden_dim
        
        
        if self.feature_type in (FeatureType.VISUAL, FeatureType.MULTIMODAL):
            self.visual_proj = FeatureProjection(
                input_dim=visual_dim,
                output_dim=config.visual_proj_dim,
                dropout=config.dropout
            )
            self.visual_to_model = nn.Linear(config.visual_proj_dim, self.d_model)
        
        if self.feature_type in (FeatureType.POSE, FeatureType.MULTIMODAL):
            self.pose_proj = FeatureProjection(
                input_dim=pose_dim,
                output_dim=config.pose_proj_dim,
                dropout=config.dropout
            )
            self.pose_to_model = nn.Linear(config.pose_proj_dim, self.d_model)
        
        self.pos_encoding = PositionalEncoding(
            d_model=self.d_model,
            max_len=config.max_position_embeddings,
            dropout=config.dropout,
            learnable=config.use_learnable_pos_encoding
        )
        
        self.cross_modal_layers = None
        if (self.feature_type == FeatureType.MULTIMODAL and 
            config.use_cross_modal_attention):
            self.cross_modal_layers = nn.ModuleList([
                CrossModalAttention(
                    d_model=self.d_model,
                    num_heads=config.num_heads,
                    dropout=config.attention_dropout
                )
                for _ in range(config.cross_modal_layers)
            ])
        elif self.feature_type == FeatureType.MULTIMODAL:
            self.fusion = nn.Sequential(
                nn.Linear(self.d_model * 2, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.GELU(),
                nn.Dropout(config.dropout)
            )
        
        # Transformer encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=self.d_model,
                num_heads=config.num_heads,
                ff_dim=config.ff_dim,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout
            )
            for _ in range(config.num_layers)
        ])
        
        self.encoder_norm = nn.LayerNorm(self.d_model)
        
        # Classification head
        self.classifier = ClassificationHead(
            d_model=self.d_model,
            num_classes=num_classes,
            hidden_dim=config.classifier_hidden_dim,
            dropout=config.dropout,
            pooling=config.use_pooling
        )
        
        self._init_weights()
        self._log_model_info()
    
    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _log_model_info(self) -> None:
        num_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"SignLanguageTransformer ({self.feature_type.value} mode):\n"
            f"- Hidden dim: {self.d_model}\n"
            f"- Num layers: {self.config.num_layers}\n"
            f"- Num heads: {self.config.num_heads}\n"
            f"- Num classes: {self.num_classes}\n"
            f"- Total params: {num_params:,}\n"
            f"- Trainable params: {trainable_params:,}\n"
            f"- Has visual branch: {self.feature_type != FeatureType.POSE}\n"
            f"- Has pose branch: {self.feature_type != FeatureType.VISUAL}\n"
            f"- Has cross-modal attention: {self.cross_modal_layers is not None}"
        )
    
    def forward(
        self,
        visual_features: Optional[torch.Tensor] = None,
        pose_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> ModelOutput:

        
        if self.feature_type == FeatureType.POSE:
            assert pose_features is not None, "pose_features required for pose mode"
            
            pose = self.pose_proj(pose_features)
            pose = self.pose_to_model(pose)
            hidden = self.pos_encoding(pose)
            
        elif self.feature_type == FeatureType.VISUAL:
            assert visual_features is not None, "visual_features required for visual mode"
            
            visual = self.visual_proj(visual_features)
            visual = self.visual_to_model(visual)
            hidden = self.pos_encoding(visual)
            
        elif self.feature_type == FeatureType.MULTIMODAL:
            assert visual_features is not None and pose_features is not None, \
                "Both visual and pose features required for multimodal mode"
            
            visual = self.visual_proj(visual_features)
            pose = self.pose_proj(pose_features)
            visual = self.visual_to_model(visual)
            pose = self.pose_to_model(pose)
            visual = self.pos_encoding(visual)
            pose = self.pos_encoding(pose)
            
            if self.cross_modal_layers is not None:
                hidden = visual
                for cross_modal in self.cross_modal_layers:
                    hidden = cross_modal(hidden, pose, attention_mask)
            else:
                hidden = torch.cat([visual, pose], dim=-1)
                hidden = self.fusion(hidden)
        
        # Transformer encoding
        for layer in self.encoder_layers:
            hidden = layer(hidden, attention_mask)
        
        hidden = self.encoder_norm(hidden)
        
        # Classification
        logits = self.classifier(hidden, attention_mask)
        
        output = ModelOutput(logits=logits)
        if return_features:
            output.features = hidden
            if hasattr(self.classifier, 'pooling') and self.classifier.pooling is not None:
                output.pooled = self.classifier.pooling(hidden, attention_mask)
            else:
                if attention_mask is not None:
                    mask = attention_mask.unsqueeze(-1).float()
                    output.pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                else:
                    output.pooled = hidden.mean(dim=1)
        
        return output


def create_model(
    model_config: ModelConfig,
    data_config: DataConfig,
    device: Optional[torch.device] = None
) -> SignLanguageTransformer:

    pose_dim = data_config.pose_feature_dim
    if data_config.add_velocity:
        pose_dim *= 2
    
    visual_dim = data_config.visual_feature_dim
    
    model = SignLanguageTransformer(
        config=model_config,
        num_classes=data_config.num_classes,
        visual_dim=visual_dim,
        pose_dim=pose_dim
    )
    
    if device is not None:
        model = model.to(device)
    
    return model