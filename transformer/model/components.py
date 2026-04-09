import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import logging


logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_len: int = 128,
        dropout: float = 0.1,
        learnable: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)
        self.learnable = learnable
        
        if learnable:
            self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        else:
            pe = self._create_sinusoidal_encoding(max_len, d_model)
            self.register_buffer('pe', pe)
            
    def _create_sinusoidal_encoding(
        self,
        max_len: int,
        d_model: int
    ) -> torch.Tensor:

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        seq_len = x.size(1)
        
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum {self.max_len}"
            )
            
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class FeatureProjection(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class CrossModalAttention(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Visual attends to pose
        self.visual_to_pose = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        # Pose attends to visual
        self.pose_to_visual = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm_visual = nn.LayerNorm(d_model)
        self.norm_pose = nn.LayerNorm(d_model)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.output_norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        visual: torch.Tensor,
        pose: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        # Convert mask to attention format if provided
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask  # Invert for nn.MultiheadAttention
            
        # Visual attends to pose (query=visual, key/value=pose)
        visual_attended, _ = self.visual_to_pose(
            query=visual,
            key=pose,
            value=pose,
            key_padding_mask=key_padding_mask
        )
        visual_attended = self.norm_visual(visual + visual_attended)
        
        # Pose attends to visual
        pose_attended, _ = self.pose_to_visual(
            query=pose,
            key=visual,
            value=visual,
            key_padding_mask=key_padding_mask
        )
        pose_attended = self.norm_pose(pose + pose_attended)
        
        # Fuse modalities
        fused = torch.cat([visual_attended, pose_attended], dim=-1)
        fused = self.fusion(fused)
        
        return self.output_norm(fused)

class AttentionPooling(nn.Module):
    
    def __init__(self, d_model: int):

        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        # Compute attention scores
        scores = self.attention(x).squeeze(-1)  # [batch, seq_len]
        
        # Mask invalid positions
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask, float('-inf'))
            
        # Softmax over sequence
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # [batch, seq_len, 1]
        
        # Weighted sum
        pooled = (x * weights).sum(dim=1)  # [batch, d_model]
        
        return pooled

class TransformerEncoderLayer(nn.Module):

    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1
    ):

        super().__init__()
        
        # Pre-normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        # Self-attention with pre-norm
        normed = self.norm1(x)
        
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask
            
        attended, _ = self.self_attn(
            query=normed,
            key=normed,
            value=normed,
            key_padding_mask=key_padding_mask
        )
        x = x + self.dropout(attended)
        
        # Feed-forward with pre-norm
        normed = self.norm2(x)
        x = x + self.ff(normed)
        
        return x


class ClassificationHead(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        pooling: str = "attention"
    ):

        super().__init__()
        
        self.pooling_type = pooling
        
        if pooling == "attention":
            self.pooling = AttentionPooling(d_model)
        elif pooling == "mean":
            self.pooling = None
        elif pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.pooling = None
        else:
            raise ValueError(f"Unknown pooling type: {pooling}")
            
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        pooled = self.pooling(x, attention_mask)
        
        return self.classifier(pooled)