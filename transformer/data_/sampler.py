import numpy as np
import torch
from torch.utils.data import Sampler
from typing import List, Iterator, Optional
import logging

logger = logging.getLogger(__name__)

class ClassWeightedSampler(Sampler):
    
    def __init__(
        self,
        labels: List[int],
        num_samples: Optional[int] = None,
        power: float = 0.5,
        seed: Optional[int] = None
    ):

        super().__init__(labels)
        
        self.labels = np.array(labels)
        self.num_samples = num_samples or len(labels)
        self.rng = np.random.default_rng(seed)
        
        # Calculate class frequencies
        unique, counts = np.unique(labels, return_counts=True)
        class_counts = dict(zip(unique, counts))
        
        # Calculate weights (inverse frequency with power scaling)
        max_count = max(counts)
        self.weights = np.array([
            (max_count / class_counts[label]) ** power
            for label in labels
        ])
        
        # Normalize
        self.weights = self.weights / self.weights.sum()
        
        logger.info(
            f"ClassWeightedSampler: {len(unique)} classes, "
            f"power={power}, num_samples={self.num_samples}"
        )
        
    def __iter__(self) -> Iterator[int]:

        indices = self.rng.choice(
            len(self.labels),
            size=self.num_samples,
            replace=True,
            p=self.weights
        )
        
        yield from indices.tolist()
        
    def __len__(self) -> int:
        return self.num_samples


def compute_class_weights(
    labels: List[int],
    power: float = 0.5,
    normalize: bool = True
) -> torch.Tensor:

    labels_np = np.array(labels)
    unique, counts = np.unique(labels_np, return_counts=True)
    
    num_classes = len(unique)
    max_count = max(counts)
    
    # Inverse frequency with power scaling
    weights = np.zeros(num_classes)
    for cls, count in zip(unique, counts):
        weights[cls] = (max_count / count) ** power
        
    if normalize:
        weights = weights * num_classes / weights.sum()
        
    logger.info(
        f"Class weights computed: min={weights.min():.4f}, "
        f"max={weights.max():.4f}, mean={weights.mean():.4f}"
    )
        
    return torch.tensor(weights, dtype=torch.float32)
