import torch
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class MetricsResult:

    accuracy: float
    top5_accuracy: float
    top10_accuracy: float
    loss: float
    per_class_metrics: Optional[Dict[int, Dict[str, float]]] = None
    confusion_matrix: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "accuracy": self.accuracy,
            "top5_accuracy": self.top5_accuracy,
            "top10_accuracy": self.top10_accuracy,
            "loss": self.loss
        }
        
        if self.per_class_metrics:
            # Summary statistics
            precisions = [m["precision"] for m in self.per_class_metrics.values()]
            recalls = [m["recall"] for m in self.per_class_metrics.values()]
            f1s = [m["f1"] for m in self.per_class_metrics.values()]
            
            result["macro_precision"] = np.mean(precisions)
            result["macro_recall"] = np.mean(recalls)
            result["macro_f1"] = np.mean(f1s)
            
        return result

class MetricsTracker:
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self) -> None:
        self.predictions: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []
        self.losses: List[float] = []
        self.num_samples = 0
        
    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss: Optional[float] = None
    ) -> None:
        # Store predictions (top-k indices for efficiency)
        _, top_k = logits.topk(min(10, logits.size(-1)), dim=-1)
        self.predictions.append(top_k.cpu())
        self.labels.append(labels.cpu())
        
        if loss is not None:
            self.losses.append(loss)
            
        self.num_samples += labels.size(0)
        
    def compute(
        self,
        compute_per_class: bool = False,
        compute_confusion_matrix: bool = False
    ) -> MetricsResult:
        if not self.predictions:
            logger.warning("No predictions accumulated, returning zero metrics")
            return MetricsResult(
                accuracy=0.0,
                top5_accuracy=0.0,
                top10_accuracy=0.0,
                loss=0.0
            )
            
        # Concatenate all batches
        all_preds = torch.cat(self.predictions, dim=0)  # [N, 10]
        all_labels = torch.cat(self.labels, dim=0)  # [N]
        
        # Compute accuracies
        accuracy = self._compute_accuracy(all_preds[:, 0], all_labels)
        top5_accuracy = self._compute_topk_accuracy(all_preds[:, :5], all_labels)
        top10_accuracy = self._compute_topk_accuracy(all_preds, all_labels)
        
        # Average loss
        avg_loss = np.mean(self.losses) if self.losses else 0.0
        
        # Per-class metrics
        per_class = None
        if compute_per_class:
            per_class = self._compute_per_class_metrics(all_preds[:, 0], all_labels)
            
        # Confusion matrix
        conf_matrix = None
        if compute_confusion_matrix:
            conf_matrix = self._compute_confusion_matrix(all_preds[:, 0], all_labels)
            
        return MetricsResult(
            accuracy=accuracy,
            top5_accuracy=top5_accuracy,
            top10_accuracy=top10_accuracy,
            loss=avg_loss,
            per_class_metrics=per_class,
            confusion_matrix=conf_matrix
        )
        
    def _compute_accuracy(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        correct = (predictions == labels).sum().item()
        return correct / len(labels) if len(labels) > 0 else 0.0
        
    def _compute_topk_accuracy(
        self,
        top_k_preds: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        labels_expanded = labels.unsqueeze(-1).expand_as(top_k_preds)
        correct = (top_k_preds == labels_expanded).any(dim=-1).sum().item()
        return correct / len(labels) if len(labels) > 0 else 0.0
        
    def _compute_per_class_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[int, Dict[str, float]]:

        predictions = predictions.numpy()
        labels = labels.numpy()
        
        per_class = {}
        
        for cls in range(self.num_classes):
            # True positives, false positives, false negatives
            tp = np.sum((predictions == cls) & (labels == cls))
            fp = np.sum((predictions == cls) & (labels != cls))
            fn = np.sum((predictions != cls) & (labels == cls))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_class[cls] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(np.sum(labels == cls))
            }
            
        return per_class
        
    def _compute_confusion_matrix(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> np.ndarray:
        predictions = predictions.numpy()
        labels = labels.numpy()
        
        conf_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        
        for pred, label in zip(predictions, labels):
            conf_matrix[label, pred] += 1
            
        return conf_matrix

class MovingAverage:
    
    def __init__(self, alpha: float = 0.1):

        self.alpha = alpha
        self.value: Optional[float] = None
        
    def update(self, value: float) -> float:

        if self.value is None:
            self.value = value
        else:
            self.value = self.alpha * value + (1 - self.alpha) * self.value
        return self.value
        
    def reset(self) -> None:
        """Reset moving average."""
        self.value = None

class TrainingHistory:
    
    def __init__(self):
        self.train_metrics: List[Dict[str, float]] = []
        self.val_metrics: List[Dict[str, float]] = []
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        
    def add_epoch(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None
    ) -> None:

        self.train_metrics.append(train_metrics)
        
        if val_metrics:
            self.val_metrics.append(val_metrics)
            
            # Track best
            if val_metrics.get("accuracy", 0) > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics["accuracy"]
                self.best_epoch = len(self.val_metrics) - 1
                
    def get_metric_history(
        self,
        metric_name: str,
        split: str = "train"
    ) -> List[float]:

        metrics = self.train_metrics if split == "train" else self.val_metrics
        return [m.get(metric_name, 0.0) for m in metrics]
        
    def save(self, path: Path) -> None:

        data = {
            "train": self.train_metrics,
            "val": self.val_metrics,
            "best_val_accuracy": self.best_val_accuracy,
            "best_epoch": self.best_epoch
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
            
    @classmethod
    def load(cls, path: Path) -> "TrainingHistory":

        with open(path, 'r') as f:
            data = json.load(f)
            
        history = cls()
        history.train_metrics = data["train"]
        history.val_metrics = data["val"]
        history.best_val_accuracy = data["best_val_accuracy"]
        history.best_epoch = data["best_epoch"]
        
        return history