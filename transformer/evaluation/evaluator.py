import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

from transformer.core.config import Config
from transformer.model.transformer import SignLanguageTransformer, create_model
from transformer.training.metrics import MetricsTracker, MetricsResult
from transformer.training.callbacks import ModelCheckpoint
    


logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    overall_metrics: Dict[str, float]
    per_class_metrics: Dict[int, Dict[str, float]]
    error_analysis: Dict[str, Any]
    model_info: Dict[str, Any]
    dataset_info: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
        
    def save(self, path: Path) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class Evaluator:
    
    def __init__(
        self,
        model: SignLanguageTransformer,
        dataloader: DataLoader,
        config: Config,
        device: Optional[torch.device] = None,
        label_map: Optional[Dict[int, str]] = None
    ):

        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.label_map = label_map or {}
        
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Storage for predictions
        self.all_predictions: List[torch.Tensor] = []
        self.all_labels: List[torch.Tensor] = []
        self.all_probs: List[torch.Tensor] = []
        self.sample_ids: List[str] = []
        
        logger.info(
            f"Evaluator initialized: device={self.device}, "
            f"samples={len(dataloader.dataset)}"
        )
        
    @torch.no_grad()
    def evaluate(
        self,
        compute_confidence: bool = True
    ) -> EvaluationReport:

        logger.info("Starting evaluation...")
        
        # Reset storage
        self.all_predictions = []
        self.all_labels = []
        self.all_probs = []
        self.sample_ids = []
        
        # Metrics tracker
        metrics_tracker = MetricsTracker(self.config.data.num_classes)
        
        # Run inference
        for batch in tqdm(self.dataloader, desc="Evaluating"):
            batch = batch.to(self.device)
            
            with autocast(enabled=self.config.training.use_amp):
                output = self.model(
                    batch.visual_features,
                    batch.pose_features,
                    batch.attention_mask
                )
                
            # Get predictions
            probs = torch.softmax(output.logits, dim=-1)
            predictions = output.logits.argmax(dim=-1)
            
            # Store
            self.all_predictions.append(predictions.cpu())
            self.all_labels.append(batch.labels.cpu())
            self.all_probs.append(probs.cpu())
            
            if batch.sample_ids:
                self.sample_ids.extend(batch.sample_ids)
                
            # Update metrics
            metrics_tracker.update(output.logits.detach(), batch.labels)
            
        # Compute metrics
        metrics_result = metrics_tracker.compute(
            compute_per_class=True,
            compute_confusion_matrix=True
        )
        
        # Concatenate all results
        all_preds = torch.cat(self.all_predictions)
        all_labels = torch.cat(self.all_labels)
        all_probs = torch.cat(self.all_probs)
        
        # Error analysis
        error_analysis = self._analyze_errors(all_preds, all_labels, all_probs)
        
        # Build report
        report = EvaluationReport(
            overall_metrics=metrics_result.to_dict(),
            per_class_metrics=metrics_result.per_class_metrics or {},
            error_analysis=error_analysis,
            model_info=self._get_model_info(),
            dataset_info=self._get_dataset_info()
        )
        
        logger.info(
            f"Evaluation complete: "
            f"accuracy={metrics_result.accuracy:.4f}, "
            f"top5={metrics_result.top5_accuracy:.4f}"
        )
        
        return report
        
    def _analyze_errors(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        probs: torch.Tensor
    ) -> Dict[str, Any]:

        predictions = predictions.numpy()
        labels = labels.numpy()
        probs = probs.numpy()
        
        # Find errors
        errors = predictions != labels
        error_indices = np.where(errors)[0]
        
        # Get confidences for errors
        error_confidences = probs[error_indices, predictions[error_indices]]
        correct_confidences = probs[~errors, predictions[~errors]]
        
        # Most confused pairs
        confusion_pairs = defaultdict(int)
        for idx in error_indices:
            true_label = labels[idx]
            pred_label = predictions[idx]
            confusion_pairs[(int(true_label), int(pred_label))] += 1
            
        # Sort by frequency
        top_confusions = sorted(
            confusion_pairs.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        # High confidence errors (most concerning)
        high_conf_errors = error_indices[error_confidences > 0.8]
        
        analysis = {
            "total_samples": len(labels),
            "total_errors": len(error_indices),
            "error_rate": len(error_indices) / len(labels),
            "mean_error_confidence": float(error_confidences.mean()) if len(error_confidences) > 0 else 0,
            "mean_correct_confidence": float(correct_confidences.mean()) if len(correct_confidences) > 0 else 0,
            "high_confidence_errors": len(high_conf_errors),
            "top_confusion_pairs": [
                {
                    "true": int(pair[0]),
                    "predicted": int(pair[1]),
                    "true_name": self.label_map.get(pair[0], str(pair[0])),
                    "pred_name": self.label_map.get(pair[1], str(pair[1])),
                    "count": count
                }
                for pair, count in top_confusions
            ]
        }
        
        return analysis
        
    def _get_model_info(self) -> Dict[str, Any]:
        num_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        
        return {
            "architecture": self.model.__class__.__name__,
            "total_parameters": num_params,
            "trainable_parameters": trainable_params,
            "hidden_dim": self.model.d_model,
            "num_layers": self.config.model.num_layers,
            "num_heads": self.config.model.num_heads,
            "num_classes": self.model.num_classes
        }
        
    def _get_dataset_info(self) -> Dict[str, Any]:
        return {
            "num_samples": len(self.dataloader.dataset),
            "num_classes": self.config.data.num_classes,
            "max_seq_length": self.config.data.max_seq_length,
            "visual_feature_dim": self.config.data.visual_feature_dim,
            "pose_feature_dim": self.config.data.pose_feature_dim
        }
        
    def get_predictions_dataframe(self) -> Any:

            
        all_preds = torch.cat(self.all_predictions).numpy()
        all_labels = torch.cat(self.all_labels).numpy()
        all_probs = torch.cat(self.all_probs).numpy()
        
        df = pd.DataFrame({
            "sample_id": self.sample_ids if self.sample_ids else range(len(all_preds)),
            "true_label": all_labels,
            "predicted_label": all_preds,
            "confidence": all_probs[np.arange(len(all_preds)), all_preds],
            "correct": all_preds == all_labels
        })
        
        # Add label names if available
        if self.label_map:
            df["true_name"] = df["true_label"].map(self.label_map)
            df["predicted_name"] = df["predicted_label"].map(self.label_map)
            
        return df
        
    def _get_predictions_dict(self) -> Dict[str, List]:
        all_preds = torch.cat(self.all_predictions).numpy()
        all_labels = torch.cat(self.all_labels).numpy()
        
        return {
            "sample_ids": self.sample_ids,
            "true_labels": all_labels.tolist(),
            "predicted_labels": all_preds.tolist(),
            "correct": (all_preds == all_labels).tolist()
        }


def evaluate_model(
    model_path: Path,
    config: Config,
    test_loader: DataLoader,
    label_map: Optional[Dict[int, str]] = None
) -> EvaluationReport:

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(config.model, config.data, device)
    
    # Load weights
    ModelCheckpoint.load_checkpoint(model_path, model)
    
    # Evaluate
    evaluator = Evaluator(model, test_loader, config, device, label_map)
    return evaluator.evaluate()
