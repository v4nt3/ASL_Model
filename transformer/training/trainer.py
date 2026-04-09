import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Tuple, Any
from pathlib import Path
import logging
import time
import math
from tqdm import tqdm
from transformer.model.transformer import create_model

from transformer.core.config import Config, FeatureType
from transformer.core.exceptions import TrainingError
from transformer.data_.dataset import BatchData
from transformer.training.metrics import MetricsTracker, TrainingHistory, MovingAverage
from transformer.training.callbacks import (
    Callback, CallbackHandler, CallbackContext,
    EarlyStopping, ModelCheckpoint, LearningRateLogger
)

logger = logging.getLogger(__name__)

class Trainer:
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None,
        callbacks: Optional[List[Callback]] = None,
        device: Optional[torch.device] = None,
        resume_from: Optional[Path] = None
    ):
        self.model = model
        self.config = config
        self.train_config = config.training
        self.feature_type = config.model.feature_type
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        
        self.criterion = self._create_criterion(class_weights)
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        self.scaler = GradScaler() if self.train_config.use_amp else None
        
        default_callbacks = self._create_default_callbacks()
        all_callbacks = default_callbacks + (callbacks or [])
        self.callback_handler = CallbackHandler(all_callbacks)
        
        self.train_metrics = MetricsTracker(config.data.num_classes)
        self.val_metrics = MetricsTracker(config.data.num_classes)
        self.history = TrainingHistory()
        
        self.current_epoch = 0
        self.global_step = 0
        self.loss_avg = MovingAverage(alpha=0.1)
        
        if resume_from is not None:
            self._resume_training(resume_from)
        
        self._log_setup()
    
    def _forward_batch(self, batch: BatchData):

        if self.feature_type == FeatureType.POSE:
            output = self.model(
                pose_features=batch.pose_features,
                attention_mask=batch.attention_mask
            )
        elif self.feature_type == FeatureType.MULTIMODAL:
            output = self.model(
                visual_features=batch.visual_features,
                pose_features=batch.pose_features,
                attention_mask=batch.attention_mask
            )
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")
        
        return output
    
    def _create_criterion(self, class_weights):
        weight = None
        if self.train_config.use_class_weights and class_weights is not None:
            weight = class_weights.to(self.device)
            logger.info("Using class-weighted cross entropy loss")
        return nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=self.train_config.label_smoothing
        )
    
    def _create_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
        param_groups = [
            {
                'params': [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': self.train_config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0
            }
        ]
        
        opt_type = self.train_config.optimizer.value
        if opt_type == "adamw":
            return optim.AdamW(param_groups, lr=self.train_config.learning_rate, betas=(0.9, 0.999))
        elif opt_type == "adam":
            return optim.Adam(param_groups, lr=self.train_config.learning_rate, betas=(0.9, 0.999))
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")
    
    def _create_scheduler(self):
        num_training_steps = len(self.train_loader) * self.train_config.num_epochs
        num_warmup_steps = int(num_training_steps * self.train_config.warmup_ratio)
        
        scheduler_type = self.train_config.scheduler.value
        
        if scheduler_type == "cosine_warmup":
            def lr_lambda(step):
                if step < num_warmup_steps:
                    return float(step) / float(max(1, num_warmup_steps))
                progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                return max(
                    self.train_config.min_learning_rate / self.train_config.learning_rate,
                    0.5 * (1.0 + math.cos(math.pi * progress))
                )
            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        elif scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=num_training_steps,
                eta_min=self.train_config.min_learning_rate
            )
        elif scheduler_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=5,
                min_lr=self.train_config.min_learning_rate
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    def _create_default_callbacks(self):
        return [
            EarlyStopping(
                patience=self.train_config.early_stopping_patience,
                min_delta=self.train_config.early_stopping_min_delta,
                monitor="accuracy", mode="max"
            ),
            ModelCheckpoint(
                save_dir=self.train_config.checkpoint_dir,
                monitor="accuracy", mode="max",
                save_top_k=self.train_config.save_top_k,
                save_every_n_epochs=self.train_config.save_every_n_epochs
            ),
            LearningRateLogger()
        ]
    
    def _resume_training(self, checkpoint_path):
        checkpoint = ModelCheckpoint.load_checkpoint(
            checkpoint_path, self.model, self.optimizer, self.scheduler
        )
        self.current_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["step"]
        logger.info(f"Resumed from epoch {self.current_epoch}")
    
    def _log_setup(self):
        logger.info(
            f"\nTraining Setup ({self.feature_type.value} mode):\n"
            f"  Device: {self.device}\n"
            f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}\n"
            f"  Training samples: {len(self.train_loader.dataset)}\n"
            f"  Validation samples: {len(self.val_loader.dataset)}\n"
            f"  Batch size: {self.train_config.batch_size}\n"
            f"  Effective batch size: {self.train_config.effective_batch_size}\n"
            f"  Mixed precision: {self.train_config.use_amp}\n"
            f"  Feature type: {self.feature_type.value}\n"
            f"  Epochs: {self.train_config.num_epochs}\n"
        )
    
    def train(self) -> TrainingHistory:
        logger.info(f"Starting training ({self.feature_type.value} mode)")
        
        context = CallbackContext(
            epoch=0, step=self.global_step,
            model=self.model, optimizer=self.optimizer, scheduler=self.scheduler
        )
        self.callback_handler.on_train_begin(context)
        
        try:
            for epoch in range(self.current_epoch, self.train_config.num_epochs):
                self.current_epoch = epoch
                
                train_metrics = self._train_epoch(epoch)
                val_metrics = self._validate(epoch)
                
                self.history.add_epoch(
                    train_metrics.to_dict(),
                    val_metrics.to_dict()
                )
                
                self._log_epoch_metrics(epoch, train_metrics, val_metrics)
                
                context = CallbackContext(
                    epoch=epoch, step=self.global_step,
                    train_metrics=train_metrics.to_dict(),
                    val_metrics=val_metrics.to_dict(),
                    model=self.model, optimizer=self.optimizer,
                    scheduler=self.scheduler
                )
                
                if not self.callback_handler.on_epoch_end(context):
                    logger.info("Training stopped by callback")
                    break
                    
        except KeyboardInterrupt:
            logger.warning("Training interrupted")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise TrainingError("Training failed", details={"error": str(e)})
        finally:
            self.callback_handler.on_train_end(context)
        
        logger.info(
            f"\nTraining completed!\n"
            f"  Best val accuracy: {self.history.best_val_accuracy:.4f}\n"
            f"  Best epoch: {self.history.best_epoch}"
        )
        
        return self.history
    
    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        self.optimizer.zero_grad()
        
        accumulation_steps = self.train_config.gradient_accumulation_steps
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.train_config.num_epochs}",
            leave=True
        )
        
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)
            
            with autocast(enabled=self.train_config.use_amp):
                output = self._forward_batch(batch)
                loss = self.criterion(output.logits, batch.labels)
                loss = loss / accumulation_steps
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config.max_grad_norm
                )
                
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                if not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            actual_loss = loss.item() * accumulation_steps
            self.train_metrics.update(output.logits.detach(), batch.labels, actual_loss)
            
            smoothed_loss = self.loss_avg.update(actual_loss)
            pbar.set_postfix({
                "loss": f"{smoothed_loss:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        return self.train_metrics.compute()
    
    @torch.no_grad()
    def _validate(self, epoch):
        self.model.eval()
        self.val_metrics.reset()
        
        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            batch = batch.to(self.device)
            
            with autocast(enabled=self.train_config.use_amp):
                output = self._forward_batch(batch)
                loss = self.criterion(output.logits, batch.labels)
            
            self.val_metrics.update(output.logits, batch.labels, loss.item())
        
        metrics = self.val_metrics.compute(compute_per_class=True)
        
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(metrics.accuracy)
        
        return metrics
    
    def _log_epoch_metrics(self, epoch, train_metrics, val_metrics):
        logger.info(
            f"\nEpoch {epoch + 1} Summary:\n"
            f"Train - Loss: {train_metrics.loss:.4f}, "
            f"Acc: {train_metrics.accuracy:.4f}, "
            f"Top-5: {train_metrics.top5_accuracy:.4f}\n"
            f"Val - Loss: {val_metrics.loss:.4f}, "
            f"Acc: {val_metrics.accuracy:.4f}, "
            f"Top-5: {val_metrics.top5_accuracy:.4f}, "
            f"Top-10: {val_metrics.top10_accuracy:.4f}"
        )


def train_model(
    config: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: Optional[torch.Tensor] = None,
    resume_from: Optional[Path] = None
) -> Tuple[nn.Module, TrainingHistory]:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(config.model, config.data, device)
    
    trainer = Trainer(
        model=model, config=config,
        train_loader=train_loader, val_loader=val_loader,
        class_weights=class_weights, resume_from=resume_from
    )
    
    history = trainer.train()
    return model, history
