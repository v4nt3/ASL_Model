import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, List, Any, Callable
import logging
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
import shutil

logger = logging.getLogger(__name__)

@dataclass
class CallbackContext:
    epoch: int
    step: int
    train_metrics: Optional[Dict[str, float]] = None
    val_metrics: Optional[Dict[str, float]] = None
    model: Optional[nn.Module] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    scheduler: Optional[Any] = None


class Callback(ABC):
    def on_train_begin(self, context: CallbackContext) -> None:
        """Called at the beginning of training."""
        pass
        
    def on_train_end(self, context: CallbackContext) -> None:
        """Called at the end of training."""
        pass
        
    def on_epoch_begin(self, context: CallbackContext) -> None:
        """Called at the beginning of each epoch."""
        pass
        
    def on_epoch_end(self, context: CallbackContext) -> bool:
        """
        Called at the end of each epoch.
        """
        return True
        
    def on_batch_begin(self, context: CallbackContext) -> None:
        """Called before each batch."""
        pass
        
    def on_batch_end(self, context: CallbackContext) -> None:
        """Called after each batch."""
        pass


class EarlyStopping(Callback):
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        monitor: str = "accuracy",
        mode: str = "max"
    ):

        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        
        self.best_value: Optional[float] = None
        self.counter = 0
        self.should_stop = False
        
    def _is_improvement(self, current: float, best: float) -> bool:
        if self.mode == "max":
            return current > best + self.min_delta
        else:
            return current < best - self.min_delta
            
    def on_train_begin(self, context: CallbackContext) -> None:
        self.best_value = None
        self.counter = 0
        self.should_stop = False
        
    def on_epoch_end(self, context: CallbackContext) -> bool:

        if context.val_metrics is None:
            return True
            
        current = context.val_metrics.get(self.monitor)
        
        if current is None:
            logger.warning(f"Early stopping metric '{self.monitor}' not found")
            return True
            
        if self.best_value is None:
            self.best_value = current
            return True
            
        if self._is_improvement(current, self.best_value):
            self.best_value = current
            self.counter = 0
            logger.info(
                f"EarlyStopping: {self.monitor} improved to {current:.4f}"
            )
        else:
            self.counter += 1
            logger.info(
                f"EarlyStopping: No improvement for {self.counter}/{self.patience} epochs"
            )
            
            if self.counter >= self.patience:
                logger.info(
                    f"EarlyStopping triggered after {self.patience} epochs without improvement"
                )
                self.should_stop = True
                return False
                
        return True


class ModelCheckpoint(Callback):
    
    def __init__(
        self,
        save_dir: str,
        monitor: str = "accuracy",
        mode: str = "max",
        save_top_k: int = 3,
        save_every_n_epochs: int = 5,
        filename_format: str = "checkpoint_epoch{epoch:03d}_{monitor}_{value:.4f}.pt"
    ):

        self.save_dir = Path(save_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_every_n_epochs = save_every_n_epochs
        self.filename_format = filename_format
        
        self.checkpoints: List[tuple] = []
        self.best_value: Optional[float] = None
        self.best_path: Optional[Path] = None
        
    def _is_improvement(self, current: float, best: float) -> bool:
        if self.mode == "max":
            return current > best
        else:
            return current < best
            
    def on_train_begin(self, context: CallbackContext) -> None:
        """Create save directory."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints = []
        self.best_value = None
        self.best_path = None
        
    def on_epoch_end(self, context: CallbackContext) -> bool:

        if context.val_metrics is None or context.model is None:
            return True
            
        current = context.val_metrics.get(self.monitor)
        
        if current is None:
            return True
            
        is_best = False
        if self.best_value is None or self._is_improvement(current, self.best_value):
            is_best = True
            self.best_value = current
            
        # Save checkpoint
        filename = self.filename_format.format(
            epoch=context.epoch,
            monitor=self.monitor,
            value=current
        )
        filepath = self.save_dir / filename
        
        self._save_checkpoint(context, filepath, is_best)
        
        # Update best path
        if is_best:
            self.best_path = filepath
            logger.info(f"New best model saved: {filepath}")
            
            # Also save as 'best_model.pt'
            best_path = self.save_dir / "best_model.pt"
            shutil.copy(filepath, best_path)
            
        self.checkpoints.append((current, filepath))
        
        # Periodic save
        if context.epoch % self.save_every_n_epochs == 0:
            periodic_path = self.save_dir / f"checkpoint_epoch{context.epoch:03d}.pt"
            if not periodic_path.exists():
                shutil.copy(filepath, periodic_path)
                logger.info(f"Periodic checkpoint saved: {periodic_path}")
                
        return True
        
    def _save_checkpoint(
        self,
        context: CallbackContext,
        filepath: Path,
        is_best: bool
    ) -> None:
        checkpoint = {
            "epoch": context.epoch,
            "step": context.step,
            "model_state_dict": context.model.state_dict(),
            "train_metrics": context.train_metrics,
            "val_metrics": context.val_metrics,
            "is_best": is_best
        }
        
        if context.optimizer is not None:
            checkpoint["optimizer_state_dict"] = context.optimizer.state_dict()
            
        if context.scheduler is not None:
            checkpoint["scheduler_state_dict"] = context.scheduler.state_dict()
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, filepath)
        
    def _cleanup_checkpoints(self) -> None:
        if len(self.checkpoints) <= self.save_top_k:
            return
        if self.mode == "max":
            self.checkpoints.sort(key=lambda x: x[0], reverse=True)
        else:
            self.checkpoints.sort(key=lambda x: x[0])
        to_remove = self.checkpoints[self.save_top_k:]
        self.checkpoints = self.checkpoints[:self.save_top_k]
        for _, path in to_remove:
            try:
                if path.exists() and path != self.best_path:
                    path.unlink()
                    logger.debug(f"Removed checkpoint: {path}")
            except OSError as e:
                logger.warning(f"Could not remove checkpoint {path}: {e}")
                
    @staticmethod
    def load_checkpoint(
        path: Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None
    ) -> Dict[str, Any]:

        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
        logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
        
        return checkpoint


class LearningRateLogger(Callback):

    def __init__(self):
        self.history: List[tuple] = []
        
    def on_batch_end(self, context: CallbackContext) -> None:
        if context.optimizer is not None:
            lr = context.optimizer.param_groups[0]['lr']
            self.history.append((context.step, lr))
            
    def on_epoch_end(self, context: CallbackContext) -> bool:
        if context.optimizer is not None:
            lr = context.optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {context.epoch} - Learning rate: {lr:.2e}")
        return True
        
    def get_history(self) -> List[tuple]:
        return self.history


class CallbackHandler:
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):

        self.callbacks = callbacks or []
        
    def add_callback(self, callback: Callback) -> None:
        self.callbacks.append(callback)
        
    def on_train_begin(self, context: CallbackContext) -> None:
        for callback in self.callbacks:
            callback.on_train_begin(context)
            
    def on_train_end(self, context: CallbackContext) -> None:
        for callback in self.callbacks:
            callback.on_train_end(context)
            
    def on_epoch_begin(self, context: CallbackContext) -> None:
        for callback in self.callbacks:
            callback.on_epoch_begin(context)
            
    def on_epoch_end(self, context: CallbackContext) -> bool:
        continue_training = True
        for callback in self.callbacks:
            result = callback.on_epoch_end(context)
            if result is False:
                continue_training = False
        return continue_training
        
    def on_batch_begin(self, context: CallbackContext) -> None:
        for callback in self.callbacks:
            callback.on_batch_begin(context)
            
    def on_batch_end(self, context: CallbackContext) -> None:
        for callback in self.callbacks:
            callback.on_batch_end(context)