
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union
from logging.handlers import RotatingFileHandler
import json


class StructuredFormatter(logging.Formatter):
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def __init__(
        self, 
        use_json: bool = False, 
        include_extra: bool = True,
        use_colors: bool = True
    ):
        
        super().__init__()
        self.use_json = use_json
        self.include_extra = include_extra
        self.use_colors = use_colors and not use_json
        
    def format(self, record: logging.LogRecord) -> str:
        # Extract extra fields
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord(
                "", 0, "", 0, "", (), None
            ).__dict__ and not key.startswith('_'):
                extra_fields[key] = value
        
        if self.use_json:
            log_data = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
            }
            
            if self.include_extra and extra_fields:
                log_data["extra"] = extra_fields
                
            if record.exc_info:
                log_data["exception"] = self.formatException(record.exc_info)
                
            return json.dumps(log_data, default=str)
        else:
            timestamp = datetime.fromtimestamp(record.created).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            
            level = record.levelname
            if self.use_colors:
                color = self.COLORS.get(level, self.COLORS['RESET'])
                reset = self.COLORS['RESET']
                level = f"{color}{level:8s}{reset}"
            else:
                level = f"{level:8s}"
            
            message = record.getMessage()
            
            # Add extra fields if present
            extra_str = ""
            if self.include_extra and extra_fields:
                extra_str = " | " + " ".join(
                    f"{k}={v}" for k, v in extra_fields.items()
                )
            
            formatted = f"[{timestamp}] {level} {record.name}: {message}{extra_str}"
            
            if record.exc_info:
                formatted += "\n" + self.formatException(record.exc_info)
                
            return formatted


class MetricsLogger:
    """
    logger for training metrics.
    """
    
    def __init__(self, logger: logging.Logger, experiment_name: str):

        self.logger = logger
        self.experiment_name = experiment_name
        self.metrics_history: Dict[str, list] = {
            "train": [],
            "val": [],
            "test": []
        }
        
    def log_epoch(
        self,
        epoch: int,
        phase: str,
        metrics: Dict[str, float],
        extra: Optional[Dict[str, Any]] = None
    ) -> None:

        log_extra = {
            "epoch": epoch,
            "phase": phase,
            **metrics,
            **(extra or {})
        }
        
        # Store in history
        self.metrics_history[phase].append({
            "epoch": epoch,
            **metrics
        })
        
        # Format metrics string
        metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        self.logger.info(
            f"Epoch {epoch} [{phase}] {metrics_str}",
            extra=log_extra
        )
        
    def log_batch(
        self,
        epoch: int,
        batch: int,
        total_batches: int,
        loss: float,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:

        progress = (batch / total_batches) * 100
        metrics_str = f"loss: {loss:.4f}"
        
        if metrics:
            metrics_str += " | " + " | ".join(
                f"{k}: {v:.4f}" for k, v in metrics.items()
            )
            
        self.logger.debug(
            f"Epoch {epoch} [{progress:5.1f}%] {metrics_str}",
            extra={"epoch": epoch, "batch": batch, "loss": loss, **(metrics or {})}
        )
        
    def get_best_metrics(self, phase: str = "val") -> Optional[Dict[str, Any]]:

        if not self.metrics_history[phase]:
            return None
            
        # Assume accuracy is the primary metric
        return max(
            self.metrics_history[phase],
            key=lambda x: x.get("accuracy", x.get("acc", 0))
        )


# Global logger registry
_loggers: Dict[str, logging.Logger] = {}
_metrics_logger: Optional[MetricsLogger] = None


def setup_logger(
    log_dir: Optional[Union[str, Path]] = None,
    experiment_name: Optional[str] = None,
    level: int = logging.INFO,
    console_level: Optional[int] = None,
    file_level: Optional[int] = None,
    use_json_file: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5
) -> logging.Logger:

    global _metrics_logger
    
    # Create root logger for the package
    root_logger = logging.getLogger("sign_language_recognition")
    root_logger.setLevel(logging.DEBUG)  # Capture all, filter at handlers
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level or level)
    console_handler.setFormatter(StructuredFormatter(
        use_json=False,
        use_colors=True
    ))
    root_logger.addHandler(console_handler)
    
    # File handlers
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Generate experiment name if not provided
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Main log file (JSON for parsing)
        main_log_file = log_path / f"{experiment_name}.log"
        file_handler = RotatingFileHandler(
            main_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(file_level or logging.DEBUG)
        file_handler.setFormatter(StructuredFormatter(
            use_json=use_json_file,
            include_extra=True
        ))
        root_logger.addHandler(file_handler)
        
        # Error log file (separate for easy error tracking)
        error_log_file = log_path / f"{experiment_name}_errors.log"
        error_handler = RotatingFileHandler(
            error_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter(
            use_json=True,
            include_extra=True
        ))
        root_logger.addHandler(error_handler)
        
        root_logger.info(
            f"Logging initialized",
            extra={
                "log_dir": str(log_path),
                "experiment_name": experiment_name,
                "log_file": str(main_log_file)
            }
        )
    
    # Setup metrics logger
    _metrics_logger = MetricsLogger(root_logger, experiment_name or "default")
    
    return root_logger


def get_logger(name: str) -> logging.Logger:

    if name not in _loggers:
        # Create child logger under the package namespace
        if not name.startswith("sign_language_recognition"):
            full_name = f"sign_language_recognition.{name}"
        else:
            full_name = name
            
        _loggers[name] = logging.getLogger(full_name)
        
    return _loggers[name]


def get_metrics_logger() -> Optional[MetricsLogger]:
    """
    Get the metrics logger instance.
    
    Returns:
        MetricsLogger instance or None if not initialized
    """
    return _metrics_logger
