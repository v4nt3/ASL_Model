from typing import Optional, Any, Dict


class SignLanguageRecognitionError(Exception):
    """
    Base exception for all sign language recognition errors.
    
    All custom exceptions inherit from this class, enabling catch-all
    handling when needed while maintaining specific exception types.
    
    Attributes:
        message: Human-readable error description
        details: Additional context about the error
        recovery_hint: Suggested action to resolve the error
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        recovery_hint: Optional[str] = None
    ):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            details: Dictionary with additional error context
            recovery_hint: Suggestion for resolving the error
        """
        self.message = message
        self.details = details or {}
        self.recovery_hint = recovery_hint
        super().__init__(self._format_message())
        
    def _format_message(self) -> str:
        """Format the complete error message."""
        parts = [self.message]
        
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            parts.append(f"Details: {details_str}")
            
        if self.recovery_hint:
            parts.append(f"Hint: {self.recovery_hint}")
            
        return " | ".join(parts)


# Configuration Errors
class ConfigurationError(SignLanguageRecognitionError):
    """
    Raised when there is a configuration-related error.
    
    Examples:
        - Invalid configuration values
        - Missing required configuration
        - Configuration file parsing errors
    """
    pass


# Data Errors
class DataError(SignLanguageRecognitionError):
    """Base class for data-related errors."""
    pass


class DataLoadError(DataError):
    """
    Raised when data cannot be loaded.
    
    Examples:
        - File not found
        - Corrupted data files
        - Permission errors
    """
    pass


class DataValidationError(DataError):
    """
    Raised when data validation fails.
    
    Examples:
        - Invalid data format
        - Missing required fields
        - Data integrity issues
    """
    pass


class FeatureExtractionError(DataError):
    """
    Raised when feature extraction fails.
    
    Examples:
        - Model loading failures
        - GPU memory errors during extraction
        - Invalid input dimensions
    """
    pass


# Model Errors
class ModelError(SignLanguageRecognitionError):
    """Base class for model-related errors."""
    pass


class ModelInitializationError(ModelError):
    """
    Raised when model initialization fails.
    
    Examples:
        - Invalid architecture parameters
        - Weight loading failures
        - Incompatible configurations
    """
    pass


class CheckpointError(ModelError):
    """
    Raised when checkpoint operations fail.
    
    Examples:
        - Checkpoint not found
        - Incompatible checkpoint format
        - Corrupted checkpoint file
    """
    pass


class InferenceError(ModelError):
    """
    Raised when inference fails.
    
    Examples:
        - Invalid input format
        - Model not in evaluation mode
        - Output processing errors
    """
    pass


# Training Errors
class TrainingError(SignLanguageRecognitionError):
    """Base class for training-related errors."""
    pass


class OptimizationError(TrainingError):
    """
    Raised when optimization encounters issues.
    
    Examples:
        - NaN loss values
        - Gradient explosion
        - Invalid learning rate
    """
    pass


class EarlyStoppingError(TrainingError):
    """
    Raised when early stopping criteria are met unexpectedly.
    
    This is typically caught internally and used for flow control,
    but can be raised if early stopping occurs too early.
    """
    pass


class ResourceError(SignLanguageRecognitionError):
    """
    Raised when system resources are insufficient.
    
    Examples:
        - Out of GPU memory
        - Disk space exhausted
        - Too many open files
    """
    pass
