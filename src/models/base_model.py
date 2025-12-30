"""Base model class for stock price prediction."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any
import numpy as np
import joblib
import os

from src.utils.logger import get_logger

logger = get_logger(__name__)


class StockPredictionModel(ABC):
    """
    Base class for stock prediction models.
    
    Attributes:
        model_type: Type of the model
        model: The underlying model object
        history: Training history (for neural networks)
        is_fitted: Whether the model has been trained
    """
    
    def __init__(self, model_type: str = "base"):
        """
        Initialize the model.
        
        Args:
            model_type: Type identifier for the model
        """
        self.model_type = model_type
        self.model = None
        self.history = None
        self.is_fitted = False
    
    @abstractmethod
    def build_model(self, input_shape: Tuple) -> None:
        """
        Build the model architecture.
        
        Args:
            input_shape: Shape of input data
        """
        pass
    
    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **kwargs: Additional training parameters
        
        Returns:
            Training history dictionary
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
        
        Returns:
            Predictions array
        """
        pass
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save based on model type
        if self.model_type in ["random_forest", "xgboost"]:
            joblib.dump(self.model, filepath)
        else:
            # For Keras models, use .h5 format
            if hasattr(self.model, "save"):
                self.model.save(filepath)
            else:
                joblib.dump(self.model, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load based on file extension
        if filepath.endswith(".h5"):
            # Keras model
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(filepath)
            except ImportError:
                raise ImportError("TensorFlow is required to load .h5 models")
        else:
            self.model = joblib.load(filepath)
        
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        return {
            "model_type": self.model_type,
            "is_fitted": self.is_fitted,
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_type='{self.model_type}', is_fitted={self.is_fitted})"
