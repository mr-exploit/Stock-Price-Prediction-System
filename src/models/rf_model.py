"""Random Forest model for stock price prediction."""

import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor

from src.models.base_model import StockPredictionModel
from src.utils.logger import get_logger
from src.exceptions import ModelTrainingError

logger = get_logger(__name__)


class RandomForestPredictor(StockPredictionModel):
    """
    Random Forest-based stock price predictor.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
    ):
        """
        Initialize Random Forest predictor.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            min_samples_split: Minimum samples required to split an internal node
            min_samples_leaf: Minimum samples required to be at a leaf node
            random_state: Random seed for reproducibility
        """
        super().__init__(model_type="random_forest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
    
    def build_model(self, input_shape: Tuple = None) -> None:
        """
        Build the Random Forest model.
        
        Args:
            input_shape: Not used for Random Forest, kept for interface consistency
        """
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,  # Use all available cores
        )
        logger.info(f"Built Random Forest model with {self.n_estimators} estimators")
    
    def _prepare_data(self, X: np.ndarray) -> np.ndarray:
        """
        Prepare data for Random Forest (flatten sequences).
        
        Args:
            X: Input data (can be 3D for sequences or 2D for features)
        
        Returns:
            2D array suitable for Random Forest
        """
        if len(X.shape) == 3:
            # Flatten sequences: (n_samples, sequence_length, n_features) -> (n_samples, sequence_length * n_features)
            return X.reshape(X.shape[0], -1)
        return X
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict:
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (not used for RF, kept for consistency)
            y_val: Validation targets (not used for RF, kept for consistency)
        
        Returns:
            Training metrics dictionary
        """
        if self.model is None:
            self.build_model()
        
        try:
            # Prepare data
            X_train_flat = self._prepare_data(X_train)
            
            # Train model
            self.model.fit(X_train_flat, y_train)
            self.is_fitted = True
            
            # Calculate training metrics
            train_predictions = self.model.predict(X_train_flat)
            train_mse = np.mean((train_predictions - y_train) ** 2)
            train_mae = np.mean(np.abs(train_predictions - y_train))
            
            self.history = {
                "train_mse": [train_mse],
                "train_mae": [train_mae],
            }
            
            # Validation metrics if provided
            if X_val is not None and y_val is not None:
                X_val_flat = self._prepare_data(X_val)
                val_predictions = self.model.predict(X_val_flat)
                val_mse = np.mean((val_predictions - y_val) ** 2)
                val_mae = np.mean(np.abs(val_predictions - y_val))
                self.history["val_mse"] = [val_mse]
                self.history["val_mae"] = [val_mae]
            
            logger.info(f"Training completed. Train MSE: {train_mse:.6f}, Train MAE: {train_mae:.6f}")
            
            return self.history
        
        except Exception as e:
            raise ModelTrainingError(f"Training failed: {str(e)}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
        
        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        X_flat = self._prepare_data(X)
        return self.model.predict(X_flat)
    
    def get_feature_importances(self) -> np.ndarray:
        """
        Get feature importances from the trained model.
        
        Returns:
            Array of feature importances
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.feature_importances_
    
    def get_params(self) -> Dict:
        """Get model parameters."""
        params = super().get_params()
        params.update({
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
        })
        return params
