"""LSTM model for stock price prediction."""

import numpy as np
from typing import Dict, Optional, Tuple

from src.models.base_model import StockPredictionModel
from src.utils.logger import get_logger
from src.exceptions import ModelTrainingError

logger = get_logger(__name__)


class LSTMPredictor(StockPredictionModel):
    """
    LSTM-based stock price predictor.
    
    Architecture:
        - Input Layer: (sequence_length, n_features)
        - LSTM Layer 1: 50 units, return_sequences=True
        - Dropout: 0.2
        - LSTM Layer 2: 50 units, return_sequences=False
        - Dropout: 0.2
        - Dense Layer 1: 25 units, activation='relu'
        - Output Layer: 1 unit (predicted price)
    """
    
    def __init__(
        self,
        units: Tuple[int, int] = (50, 50),
        dropout: float = 0.2,
        learning_rate: float = 0.001,
    ):
        """
        Initialize LSTM predictor.
        
        Args:
            units: Number of units for each LSTM layer
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
        """
        super().__init__(model_type="lstm")
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
    
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build the LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam
        except ImportError:
            raise ImportError("TensorFlow is required for LSTM model. Install with: pip install tensorflow")
        
        model = Sequential([
            # First LSTM layer
            LSTM(
                units=self.units[0],
                return_sequences=True,
                input_shape=input_shape
            ),
            Dropout(self.dropout),
            
            # Second LSTM layer
            LSTM(
                units=self.units[1],
                return_sequences=False
            ),
            Dropout(self.dropout),
            
            # Dense layers
            Dense(25, activation="relu"),
            Dense(1)  # Output layer
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="mean_squared_error",
            metrics=["mae", "mape"]
        )
        
        self.model = model
        logger.info(f"Built LSTM model with input shape {input_shape}")
        logger.info(f"Model summary: {model.summary()}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1,
        early_stopping: bool = True,
        patience: int = 10,
    ) -> Dict:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level
            early_stopping: Whether to use early stopping
            patience: Patience for early stopping
        
        Returns:
            Training history dictionary
        """
        if self.model is None:
            raise ModelTrainingError("Model not built. Call build_model() first.")
        
        try:
            from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        except ImportError:
            raise ImportError("TensorFlow is required for LSTM model.")
        
        callbacks = []
        
        if early_stopping:
            early_stop = EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=patience,
                restore_best_weights=True
            )
            callbacks.append(early_stop)
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        try:
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=verbose
            )
            
            self.history = history.history
            self.is_fitted = True
            
            logger.info(f"Training completed. Final loss: {history.history['loss'][-1]:.6f}")
            
            return self.history
        
        except Exception as e:
            raise ModelTrainingError(f"Training failed: {str(e)}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input sequences
        
        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def predict_future(
        self,
        last_sequence: np.ndarray,
        n_days: int,
        scaler=None,
    ) -> np.ndarray:
        """
        Predict future stock prices.
        
        Args:
            last_sequence: Most recent data sequence
            n_days: Number of days to predict
            scaler: Scaler for inverse transformation (optional)
        
        Returns:
            Array of predicted future prices
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(n_days):
            # Reshape for prediction
            X = current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1])
            
            # Predict next value
            pred = self.model.predict(X, verbose=0)[0, 0]
            predictions.append(pred)
            
            # Update sequence: shift and add new prediction
            new_row = current_sequence[-1].copy()
            new_row[0] = pred  # Update close price (first feature)
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        return np.array(predictions)
    
    def get_params(self) -> Dict:
        """Get model parameters."""
        params = super().get_params()
        params.update({
            "units": self.units,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
        })
        return params


def build_lstm_model(sequence_length: int, n_features: int) -> "LSTMPredictor":
    """
    Build a default LSTM model.
    
    Args:
        sequence_length: Length of input sequences
        n_features: Number of features
    
    Returns:
        Built LSTMPredictor instance
    """
    predictor = LSTMPredictor()
    predictor.build_model(input_shape=(sequence_length, n_features))
    return predictor
