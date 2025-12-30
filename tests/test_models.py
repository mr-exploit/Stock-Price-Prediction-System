"""Tests for prediction models."""

import pytest
import numpy as np
import pandas as pd
import os
import tempfile

from src.models.base_model import StockPredictionModel
from src.models.lstm_model import LSTMPredictor, build_lstm_model
from src.models.rf_model import RandomForestPredictor
from src.models.arima_model import ARIMAPredictor
from src.exceptions import ModelTrainingError


@pytest.fixture
def sample_training_data():
    """Create sample training data."""
    np.random.seed(42)
    
    # Create sequences (samples, timesteps, features)
    n_samples = 100
    sequence_length = 30
    n_features = 5
    
    X = np.random.randn(n_samples, sequence_length, n_features)
    y = np.random.randn(n_samples)
    
    return X, y


@pytest.fixture
def sample_time_series():
    """Create sample time series for ARIMA."""
    np.random.seed(42)
    return np.cumsum(np.random.randn(200)) + 100


class TestRandomForestPredictor:
    """Tests for Random Forest predictor."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = RandomForestPredictor(n_estimators=50, max_depth=5)
        
        assert model.model_type == "random_forest"
        assert model.n_estimators == 50
        assert model.max_depth == 5
        assert model.is_fitted is False
    
    def test_build_model(self):
        """Test model building."""
        model = RandomForestPredictor()
        model.build_model()
        
        assert model.model is not None
    
    def test_train(self, sample_training_data):
        """Test model training."""
        X, y = sample_training_data
        
        model = RandomForestPredictor(n_estimators=10)
        model.build_model()
        history = model.train(X, y)
        
        assert model.is_fitted is True
        assert "train_mse" in history
        assert "train_mae" in history
    
    def test_predict(self, sample_training_data):
        """Test model prediction."""
        X, y = sample_training_data
        
        model = RandomForestPredictor(n_estimators=10)
        model.build_model()
        model.train(X, y)
        
        predictions = model.predict(X[:10])
        
        assert len(predictions) == 10
    
    def test_predict_without_training_raises_error(self):
        """Test that predicting without training raises error."""
        model = RandomForestPredictor()
        model.build_model()
        
        with pytest.raises(ValueError):
            model.predict(np.random.randn(10, 30, 5))
    
    def test_feature_importances(self, sample_training_data):
        """Test feature importance extraction."""
        X, y = sample_training_data
        
        model = RandomForestPredictor(n_estimators=10)
        model.build_model()
        model.train(X, y)
        
        importances = model.get_feature_importances()
        
        assert len(importances) == X.shape[1] * X.shape[2]  # Flattened features
    
    def test_save_and_load(self, sample_training_data):
        """Test model saving and loading."""
        X, y = sample_training_data
        
        model = RandomForestPredictor(n_estimators=10)
        model.build_model()
        model.train(X, y)
        
        # Create a temp file and close it before using
        fd, temp_path = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)
        
        try:
            model.save_model(temp_path)
            
            # Load into new model
            new_model = RandomForestPredictor()
            new_model.load_model(temp_path)
            
            # Predictions should be the same
            pred1 = model.predict(X[:5])
            pred2 = new_model.predict(X[:5])
            
            np.testing.assert_array_almost_equal(pred1, pred2)
        finally:
            os.unlink(temp_path)


class TestARIMAPredictor:
    """Tests for ARIMA predictor."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = ARIMAPredictor(order=(2, 1, 1))
        
        assert model.model_type == "arima"
        assert model.order == (2, 1, 1)
        assert model.is_fitted is False
    
    def test_train(self, sample_time_series):
        """Test model training."""
        model = ARIMAPredictor(order=(2, 1, 0))
        model.build_model()
        history = model.train(None, sample_time_series)
        
        assert model.is_fitted is True
        assert "aic" in history
        assert "bic" in history
    
    def test_forecast(self, sample_time_series):
        """Test forecasting."""
        model = ARIMAPredictor(order=(2, 1, 0))
        model.build_model()
        model.train(None, sample_time_series)
        
        forecast = model.forecast(steps=10)
        
        assert len(forecast) == 10
    
    def test_predict_with_confidence(self, sample_time_series):
        """Test prediction with confidence intervals."""
        model = ARIMAPredictor(order=(2, 1, 0))
        model.build_model()
        model.train(None, sample_time_series)
        
        forecast, lower, upper = model.predict_with_confidence(steps=10)
        
        assert len(forecast) == 10
        assert len(lower) == 10
        assert len(upper) == 10
        # Lower bound should be less than upper bound
        assert (lower < upper).all()


class TestLSTMPredictor:
    """Tests for LSTM predictor (requires TensorFlow)."""
    
    @pytest.fixture(autouse=True)
    def check_tensorflow(self):
        """Check if TensorFlow is available."""
        try:
            import tensorflow
        except ImportError:
            pytest.skip("TensorFlow not available")
    
    def test_initialization(self):
        """Test model initialization."""
        model = LSTMPredictor(units=(50, 50), dropout=0.2)
        
        assert model.model_type == "lstm"
        assert model.units == (50, 50)
        assert model.dropout == 0.2
    
    def test_build_model(self):
        """Test model building."""
        model = LSTMPredictor()
        model.build_model(input_shape=(30, 5))
        
        assert model.model is not None
    
    def test_train(self, sample_training_data):
        """Test model training."""
        X, y = sample_training_data
        
        model = LSTMPredictor()
        model.build_model(input_shape=(X.shape[1], X.shape[2]))
        history = model.train(X, y, epochs=2, verbose=0)
        
        assert model.is_fitted is True
        assert "loss" in history
    
    def test_predict(self, sample_training_data):
        """Test model prediction."""
        X, y = sample_training_data
        
        model = LSTMPredictor()
        model.build_model(input_shape=(X.shape[1], X.shape[2]))
        model.train(X, y, epochs=2, verbose=0)
        
        predictions = model.predict(X[:10])
        
        assert len(predictions) == 10
    
    def test_predict_future(self, sample_training_data):
        """Test future prediction."""
        X, y = sample_training_data
        
        model = LSTMPredictor()
        model.build_model(input_shape=(X.shape[1], X.shape[2]))
        model.train(X, y, epochs=2, verbose=0)
        
        last_sequence = X[-1]
        future = model.predict_future(last_sequence, n_days=5)
        
        assert len(future) == 5


class TestBuildLSTMModel:
    """Tests for build_lstm_model helper function."""
    
    @pytest.fixture(autouse=True)
    def check_tensorflow(self):
        """Check if TensorFlow is available."""
        try:
            import tensorflow
        except ImportError:
            pytest.skip("TensorFlow not available")
    
    def test_build_lstm_model(self):
        """Test the helper function."""
        model = build_lstm_model(sequence_length=30, n_features=5)
        
        assert isinstance(model, LSTMPredictor)
        assert model.model is not None
