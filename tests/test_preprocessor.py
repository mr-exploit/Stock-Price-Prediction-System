"""Tests for data preprocessor module."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.data.preprocessor import (
    calculate_technical_indicators,
    create_lag_features,
    handle_missing_values,
    create_sequences,
    preprocess_data,
    inverse_transform_predictions,
)
from src.exceptions import PreprocessingError


@pytest.fixture
def sample_stock_data():
    """Create sample stock data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=300, freq="D")
    np.random.seed(42)
    
    # Generate realistic price data
    close = 100 + np.cumsum(np.random.randn(300) * 0.5)
    high = close + np.abs(np.random.randn(300))
    low = close - np.abs(np.random.randn(300))
    open_price = close + np.random.randn(300) * 0.3
    volume = np.random.randint(1000000, 10000000, 300)
    
    df = pd.DataFrame({
        "Date": dates,
        "Open": open_price,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })
    
    return df


class TestCalculateTechnicalIndicators:
    """Tests for calculate_technical_indicators function."""
    
    def test_calculates_sma(self, sample_stock_data):
        """Test that SMA indicators are calculated."""
        df = calculate_technical_indicators(sample_stock_data)
        
        assert "SMA_20" in df.columns
        assert "SMA_50" in df.columns
        assert "SMA_200" in df.columns
    
    def test_calculates_ema(self, sample_stock_data):
        """Test that EMA indicators are calculated."""
        df = calculate_technical_indicators(sample_stock_data)
        
        assert "EMA_12" in df.columns
        assert "EMA_26" in df.columns
    
    def test_calculates_macd(self, sample_stock_data):
        """Test that MACD is calculated."""
        df = calculate_technical_indicators(sample_stock_data)
        
        assert "MACD" in df.columns
        assert "Signal_Line" in df.columns
        assert "MACD_Histogram" in df.columns
    
    def test_calculates_rsi(self, sample_stock_data):
        """Test that RSI is calculated."""
        df = calculate_technical_indicators(sample_stock_data)
        
        assert "RSI_14" in df.columns
        # RSI should be between 0 and 100
        valid_rsi = df["RSI_14"].dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()
    
    def test_calculates_bollinger_bands(self, sample_stock_data):
        """Test that Bollinger Bands are calculated."""
        df = calculate_technical_indicators(sample_stock_data)
        
        assert "BB_upper" in df.columns
        assert "BB_lower" in df.columns
        assert "BB_middle" in df.columns
    
    def test_missing_column_raises_error(self):
        """Test that missing required column raises error."""
        df = pd.DataFrame({"Open": [1, 2, 3]})
        
        with pytest.raises(PreprocessingError):
            calculate_technical_indicators(df)


class TestCreateLagFeatures:
    """Tests for create_lag_features function."""
    
    def test_creates_lag_features(self, sample_stock_data):
        """Test that lag features are created."""
        df = calculate_technical_indicators(sample_stock_data)
        df = create_lag_features(df, lags=[1, 2, 3])
        
        assert "Close_Lag_1" in df.columns
        assert "Close_Lag_2" in df.columns
        assert "Close_Lag_3" in df.columns
    
    def test_lag_values_are_correct(self, sample_stock_data):
        """Test that lag values are calculated correctly."""
        df = calculate_technical_indicators(sample_stock_data)
        df = create_lag_features(df, lags=[1])
        
        # The lag should be the previous value
        for i in range(1, len(df)):
            if not np.isnan(df["Close_Lag_1"].iloc[i]):
                assert df["Close_Lag_1"].iloc[i] == df["Close"].iloc[i - 1]


class TestHandleMissingValues:
    """Tests for handle_missing_values function."""
    
    def test_forward_fill(self, sample_stock_data):
        """Test forward fill method."""
        df = sample_stock_data.copy()
        df.loc[5:10, "Close"] = np.nan
        
        result = handle_missing_values(df, method="forward_fill")
        
        assert result["Close"].isna().sum() == 0
    
    def test_interpolate(self, sample_stock_data):
        """Test interpolation method."""
        df = sample_stock_data.copy()
        df.loc[5:10, "Close"] = np.nan
        
        result = handle_missing_values(df, method="interpolate")
        
        assert result["Close"].isna().sum() == 0
    
    def test_drop_method(self, sample_stock_data):
        """Test drop method."""
        df = sample_stock_data.copy()
        df.loc[5:10, "Close"] = np.nan
        
        result = handle_missing_values(df, method="drop")
        
        assert len(result) < len(df)
        assert result["Close"].isna().sum() == 0


class TestCreateSequences:
    """Tests for create_sequences function."""
    
    def test_sequence_shape(self):
        """Test that sequences have correct shape."""
        data = np.random.randn(100, 5)
        sequence_length = 10
        
        X, y = create_sequences(data, sequence_length)
        
        assert X.shape == (90, 10, 5)  # 100 - 10 = 90 sequences
        assert y.shape == (90,)
    
    def test_sequence_values(self):
        """Test that sequence values are correct."""
        data = np.arange(20).reshape(20, 1)
        sequence_length = 5
        
        X, y = create_sequences(data, sequence_length)
        
        # First sequence should be [0,1,2,3,4], target should be 5
        np.testing.assert_array_equal(X[0], [[0], [1], [2], [3], [4]])
        assert y[0] == 5


class TestPreprocessData:
    """Tests for preprocess_data function."""
    
    def test_returns_correct_tuple(self, sample_stock_data):
        """Test that preprocess_data returns correct tuple."""
        result = preprocess_data(sample_stock_data, sequence_length=30)
        
        assert len(result) == 6  # X_train, X_test, y_train, y_test, scaler, features
    
    def test_train_test_split(self, sample_stock_data):
        """Test train/test split."""
        X_train, X_test, y_train, y_test, scaler, features = preprocess_data(
            sample_stock_data,
            sequence_length=30,
            test_size=0.2,
        )
        
        total_samples = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total_samples
        
        assert 0.15 < test_ratio < 0.25  # Allow some tolerance
    
    def test_scaled_data_range(self, sample_stock_data):
        """Test that data is properly scaled."""
        X_train, X_test, y_train, y_test, scaler, features = preprocess_data(
            sample_stock_data,
            sequence_length=30,
            scaler_type="minmax",
        )
        
        # MinMaxScaler should scale data between 0 and 1 (with floating-point tolerance)
        assert X_train.min() >= 0
        assert X_train.max() <= 1 + 1e-10


class TestInverseTransformPredictions:
    """Tests for inverse_transform_predictions function."""
    
    def test_inverse_transform(self, sample_stock_data):
        """Test inverse transformation."""
        X_train, X_test, y_train, y_test, scaler, features = preprocess_data(
            sample_stock_data,
            sequence_length=30,
        )
        
        # Create some dummy predictions
        predictions = y_test
        
        # Inverse transform
        original_scale = inverse_transform_predictions(predictions, scaler, len(features))
        
        assert len(original_scale) == len(predictions)
        # Values should be in original scale (around 100 for our test data)
        assert original_scale.mean() > 10  # Should be much larger than scaled values
