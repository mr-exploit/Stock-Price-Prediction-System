"""Data preprocessing and feature engineering module."""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger
from src.exceptions import PreprocessingError

logger = get_logger(__name__)


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for stock data.
    
    Indicators:
        - Simple Moving Average (SMA): 20, 50, 200 days
        - Exponential Moving Average (EMA): 12, 26 days
        - Relative Strength Index (RSI): 14 days
        - MACD: (12, 26, 9)
        - Bollinger Bands: 20 days, 2 std
        - Volume Moving Average
        - Price Rate of Change (ROC)
    
    Args:
        df: DataFrame with stock data (must have 'Close' and 'Volume' columns)
    
    Returns:
        DataFrame with technical indicators added
    """
    df = df.copy()
    
    # Validate required columns
    required_columns = ["Close", "Volume"]
    for col in required_columns:
        if col not in df.columns:
            raise PreprocessingError(f"Missing required column: {col}")
    
    # Simple Moving Averages
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    
    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["Signal_Line"]
    
    # RSI (Relative Strength Index)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI_14"] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df["BB_middle"] = df["Close"].rolling(window=20).mean()
    bb_std = df["Close"].rolling(window=20).std()
    df["BB_upper"] = df["BB_middle"] + (bb_std * 2)
    df["BB_lower"] = df["BB_middle"] - (bb_std * 2)
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_middle"]
    
    # Volume Moving Average
    df["Volume_MA"] = df["Volume"].rolling(window=20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA"]
    
    # Price Rate of Change
    df["ROC"] = df["Close"].pct_change(periods=10) * 100
    
    # Daily returns
    df["Daily_Return"] = df["Close"].pct_change()
    
    # Volatility (20-day)
    df["Volatility"] = df["Daily_Return"].rolling(window=20).std() * np.sqrt(252)
    
    logger.info("Calculated technical indicators")
    return df


def create_lag_features(df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
    """
    Create lag features for time series prediction.
    
    Args:
        df: DataFrame with stock data
        lags: List of lag periods to create
    
    Returns:
        DataFrame with lag features added
    """
    df = df.copy()
    
    for lag in lags:
        df[f"Close_Lag_{lag}"] = df["Close"].shift(lag)
        df[f"Volume_Lag_{lag}"] = df["Volume"].shift(lag)
        df[f"Return_Lag_{lag}"] = df["Daily_Return"].shift(lag) if "Daily_Return" in df.columns else df["Close"].pct_change().shift(lag)
    
    logger.info(f"Created lag features for lags: {lags}")
    return df


def handle_missing_values(df: pd.DataFrame, method: str = "forward_fill") -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.
    
    Args:
        df: DataFrame with potential missing values
        method: Method to handle missing values ('forward_fill', 'interpolate', 'drop')
    
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    if method == "forward_fill":
        df = df.ffill()
        df = df.bfill()  # Fill any remaining at the beginning
    elif method == "interpolate":
        df = df.interpolate(method="linear")
        df = df.bfill()
    elif method == "drop":
        df = df.dropna()
    else:
        raise PreprocessingError(f"Unknown method: {method}")
    
    logger.info(f"Handled missing values using {method}")
    return df


def create_sequences(
    data: np.ndarray,
    sequence_length: int = 60,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training.
    
    Args:
        data: Numpy array with features (rows are samples, columns are features)
        sequence_length: Number of time steps in each sequence
    
    Returns:
        Tuple of (X, y) where X is sequences and y is targets
    """
    X, y = [], []
    
    for i in range(sequence_length, len(data)):
        X.append(data[i - sequence_length:i])
        y.append(data[i, 0])  # Assuming first column is the target (Close price)
    
    return np.array(X), np.array(y)


def preprocess_data(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    target_column: str = "Close",
    test_size: float = 0.2,
    sequence_length: int = 60,
    scaler_type: str = "minmax",
    include_technical_indicators: bool = True,
    lag_features: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object, List[str]]:
    """
    Preprocess stock data for machine learning.
    
    Args:
        df: Raw stock data DataFrame
        features: List of feature column names (if None, will be auto-selected)
        target_column: Name of the target column
        test_size: Proportion of data for testing
        sequence_length: Number of time steps for LSTM sequences
        scaler_type: Type of scaler ('minmax' or 'standard')
        include_technical_indicators: Whether to calculate technical indicators
        lag_features: List of lag periods for lag features
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler, feature_names)
    """
    df = df.copy()
    
    # Calculate technical indicators if requested
    if include_technical_indicators:
        df = calculate_technical_indicators(df)
    
    # Create lag features if requested
    if lag_features:
        df = create_lag_features(df, lag_features)
    
    # Handle missing values
    df = handle_missing_values(df, method="drop")
    
    # Select features
    if features is None:
        # Default features
        features = [target_column, "Open", "High", "Low", "Volume"]
        
        # Add technical indicators if available
        indicator_columns = [
            "SMA_20", "SMA_50", "EMA_12", "EMA_26", "RSI_14", 
            "MACD", "BB_upper", "BB_lower", "Volume_MA", "ROC"
        ]
        features.extend([col for col in indicator_columns if col in df.columns])
    
    # Ensure target column is first
    if target_column in features:
        features.remove(target_column)
    features = [target_column] + features
    
    # Filter to available columns
    available_features = [f for f in features if f in df.columns]
    
    logger.info(f"Using features: {available_features}")
    
    # Get feature data
    feature_data = df[available_features].values
    
    # Scale the data
    if scaler_type == "minmax":
        scaler = MinMaxScaler(feature_range=(0, 1))
    elif scaler_type == "standard":
        scaler = StandardScaler()
    else:
        raise PreprocessingError(f"Unknown scaler type: {scaler_type}")
    
    scaled_data = scaler.fit_transform(feature_data)
    
    # Create sequences
    X, y = create_sequences(scaled_data, sequence_length)
    
    # Split into train and test sets
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler, available_features


def inverse_transform_predictions(
    predictions: np.ndarray,
    scaler: object,
    n_features: int,
) -> np.ndarray:
    """
    Inverse transform scaled predictions back to original scale.
    
    Args:
        predictions: Scaled predictions
        scaler: Fitted scaler object
        n_features: Number of features used during scaling
    
    Returns:
        Predictions in original scale
    """
    # Create a dummy array to inverse transform
    dummy = np.zeros((len(predictions), n_features))
    dummy[:, 0] = predictions.flatten()
    
    # Inverse transform
    inverse_transformed = scaler.inverse_transform(dummy)
    
    return inverse_transformed[:, 0]
