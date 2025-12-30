"""Evaluation metrics for stock price prediction models."""

import numpy as np
from typing import Dict

from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        RMSE value
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        MAE value
    """
    return np.mean(np.abs(y_true - y_pred))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        MAPE value (as percentage)
    """
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R-squared (coefficient of determination).
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        R² score
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return 1 - (ss_res / ss_tot)


def calculate_direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate direction accuracy (percentage of correct up/down predictions).
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        Direction accuracy as percentage
    """
    # Calculate actual and predicted direction changes
    actual_direction = np.diff(y_true) > 0
    predicted_direction = np.diff(y_pred) > 0
    
    # Calculate accuracy
    correct = np.sum(actual_direction == predicted_direction)
    total = len(actual_direction)
    
    return (correct / total) * 100 if total > 0 else 0.0


def calculate_max_drawdown(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate maximum drawdown in predictions.
    
    Args:
        y_true: Actual values (not used, kept for consistency)
        y_pred: Predicted values
    
    Returns:
        Maximum drawdown as percentage
    """
    peak = y_pred[0]
    max_dd = 0
    
    for value in y_pred:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, drawdown)
    
    return max_dd


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.
    
    Metrics:
        - RMSE (Root Mean Squared Error)
        - MAE (Mean Absolute Error)
        - MAPE (Mean Absolute Percentage Error)
        - R² Score
        - Direction Accuracy
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        "rmse": calculate_rmse(y_true, y_pred),
        "mae": calculate_mae(y_true, y_pred),
        "mape": calculate_mape(y_true, y_pred),
        "r2": calculate_r2(y_true, y_pred),
        "direction_accuracy": calculate_direction_accuracy(y_true, y_pred),
    }
    
    logger.info(
        f"Model Evaluation - RMSE: {metrics['rmse']:.4f}, "
        f"MAE: {metrics['mae']:.4f}, MAPE: {metrics['mape']:.2f}%, "
        f"R²: {metrics['r2']:.4f}, Direction Accuracy: {metrics['direction_accuracy']:.2f}%"
    )
    
    return metrics


def compare_models(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models' predictions.
    
    Args:
        y_true: Actual values
        predictions: Dictionary mapping model name to predictions
    
    Returns:
        Dictionary mapping model name to metrics
    """
    results = {}
    
    for model_name, y_pred in predictions.items():
        results[model_name] = evaluate_model(y_true, y_pred)
        logger.info(f"Evaluated {model_name}")
    
    return results


def print_evaluation_report(metrics: Dict[str, float], model_name: str = "Model") -> str:
    """
    Generate a formatted evaluation report.
    
    Args:
        metrics: Dictionary of evaluation metrics
        model_name: Name of the model
    
    Returns:
        Formatted report string
    """
    report = f"""
{'='*50}
{model_name} Evaluation Report
{'='*50}

Performance Metrics:
--------------------
RMSE (Root Mean Squared Error):     {metrics.get('rmse', 0):.4f}
MAE (Mean Absolute Error):          {metrics.get('mae', 0):.4f}
MAPE (Mean Absolute % Error):       {metrics.get('mape', 0):.2f}%
R² Score:                           {metrics.get('r2', 0):.4f}
Direction Accuracy:                 {metrics.get('direction_accuracy', 0):.2f}%

{'='*50}
"""
    return report
