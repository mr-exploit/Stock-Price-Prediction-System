"""Visualization module for stock price predictions."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import os

from src.utils.logger import get_logger

logger = get_logger(__name__)


def plot_predictions(
    df: pd.DataFrame,
    predictions: np.ndarray,
    ticker: str,
    test_dates: Optional[pd.DatetimeIndex] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> None:
    """
    Create visualization of historical prices vs predicted prices.
    
    Args:
        df: DataFrame with historical stock data
        predictions: Array of predicted prices
        ticker: Stock ticker symbol
        test_dates: Dates corresponding to predictions
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib is required for plotting. Install with: pip install matplotlib")
        return
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot historical data
    if "Date" in df.columns:
        ax.plot(df["Date"], df["Close"], label="Actual Price", color="blue", alpha=0.7)
    else:
        ax.plot(df.index, df["Close"], label="Actual Price", color="blue", alpha=0.7)
    
    # Plot predictions
    if test_dates is not None:
        ax.plot(test_dates, predictions, label="Predicted Price", color="red", alpha=0.7)
    else:
        # Use last n dates
        n_predictions = len(predictions)
        if "Date" in df.columns:
            pred_dates = df["Date"].iloc[-n_predictions:]
        else:
            pred_dates = df.index[-n_predictions:]
        ax.plot(pred_dates, predictions, label="Predicted Price", color="red", alpha=0.7)
    
    ax.set_title(f"{ticker} Stock Price - Actual vs Predicted", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price (USD)", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> None:
    """
    Plot training and validation loss curves.
    
    Args:
        history: Training history dictionary (from model.fit())
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib is required for plotting.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    if "loss" in history:
        axes[0].plot(history["loss"], label="Training Loss", color="blue")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="Validation Loss", color="red")
    axes[0].set_title("Model Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (MSE)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE plot
    if "mae" in history:
        axes[1].plot(history["mae"], label="Training MAE", color="blue")
    if "val_mae" in history:
        axes[1].plot(history["val_mae"], label="Validation MAE", color="red")
    axes[1].set_title("Model MAE", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Training history plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_technical_indicators(
    df: pd.DataFrame,
    ticker: str,
    indicators: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> None:
    """
    Plot price with technical indicators.
    
    Args:
        df: DataFrame with stock data and technical indicators
        ticker: Stock ticker symbol
        indicators: List of indicators to plot (default: SMA, MACD, RSI)
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib is required for plotting.")
        return
    
    if indicators is None:
        indicators = ["SMA_20", "SMA_50", "MACD", "RSI_14"]
    
    # Create figure with subplots
    n_subplots = 3  # Price, MACD, RSI
    fig, axes = plt.subplots(n_subplots, 1, figsize=(14, 10), 
                             gridspec_kw={"height_ratios": [3, 1, 1]})
    
    # Get x-axis values
    x_values = df["Date"] if "Date" in df.columns else df.index
    
    # Price and Moving Averages
    axes[0].plot(x_values, df["Close"], label="Close", color="blue", linewidth=1.5)
    if "SMA_20" in df.columns:
        axes[0].plot(x_values, df["SMA_20"], label="SMA 20", color="orange", alpha=0.7)
    if "SMA_50" in df.columns:
        axes[0].plot(x_values, df["SMA_50"], label="SMA 50", color="green", alpha=0.7)
    if "BB_upper" in df.columns and "BB_lower" in df.columns:
        axes[0].fill_between(x_values, df["BB_lower"], df["BB_upper"], 
                            alpha=0.1, color="gray", label="Bollinger Bands")
    axes[0].set_title(f"{ticker} Price with Technical Indicators", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Price")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)
    
    # MACD
    if "MACD" in df.columns and "Signal_Line" in df.columns:
        axes[1].plot(x_values, df["MACD"], label="MACD", color="blue")
        axes[1].plot(x_values, df["Signal_Line"], label="Signal", color="red")
        if "MACD_Histogram" in df.columns:
            colors = ["green" if v >= 0 else "red" for v in df["MACD_Histogram"]]
            axes[1].bar(x_values, df["MACD_Histogram"], color=colors, alpha=0.3)
        axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        axes[1].set_ylabel("MACD")
        axes[1].legend(loc="upper left")
        axes[1].grid(True, alpha=0.3)
    
    # RSI
    if "RSI_14" in df.columns:
        axes[2].plot(x_values, df["RSI_14"], label="RSI 14", color="purple")
        axes[2].axhline(y=70, color="red", linestyle="--", alpha=0.5)
        axes[2].axhline(y=30, color="green", linestyle="--", alpha=0.5)
        axes[2].fill_between(x_values, 30, 70, alpha=0.1, color="gray")
        axes[2].set_ylabel("RSI")
        axes[2].set_ylim(0, 100)
        axes[2].legend(loc="upper left")
        axes[2].grid(True, alpha=0.3)
    
    plt.xlabel("Date")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Technical indicators plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_interactive_dashboard(
    ticker: str,
    df: pd.DataFrame,
    predictions: Optional[np.ndarray] = None,
    metrics: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Create interactive Plotly dashboard with candlestick chart and indicators.
    
    Args:
        ticker: Stock ticker symbol
        df: DataFrame with stock data and indicators
        predictions: Predicted prices (optional)
        metrics: Evaluation metrics (optional)
        save_path: Path to save HTML file (optional)
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.error("plotly is required for interactive dashboard. Install with: pip install plotly")
        return
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=(f"{ticker} Stock Price", "Volume", "MACD", "RSI")
    )
    
    # Get x-axis values
    x_values = df["Date"] if "Date" in df.columns else df.index
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=x_values,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC"
        ),
        row=1, col=1
    )
    
    # Moving averages
    if "SMA_20" in df.columns:
        fig.add_trace(
            go.Scatter(x=x_values, y=df["SMA_20"], name="SMA 20", 
                      line=dict(color="orange", width=1)),
            row=1, col=1
        )
    if "SMA_50" in df.columns:
        fig.add_trace(
            go.Scatter(x=x_values, y=df["SMA_50"], name="SMA 50",
                      line=dict(color="blue", width=1)),
            row=1, col=1
        )
    
    # Predictions
    if predictions is not None:
        n_pred = len(predictions)
        pred_x = x_values[-n_pred:]
        fig.add_trace(
            go.Scatter(x=pred_x, y=predictions, name="Predictions",
                      line=dict(color="red", width=2, dash="dash")),
            row=1, col=1
        )
    
    # Volume
    colors = ["green" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "red" 
              for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=x_values, y=df["Volume"], name="Volume", marker_color=colors),
        row=2, col=1
    )
    
    # MACD
    if "MACD" in df.columns:
        fig.add_trace(
            go.Scatter(x=x_values, y=df["MACD"], name="MACD",
                      line=dict(color="blue", width=1)),
            row=3, col=1
        )
        if "Signal_Line" in df.columns:
            fig.add_trace(
                go.Scatter(x=x_values, y=df["Signal_Line"], name="Signal",
                          line=dict(color="orange", width=1)),
                row=3, col=1
            )
    
    # RSI
    if "RSI_14" in df.columns:
        fig.add_trace(
            go.Scatter(x=x_values, y=df["RSI_14"], name="RSI 14",
                      line=dict(color="purple", width=1)),
            row=4, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{ticker} Stock Analysis Dashboard",
            font=dict(size=20)
        ),
        xaxis_rangeslider_visible=False,
        height=900,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    
    # Add metrics annotation if available
    if metrics:
        metrics_text = (
            f"RMSE: {metrics.get('rmse', 0):.4f}<br>"
            f"MAE: {metrics.get('mae', 0):.4f}<br>"
            f"R²: {metrics.get('r2', 0):.4f}<br>"
            f"Direction Acc: {metrics.get('direction_accuracy', 0):.1f}%"
        )
        fig.add_annotation(
            text=metrics_text,
            xref="paper", yref="paper",
            x=1.02, y=0.98,
            showarrow=False,
            font=dict(size=10),
            align="left",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            bgcolor="white",
        )
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.write_html(save_path)
        logger.info(f"Interactive dashboard saved to {save_path}")
    
    fig.show()


def plot_model_comparison(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    model_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> None:
    """
    Compare predictions from multiple models.
    
    Args:
        y_true: Actual values
        predictions: Dictionary mapping model name to predictions
        model_names: List of model names to include (optional)
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib is required for plotting.")
        return
    
    if model_names is None:
        model_names = list(predictions.keys())
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot actual values
    ax.plot(y_true, label="Actual", color="black", linewidth=2)
    
    # Plot predictions for each model
    colors = ["red", "blue", "green", "orange", "purple"]
    for i, name in enumerate(model_names):
        if name in predictions:
            color = colors[i % len(colors)]
            ax.plot(predictions[name], label=name, color=color, alpha=0.7)
    
    ax.set_title("Model Comparison - Actual vs Predicted", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Model comparison plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
