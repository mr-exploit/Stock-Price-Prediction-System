#!/usr/bin/env python
"""
Stock Price Prediction System - Main Entry Point

Usage:
    python main.py --mode train --ticker AAPL --model lstm --epochs 100
    python main.py --mode predict --ticker AAPL --days 30
    python main.py --mode evaluate --ticker AAPL --model lstm
    python main.py --mode compare --ticker AAPL --models lstm,rf,xgboost
"""

import argparse
import sys
import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.data.fetcher import fetch_stock_data, get_ticker_info
from src.data.preprocessor import (
    preprocess_data,
    calculate_technical_indicators,
    inverse_transform_predictions,
)
from src.models.lstm_model import LSTMPredictor
from src.models.rf_model import RandomForestPredictor
from src.models.arima_model import ARIMAPredictor
from src.evaluation.metrics import evaluate_model, print_evaluation_report, compare_models
from src.visualization.plotter import (
    plot_predictions,
    plot_training_history,
    plot_technical_indicators,
    create_interactive_dashboard,
    plot_model_comparison,
)
from src.utils.logger import setup_logger, get_logger
from src.utils.helpers import load_config, create_directories
from src.exceptions import (
    StockDataError,
    InvalidTickerError,
    ModelTrainingError,
    ConfigurationError,
)

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup logger
logger = setup_logger("stock_prediction")


def get_model(model_type: str, config: dict):
    """
    Get model instance based on type.
    
    Args:
        model_type: Type of model ('lstm', 'rf', 'random_forest', 'xgboost', 'arima')
        config: Configuration dictionary
    
    Returns:
        Model instance
    """
    model_config = config.get("model", {})
    
    if model_type == "lstm":
        lstm_config = model_config.get("lstm", {})
        return LSTMPredictor(
            units=tuple(lstm_config.get("units", [50, 50])),
            dropout=lstm_config.get("dropout", 0.2),
            learning_rate=lstm_config.get("learning_rate", 0.001),
        )
    elif model_type in ["rf", "random_forest"]:
        rf_config = model_config.get("random_forest", {})
        return RandomForestPredictor(
            n_estimators=rf_config.get("n_estimators", 100),
            max_depth=rf_config.get("max_depth", 10),
        )
    elif model_type == "arima":
        return ARIMAPredictor(order=(5, 1, 0))
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_model(args, config: dict):
    """
    Train a model on stock data.
    
    Args:
        args: Command line arguments
        config: Configuration dictionary
    """
    logger.info(f"Training {args.model} model for {args.ticker}")
    
    # Load data
    data_config = config.get("data", {})
    start_date = args.start_date or data_config.get("start_date", "2018-01-01")
    end_date = args.end_date or data_config.get("end_date", datetime.now().strftime("%Y-%m-%d"))
    
    try:
        df = fetch_stock_data(args.ticker, start_date, end_date)
    except (InvalidTickerError, StockDataError) as e:
        logger.error(f"Failed to fetch data: {e}")
        return
    
    # Display ticker info
    ticker_info = get_ticker_info(args.ticker)
    if ticker_info:
        logger.info(f"Company: {ticker_info['name']} ({ticker_info['symbol']})")
        logger.info(f"Sector: {ticker_info['sector']}, Industry: {ticker_info['industry']}")
    
    # Preprocess data
    model_config = config.get("model", {})
    lstm_config = model_config.get("lstm", {})
    sequence_length = lstm_config.get("sequence_length", 60)
    test_size = 1 - data_config.get("train_test_split", 0.8)
    
    X_train, X_test, y_train, y_test, scaler, features = preprocess_data(
        df,
        sequence_length=sequence_length,
        test_size=test_size,
        include_technical_indicators=True,
    )
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Testing data shape: {X_test.shape}")
    
    # Build and train model
    model = get_model(args.model, config)
    
    if args.model == "lstm":
        model.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        epochs = args.epochs or lstm_config.get("epochs", 100)
        batch_size = lstm_config.get("batch_size", 32)
        
        history = model.train(
            X_train, y_train,
            X_val=X_test, y_val=y_test,
            epochs=epochs,
            batch_size=batch_size,
            early_stopping=True,
            patience=10,
        )
        
        # Plot training history
        if args.plot:
            plot_training_history(
                history,
                save_path=f"output/plots/{args.ticker}_{args.model}_history.png",
                show_plot=args.show_plots,
            )
    
    elif args.model in ["rf", "random_forest"]:
        model.build_model()
        model.train(X_train, y_train, X_val=X_test, y_val=y_test)
    
    elif args.model == "arima":
        # ARIMA uses only target values
        model.build_model()
        model.train(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Inverse transform predictions if using LSTM/RF
    if args.model != "arima":
        predictions_original = inverse_transform_predictions(
            predictions, scaler, len(features)
        )
        y_test_original = inverse_transform_predictions(
            y_test, scaler, len(features)
        )
    else:
        predictions_original = predictions
        y_test_original = y_test
    
    # Evaluate model
    metrics = evaluate_model(y_test_original, predictions_original)
    print(print_evaluation_report(metrics, f"{args.ticker} - {args.model.upper()}"))
    
    # Save model
    model_path = f"data/models/{args.ticker}_{args.model}_model"
    if args.model == "lstm":
        model_path += ".h5"
    else:
        model_path += ".pkl"
    
    model.save_model(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Plot predictions
    if args.plot:
        plot_predictions(
            df,
            predictions_original,
            args.ticker,
            save_path=f"output/plots/{args.ticker}_{args.model}_predictions.png",
            show_plot=args.show_plots,
        )
        
        # Create interactive dashboard
        df_with_indicators = calculate_technical_indicators(df)
        create_interactive_dashboard(
            args.ticker,
            df_with_indicators,
            predictions=predictions_original,
            metrics=metrics,
            save_path=f"output/plots/{args.ticker}_dashboard.html",
        )
    
    logger.info("Training completed successfully!")


def predict_future(args, config: dict):
    """
    Predict future stock prices.
    
    Args:
        args: Command line arguments
        config: Configuration dictionary
    """
    logger.info(f"Predicting {args.days} days ahead for {args.ticker}")
    
    # Load data
    data_config = config.get("data", {})
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")
    
    try:
        df = fetch_stock_data(args.ticker, start_date, end_date)
    except (InvalidTickerError, StockDataError) as e:
        logger.error(f"Failed to fetch data: {e}")
        return
    
    # Preprocess data
    model_config = config.get("model", {})
    lstm_config = model_config.get("lstm", {})
    sequence_length = lstm_config.get("sequence_length", 60)
    
    X_train, X_test, y_train, y_test, scaler, features = preprocess_data(
        df,
        sequence_length=sequence_length,
        test_size=0.1,  # Use most data for training
        include_technical_indicators=True,
    )
    
    # Load or train model
    model_path = f"data/models/{args.ticker}_{args.model}_model"
    if args.model == "lstm":
        model_path += ".h5"
    else:
        model_path += ".pkl"
    
    model = get_model(args.model, config)
    
    if os.path.exists(model_path):
        logger.info(f"Loading existing model from {model_path}")
        model.load_model(model_path)
    else:
        logger.info("No saved model found. Training new model...")
        if args.model == "lstm":
            model.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            model.train(X_train, y_train, epochs=50, early_stopping=True)
        else:
            model.build_model()
            model.train(X_train, y_train)
    
    # Predict future
    if args.model == "lstm":
        last_sequence = X_test[-1]  # Last available sequence
        future_predictions = model.predict_future(last_sequence, args.days)
        
        # Inverse transform
        future_prices = inverse_transform_predictions(
            future_predictions, scaler, len(features)
        )
    elif args.model == "arima":
        future_prices = model.forecast(args.days)
    else:
        # For RF, we can only do one-step predictions
        logger.warning("Random Forest doesn't support multi-step forecasting well")
        future_prices = np.array([model.predict(X_test[-1:]) for _ in range(args.days)]).flatten()
    
    # Create future dates
    last_date = df["Date"].iloc[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=args.days, freq="B")
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted_Price": future_prices,
    })
    
    print("\n" + "=" * 50)
    print(f"Future Price Predictions for {args.ticker}")
    print("=" * 50)
    print(predictions_df.to_string(index=False))
    print("=" * 50)
    
    # Save predictions
    predictions_df.to_csv(f"output/plots/{args.ticker}_predictions.csv", index=False)
    logger.info(f"Predictions saved to output/plots/{args.ticker}_predictions.csv")
    
    return predictions_df


def evaluate_model_performance(args, config: dict):
    """
    Evaluate a trained model.
    
    Args:
        args: Command line arguments
        config: Configuration dictionary
    """
    logger.info(f"Evaluating {args.model} model for {args.ticker}")
    
    # Load data
    data_config = config.get("data", {})
    start_date = args.start_date or data_config.get("start_date", "2018-01-01")
    end_date = args.end_date or data_config.get("end_date", datetime.now().strftime("%Y-%m-%d"))
    
    try:
        df = fetch_stock_data(args.ticker, start_date, end_date)
    except (InvalidTickerError, StockDataError) as e:
        logger.error(f"Failed to fetch data: {e}")
        return
    
    # Preprocess data
    model_config = config.get("model", {})
    lstm_config = model_config.get("lstm", {})
    sequence_length = lstm_config.get("sequence_length", 60)
    
    X_train, X_test, y_train, y_test, scaler, features = preprocess_data(
        df,
        sequence_length=sequence_length,
        test_size=0.2,
        include_technical_indicators=True,
    )
    
    # Load model
    model_path = f"data/models/{args.ticker}_{args.model}_model"
    if args.model == "lstm":
        model_path += ".h5"
    else:
        model_path += ".pkl"
    
    model = get_model(args.model, config)
    
    if os.path.exists(model_path):
        model.load_model(model_path)
    else:
        logger.error(f"No saved model found at {model_path}. Train a model first.")
        return
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Inverse transform
    predictions_original = inverse_transform_predictions(predictions, scaler, len(features))
    y_test_original = inverse_transform_predictions(y_test, scaler, len(features))
    
    # Evaluate
    metrics = evaluate_model(y_test_original, predictions_original)
    print(print_evaluation_report(metrics, f"{args.ticker} - {args.model.upper()}"))
    
    # Plot
    if args.plot:
        plot_predictions(
            df,
            predictions_original,
            args.ticker,
            save_path=f"output/plots/{args.ticker}_{args.model}_evaluation.png",
            show_plot=args.show_plots,
        )


def compare_models_performance(args, config: dict):
    """
    Compare multiple models.
    
    Args:
        args: Command line arguments
        config: Configuration dictionary
    """
    models_to_compare = args.models.split(",")
    logger.info(f"Comparing models: {models_to_compare} for {args.ticker}")
    
    # Load data
    data_config = config.get("data", {})
    start_date = data_config.get("start_date", "2018-01-01")
    end_date = data_config.get("end_date", datetime.now().strftime("%Y-%m-%d"))
    
    try:
        df = fetch_stock_data(args.ticker, start_date, end_date)
    except (InvalidTickerError, StockDataError) as e:
        logger.error(f"Failed to fetch data: {e}")
        return
    
    # Preprocess data
    model_config = config.get("model", {})
    lstm_config = model_config.get("lstm", {})
    sequence_length = lstm_config.get("sequence_length", 60)
    
    X_train, X_test, y_train, y_test, scaler, features = preprocess_data(
        df,
        sequence_length=sequence_length,
        test_size=0.2,
        include_technical_indicators=True,
    )
    
    # Get predictions from each model
    all_predictions = {}
    all_metrics = {}
    
    for model_type in models_to_compare:
        logger.info(f"Training/evaluating {model_type}...")
        
        try:
            model = get_model(model_type.strip(), config)
            
            if model_type.strip() == "lstm":
                model.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
                model.train(X_train, y_train, epochs=50, early_stopping=True, verbose=0)
            elif model_type.strip() == "arima":
                model.build_model()
                model.train(X_train, y_train)
            else:
                model.build_model()
                model.train(X_train, y_train)
            
            predictions = model.predict(X_test)
            
            # Inverse transform
            if model_type.strip() != "arima":
                predictions = inverse_transform_predictions(predictions, scaler, len(features))
            
            all_predictions[model_type.strip().upper()] = predictions
            
        except Exception as e:
            logger.warning(f"Failed to train {model_type}: {e}")
    
    # Get actual values in original scale
    y_test_original = inverse_transform_predictions(y_test, scaler, len(features))
    
    # Compare all models
    comparison_results = compare_models(y_test_original, all_predictions)
    
    # Print comparison
    print("\n" + "=" * 70)
    print(f"Model Comparison for {args.ticker}")
    print("=" * 70)
    print(f"{'Model':<15} {'RMSE':<12} {'MAE':<12} {'MAPE':<12} {'R²':<12} {'Dir. Acc.':<12}")
    print("-" * 70)
    
    for model_name, metrics in comparison_results.items():
        print(
            f"{model_name:<15} "
            f"{metrics['rmse']:<12.4f} "
            f"{metrics['mae']:<12.4f} "
            f"{metrics['mape']:<12.2f}% "
            f"{metrics['r2']:<12.4f} "
            f"{metrics['direction_accuracy']:<12.2f}%"
        )
    
    print("=" * 70)
    
    # Plot comparison
    if args.plot:
        plot_model_comparison(
            y_test_original,
            all_predictions,
            save_path=f"output/plots/{args.ticker}_model_comparison.png",
            show_plot=args.show_plots,
        )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Stock Price Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train LSTM model
    python main.py --mode train --ticker AAPL --model lstm --epochs 100

    # Predict future prices
    python main.py --mode predict --ticker AAPL --days 30

    # Evaluate model
    python main.py --mode evaluate --ticker AAPL --model lstm

    # Compare models
    python main.py --mode compare --ticker AAPL --models lstm,rf,arima
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "predict", "evaluate", "compare"],
        help="Mode of operation"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g., AAPL, TSLA, GOOGL)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lstm",
        choices=["lstm", "rf", "random_forest", "arima", "xgboost"],
        help="Model type for training/prediction"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="lstm,rf,arima",
        help="Comma-separated list of models to compare"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs for neural networks"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to predict into the future"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for data (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for data (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="Generate plots"
    )
    parser.add_argument(
        "--no-plot",
        action="store_false",
        dest="plot",
        help="Disable plot generation"
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        default=False,
        help="Display plots interactively"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {args.config}. Using defaults.")
        config = {}
    
    # Create necessary directories
    os.makedirs("data/models", exist_ok=True)
    os.makedirs("output/plots", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Execute based on mode
    if args.mode == "train":
        train_model(args, config)
    elif args.mode == "predict":
        predict_future(args, config)
    elif args.mode == "evaluate":
        evaluate_model_performance(args, config)
    elif args.mode == "compare":
        compare_models_performance(args, config)


if __name__ == "__main__":
    main()
