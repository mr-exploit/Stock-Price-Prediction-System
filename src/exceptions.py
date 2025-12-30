"""Custom exceptions for Stock Price Prediction System."""


class StockPredictionError(Exception):
    """Base exception for Stock Price Prediction System."""
    pass


class StockDataError(StockPredictionError):
    """Raised when stock data cannot be fetched."""
    pass


class InvalidTickerError(StockPredictionError):
    """Raised when ticker symbol is invalid."""
    pass


class ModelTrainingError(StockPredictionError):
    """Raised when model training fails."""
    pass


class PreprocessingError(StockPredictionError):
    """Raised when data preprocessing fails."""
    pass


class ConfigurationError(StockPredictionError):
    """Raised when configuration is invalid or missing."""
    pass


class PredictionError(StockPredictionError):
    """Raised when prediction fails."""
    pass
