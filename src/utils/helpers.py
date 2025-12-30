"""Helper functions for Stock Price Prediction System."""

import os
import pickle
from typing import Any, Optional
import yaml
import hashlib
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
    
    Returns:
        Configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def generate_cache_key(ticker: str, start_date: str, end_date: str, interval: str) -> str:
    """
    Generate a unique cache key for stock data.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date
        end_date: End date
        interval: Data interval
    
    Returns:
        Cache key string
    """
    key_string = f"{ticker}_{start_date}_{end_date}_{interval}"
    return hashlib.md5(key_string.encode()).hexdigest()


def save_to_cache(data: Any, cache_key: str, cache_dir: str = "data/raw") -> str:
    """
    Save data to cache.
    
    Args:
        data: Data to cache
        cache_key: Unique cache key
        cache_dir: Directory to store cache files
    
    Returns:
        Path to cached file
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    
    logger.debug(f"Saved data to cache: {cache_path}")
    return cache_path


def load_from_cache(cache_key: str, cache_dir: str = "data/raw") -> Optional[Any]:
    """
    Load data from cache if it exists.
    
    Args:
        cache_key: Unique cache key
        cache_dir: Directory to look for cache files
    
    Returns:
        Cached data or None if not found
    """
    cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        logger.debug(f"Loaded data from cache: {cache_path}")
        return data
    
    return None


def create_directories(config: dict) -> None:
    """
    Create necessary directories based on configuration.
    
    Args:
        config: Configuration dictionary
    """
    paths = config.get("paths", {})
    for path_name, path_value in paths.items():
        os.makedirs(path_value, exist_ok=True)
        logger.debug(f"Created directory: {path_value}")


def format_percentage(value: float) -> str:
    """
    Format a float as a percentage string.
    
    Args:
        value: Float value
    
    Returns:
        Formatted percentage string
    """
    return f"{value:.2f}%"


def validate_date_format(date_str: str) -> bool:
    """
    Validate date string format (YYYY-MM-DD).
    
    Args:
        date_str: Date string to validate
    
    Returns:
        True if valid, False otherwise
    """
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False
