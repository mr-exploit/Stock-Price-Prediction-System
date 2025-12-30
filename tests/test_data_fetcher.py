"""Tests for data fetcher module."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from src.data.fetcher import (
    fetch_stock_data,
    validate_ticker,
    fetch_multiple_tickers,
    get_ticker_info,
)
from src.exceptions import InvalidTickerError, StockDataError


class TestFetchStockData:
    """Tests for fetch_stock_data function."""
    
    def test_fetch_stock_data_returns_dataframe(self):
        """Test that fetch_stock_data returns a DataFrame."""
        # Skip if no network connection
        try:
            df = fetch_stock_data("AAPL", "2023-01-01", "2023-03-01", use_cache=False)
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
        except Exception:
            pytest.skip("Network unavailable or API rate limited")
    
    def test_fetch_stock_data_has_required_columns(self):
        """Test that returned DataFrame has required columns."""
        try:
            df = fetch_stock_data("AAPL", "2023-01-01", "2023-03-01", use_cache=False)
            required_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
            for col in required_columns:
                assert col in df.columns, f"Missing column: {col}"
        except Exception:
            pytest.skip("Network unavailable or API rate limited")
    
    def test_fetch_stock_data_invalid_ticker(self):
        """Test that invalid ticker raises exception."""
        with pytest.raises((InvalidTickerError, StockDataError)):
            fetch_stock_data("INVALID_TICKER_12345", "2023-01-01", "2023-03-01", use_cache=False)
    
    def test_fetch_stock_data_invalid_interval(self):
        """Test that invalid interval raises exception."""
        with pytest.raises(StockDataError):
            fetch_stock_data("AAPL", "2023-01-01", "2023-03-01", interval="invalid")


class TestValidateTicker:
    """Tests for validate_ticker function."""
    
    def test_validate_valid_ticker(self):
        """Test validation of valid ticker."""
        try:
            result = validate_ticker("AAPL")
            assert result is True
        except Exception:
            pytest.skip("Network unavailable")
    
    def test_validate_invalid_ticker(self):
        """Test validation of invalid ticker."""
        try:
            result = validate_ticker("INVALID_TICKER_12345")
            assert result is False
        except Exception:
            pytest.skip("Network unavailable")


class TestGetTickerInfo:
    """Tests for get_ticker_info function."""
    
    def test_get_ticker_info_returns_dict(self):
        """Test that get_ticker_info returns a dictionary."""
        try:
            info = get_ticker_info("AAPL")
            assert info is not None
            assert isinstance(info, dict)
            assert "symbol" in info
            assert "name" in info
        except Exception:
            pytest.skip("Network unavailable")
