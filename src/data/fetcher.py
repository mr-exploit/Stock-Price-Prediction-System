"""Yahoo Finance data fetching module."""

import yfinance as yf
import pandas as pd
from typing import Optional, List
from datetime import datetime

from src.utils.logger import get_logger
from src.utils.helpers import generate_cache_key, save_to_cache, load_from_cache
from src.exceptions import InvalidTickerError, StockDataError

logger = get_logger(__name__)


def validate_ticker(ticker: str) -> bool:
    """
    Validate if a ticker symbol exists.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        True if valid, False otherwise
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        # Check if we got valid data
        return info.get("regularMarketPrice") is not None or info.get("previousClose") is not None
    except Exception:
        return False


def fetch_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL', 'TSLA', 'GOOGL')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        interval: Data interval ('1d', '1wk', '1mo')
        use_cache: Whether to use cached data if available
    
    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume, Adj Close
    
    Raises:
        InvalidTickerError: If ticker symbol is invalid
        StockDataError: If data cannot be fetched
    """
    # Validate interval
    valid_intervals = ["1d", "1wk", "1mo"]
    if interval not in valid_intervals:
        raise StockDataError(f"Invalid interval: {interval}. Must be one of {valid_intervals}")
    
    # Check cache first
    if use_cache:
        cache_key = generate_cache_key(ticker, start_date, end_date, interval)
        cached_data = load_from_cache(cache_key)
        if cached_data is not None:
            logger.info(f"Loaded {ticker} data from cache")
            return cached_data
    
    try:
        logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
        
        # Create ticker object
        stock = yf.Ticker(ticker)
        
        # Fetch historical data
        df = stock.history(start=start_date, end=end_date, interval=interval)
        
        if df.empty:
            raise InvalidTickerError(f"No data found for ticker: {ticker}")
        
        # Reset index to make Date a column
        df.reset_index(inplace=True)
        
        # Rename 'Date' or 'Datetime' column
        if "Datetime" in df.columns:
            df.rename(columns={"Datetime": "Date"}, inplace=True)
        
        # Ensure Date column is datetime
        df["Date"] = pd.to_datetime(df["Date"])
        
        # Select and order columns
        columns_to_keep = ["Date", "Open", "High", "Low", "Close", "Volume"]
        if "Adj Close" in df.columns:
            columns_to_keep.append("Adj Close")
        
        df = df[[col for col in columns_to_keep if col in df.columns]]
        
        # Cache the data
        if use_cache:
            cache_key = generate_cache_key(ticker, start_date, end_date, interval)
            save_to_cache(df, cache_key)
        
        logger.info(f"Successfully fetched {len(df)} records for {ticker}")
        return df
    
    except InvalidTickerError:
        raise
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        raise StockDataError(f"Failed to fetch data for {ticker}: {str(e)}")


def fetch_multiple_tickers(
    tickers: List[str],
    start_date: str,
    end_date: str,
    interval: str = "1d",
) -> dict:
    """
    Fetch historical data for multiple tickers.
    
    Args:
        tickers: List of stock ticker symbols
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        interval: Data interval ('1d', '1wk', '1mo')
    
    Returns:
        Dictionary mapping ticker to DataFrame
    """
    results = {}
    for ticker in tickers:
        try:
            results[ticker] = fetch_stock_data(ticker, start_date, end_date, interval)
        except (InvalidTickerError, StockDataError) as e:
            logger.warning(f"Skipping {ticker}: {str(e)}")
    
    return results


def get_ticker_info(ticker: str) -> Optional[dict]:
    """
    Get information about a stock ticker.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with ticker information or None if not found
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "symbol": ticker,
            "name": info.get("longName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "currency": info.get("currency", "USD"),
        }
    except Exception as e:
        logger.error(f"Error getting info for {ticker}: {str(e)}")
        return None
