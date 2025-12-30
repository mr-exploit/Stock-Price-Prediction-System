# Stock Price Prediction System

A Python-based stock price prediction system that fetches real-time data from Yahoo Finance and uses machine learning models to predict future stock prices for international markets.

## Features

- **Data Collection**: Fetch historical stock data from Yahoo Finance with caching support
- **Technical Indicators**: Calculate SMA, EMA, RSI, MACD, Bollinger Bands, and more
- **Multiple ML Models**: 
  - LSTM (Long Short-Term Memory) neural network
  - Random Forest Regressor
  - ARIMA/SARIMA statistical models
- **Model Evaluation**: RMSE, MAE, MAPE, R², and Direction Accuracy metrics
- **Visualization**: Static plots with Matplotlib and interactive dashboards with Plotly

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/mr-exploit/Stock-Price-Prediction-System.git
cd Stock-Price-Prediction-System
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from src.data.fetcher import fetch_stock_data
from src.data.preprocessor import preprocess_data
from src.models.lstm_model import LSTMPredictor
from src.visualization.plotter import plot_predictions

# Fetch data
df = fetch_stock_data('AAPL', '2018-01-01', '2024-12-31')

# Preprocess
X_train, X_test, y_train, y_test, scaler, features = preprocess_data(df)

# Train model
model = LSTMPredictor()
model.build_model(input_shape=(60, X_train.shape[2]))
history = model.train(X_train, y_train, epochs=100)

# Predict
predictions = model.predict(X_test)

# Visualize
plot_predictions(df, predictions, 'AAPL')
```

### Command Line Interface

```bash
# Train a model
python main.py --mode train --ticker AAPL --model lstm --epochs 100

# Predict future prices
python main.py --mode predict --ticker AAPL --days 30

# Evaluate a trained model
python main.py --mode evaluate --ticker AAPL --model lstm

# Compare multiple models
python main.py --mode compare --ticker AAPL --models lstm,rf,arima
```

## Project Structure

```
stock-prediction-system/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── config/
│   ├── __init__.py
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
├── src/
│   ├── __init__.py
│   ├── exceptions.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetcher.py          # Yahoo Finance data fetching
│   │   └── preprocessor.py     # Data preprocessing & feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py       # Base model class
│   │   ├── lstm_model.py       # LSTM implementation
│   │   ├── rf_model.py         # Random Forest implementation
│   │   └── arima_model.py      # ARIMA implementation
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py          # Evaluation metrics
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plotter.py          # Plotting functions
│   └── utils/
│       ├── __init__.py
│       ├── logger.py           # Logging utilities
│       └── helpers.py          # Helper functions
├── tests/
│   ├── __init__.py
│   ├── test_data_fetcher.py
│   ├── test_preprocessor.py
│   └── test_models.py
├── notebooks/
└── main.py                     # Main execution script
```

## Configuration

Edit `config/config.yaml` to customize:

```yaml
# Stock tickers to analyze
tickers:
  - AAPL
  - MSFT
  - GOOGL

# Data configuration
data:
  start_date: "2018-01-01"
  end_date: "2024-12-31"
  interval: "1d"
  train_test_split: 0.8

# Model configuration
model:
  type: "lstm"
  lstm:
    sequence_length: 60
    units: [50, 50]
    dropout: 0.2
    epochs: 100
    batch_size: 32
```

## Available Models

### 1. LSTM (Long Short-Term Memory)
Neural network designed for sequence prediction. Best for capturing long-term dependencies in time series data.

```python
from src.models.lstm_model import LSTMPredictor

model = LSTMPredictor(units=(50, 50), dropout=0.2)
model.build_model(input_shape=(sequence_length, n_features))
model.train(X_train, y_train, epochs=100)
```

### 2. Random Forest
Ensemble learning method using multiple decision trees.

```python
from src.models.rf_model import RandomForestPredictor

model = RandomForestPredictor(n_estimators=100, max_depth=10)
model.build_model()
model.train(X_train, y_train)
```

### 3. ARIMA
Classical statistical model for time series forecasting.

```python
from src.models.arima_model import ARIMAPredictor

model = ARIMAPredictor(order=(5, 1, 0))
model.build_model()
model.train(None, y_train)  # ARIMA only uses target values
forecast = model.forecast(steps=30)
```

## Technical Indicators

The system calculates the following technical indicators:

| Indicator | Description |
|-----------|-------------|
| SMA (20, 50, 200) | Simple Moving Average |
| EMA (12, 26) | Exponential Moving Average |
| MACD | Moving Average Convergence Divergence |
| RSI (14) | Relative Strength Index |
| Bollinger Bands | Price volatility bands |
| Volume MA | Volume Moving Average |
| ROC | Rate of Change |

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |
| MAPE | Mean Absolute Percentage Error |
| R² | Coefficient of Determination |
| Direction Accuracy | Percentage of correct up/down predictions |

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Performance Targets

- **RMSE**: < 5% of average stock price
- **MAPE**: < 10%
- **Direction Accuracy**: > 60%
- **R² Score**: > 0.85

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature/new-feature`
5. Submit a pull request

## Disclaimer

This software is for educational and research purposes only. Stock price predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always consult with a qualified financial advisor before making investment decisions.

## License

MIT License

## Author

Stock Prediction Team

---

**Document Version**: 1.0  
**Last Updated**: December 2024
