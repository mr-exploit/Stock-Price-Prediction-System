"""ARIMA model for stock price prediction."""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import warnings

from src.models.base_model import StockPredictionModel
from src.utils.logger import get_logger
from src.exceptions import ModelTrainingError

logger = get_logger(__name__)

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=UserWarning)


class ARIMAPredictor(StockPredictionModel):
    """
    ARIMA-based stock price predictor.
    
    ARIMA (AutoRegressive Integrated Moving Average) is a classical
    statistical model for time series forecasting.
    """
    
    def __init__(
        self,
        order: Tuple[int, int, int] = (5, 1, 0),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    ):
        """
        Initialize ARIMA predictor.
        
        Args:
            order: (p, d, q) order of the model
                p: number of AR terms
                d: number of differences
                q: number of MA terms
            seasonal_order: (P, D, Q, s) seasonal order (for SARIMA)
        """
        super().__init__(model_type="arima")
        self.order = order
        self.seasonal_order = seasonal_order
        self.fitted_model = None
    
    def build_model(self, input_shape: Tuple = None) -> None:
        """
        Build the ARIMA model structure.
        
        Note: ARIMA models are built during fitting, this method
        is kept for interface consistency.
        
        Args:
            input_shape: Not used for ARIMA
        """
        logger.info(f"ARIMA model configured with order {self.order}")
        if self.seasonal_order:
            logger.info(f"Seasonal order: {self.seasonal_order}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict:
        """
        Train the ARIMA model.
        
        Note: ARIMA uses only the target values (y_train), not features.
        
        Args:
            X_train: Not used for ARIMA (kept for interface consistency)
            y_train: Training target values (time series)
            X_val: Not used
            y_val: Validation targets for metrics
        
        Returns:
            Training metrics dictionary
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError:
            raise ImportError("statsmodels is required for ARIMA. Install with: pip install statsmodels")
        
        try:
            # Fit ARIMA or SARIMA model
            if self.seasonal_order:
                model = SARIMAX(
                    y_train,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
            else:
                model = ARIMA(
                    y_train,
                    order=self.order,
                )
            
            self.fitted_model = model.fit(disp=False)
            self.model = self.fitted_model
            self.is_fitted = True
            
            # Get model metrics
            aic = self.fitted_model.aic
            bic = self.fitted_model.bic
            
            # Calculate in-sample metrics
            fitted_values = self.fitted_model.fittedvalues
            residuals = y_train - fitted_values
            mse = np.mean(residuals ** 2)
            mae = np.mean(np.abs(residuals))
            
            self.history = {
                "aic": aic,
                "bic": bic,
                "train_mse": mse,
                "train_mae": mae,
            }
            
            logger.info(f"ARIMA training completed. AIC: {aic:.2f}, BIC: {bic:.2f}")
            
            return self.history
        
        except Exception as e:
            raise ModelTrainingError(f"ARIMA training failed: {str(e)}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make in-sample predictions (fitted values).
        
        Args:
            X: Not used (kept for interface consistency)
        
        Returns:
            Fitted values
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.fitted_model.fittedvalues
    
    def forecast(self, steps: int) -> np.ndarray:
        """
        Forecast future values.
        
        Args:
            steps: Number of steps to forecast
        
        Returns:
            Forecasted values
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        forecast = self.fitted_model.forecast(steps=steps)
        return np.array(forecast)
    
    def predict_with_confidence(
        self,
        steps: int,
        alpha: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forecast with confidence intervals.
        
        Args:
            steps: Number of steps to forecast
            alpha: Significance level (default 0.05 for 95% CI)
        
        Returns:
            Tuple of (forecast, lower_bound, upper_bound)
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        forecast_result = self.fitted_model.get_forecast(steps=steps)
        forecast = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=alpha)
        
        return (
            np.array(forecast),
            np.array(conf_int.iloc[:, 0]),
            np.array(conf_int.iloc[:, 1]),
        )
    
    def get_params(self) -> Dict:
        """Get model parameters."""
        params = super().get_params()
        params.update({
            "order": self.order,
            "seasonal_order": self.seasonal_order,
        })
        if self.is_fitted:
            params["aic"] = self.fitted_model.aic
            params["bic"] = self.fitted_model.bic
        return params


def auto_arima(
    y: np.ndarray,
    max_p: int = 5,
    max_d: int = 2,
    max_q: int = 5,
    criterion: str = "aic",
) -> ARIMAPredictor:
    """
    Automatically find the best ARIMA order.
    
    Args:
        y: Time series data
        max_p: Maximum AR order
        max_d: Maximum difference order
        max_q: Maximum MA order
        criterion: Selection criterion ('aic' or 'bic')
    
    Returns:
        ARIMAPredictor with best order
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        raise ImportError("statsmodels is required for auto_arima")
    
    best_score = np.inf
    best_order = (1, 1, 0)
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(y, order=(p, d, q))
                    fitted = model.fit(disp=False)
                    
                    score = fitted.aic if criterion == "aic" else fitted.bic
                    
                    if score < best_score:
                        best_score = score
                        best_order = (p, d, q)
                except Exception:
                    continue
    
    logger.info(f"Best ARIMA order: {best_order} ({criterion}: {best_score:.2f})")
    
    predictor = ARIMAPredictor(order=best_order)
    return predictor
