import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import pickle

# Time series forecasting models
import statsmodels.api as sm
from prophet import Prophet
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

from config.settings import MODEL_CACHE_DIR

# Configure logging
logger = logging.getLogger(__name__)


class PriceModel:
    """Base class for price forecasting models."""
    
    def __init__(self, model_type: str = "prophet"):
        """
        Initialize the price model.
        
        Args:
            model_type: Type of model to use. Options: "arima", "prophet", "lstm"
        """
        self.model_type = model_type
        self.model = None
        self.scaler = None  # For LSTM
        
        # Create cache directory if it doesn't exist
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        
        logger.info(f"Initialized price model of type: {model_type}")
    
    def prepare_data(self, 
                     df: pd.DataFrame, 
                     date_col: str = "date", 
                     price_col: str = "price") -> pd.DataFrame:
        """
        Prepare data for training and prediction.
        
        Args:
            df: DataFrame with price history.
            date_col: Name of the date column.
            price_col: Name of the price column.
            
        Returns:
            Prepared DataFrame.
        """
        # Ensure date column is datetime type
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Sort by date
        df = df.sort_values(by=date_col)
        
        # Reset index
        df = df.reset_index(drop=True)
        
        # Handle missing values if any
        df[price_col] = df[price_col].fillna(method="ffill")
        
        # Handle duplicate dates by taking the mean
        df = df.groupby(date_col)[price_col].mean().reset_index()
        
        return df
    
    def train_arima(self, 
                    df: pd.DataFrame, 
                    date_col: str = "date", 
                    price_col: str = "price") -> Dict[str, Any]:
        """
        Train ARIMA model.
        
        Args:
            df: DataFrame with price history.
            date_col: Name of the date column.
            price_col: Name of the price column.
            
        Returns:
            Dictionary with training results.
        """
        try:
            # Prepare data
            df = self.prepare_data(df, date_col, price_col)
            
            # Set date as index
            ts = df.set_index(date_col)[price_col]
            
            # Fit ARIMA model
            model = sm.tsa.ARIMA(ts, order=(5, 1, 0))
            self.model = model.fit()
            
            return {
                "status": "success",
                "model_type": "arima",
                "model_params": {
                    "order": (5, 1, 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error training ARIMA model: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def train_prophet(self, 
                      df: pd.DataFrame, 
                      date_col: str = "date", 
                      price_col: str = "price") -> Dict[str, Any]:
        """
        Train Prophet model.
        
        Args:
            df: DataFrame with price history.
            date_col: Name of the date column.
            price_col: Name of the price column.
            
        Returns:
            Dictionary with training results.
        """
        try:
            # Prepare data
            df = self.prepare_data(df, date_col, price_col)
            
            # Rename columns for Prophet
            prophet_df = df.rename(columns={date_col: "ds", price_col: "y"})
            
            # Fit Prophet model
            model = Prophet(daily_seasonality=False, yearly_seasonality=True)
            self.model = model.fit(prophet_df)
            
            return {
                "status": "success",
                "model_type": "prophet",
                "model_params": {
                    "daily_seasonality": False,
                    "yearly_seasonality": True
                }
            }
            
        except Exception as e:
            logger.error(f"Error training Prophet model: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def train_lstm(self, 
                   df: pd.DataFrame, 
                   date_col: str = "date", 
                   price_col: str = "price",
                   seq_length: int = 5,
                   epochs: int = 50,
                   batch_size: int = 32) -> Dict[str, Any]:
        """
        Train LSTM model.
        
        Args:
            df: DataFrame with price history.
            date_col: Name of the date column.
            price_col: Name of the price column.
            seq_length: Length of input sequences.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            
        Returns:
            Dictionary with training results.
        """
        try:
            # Prepare data
            df = self.prepare_data(df, date_col, price_col)
            
            # Check if we have enough data
            if len(df) < seq_length + 2:
                return {
                    "status": "error",
                    "message": f"Not enough data for LSTM. Need at least {seq_length + 2} data points."
                }
            
            # Extract prices
            prices = df[price_col].values.reshape(-1, 1)
            
            # Normalize the data
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            prices_scaled = self.scaler.fit_transform(prices)
            
            # Create sequences
            X = []
            y = []
            
            for i in range(len(prices_scaled) - seq_length):
                X.append(prices_scaled[i:i + seq_length, 0])
                y.append(prices_scaled[i + seq_length, 0])
            
            X = np.array(X)
            y = np.array(y)
            
            # Reshape input to be [samples, time steps, features]
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            # Build LSTM model
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
            model.add(LSTM(50))
            model.add(Dropout(0.2))
            model.add(Dense(1))
            
            # Compile model
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Train model
            history = model.fit(
                X, 
                y, 
                epochs=epochs, 
                batch_size=batch_size, 
                validation_split=0.1,
                verbose=0
            )
            
            self.model = model
            
            return {
                "status": "success",
                "model_type": "lstm",
                "model_params": {
                    "seq_length": seq_length,
                    "epochs": epochs,
                    "batch_size": batch_size
                },
                "training_loss": float(history.history["loss"][-1]),
                "validation_loss": float(history.history["val_loss"][-1]) if "val_loss" in history.history else None
            }
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def train(self, 
              df: pd.DataFrame, 
              date_col: str = "date", 
              price_col: str = "price") -> Dict[str, Any]:
        """
        Train price forecasting model.
        
        Args:
            df: DataFrame with price history.
            date_col: Name of the date column.
            price_col: Name of the price column.
            
        Returns:
            Dictionary with training results.
        """
        if self.model_type == "arima":
            return self.train_arima(df, date_col, price_col)
        elif self.model_type == "lstm":
            return self.train_lstm(df, date_col, price_col)
        else:  # Default to Prophet
            return self.train_prophet(df, date_col, price_col)
    
    def predict_arima(self, 
                     steps: int = 30) -> Dict[str, Any]:
        """
        Make predictions using ARIMA model.
        
        Args:
            steps: Number of steps to forecast.
            
        Returns:
            Dictionary with prediction results.
        """
        if self.model is None:
            return {"status": "error", "message": "Model not trained"}
            
        try:
            # Make predictions
            forecast = self.model.forecast(steps=steps)
            
            # Create prediction dates
            last_date = self.model.data.dates[-1]
            pred_dates = pd.date_range(start=last_date, periods=steps + 1)[1:]
            
            # Create DataFrame with predictions
            predictions = pd.DataFrame({
                "date": pred_dates,
                "predicted_price": forecast.values
            })
            
            return {
                "status": "success",
                "predictions": predictions.to_dict("records")
            }
            
        except Exception as e:
            logger.error(f"Error making ARIMA predictions: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def predict_prophet(self, 
                       steps: int = 30) -> Dict[str, Any]:
        """
        Make predictions using Prophet model.
        
        Args:
            steps: Number of steps to forecast.
            
        Returns:
            Dictionary with prediction results.
        """
        if self.model is None:
            return {"status": "error", "message": "Model not trained"}
            
        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=steps)
            
            # Make predictions
            forecast = self.model.predict(future)
            
            # Get only future predictions
            predictions = forecast[["ds", "yhat"]].tail(steps)
            predictions = predictions.rename(columns={"ds": "date", "yhat": "predicted_price"})
            
            return {
                "status": "success",
                "predictions": predictions.to_dict("records")
            }
            
        except Exception as e:
            logger.error(f"Error making Prophet predictions: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def predict_lstm(self, 
                    last_sequence: np.ndarray,
                    steps: int = 30) -> Dict[str, Any]:
        """
        Make predictions using LSTM model.
        
        Args:
            last_sequence: Last sequence of prices (scaled).
            steps: Number of steps to forecast.
            
        Returns:
            Dictionary with prediction results.
        """
        if self.model is None or self.scaler is None:
            return {"status": "error", "message": "Model not trained or scaler not initialized"}
            
        try:
            # Make sure last_sequence is properly shaped
            if len(last_sequence.shape) < 3:
                last_sequence = last_sequence.reshape(1, -1, 1)
            
            # Generate predictions
            pred_dates = [datetime.now() + timedelta(days=i) for i in range(1, steps + 1)]
            predicted_prices = []
            
            current_sequence = last_sequence.copy()
            
            for _ in range(steps):
                # Predict next value
                next_pred = self.model.predict(current_sequence, verbose=0)[0, 0]
                predicted_prices.append(next_pred)
                
                # Update sequence for next prediction
                current_sequence = np.append(current_sequence[:, 1:, :], 
                                           [[next_pred]], 
                                           axis=1)
            
            # Inverse transform to get actual prices
            predicted_prices = np.array(predicted_prices).reshape(-1, 1)
            predicted_prices = self.scaler.inverse_transform(predicted_prices)
            
            # Create DataFrame with predictions
            predictions = pd.DataFrame({
                "date": pred_dates,
                "predicted_price": predicted_prices.flatten()
            })
            
            return {
                "status": "success",
                "predictions": predictions.to_dict("records")
            }
            
        except Exception as e:
            logger.error(f"Error making LSTM predictions: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def predict(self, 
               df: Optional[pd.DataFrame] = None,
               date_col: str = "date", 
               price_col: str = "price",
               steps: int = 30) -> Dict[str, Any]:
        """
        Make predictions using the trained model.
        
        Args:
            df: Optional DataFrame with price history (for preparing LSTM sequence).
            date_col: Name of the date column.
            price_col: Name of the price column.
            steps: Number of steps to forecast.
            
        Returns:
            Dictionary with prediction results.
        """
        if self.model is None:
            return {"status": "error", "message": "Model not trained"}
            
        if self.model_type == "arima":
            return self.predict_arima(steps)
        elif self.model_type == "lstm":
            if df is None:
                return {"status": "error", "message": "DataFrame required for LSTM predictions"}
                
            # Prepare data for LSTM
            df = self.prepare_data(df, date_col, price_col)
            prices = df[price_col].values.reshape(-1, 1)
            prices_scaled = self.scaler.transform(prices)
            
            seq_length = 5  # Same as in training
            last_sequence = prices_scaled[-seq_length:].reshape(1, seq_length, 1)
            
            return self.predict_lstm(last_sequence, steps)
        else:  # Default to Prophet
            return self.predict_prophet(steps)
    
    def save_model(self, path: str) -> Dict[str, Any]:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model.
            
        Returns:
            Dictionary with saving results.
        """
        if self.model is None:
            return {"status": "error", "message": "No model to save"}
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
            
            # Save model based on type
            if self.model_type == "arima":
                # Save ARIMA model
                with open(os.path.join(path, "arima_model.pkl"), "wb") as f:
                    pickle.dump(self.model, f)
                
            elif self.model_type == "prophet":
                # Save Prophet model
                with open(os.path.join(path, "prophet_model.pkl"), "wb") as f:
                    pickle.dump(self.model, f)
                
            elif self.model_type == "lstm":
                # Save LSTM model
                self.model.save(os.path.join(path, "lstm_model"))
                
                # Save scaler
                with open(os.path.join(path, "lstm_scaler.pkl"), "wb") as f:
                    pickle.dump(self.scaler, f)
            
            # Save model type
            with open(os.path.join(path, "model_type.txt"), "w") as f:
                f.write(self.model_type)
            
            return {
                "status": "success",
                "model_path": path,
                "model_type": self.model_type
            }
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def load_model(self, path: str) -> Dict[str, Any]:
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from.
            
        Returns:
            Dictionary with loading results.
        """
        try:
            # Load model type
            with open(os.path.join(path, "model_type.txt"), "r") as f:
                self.model_type = f.read().strip()
            
            # Load model based on type
            if self.model_type == "arima":
                # Load ARIMA model
                with open(os.path.join(path, "arima_model.pkl"), "rb") as f:
                    self.model = pickle.load(f)
                
            elif self.model_type == "prophet":
                # Load Prophet model
                with open(os.path.join(path, "prophet_model.pkl"), "rb") as f:
                    self.model = pickle.load(f)
                
            elif self.model_type == "lstm":
                # Load LSTM model
                self.model = load_model(os.path.join(path, "lstm_model"))
                
                # Load scaler
                with open(os.path.join(path, "lstm_scaler.pkl"), "rb") as f:
                    self.scaler = pickle.load(f)
            
            return {
                "status": "success",
                "model_path": path,
                "model_type": self.model_type
            }
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return {"status": "error", "message": str(e)}