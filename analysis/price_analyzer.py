import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import statsmodels.api as sm
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from config.settings import PRICE_PREDICTION_MODEL
from database.connection import get_db_session
from database.models import Product, PriceHistory, PricePrediction

# Configure logging
logger = logging.getLogger(__name__)


class PriceAnalyzer:
    """Class for analyzing product prices and making predictions."""
    
    def __init__(self, model_type: str = PRICE_PREDICTION_MODEL):
        """
        Initialize the price analyzer.
        
        Args:
            model_type: Type of model to use for price prediction.
                        Options: "arima", "prophet", "lstm"
        """
        self.model_type = model_type
        logger.info(f"Initialized PriceAnalyzer with model type: {model_type}")
    
    def get_price_history(self, product_id: int) -> pd.DataFrame:
        """
        Get price history for a specific product.
        
        Args:
            product_id: Database ID of the product.
            
        Returns:
            DataFrame with price history.
        """
        session = get_db_session()
        if not session:
            return pd.DataFrame()
        
        try:
            # Get price history from database
            price_history = session.query(PriceHistory).filter(
                PriceHistory.product_id == product_id
            ).order_by(PriceHistory.date).all()
            
            # Convert to DataFrame
            history_data = [
                {"date": ph.date, "price": ph.price}
                for ph in price_history
            ]
            
            df = pd.DataFrame(history_data)
            
            # Get current product price if history is empty
            if df.empty:
                product = session.query(Product).filter(Product.id == product_id).first()
                if product:
                    df = pd.DataFrame([{
                        "date": datetime.now(),
                        "price": product.price
                    }])
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting price history: {str(e)}")
            return pd.DataFrame()
            
        finally:
            session.close()
    
    def analyze_pricing_trends(self, product_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze pricing trends for multiple products.
        
        Args:
            product_data: List of dictionaries containing product data.
            
        Returns:
            Dictionary with pricing analysis results.
        """
        if not product_data:
            return {"status": "error", "message": "No product data provided"}
        
        df = pd.DataFrame(product_data)
        
        try:
            # Basic statistics
            avg_price = df["price"].mean()
            min_price = df["price"].min()
            max_price = df["price"].max()
            median_price = df["price"].median()
            
            # Identify competitively priced products (below average)
            competitive_products = df[df["price"] < avg_price]
            
            # Identify potentially overpriced products (above average + std dev)
            price_threshold = avg_price + df["price"].std()
            overpriced_products = df[df["price"] > price_threshold]
            
            # Calculate price range by product name (for similar products)
            df["product_key"] = df["product_name"].str.extract(r"^([a-zA-Z0-9\s]+)")
            price_ranges = df.groupby("product_key").agg(
                min_price=("price", "min"),
                max_price=("price", "max"),
                avg_price=("price", "mean"),
                count=("price", "count")
            ).reset_index()
            
            # Filter out product_keys with only one product
            price_ranges = price_ranges[price_ranges["count"] > 1]
            
            results = {
                "status": "success",
                "statistics": {
                    "average_price": round(avg_price, 2),
                    "median_price": round(median_price, 2),
                    "min_price": round(min_price, 2),
                    "max_price": round(max_price, 2),
                    "price_range": round(max_price - min_price, 2)
                },
                "competitive_products": competitive_products[["product_id", "product_name", "price"]].to_dict("records"),
                "overpriced_products": overpriced_products[["product_id", "product_name", "price"]].to_dict("records"),
                "price_ranges": price_ranges.to_dict("records") if not price_ranges.empty else []
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing pricing trends: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def predict_price_arima(self, price_history: pd.DataFrame, forecast_steps: int = 30) -> pd.DataFrame:
        """
        Predict future prices using ARIMA model.
        
        Args:
            price_history: DataFrame with price history.
            forecast_steps: Number of days to forecast.
            
        Returns:
            DataFrame with price predictions.
        """
        try:
            # Check if we have enough data
            if len(price_history) < 5:
                # Not enough data, use simple moving average
                last_price = price_history["price"].iloc[-1]
                dates = [price_history["date"].iloc[-1] + timedelta(days=i) for i in range(1, forecast_steps + 1)]
                predictions = [last_price] * forecast_steps
                return pd.DataFrame({"date": dates, "predicted_price": predictions})
            
            # Prepare data for ARIMA
            ts = price_history.set_index("date")["price"]
            
            # Fit ARIMA model
            model = sm.tsa.ARIMA(ts, order=(5, 1, 0))
            model_fit = model.fit()
            
            # Make predictions
            forecast = model_fit.forecast(steps=forecast_steps)
            
            # Create DataFrame with predictions
            pred_dates = [price_history["date"].iloc[-1] + timedelta(days=i) for i in range(1, forecast_steps + 1)]
            predictions = pd.DataFrame({
                "date": pred_dates,
                "predicted_price": forecast.values
            })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting prices with ARIMA: {str(e)}")
            # Fallback to simple prediction
            last_price = price_history["price"].iloc[-1]
            dates = [price_history["date"].iloc[-1] + timedelta(days=i) for i in range(1, forecast_steps + 1)]
            predictions = [last_price] * forecast_steps
            return pd.DataFrame({"date": dates, "predicted_price": predictions})
    
    def predict_price_prophet(self, price_history: pd.DataFrame, forecast_steps: int = 30) -> pd.DataFrame:
        """
        Predict future prices using Facebook Prophet.
        
        Args:
            price_history: DataFrame with price history.
            forecast_steps: Number of days to forecast.
            
        Returns:
            DataFrame with price predictions.
        """
        try:
            # Check if we have enough data
            if len(price_history) < 2:
                # Not enough data, use simple moving average
                last_price = price_history["price"].iloc[-1]
                dates = [price_history["date"].iloc[-1] + timedelta(days=i) for i in range(1, forecast_steps + 1)]
                predictions = [last_price] * forecast_steps
                return pd.DataFrame({"date": dates, "predicted_price": predictions})
            
            # Prepare data for Prophet
            prophet_df = price_history.rename(columns={"date": "ds", "price": "y"})
            
            # Fit Prophet model
            model = Prophet(daily_seasonality=False, yearly_seasonality=True)
            model.fit(prophet_df)
            
            # Make predictions
            future = model.make_future_dataframe(periods=forecast_steps)
            forecast = model.predict(future)
            
            # Create DataFrame with predictions
            predictions = forecast[["ds", "yhat"]].tail(forecast_steps)
            predictions = predictions.rename(columns={"ds": "date", "yhat": "predicted_price"})
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting prices with Prophet: {str(e)}")
            # Fallback to simple prediction
            last_price = price_history["price"].iloc[-1]
            dates = [price_history["date"].iloc[-1] + timedelta(days=i) for i in range(1, forecast_steps + 1)]
            predictions = [last_price] * forecast_steps
            return pd.DataFrame({"date": dates, "predicted_price": predictions})
    
    def predict_price_lstm(self, price_history: pd.DataFrame, forecast_steps: int = 30) -> pd.DataFrame:
        """
        Predict future prices using LSTM neural network.
        
        Args:
            price_history: DataFrame with price history.
            forecast_steps: Number of days to forecast.
            
        Returns:
            DataFrame with price predictions.
        """
        try:
            # Check if we have enough data
            if len(price_history) < 10:
                # Not enough data for LSTM, fallback to Prophet
                return self.predict_price_prophet(price_history, forecast_steps)
            
            # Prepare data for LSTM
            prices = price_history["price"].values.reshape(-1, 1)
            
            # Normalize the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            prices_scaled = scaler.fit_transform(prices)
            
            # Create sequences
            seq_length = 5
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
            model.add(Dense(1))
            
            # Compile and fit model
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X, y, epochs=20, batch_size=1, verbose=0)
            
            # Generate predictions
            last_sequence = prices_scaled[-seq_length:].reshape(1, seq_length, 1)
            predicted_prices = []
            
            for _ in range(forecast_steps):
                next_pred = model.predict(last_sequence, verbose=0)[0, 0]
                predicted_prices.append(next_pred)
                
                # Update sequence for next prediction
                last_sequence = np.append(last_sequence[:, 1:, :], [[next_pred]], axis=1)
            
            # Inverse transform to get actual prices
            predicted_prices = np.array(predicted_prices).reshape(-1, 1)
            predicted_prices = scaler.inverse_transform(predicted_prices)
            
            # Create DataFrame with predictions
            pred_dates = [price_history["date"].iloc[-1] + timedelta(days=i) for i in range(1, forecast_steps + 1)]
            predictions = pd.DataFrame({
                "date": pred_dates,
                "predicted_price": predicted_prices.flatten()
            })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting prices with LSTM: {str(e)}")
            # Fallback to Prophet
            return self.predict_price_prophet(price_history, forecast_steps)
    
    def predict_price(self, product_id: int, forecast_days: int = 30) -> Dict[str, Any]:
        """
        Predict future prices for a specific product.
        
        Args:
            product_id: Database ID of the product.
            forecast_days: Number of days to forecast.
            
        Returns:
            Dictionary with price prediction results.
        """
        try:
            # Get price history
            price_history = self.get_price_history(product_id)
            
            if price_history.empty:
                return {
                    "status": "error",
                    "message": "No price history available for this product"
                }
            
            # Make predictions based on selected model
            if self.model_type == "arima":
                predictions = self.predict_price_arima(price_history, forecast_days)
            elif self.model_type == "lstm":
                predictions = self.predict_price_lstm(price_history, forecast_days)
            else:  # Default to Prophet
                predictions = self.predict_price_prophet(price_history, forecast_days)
            
            # Get current price
            current_price = price_history["price"].iloc[-1]
            
            # Format dates
            predictions["date_str"] = predictions["date"].dt.strftime("%Y-%m-%d")
            
            # Save predictions to database
            session = get_db_session()
            if session:
                try:
                    # Get latest prediction
                    next_month_pred = predictions.iloc[29]["predicted_price"] if len(predictions) >= 30 else predictions.iloc[-1]["predicted_price"]
                    
                    # Create new prediction record
                    new_prediction = PricePrediction(
                        product_id=product_id,
                        original_price=current_price,
                        predicted_price=float(next_month_pred),
                        prediction_date=datetime.now() + timedelta(days=30)
                    )
                    
                    session.add(new_prediction)
                    session.commit()
                    
                except Exception as e:
                    logger.error(f"Error saving prediction to database: {str(e)}")
                    session.rollback()
                    
                finally:
                    session.close()
            
            return {
                "status": "success",
                "current_price": float(current_price),
                "predictions": predictions[["date_str", "predicted_price"]].rename(
                    columns={"date_str": "date"}
                ).to_dict("records"),
                "predicted_price_next_month": float(predictions.iloc[29]["predicted_price"] if len(predictions) >= 30 else predictions.iloc[-1]["predicted_price"]),
                "model_used": self.model_type
            }
            
        except Exception as e:
            logger.error(f"Error predicting price: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def compare_product_prices(self, product_ids: List[int]) -> Dict[str, Any]:
        """
        Compare prices of multiple products.
        
        Args:
            product_ids: List of product IDs to compare.
            
        Returns:
            Dictionary with comparison results.
        """
        session = get_db_session()
        if not session:
            return {"status": "error", "message": "Database connection failed"}
        
        try:
            products = session.query(Product).filter(Product.id.in_(product_ids)).all()
            
            if not products:
                return {"status": "error", "message": "No products found with the given IDs"}
            
            # Get product details
            product_details = [
                {
                    "id": p.id,
                    "product_id": p.product_id,
                    "product_name": p.product_name,
                    "price": p.price,
                    "rating": p.rating,
                    "source": p.source
                }
                for p in products
            ]
            
            # Calculate statistics
            df = pd.DataFrame(product_details)
            avg_price = df["price"].mean()
            price_range = df["price"].max() - df["price"].min()
            price_std = df["price"].std()
            
            # Find best value (highest rating/price ratio)
            df["value_score"] = df["rating"] / df["price"]
            best_value = df.loc[df["value_score"].idxmax()]
            
            # Find cheapest and most expensive
            cheapest = df.loc[df["price"].idxmin()]
            most_expensive = df.loc[df["price"].idxmax()]
            
            return {
                "status": "success",
                "products": product_details,
                "statistics": {
                    "average_price": round(avg_price, 2),
                    "price_range": round(price_range, 2),
                    "price_std": round(price_std, 2)
                },
                "best_value": {
                    "product_id": best_value["product_id"],
                    "product_name": best_value["product_name"],
                    "price": best_value["price"],
                    "rating": best_value["rating"],
                    "value_score": round(best_value["value_score"], 4)
                },
                "cheapest": {
                    "product_id": cheapest["product_id"],
                    "product_name": cheapest["product_name"],
                    "price": cheapest["price"]
                },
                "most_expensive": {
                    "product_id": most_expensive["product_id"],
                    "product_name": most_expensive["product_name"],
                    "price": most_expensive["price"]
                }
            }
            
        except Exception as e:
            logger.error(f"Error comparing product prices: {str(e)}")
            return {"status": "error", "message": str(e)}
            
        finally:
            session.close()