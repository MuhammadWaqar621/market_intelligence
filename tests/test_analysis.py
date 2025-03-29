import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from analysis.price_analyzer import PriceAnalyzer
from analysis.sentiment_analyzer import SentimentAnalyzer


class TestPriceAnalyzer:
    """Test cases for PriceAnalyzer."""
    
    def test_init(self):
        """Test initialization."""
        analyzer = PriceAnalyzer("prophet")
        assert analyzer.model_type == "prophet"
    
    @patch("analysis.price_analyzer.get_db_session")
    def test_get_price_history_empty(self, mock_get_db):
        """Test getting empty price history."""
        # Setup mock
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = []
        mock_get_db.return_value = mock_session
        
        # Test
        analyzer = PriceAnalyzer()
        df = analyzer.get_price_history(1)
        
        # Verify
        assert df.empty
        mock_get_db.assert_called_once()
    
    def test_analyze_pricing_trends(self):
        """Test analyzing pricing trends."""
        # Setup test data
        product_data = [
            {"product_id": "1", "product_name": "Test 1", "price": 99.99, "rating": 4.5},
            {"product_id": "2", "product_name": "Test 2", "price": 199.99, "rating": 4.0},
            {"product_id": "3", "product_name": "Test 3", "price": 149.99, "rating": 4.8}
        ]
        
        # Test
        analyzer = PriceAnalyzer()
        results = analyzer.analyze_pricing_trends(product_data)
        
        # Verify
        assert results["status"] == "success"
        assert "statistics" in results
        assert "competitive_products" in results
        assert "overpriced_products" in results
        assert abs(results["statistics"]["average_price"] - 149.99) < 0.01
    
    @patch("analysis.price_analyzer.Prophet")
    def test_predict_price_prophet(self, mock_prophet):
        """Test predicting prices with Prophet."""
        # Setup mock
        mock_model = MagicMock()
        mock_prophet.return_value = mock_model
        mock_model.fit.return_value = mock_model
        
        # Mock future dataframe
        future_df = pd.DataFrame({
            "ds": [datetime.now() + timedelta(days=i) for i in range(1, 31)],
            "yhat": [100.0 + i for i in range(30)]
        })
        mock_model.make_future_dataframe.return_value = future_df
        mock_model.predict.return_value = future_df
        
        # Setup test data
        price_history = pd.DataFrame({
            "date": [datetime.now() - timedelta(days=i) for i in range(10)],
            "price": [100.0 - i for i in range(10)]
        })
        
        # Test
        analyzer = PriceAnalyzer("prophet")
        results = analyzer.predict_price_prophet(30)
        
        # Verify
        assert results["status"] == "success"
        assert "predictions" in results
        assert len(results["predictions"]) == 30
    
    @patch("analysis.price_analyzer.get_db_session")
    def test_predict_price(self, mock_get_db):
        """Test predicting prices."""
        # Setup mocks
        mock_session = MagicMock()
        mock_product = MagicMock()
        mock_product.price = 100.0
        
        mock_session.query.return_value.filter.return_value.first.return_value = mock_product
        mock_get_db.return_value = mock_session
        
        # Mock price history
        with patch("analysis.price_analyzer.PriceAnalyzer.get_price_history") as mock_get_history:
            mock_history = pd.DataFrame({
                "date": [datetime.now() - timedelta(days=i) for i in range(10)],
                "price": [100.0 - i for i in range(10)]
            })
            mock_get_history.return_value = mock_history
            
            # Mock predict_price_prophet
            with patch("analysis.price_analyzer.PriceAnalyzer.predict_price_prophet") as mock_predict:
                mock_predictions = pd.DataFrame({
                    "date": [datetime.now() + timedelta(days=i) for i in range(1, 31)],
                    "predicted_price": [100.0 + i for i in range(30)]
                })
                mock_predict.return_value = {
                    "status": "success",
                    "predictions": mock_predictions.to_dict("records")
                }
                
                # Test
                analyzer = PriceAnalyzer()
                results = analyzer.predict_price(1)
                
                # Verify
                assert results["status"] == "success"
                assert "current_price" in results
                assert "predictions" in results
                assert "predicted_price_next_month" in results


class TestSentimentAnalyzer:
    """Test cases for SentimentAnalyzer."""
    
    @patch("analysis.sentiment_analyzer.AutoTokenizer.from_pretrained")
    @patch("analysis.sentiment_analyzer.AutoModelForSequenceClassification.from_pretrained")
    def test_init(self, mock_model, mock_tokenizer):
        """Test initialization."""
        # Setup mocks
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        
        # Test
        analyzer = SentimentAnalyzer("distilbert-base-uncased")
        
        # Verify
        assert analyzer.model_name == "distilbert-base-uncased"
        assert analyzer.use_transformers is True
        assert analyzer.tokenizer is not None
        assert analyzer.model is not None
    
    def test_preprocess_text(self):
        """Test text preprocessing."""
        analyzer = SentimentAnalyzer(use_transformers=False)
        
        # Test
        preprocessed = analyzer.preprocess_text("This is a test! 123")
        
        # Verify
        assert "test" in preprocessed
        assert "!" not in preprocessed
        assert "123" not in preprocessed
    
    @patch("torch.nn.functional.softmax")
    def test_predict_sentiment_transformer(self, mock_softmax):
        """Test sentiment prediction with transformer."""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.tensor([[0.1, 0.2, 0.7]])
        mock_model.return_value = mock_outputs
        
        mock_probs = torch.tensor([[0.1, 0.2, 0.7]])
        mock_softmax.return_value = mock_probs
        
        # Initialize analyzer
        analyzer = SentimentAnalyzer(use_transformers=True)
        analyzer.tokenizer = mock_tokenizer
        analyzer.model = mock_model
        
        # Test
        label, score = analyzer.predict_sentiment_transformer("This is a positive test.")
        
        # Verify
        assert label == "Positive"
        assert score > 0
    
    @patch("analysis.sentiment_analyzer.get_db_session")
    def test_analyze_product_reviews(self, mock_get_db):
        """Test analyzing product reviews."""
        # Setup mocks
        mock_session = MagicMock()
        mock_product = MagicMock()
        mock_product.product_name = "Test Product"
        mock_product.reviews = "This is a great product! | It could be better though."
        
        mock_session.query.return_value.filter.return_value.first.return_value = mock_product
        mock_get_db.return_value = mock_session
        
        # Mock predict_sentiment
        with patch("analysis.sentiment_analyzer.SentimentAnalyzer.predict_sentiment") as mock_predict:
            mock_predict.side_effect = [("Positive", 0.8), ("Neutral", 0.2)]
            
            # Mock extract_topics
            with patch("analysis.sentiment_analyzer.SentimentAnalyzer.extract_topics") as mock_extract:
                mock_extract.side_effect = [["great", "product"], ["better"]]
                
                # Test
                analyzer = SentimentAnalyzer(use_transformers=False)
                results = analyzer.analyze_product_reviews(1)
                
                # Verify
                assert results["status"] == "success"
                assert results["product_name"] == "Test Product"
                assert results["sentiment"] == "Positive"
                assert results["sentiment_score"] > 0
                assert "great" in results["top_praises"]
                assert "better" in results["top_issues"]