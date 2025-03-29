import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from api.main import app
from database.models import Product, SentimentAnalysis, PricePrediction


# Setup test client
client = TestClient(app)


@pytest.fixture
def mock_db_session():
    """
    Fixture to mock database session.
    """
    with patch("api.routers.products.get_db_session") as mock_db, \
         patch("api.routers.price_predictions.get_db_session") as mock_price_db, \
         patch("api.routers.sentiment.get_db_session") as mock_sentiment_db:
        # Create mock session
        session = MagicMock()
        
        # Return same mock for all routes
        mock_db.return_value = session
        mock_price_db.return_value = session
        mock_sentiment_db.return_value = session
        
        yield session


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "version" in response.json()


def test_health():
    """Test health check endpoint."""
    with patch("api.main.get_db_session") as mock_db:
        mock_db.return_value = MagicMock()
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert "database" in response.json()["dependencies"]


def test_get_products(mock_db_session):
    """Test get products endpoint."""
    # Setup mock products
    mock_product = MagicMock(spec=Product)
    mock_product.id = 1
    mock_product.product_id = "123"
    mock_product.product_name = "Test Product"
    mock_product.price = 99.99
    mock_product.rating = 4.5
    mock_product.num_reviews = 100
    mock_product.source = "Amazon"
    mock_product.product_url = "https://example.com"
    mock_product.created_at = "2023-01-01T00:00:00"
    mock_product.updated_at = "2023-01-01T00:00:00"
    
    # Mock query
    mock_query = MagicMock()
    mock_query.count.return_value = 1
    mock_query.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [mock_product]
    mock_db_session.query.return_value = mock_query
    
    # Test
    response = client.get("/api/products")
    
    # Verify
    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert len(response.json()["items"]) == 1
    assert response.json()["items"][0]["product_name"] == "Test Product"


def test_get_product_by_name(mock_db_session):
    """Test get product by name endpoint."""
    # Setup mock product
    mock_product = MagicMock(spec=Product)
    mock_product.id = 1
    mock_product.product_id = "123"
    mock_product.product_name = "Test Product"
    mock_product.price = 99.99
    mock_product.rating = 4.5
    mock_product.sentiment_analyses = []
    
    # Mock query
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_product
    
    # Test
    response = client.get("/api/product?name=Test%20Product")
    
    # Verify
    assert response.status_code == 200
    assert response.json()["product_name"] == "Test Product"
    assert response.json()["price"] == 99.99
    assert response.json()["rating"] == 4.5


def test_get_product_by_name_not_found(mock_db_session):
    """Test get product by name when not found."""
    # Mock query to return None
    mock_db_session.query.return_value.filter.return_value.first.return_value = None
    
    # Test
    response = client.get("/api/product?name=Nonexistent%20Product")
    
    # Verify
    assert response.status_code == 404
    assert "detail" in response.json()


@patch("api.routers.products.AmazonScraper")
def test_scrape_products(mock_scraper, mock_db_session):
    """Test scraping products endpoint."""
    # Setup mock
    scraper_instance = MagicMock()
    mock_scraper.return_value = scraper_instance
    
    # Mock scrape method
    mock_products = [
        {
            "product_id": "123",
            "product_name": "Test Product",
            "price": 99.99,
            "rating": 4.5,
            "num_reviews": 100,
            "reviews": "Great product",
            "source": "Amazon",
            "product_url": "https://example.com"
        }
    ]
    scraper_instance.scrape_products.return_value = mock_products
    
    # Mock save to database
    with patch("api.routers.products.save_to_database") as mock_save:
        mock_save.return_value = True
        
        # Test
        response = client.post("/api/scrape?query=test&source=amazon&max_results=1")
        
        # Verify
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert response.json()["count"] == 1


def test_get_price_prediction(mock_db_session):
    """Test price prediction endpoint."""
    # Setup mock product
    mock_product = MagicMock(spec=Product)
    mock_product.id = 1
    mock_product.product_name = "Test Product"
    
    # Mock price prediction
    mock_prediction = MagicMock(spec=PricePrediction)
    mock_prediction.original_price = 99.99
    mock_prediction.predicted_price = 89.99
    
    # Mock queries
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_product
    mock_db_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = mock_prediction
    
    # Test
    response = client.get("/api/price-prediction?product=Test%20Product")
    
    # Verify
    assert response.status_code == 200
    assert response.json()["product_name"] == "Test Product"
    assert response.json()["current_price"] == 99.99
    assert response.json()["predicted_price_next_month"] == 89.99


@patch("api.routers.price_predictions.PriceAnalyzer")
def test_get_price_trend(mock_analyzer, mock_db_session):
    """Test price trend endpoint."""
    # Setup mock product
    mock_product = MagicMock(spec=Product)
    mock_product.id = 1
    mock_product.product_name = "Test Product"
    
    # Mock query
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_product
    
    # Mock price analyzer
    analyzer_instance = MagicMock()
    mock_analyzer.return_value = analyzer_instance
    
    mock_predictions = {
        "status": "success",
        "predictions": [
            {"date": "2023-01-01", "predicted_price": 99.99},
            {"date": "2023-01-02", "predicted_price": 98.99}
        ]
    }
    analyzer_instance.predict_price.return_value = mock_predictions
    
    # Test
    response = client.get("/api/price-trend/1?days=2")
    
    # Verify
    assert response.status_code == 200
    assert len(response.json()) == 2
    assert response.json()[0]["date"] == "2023-01-01"
    assert response.json()[1]["predicted_price"] == 98.99


def test_get_sentiment(mock_db_session):
    """Test sentiment analysis endpoint."""
    # Setup mock product
    mock_product = MagicMock(spec=Product)
    mock_product.id = 1
    mock_product.product_name = "Test Product"
    
    # Mock sentiment analysis
    mock_sentiment = MagicMock(spec=SentimentAnalysis)
    mock_sentiment.sentiment_label = "Positive"
    mock_sentiment.sentiment_score = 0.8
    mock_sentiment.top_issues = "battery life,heating"
    
    # Mock queries
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_product
    mock_db_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = mock_sentiment
    
    # Test
    response = client.get("/api/sentiment?product=Test%20Product")
    
    # Verify
    assert response.status_code == 200
    assert response.json()["product_name"] == "Test Product"
    assert response.json()["sentiment"] == "Positive"
    assert response.json()["sentiment_score"] == 0.8
    assert "battery life" in response.json()["top_issues"]


@patch("api.routers.sentiment.SentimentAnalyzer")
def test_analyze_text(mock_analyzer):
    """Test text sentiment analysis endpoint."""
    # Mock sentiment analyzer
    analyzer_instance = MagicMock()
    mock_analyzer.return_value = analyzer_instance
    
    analyzer_instance.predict_sentiment.return_value = ("Positive", 0.8)
    analyzer_instance.extract_topics.return_value = ["great", "product"]
    
    # Test
    response = client.post("/api/analyze-text", json={"text": "This is a great product"})
    
    # Verify
    assert response.status_code == 200
    assert response.json()["sentiment"] == "Positive"
    assert response.json()["sentiment_score"] == 0.8
    assert "great" in response.json()["topics"]