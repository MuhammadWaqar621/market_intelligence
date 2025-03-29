import pytest
import os
import pandas as pd
from unittest.mock import patch, MagicMock

from scrapers.base_scraper import BaseScraper
from scrapers.amazon_scraper import AmazonScraper
from scrapers.ebay_scraper import EbayScraper
from scrapers.utils import save_to_csv, convert_to_dataframe


class TestBaseScraper:
    """Test cases for BaseScraper."""
    
    def test_init(self):
        """Test initialization."""
        scraper = BaseScraper("Test Platform")
        assert scraper.platform_name == "Test Platform"
        assert scraper.session is not None
    
    @patch("scrapers.base_scraper.requests.Session.get")
    def test_make_request(self, mock_get):
        """Test making a request."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"<html><body>Test HTML</body></html>"
        mock_get.return_value = mock_response
        
        # Test
        scraper = BaseScraper("Test Platform")
        soup = scraper._make_request("https://example.com")
        
        # Verify
        assert soup is not None
        assert soup.find("body").text == "Test HTML"
        mock_get.assert_called_once()
    
    @patch("scrapers.base_scraper.requests.Session.get")
    def test_make_request_failed(self, mock_get):
        """Test failed request."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        # Test
        scraper = BaseScraper("Test Platform")
        soup = scraper._make_request("https://example.com")
        
        # Verify
        assert soup is None
        mock_get.assert_called_once()
    
    @patch("scrapers.base_scraper.requests.Session.get")
    def test_make_request_exception(self, mock_get):
        """Test exception during request."""
        # Setup mock
        mock_get.side_effect = Exception("Test exception")
        
        # Test
        scraper = BaseScraper("Test Platform")
        soup = scraper._make_request("https://example.com")
        
        # Verify
        assert soup is None
        mock_get.assert_called_once()


class TestAmazonScraper:
    """Test cases for AmazonScraper."""
    
    @patch("scrapers.amazon_scraper.AmazonScraper._make_request")
    def test_search_products(self, mock_make_request):
        """Test searching for products."""
        # Setup mock
        mock_soup = MagicMock()
        mock_card = MagicMock()
        mock_link = MagicMock()
        mock_link.attrs = {"href": "/dp/B07XYZ123"}
        
        mock_card.select_one.return_value = mock_link
        mock_soup.select.return_value = [mock_card]
        mock_make_request.return_value = mock_soup
        
        # Test
        scraper = AmazonScraper()
        urls = scraper.search_products("test", 1)
        
        # Verify
        assert len(urls) == 1
        assert "https://www.amazon.com/dp/B07XYZ123" in urls
        mock_make_request.assert_called_once()
    
    @patch("scrapers.amazon_scraper.AmazonScraper._make_request")
    def test_extract_product_details(self, mock_make_request):
        """Test extracting product details."""
        # Setup mock
        mock_soup = MagicMock()
        
        # Mock product name
        mock_name = MagicMock()
        mock_name.text = "Test Product"
        mock_soup.select_one.side_effect = lambda x: {
            "#productTitle": mock_name,
            ".a-price .a-offscreen": MagicMock(text="$99.99"),
            "span[data-hook=\"rating-out-of-text\"]": MagicMock(text="4.5 out of 5"),
            "#acrCustomerReviewText": MagicMock(text="100 ratings"),
            "div[data-hook=\"review-collapsed\"]": [MagicMock(text="Great product")]
        }.get(x, None)
        
        mock_make_request.return_value = mock_soup
        
        # Test
        scraper = AmazonScraper()
        product = scraper.extract_product_details("https://www.amazon.com/dp/B07XYZ123")
        
        # Verify
        assert product is not None
        assert product["product_name"] == "Test Product"
        assert product["price"] == 99.99
        assert product["rating"] == 4.5
        assert product["num_reviews"] == 100
        assert "Great product" in product["reviews"]
        assert product["source"] == "Amazon"
        mock_make_request.assert_called_once()


class TestEbayScraper:
    """Test cases for EbayScraper."""
    
    @patch("scrapers.ebay_scraper.EbayScraper._make_request")
    def test_search_products(self, mock_make_request):
        """Test searching for products."""
        # Setup mock
        mock_soup = MagicMock()
        mock_card = MagicMock()
        mock_link = MagicMock()
        mock_link.attrs = {"href": "https://www.ebay.com/itm/123456789"}
        
        mock_card.select_one.return_value = mock_link
        mock_soup.select.return_value = [mock_card]
        mock_make_request.return_value = mock_soup
        
        # Test
        scraper = EbayScraper()
        urls = scraper.search_products("test", 1)
        
        # Verify
        assert len(urls) == 1
        assert "https://www.ebay.com/itm/123456789" in urls
        mock_make_request.assert_called_once()


class TestScraperUtils:
    """Test cases for scraper utilities."""
    
    def test_save_to_csv(self, tmp_path):
        """Test saving data to CSV."""
        # Setup test data
        data = [
            {"product_id": "1", "product_name": "Test 1", "price": 99.99},
            {"product_id": "2", "product_name": "Test 2", "price": 199.99}
        ]
        
        # Test
        filepath = save_to_csv(data, "test.csv")
        
        # Verify
        assert os.path.exists(filepath)
        
        # Clean up
        os.remove(filepath)
    
    def test_convert_to_dataframe(self):
        """Test converting data to DataFrame."""
        # Setup test data
        data = [
            {"product_id": "1", "product_name": "Test 1", "price": 99.99},
            {"product_id": "2", "product_name": "Test 2", "price": 199.99}
        ]
        
        # Test
        df = convert_to_dataframe(data)
        
        # Verify
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "product_id" in df.columns
        assert "product_name" in df.columns
        assert "price" in df.columns
        assert df.iloc[0]["product_name"] == "Test 1"
        assert df.iloc[1]["price"] == 199.99