from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class ProductResponse(BaseModel):
    """Model for product response."""
    product_name: str
    price: float
    rating: Optional[float] = None
    sentiment: Optional[str] = None
    top_issues: Optional[List[str]] = []


class ProductDetail(BaseModel):
    """Model for detailed product information."""
    id: int
    product_id: str
    product_name: str
    price: float
    rating: Optional[float] = None
    num_reviews: Optional[int] = None
    source: str
    product_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class ProductListResponse(BaseModel):
    """Model for paginated product list response."""
    items: List[ProductDetail]
    total: int
    skip: int
    limit: int


class PricePredictionResponse(BaseModel):
    """Model for price prediction response."""
    product_name: str
    current_price: float
    predicted_price_next_month: float


class SentimentResponse(BaseModel):
    """Model for sentiment analysis response."""
    product_name: str
    sentiment: str
    sentiment_score: float
    top_issues: List[str]


class TopIssuesResponse(BaseModel):
    """Model for top issues response."""
    product_name: str
    top_issues: List[str]


class SentimentAnalysisResponse(BaseModel):
    """Model for sentiment analysis of arbitrary text."""
    sentiment: str
    sentiment_score: float
    topics: List[str]


class ScrapingResponse(BaseModel):
    """Model for scraping response."""
    status: str
    message: str
    count: int
    products: List[Dict[str, Any]]


class PriceComparisonResponse(BaseModel):
    """Model for price comparison response."""
    status: str
    products: List[Dict[str, Any]]
    statistics: Dict[str, float]
    best_value: Optional[Dict[str, Any]] = None
    cheapest: Optional[Dict[str, Any]] = None
    most_expensive: Optional[Dict[str, Any]] = None