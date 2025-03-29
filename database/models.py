import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

from database.connection import Base

# Define database models for PostgreSQL
class Product(Base):
    """Model for storing product information."""
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(String(50), index=True)  # Original ID from the source platform
    product_name = Column(String(255), nullable=False)
    price = Column(Float, nullable=False)
    rating = Column(Float, nullable=True)
    num_reviews = Column(Integer, nullable=True)
    reviews = Column(Text, nullable=True)
    source = Column(String(50), nullable=False)  # "Amazon", "eBay", etc.
    product_url = Column(String(1024), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Relationships
    price_histories = relationship("PriceHistory", back_populates="product")
    sentiment_analyses = relationship("SentimentAnalysis", back_populates="product")
    
    def __repr__(self):
        return f"<Product {self.product_name}>"


class PriceHistory(Base):
    """Model for storing price history information."""
    __tablename__ = "price_histories"
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    price = Column(Float, nullable=False)
    date = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    product = relationship("Product", back_populates="price_histories")
    
    def __repr__(self):
        return f"<PriceHistory {self.product_id}: ${self.price}>"


class SentimentAnalysis(Base):
    """Model for storing sentiment analysis results."""
    __tablename__ = "sentiment_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    sentiment_score = Column(Float, nullable=False)  # -1 to 1 scale (negative to positive)
    sentiment_label = Column(String(20), nullable=False)  # "Positive", "Neutral", "Negative"
    top_issues = Column(Text, nullable=True)  # Comma-separated list of top issues
    top_praises = Column(Text, nullable=True)  # Comma-separated list of top praises
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    product = relationship("Product", back_populates="sentiment_analyses")
    
    def __repr__(self):
        return f"<SentimentAnalysis {self.product_id}: {self.sentiment_label}>"


class PricePrediction(Base):
    """Model for storing price prediction results."""
    __tablename__ = "price_predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    original_price = Column(Float, nullable=False)
    predicted_price = Column(Float, nullable=False)
    prediction_date = Column(DateTime, nullable=False)  # Date for which the price is predicted
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    def __repr__(self):
        return f"<PricePrediction {self.product_id}: ${self.predicted_price}>"


# Function to create all tables
def create_tables(engine):
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)