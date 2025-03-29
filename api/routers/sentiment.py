import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.orm import Session

from database.connection import get_db_session
from database.models import Product, SentimentAnalysis
from analysis.sentiment_analyzer import SentimentAnalyzer
from api.models.response_models import (
    SentimentResponse,
    TopIssuesResponse,
    SentimentAnalysisResponse
)

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


def get_db():
    """
    Get database session.
    
    Returns:
        SQLAlchemy Session
    """
    db = get_db_session()
    try:
        yield db
    finally:
        db.close()


@router.get("/sentiment", response_model=SentimentResponse)
async def get_sentiment(
    product: str,
    db: Session = Depends(get_db)
):
    """
    Get sentiment analysis for a product.
    
    Args:
        product: Product name to search for.
        db: Database session.
        
    Returns:
        Sentiment analysis results.
    """
    try:
        # Find product by name
        product_obj = db.query(Product).filter(Product.product_name.ilike(f"%{product}%")).first()
        
        if not product_obj:
            raise HTTPException(status_code=404, detail="Product not found")
        
        # Get sentiment analysis from database
        sentiment_analysis = db.query(SentimentAnalysis).filter(
            SentimentAnalysis.product_id == product_obj.id
        ).order_by(SentimentAnalysis.created_at.desc()).first()