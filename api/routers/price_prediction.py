import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from database.connection import get_db_session
from database.models import Product, PricePrediction
from analysis.price_analyzer import PriceAnalyzer
from api.models.response_models import (
    PricePredictionResponse,
    PriceComparisonResponse
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


@router.get("/price-prediction", response_model=PricePredictionResponse)
async def get_price_prediction(
    product: str,
    db: Session = Depends(get_db)
):
    """
    Get price prediction for a product.
    
    Args:
        product: Product name to search for.
        db: Database session.
        
    Returns:
        Price prediction details.
    """
    try:
        # Find product by name
        product_obj = db.query(Product).filter(Product.product_name.ilike(f"%{product}%")).first()
        
        if not product_obj:
            raise HTTPException(status_code=404, detail="Product not found")
        
        # Get price prediction from database
        price_prediction = db.query(PricePrediction).filter(
            PricePrediction.product_id == product_obj.id
        ).order_by(PricePrediction.created_at.desc()).first()
        
        # If prediction exists and is recent, return it
        if price_prediction:
            return {
                "product_name": product_obj.product_name,
                "current_price": price_prediction.original_price,
                "predicted_price_next_month": price_prediction.predicted_price
            }
        
        # Otherwise, generate a new prediction
        price_analyzer = PriceAnalyzer()
        prediction_result = price_analyzer.predict_price(product_obj.id)
        
        if prediction_result["status"] != "success":
            raise HTTPException(status_code=500, detail="Error generating price prediction")
        
        return {
            "product_name": product_obj.product_name,
            "current_price": prediction_result["current_price"],
            "predicted_price_next_month": prediction_result["predicted_price_next_month"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting price prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving price prediction")


@router.get("/price-trend/{product_id}", response_model=List[dict])
async def get_price_trend(
    product_id: int,
    days: int = 30,
    db: Session = Depends(get_db)
):
    """
    Get price trend forecast for a product.
    
    Args:
        product_id: ID of the product.
        days: Number of days to forecast.
        db: Database session.
        
    Returns:
        List of daily price predictions.
    """
    try:
        # Find product by ID
        product = db.query(Product).filter(Product.id == product_id).first()
        
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        
        # Generate price prediction with daily trend
        price_analyzer = PriceAnalyzer()
        prediction_result = price_analyzer.predict_price(product.id, days)
        
        if prediction_result["status"] != "success":
            raise HTTPException(status_code=500, detail="Error generating price trend")
        
        return prediction_result["predictions"]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting price trend: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving price trend")


@router.get("/price-comparison", response_model=PriceComparisonResponse)
async def compare_prices(
    product_ids: List[int] = Query(...),
    db: Session = Depends(get_db)
):
    """
    Compare prices of multiple products.
    
    Args:
        product_ids: List of product IDs to compare.
        db: Database session.
        
    Returns:
        Price comparison results.
    """
    try:
        if not product_ids:
            raise HTTPException(status_code=400, detail="No product IDs provided")
        
        # Verify all products exist
        products = db.query(Product).filter(Product.id.in_(product_ids)).all()
        
        if len(products) != len(product_ids):
            raise HTTPException(status_code=404, detail="One or more products not found")
        
        # Perform price comparison
        price_analyzer = PriceAnalyzer()
        comparison_result = price_analyzer.compare_product_prices(product_ids)
        
        if comparison_result["status"] != "success":
            raise HTTPException(status_code=500, detail="Error comparing product prices")
        
        return comparison_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing prices: {str(e)}")
        raise HTTPException(status_code=500, detail="Error comparing prices")