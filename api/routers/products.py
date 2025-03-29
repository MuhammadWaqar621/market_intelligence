import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from database.connection import get_db_session
from database.models import Product
from scrapers.amazon_scraper import AmazonScraper
from scrapers.ebay_scraper import EbayScraper
from scrapers.utils import save_to_database
from api.models.response_models import (
    ProductResponse, 
    ProductListResponse,
    ScrapingResponse
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


@router.get("/products", response_model=ProductListResponse)
async def get_products(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    name: Optional[str] = None,
    source: Optional[str] = None
):
    """
    Get a list of products with optional filtering.
    
    Args:
        db: Database session.
        skip: Number of records to skip (pagination).
        limit: Maximum number of records to return.
        name: Filter by product name (optional).
        source: Filter by source platform (optional).
        
    Returns:
        List of products.
    """
    try:
        # Build query
        query = db.query(Product)
        
        # Apply filters
        if name:
            query = query.filter(Product.product_name.ilike(f"%{name}%"))
        if source:
            query = query.filter(Product.source == source)
        
        # Get total count
        total = query.count()
        
        # Apply pagination
        products = query.order_by(Product.created_at.desc()).offset(skip).limit(limit).all()
        
        # Convert to response model
        result = [
            {
                "id": p.id,
                "product_id": p.product_id,
                "product_name": p.product_name,
                "price": p.price,
                "rating": p.rating,
                "num_reviews": p.num_reviews,
                "source": p.source,
                "product_url": p.product_url,
                "created_at": p.created_at,
                "updated_at": p.updated_at
            }
            for p in products
        ]
        
        return {
            "items": result,
            "total": total,
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Error getting products: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving products")


@router.get("/product", response_model=ProductResponse)
async def get_product_by_name(
    name: str,
    db: Session = Depends(get_db)
):
    """
    Get a product by name.
    
    Args:
        name: Product name to search for.
        db: Database session.
        
    Returns:
        Product details.
    """
    try:
        # Find product by name
        product = db.query(Product).filter(Product.product_name.ilike(f"%{name}%")).first()
        
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
            
        # Get sentiment analysis if available
        sentiment = "Unknown"
        top_issues = []
        
        if product.sentiment_analyses:
            latest_sentiment = product.sentiment_analyses[-1]
            sentiment = latest_sentiment.sentiment_label
            top_issues = latest_sentiment.top_issues.split(",") if latest_sentiment.top_issues else []
        
        return {
            "product_name": product.product_name,
            "price": product.price,
            "rating": product.rating,
            "sentiment": sentiment,
            "top_issues": top_issues
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting product by name: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving product")


@router.get("/product/{product_id}", response_model=ProductResponse)
async def get_product(
    product_id: int,
    db: Session = Depends(get_db)
):
    """
    Get a product by ID.
    
    Args:
        product_id: ID of the product.
        db: Database session.
        
    Returns:
        Product details.
    """
    try:
        # Find product by ID
        product = db.query(Product).filter(Product.id == product_id).first()
        
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
            
        # Get sentiment analysis if available
        sentiment = "Unknown"
        top_issues = []
        
        if product.sentiment_analyses:
            latest_sentiment = product.sentiment_analyses[-1]
            sentiment = latest_sentiment.sentiment_label
            top_issues = latest_sentiment.top_issues.split(",") if latest_sentiment.top_issues else []
        
        return {
            "product_name": product.product_name,
            "price": product.price,
            "rating": product.rating,
            "sentiment": sentiment,
            "top_issues": top_issues
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting product: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving product")


@router.post("/scrape", response_model=ScrapingResponse)
async def scrape_products(
    query: str,
    source: str = "amazon",
    max_results: int = 10
):
    """
    Scrape products from a specified source.
    
    Args:
        query: Search query for products.
        source: Source platform (amazon, ebay).
        max_results: Maximum number of products to scrape.
        
    Returns:
        Scraping results.
    """
    try:
        # Initialize appropriate scraper
        if source.lower() == "amazon":
            scraper = AmazonScraper()
        elif source.lower() == "ebay":
            scraper = EbayScraper()
        else:
            raise HTTPException(status_code=400, detail="Unsupported source platform")
        
        # Scrape products
        products = scraper.scrape_products(query, max_results)
        
        if not products:
            return {
                "status": "success",
                "message": "No products found",
                "count": 0,
                "products": []
            }
        
        # Save to database
        save_to_database(products)
        
        # Return results
        return {
            "status": "success",
            "message": f"Successfully scraped {len(products)} products from {source}",
            "count": len(products),
            "products": products
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scraping products: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error scraping products: {str(e)}")