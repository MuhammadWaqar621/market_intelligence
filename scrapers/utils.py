import os
import csv
import json
import logging
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd

from config.settings import BASE_DIR, DATABASE_TYPE
from database.connection import get_db_session

# Configure logging
logger = logging.getLogger(__name__)


def save_to_csv(data: List[Dict[str, Any]], filename: str = None) -> str:
    """
    Save scraped data to a CSV file.
    
    Args:
        data: List of dictionaries containing product data.
        filename: Optional filename to save to.
        
    Returns:
        Path to the saved CSV file.
    """
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scraped_data_{timestamp}.csv"
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    filepath = os.path.join(data_dir, filename)
    
    if not data:
        logger.warning("No data to save to CSV")
        return filepath
    
    try:
        # Get all possible keys from all dictionaries
        fieldnames = set()
        for item in data:
            fieldnames.update(item.keys())
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(fieldnames))
            writer.writeheader()
            writer.writerows(data)
            
        logger.info(f"Successfully saved {len(data)} records to {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Error saving data to CSV: {str(e)}")
        return filepath


def save_to_database(data: List[Dict[str, Any]]) -> bool:
    """
    Save scraped data to the database.
    
    Args:
        data: List of dictionaries containing product data.
        
    Returns:
        Boolean indicating success.
    """
    if not data:
        logger.warning("No data to save to database")
        return False
    
    try:
        if DATABASE_TYPE == "postgresql":
            from database.models import Product
            session = get_db_session()
            
            for item in data:
                # Check if product already exists
                existing_product = session.query(Product).filter_by(
                    product_id=item["product_id"],
                    source=item["source"]
                ).first()
                
                if existing_product:
                    # Update existing product
                    for key, value in item.items():
                        setattr(existing_product, key, value)
                else:
                    # Create new product
                    product = Product(**item)
                    session.add(product)
            
            session.commit()
            
        elif DATABASE_TYPE == "mongodb":
            from database.connection import get_mongodb_collection
            collection = get_mongodb_collection("products")
            
            # Add timestamps
            for item in data:
                item["updated_at"] = datetime.now()
                
                # Check if product already exists
                existing_product = collection.find_one({
                    "product_id": item["product_id"],
                    "source": item["source"]
                })
                
                if existing_product:
                    # Update existing product
                    collection.update_one(
                        {"_id": existing_product["_id"]},
                        {"$set": item}
                    )
                else:
                    # Create new product
                    item["created_at"] = datetime.now()
                    collection.insert_one(item)
        
        logger.info(f"Successfully saved {len(data)} records to {DATABASE_TYPE} database")
        return True
        
    except Exception as e:
        logger.error(f"Error saving data to database: {str(e)}")
        return False


def convert_to_dataframe(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert scraped data to a pandas DataFrame.
    
    Args:
        data: List of dictionaries containing product data.
        
    Returns:
        Pandas DataFrame.
    """
    return pd.DataFrame(data)