import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import pymongo
from typing import Optional

from config.settings import DATABASE_TYPE, DATABASE_URL, MONGODB_DB_NAME

# Configure logging
logger = logging.getLogger(__name__)

# SQLAlchemy setup
if DATABASE_TYPE == "postgresql":
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()

# MongoDB setup
if DATABASE_TYPE == "mongodb":
    try:
        mongo_client = pymongo.MongoClient(DATABASE_URL)
        mongodb = mongo_client[MONGODB_DB_NAME]
        # Test connection
        mongo_client.server_info()
        logger.info("Successfully connected to MongoDB")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        mongodb = None


def get_db_session() -> Session:
    """
    Get a SQLAlchemy database session.
    
    Returns:
        SQLAlchemy Session object.
    """
    if DATABASE_TYPE != "postgresql":
        logger.error("Attempted to get SQLAlchemy session but DATABASE_TYPE is not postgresql")
        return None
        
    db = SessionLocal()
    try:
        return db
    except Exception as e:
        logger.error(f"Error getting database session: {str(e)}")
        db.close()
        return None


def get_mongodb_collection(collection_name: str):
    """
    Get a MongoDB collection.
    
    Args:
        collection_name: Name of the collection.
    
    Returns:
        MongoDB collection object.
    """
    if DATABASE_TYPE != "mongodb" or mongodb is None:
        logger.error("Attempted to get MongoDB collection but DATABASE_TYPE is not mongodb or connection failed")
        return None
        
    return mongodb[collection_name]