import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Database settings
DATABASE_TYPE = os.getenv("DATABASE_TYPE", "postgresql")  # or mongodb
if DATABASE_TYPE == "postgresql":
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/market_intelligence"
    )
elif DATABASE_TYPE == "mongodb":
    DATABASE_URL = os.getenv(
        "DATABASE_URL", 
        "mongodb://localhost:27017/"
    )
    MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "market_intelligence")

# Scraping settings
SCRAPE_DELAY = int(os.getenv("SCRAPE_DELAY", "3"))  # seconds between requests
USER_AGENT_ROTATION = os.getenv("USER_AGENT_ROTATION", "True").lower() == "true"
PROXY_ENABLED = os.getenv("PROXY_ENABLED", "False").lower() == "true"
PROXY_LIST = os.getenv("PROXY_LIST", "").split(",") if os.getenv("PROXY_LIST") else []

# Number of concurrent scraping processes
CONCURRENCY = int(os.getenv("CONCURRENCY", "4"))

# AI Model settings
SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "distilbert-base-uncased")
PRICE_PREDICTION_MODEL = os.getenv("PRICE_PREDICTION_MODEL", "prophet")  # options: arima, lstm, prophet
MODEL_CACHE_DIR = os.path.join(BASE_DIR, "models_cache")

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_DEBUG = os.getenv("API_DEBUG", "True").lower() == "true"
API_TITLE = "Market Intelligence API"
API_DESCRIPTION = "AI-driven market intelligence system for e-commerce product analysis"
API_VERSION = "1.0.0"

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.path.join(BASE_DIR, "logs", "app.log")