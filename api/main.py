import logging
import uvicorn
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from config.settings import API_HOST, API_PORT, API_DEBUG, API_TITLE, API_DESCRIPTION, API_VERSION
from api.routers import products, price_predictions, sentiment
from database.connection import get_db_session
from database.models import create_tables
from database.connection import engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(products.router, prefix="/api", tags=["Products"])
app.include_router(price_predictions.router, prefix="/api", tags=["Price Predictions"])
app.include_router(sentiment.router, prefix="/api", tags=["Sentiment Analysis"])


# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        routes=app.routes,
    )
    
    # Custom formatting or modifications to OpenAPI schema can be done here
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint to check if API is running."""
    return {
        "message": "Market Intelligence API is running",
        "version": API_VERSION,
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }


@app.get("/api/health", tags=["Health"])
async def health_check():
    """Check the health of the API and its dependencies."""
    # Check database connection
    db = get_db_session()
    db_status = "healthy" if db else "unhealthy"
    if db:
        db.close()
    
    return {
        "status": "healthy",
        "version": API_VERSION,
        "dependencies": {
            "database": db_status
        }
    }


# Create database tables on startup
@app.on_event("startup")
async def startup_event():
    """Create database tables on startup."""
    try:
        create_tables(engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")


def start():
    """Start the FastAPI server."""
    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_DEBUG
    )


if __name__ == "__main__":
    start()