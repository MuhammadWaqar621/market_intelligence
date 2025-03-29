# AI-Driven Market Intelligence System

A comprehensive system that scrapes e-commerce data, analyzes pricing trends, performs sentiment analysis, and provides market intelligence through a REST API.

## Features

- **Web Scraping**: Data collection from Amazon and eBay
- **AI Analysis**:
  - Price trend forecasting using ARIMA, Prophet, and LSTM models
  - Sentiment analysis of customer reviews using transformer models
  - Identification of competitive pricing and common issues
- **REST API**: Real-time queries for product data, price predictions, and sentiment analysis

## System Architecture

The system is designed with a modular architecture and consists of the following components:

### Web Scrapers
- Base scraper with rotating user agents and proxy handling
- Platform-specific scrapers for Amazon and eBay
- Rate limiting and robots.txt compliance

### AI/ML Analysis
- Price trend analysis using statistical and deep learning models
- Sentiment analysis using pre-trained transformer models
- Feature extraction from reviews to identify common issues and praises

### Database
- PostgreSQL (primary) or MongoDB (alternative) storage
- Models for products, price history, sentiment analysis, and predictions

### REST API
- FastAPI-based endpoints for data retrieval
- OpenAPI documentation
- Health checks and error handling

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.9+ (for local development)

### Installation

#### Using Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/MuhammadWaqar621/market_intelligence.git
   ```

2. Start the Docker containers:
   ```bash
   docker-compose up -d
   ```

3. The API will be accessible at `http://localhost:8000`

#### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/MuhammadWaqar621/market_intelligence.git
   cd market-intelligence
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  
   pip install -r requirements.txt
   ```
3. Run the API:
   ```bash
   python -m api.main
   ```

## API Endpoints

### Products

- `GET /api/products` - List all products with optional filtering
- `GET /api/product?name={name}` - Get product details by name
- `POST /api/scrape` - Scrape products from specified source

### Price Analysis

- `GET /api/price-prediction?product={name}` - Get price prediction for a product
- `GET /api/price-trend/{product_id}` - Get price trend forecast
- `GET /api/price-comparison?product_ids={ids}` - Compare prices of multiple products

### Sentiment Analysis

- `GET /api/sentiment?product={name}` - Get sentiment analysis for a product
- `GET /api/top-issues?product={name}` - Get top customer complaints
- `POST /api/analyze-text` - Analyze sentiment of provided text

## Database Schema

### Products Table
- `id`: Primary key
- `product_id`: Original ID from source platform
- `product_name`: Name of the product
- `price`: Current price
- `rating`: Average rating
- `num_reviews`: Number of reviews
- `reviews`: Text of reviews
- `source`: Source platform
- `product_url`: URL to product page
- `created_at`: Creation timestamp
- `updated_at`: Last update timestamp

### Price History Table
- `id`: Primary key
- `product_id`: Foreign key to Products
- `price`: Historical price
- `date`: Date of the price record

### Sentiment Analysis Table
- `id`: Primary key
- `product_id`: Foreign key to Products
- `sentiment_score`: Score from -1 to 1
- `sentiment_label`: "Positive", "Neutral", or "Negative"
- `top_issues`: Common complaints
- `top_praises`: Common praises
- `created_at`: Creation timestamp

### Price Prediction Table
- `id`: Primary key
- `product_id`: Foreign key to Products
- `original_price`: Current price
- `predicted_price`: Predicted future price
- `prediction_date`: Date for which price is predicted
- `created_at`: Creation timestamp

## Usage Examples

### Scraping Products

```bash
curl -X POST "http://localhost:8000/api/scrape?query=laptop&source=amazon&max_results=5"
```

### Getting Price Prediction

```bash
curl "http://localhost:8000/api/price-prediction?product=MacBook%20Air%202023"
```

### Sentiment Analysis

```bash
curl "http://localhost:8000/api/sentiment?product=Dell%20XPS%2015"
```

## Testing

Run the test suite:

```bash
pytest tests/
```

## Contact

waqarsahi621@gmail.com