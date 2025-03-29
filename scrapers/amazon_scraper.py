import re
import logging
import uuid
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from scrapers.base_scraper import BaseScraper

# Configure logging
logger = logging.getLogger(__name__)


class AmazonScraper(BaseScraper):
    """Amazon-specific scraper implementation."""
    
    def __init__(self):
        super().__init__("Amazon")
        self.base_url = "https://www.amazon.com"
        self.search_url = f"{self.base_url}/s?k="
    
    def search_products(self, query: str, max_results: int = 20) -> List[str]:
        """
        Search for products on Amazon and return a list of product URLs.
        
        Args:
            query: The search query.
            max_results: Maximum number of results to return.
            
        Returns:
            List of product URLs.
        """
        search_url = f"{self.search_url}{query.replace(' ', '+')}"
        logger.info(f"Searching Amazon for: {query}")
        
        product_urls = []
        page = 1
        
        while len(product_urls) < max_results:
            page_url = f"{search_url}&page={page}" if page > 1 else search_url
            soup = self._make_request(page_url)
            
            if not soup:
                break
                
            # Extract product URLs from search results
            product_cards = soup.select('div[data-component-type="s-search-result"]')
            
            if not product_cards:
                break
                
            for card in product_cards:
                if len(product_urls) >= max_results:
                    break
                    
                link_element = card.select_one('h2 a')
                if link_element and 'href' in link_element.attrs:
                    product_url = urljoin(self.base_url, link_element['href'])
                    if '/dp/' in product_url:  # Ensure it's a product URL
                        product_urls.append(product_url)
            
            page += 1
            if len(product_cards) < 10:  # Less than 10 results likely means end of results
                break
                
        logger.info(f"Found {len(product_urls)} product URLs on Amazon")
        return product_urls
    
    def extract_product_details(self, product_url: str) -> Optional[Dict[str, Any]]:
        """
        Extract product details from an Amazon product page.
        
        Args:
            product_url: URL of the Amazon product page.
            
        Returns:
            Dictionary with product details or None if extraction failed.
        """
        soup = self._make_request(product_url)
        if not soup:
            return None
            
        try:
            # Extract product ID (ASIN)
            product_id = None
            asin_match = re.search(r'/dp/([A-Z0-9]{10})', product_url)
            if asin_match:
                product_id = asin_match.group(1)
            else:
                product_id = str(uuid.uuid4())  # Generate random ID if ASIN not found
            
            # Extract product name
            product_name_elem = soup.select_one('#productTitle')
            product_name = product_name_elem.text.strip() if product_name_elem else "Unknown"
            
            # Extract price
            price = None
            price_elements = [
                soup.select_one('.a-price .a-offscreen'),
                soup.select_one('#priceblock_ourprice'),
                soup.select_one('#priceblock_dealprice'),
                soup.select_one('.a-price-whole')
            ]
            
            for price_elem in price_elements:
                if price_elem:
                    price_text = price_elem.text.strip()
                    # Remove currency symbol and commas, convert to float
                    price_match = re.search(r'[\d,]+\.\d+|\d+', price_text.replace(',', ''))
                    if price_match:
                        price = float(price_match.group(0))
                        break
            
            # Extract rating
            rating = None
            rating_elem = soup.select_one('span[data-hook="rating-out-of-text"]') or soup.select_one('i.a-icon-star')
            if rating_elem:
                rating_text = rating_elem.text.strip()
                rating_match = re.search(r'(\d+\.?\d*) out of \d+|\d+\.?\d* stars', rating_text)
                if rating_match:
                    rating = float(rating_match.group(1))
            
            # Extract number of reviews
            num_reviews = 0
            reviews_elem = soup.select_one('#acrCustomerReviewText') or soup.select_one('span[data-hook="total-review-count"]')
            if reviews_elem:
                reviews_text = reviews_elem.text.strip()
                reviews_match = re.search(r'([\d,]+) (?:ratings|reviews)', reviews_text)
                if reviews_match:
                    num_reviews = int(reviews_match.group(1).replace(',', ''))
            
            # Extract customer reviews (first few)
            reviews = []
            review_elements = soup.select('div[data-hook="review-collapsed"]') or soup.select('span[data-hook="review-body"]')
            for i, review_elem in enumerate(review_elements):
                if i >= 5:  # Limit to 5 reviews for efficiency
                    break
                reviews.append(review_elem.text.strip())
            
            reviews_str = " | ".join(reviews)
            
            product_data = {
                "product_id": product_id,
                "product_name": product_name,
                "price": price if price else 0.0,
                "rating": rating if rating else 0.0,
                "num_reviews": num_reviews,
                "reviews": reviews_str,
                "source": "Amazon",
                "product_url": product_url
            }
            
            logger.info(f"Successfully extracted details for product: {product_name}")
            return product_data
            
        except Exception as e:
            logger.error(f"Error extracting product details from {product_url}: {str(e)}")
            return None