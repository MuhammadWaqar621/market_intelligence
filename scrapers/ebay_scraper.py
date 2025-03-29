import re
import logging
import uuid
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from scrapers.base_scraper import BaseScraper

# Configure logging
logger = logging.getLogger(__name__)


class EbayScraper(BaseScraper):
    """eBay-specific scraper implementation."""
    
    def __init__(self):
        super().__init__("eBay")
        self.base_url = "https://www.ebay.com"
        self.search_url = f"{self.base_url}/sch/i.html?_nkw="
    
    def search_products(self, query: str, max_results: int = 20) -> List[str]:
        """
        Search for products on eBay and return a list of product URLs.
        
        Args:
            query: The search query.
            max_results: Maximum number of results to return.
            
        Returns:
            List of product URLs.
        """
        search_url = f"{self.search_url}{query.replace(' ', '+')}"
        logger.info(f"Searching eBay for: {query}")
        
        product_urls = []
        page = 1
        
        while len(product_urls) < max_results:
            page_url = f"{search_url}&_pgn={page}" if page > 1 else search_url
            soup = self._make_request(page_url)
            
            if not soup:
                break
                
            # Extract product URLs from search results
            product_cards = soup.select('li.s-item')
            
            if not product_cards:
                break
                
            for card in product_cards:
                if len(product_urls) >= max_results:
                    break
                    
                link_element = card.select_one('a.s-item__link')
                if link_element and 'href' in link_element.attrs:
                    product_url = link_element['href']
                    if '/itm/' in product_url:  # Ensure it's a product URL
                        product_urls.append(product_url)
            
            page += 1
            if len(product_cards) < 10:  # Less than 10 results likely means end of results
                break
                
        logger.info(f"Found {len(product_urls)} product URLs on eBay")
        return product_urls
    
    def extract_product_details(self, product_url: str) -> Optional[Dict[str, Any]]:
        """
        Extract product details from an eBay product page.
        
        Args:
            product_url: URL of the eBay product page.
            
        Returns:
            Dictionary with product details or None if extraction failed.
        """
        soup = self._make_request(product_url)
        if not soup:
            return None
            
        try:
            # Extract product ID
            product_id = None
            item_match = re.search(r'/itm/(\d+)', product_url)
            if item_match:
                product_id = item_match.group(1)
            else:
                product_id = str(uuid.uuid4())  # Generate random ID if item ID not found
            
            # Extract product name
            product_name_elem = soup.select_one('h1.x-item-title__mainTitle')
            product_name = product_name_elem.text.strip() if product_name_elem else "Unknown"
            
            # Extract price
            price = None
            price_elem = soup.select_one('span.x-price-primary') or soup.select_one('span.notranslate')
            if price_elem:
                price_text = price_elem.text.strip()
                # Remove currency symbol and commas, convert to float
                price_match = re.search(r'[\d,]+\.\d+|\d+', price_text.replace(',', ''))
                if price_match:
                    price = float(price_match.group(0))
            
            # Extract rating (out of 5)
            rating = None
            rating_elem = soup.select_one('div.x-star-rating span')
            if rating_elem:
                rating_text = rating_elem.text.strip()
                rating_match = re.search(r'(\d+\.?\d*)', rating_text)
                if rating_match:
                    rating = float(rating_match.group(1))
            
            # Extract number of reviews
            num_reviews = 0
            reviews_count_elem = soup.select_one('a.reviews-right-panel div.right-section span')
            if reviews_count_elem:
                reviews_text = reviews_count_elem.text.strip()
                reviews_match = re.search(r'(\d+)', reviews_text)
                if reviews_match:
                    num_reviews = int(reviews_match.group(1))
            
            # Extract customer reviews (first few)
            reviews = []
            review_elements = soup.select('div.ebay-review-section p.review-item-content')
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
                "source": "eBay",
                "product_url": product_url
            }
            
            logger.info(f"Successfully extracted details for product: {product_name}")
            return product_data
            
        except Exception as e:
            logger.error(f"Error extracting product details from {product_url}: {str(e)}")
            return None