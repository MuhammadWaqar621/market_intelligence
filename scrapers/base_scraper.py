import time
import logging
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import requests
from fake_useragent import UserAgent
from bs4 import BeautifulSoup

from config.settings import SCRAPE_DELAY, USER_AGENT_ROTATION, PROXY_ENABLED, PROXY_LIST

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """Base scraper class that all specific platform scrapers should inherit from."""
    
    def __init__(self, platform_name: str):
        self.platform_name = platform_name
        self.session = requests.Session()
        self.user_agent = UserAgent()
        self.proxies = PROXY_LIST if PROXY_ENABLED else []
        logger.info(f"Initialized {platform_name} scraper")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with a random user agent if rotation is enabled."""
        headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        if USER_AGENT_ROTATION:
            headers['User-Agent'] = self.user_agent.random
        else:
            headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36'
        
        return headers
    
    def _get_proxy(self) -> Optional[Dict[str, str]]:
        """Get a random proxy from the proxy list if proxy is enabled."""
        if not PROXY_ENABLED or not self.proxies:
            return None
        
        proxy = random.choice(self.proxies)
        return {
            'http': proxy,
            'https': proxy
        }
    
    def _make_request(self, url: str) -> Optional[BeautifulSoup]:
        """Make a request to the given URL and return the BeautifulSoup object."""
        try:
            headers = self._get_headers()
            proxies = self._get_proxy()
            
            logger.info(f"Making request to {url}")
            response = self.session.get(
                url,
                headers=headers,
                proxies=proxies,
                timeout=30
            )
            
            # Respect robots.txt and terms of service by adding delay
            time.sleep(SCRAPE_DELAY + random.uniform(0.5, 2.0))
            
            if response.status_code == 200:
                logger.info(f"Request to {url} successful")
                return BeautifulSoup(response.content, 'html.parser')
            else:
                logger.error(f"Failed to fetch {url}, status code: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error making request to {url}: {str(e)}")
            return None
    
    @abstractmethod
    def search_products(self, query: str, max_results: int = 20) -> List[str]:
        """
        Search for products and return a list of product URLs.
        
        Args:
            query: The search query.
            max_results: Maximum number of results to return.
            
        Returns:
            List of product URLs.
        """
        pass
    
    @abstractmethod
    def extract_product_details(self, product_url: str) -> Optional[Dict[str, Any]]:
        """
        Extract product details from a product page.
        
        Args:
            product_url: URL of the product page.
            
        Returns:
            Dictionary with product details or None if extraction failed.
        """
        pass
    
    def scrape_products(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """
        Search and scrape products based on the query.
        
        Args:
            query: The search query.
            max_results: Maximum number of results to return.
            
        Returns:
            List of dictionaries with product details.
        """
        logger.info(f"Scraping products for query: {query}")
        
        # Get product URLs
        product_urls = self.search_products(query, max_results)
        logger.info(f"Found {len(product_urls)} product URLs")
        
        # Extract product details
        products = []
        for url in product_urls:
            product_data = self.extract_product_details(url)
            if product_data:
                products.append(product_data)
                
        logger.info(f"Successfully scraped {len(products)} products")
        return products