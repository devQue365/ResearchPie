import os
import csv
import sys
import logging
import asyncio
import asyncpg
import aiohttp
from queue import Queue
import random
from playwright.async_api import Page
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import wraps
from dataclasses import dataclass
from typing import List, Dict, TextIO, Optional, Any
from datetime import datetime, timedelta
from ...settings import settings

@dataclass
class SearchConfig:
    """Configuration for search parameters"""
    query: Optional[str] = None
    max_results_per_query = settings.max_results_per_query
    max_pages_per_query = settings.max_pages_per_query
    concurrent_requests = settings.concurrent_requests
    # delay_between_requests: Tuple[int, int] = (2,5)
    # delay_between_pages = Tuple[int, int] = (3,8)
    proxies_enabled: bool = False
    proxy_list: Optional[List[str]] = None

# Human Behavior Simulator
class HumanBehaviorSimulator:
    """Simulates human-like behavior"""
    @staticmethod
    def random_delay(min_seconds: float = 1.0, max_seconds: float = 5.0)->float:
        """Returns random delay with the ```range(min_seconds, max_seconds)```"""
        return random.uniform(min_seconds, max_seconds)
    
    @staticmethod
    def typing_delay(text: str, wpm: int = 65):
        """Calculated typing delay based on text length and WPM"""
        chars_per_second = wpm * 5 / 60 # Avg 5 characters per minute
        return len(text) / chars_per_second + random.uniform(0.1, 0.5)
    
    @staticmethod
    async def human_like_scroll(page: Page, scroll_count: int = 3):
        """Emitates human like scroll"""
        scroll_logic_js = """
            () => {
                function humanLikeScroll() {
                    const maxScroll = document.body.scrollHeight - window.innerHeight;
                    const current = window.scrollY;

                    // Mostly scroll down (80% chance), sometimes up
                    const direction = Math.random() < 0.8 ? 1 : -1;

                    // Scroll distance: small if up, larger if down
                    const distance = direction === 1
                        ? Math.floor(Math.random() * 400) + 100   // 100–500px down
                        : Math.floor(Math.random() * 100) + 50;   // 50–150px up

                    let newY = current + direction * distance;

                    // Keep inside page bounds
                    if (newY < 0) newY = 0;
                    if (newY > maxScroll) newY = maxScroll;

                    window.scrollTo({ top: newY, behavior: 'smooth' });

                    // Decide whether to stop scrolling
                    const reachedBottom = (newY >= maxScroll - 50);
                    const stopNow = Math.random() < 0.1; // 10% chance to stop early

                    if (!reachedBottom && !stopNow) {
                        // Wait a random 1–5 seconds before next scroll
                        const delay = Math.floor(Math.random() * 4000) + 1000;
                        setTimeout(humanLikeScroll, delay);
                    }
                }

                // Start after a short random delay (simulate reading first)
                setTimeout(humanLikeScroll, Math.floor(Math.random() * 2000) + 1000);
            }

        """
        # for _ in range(scroll_count):
        await page.evaluate(scroll_logic_js)
            # await asyncio.sleep(random.uniform(0.8 , 2.0))

    @staticmethod
    async def random_mouse_movement(page: Page, movements: int = 5):
        """Emitates real mouse movements"""
        viewport = page.viewport_size
        if viewport:
            for _ in range(movements):
                # Calculate X and Y coordinates
                x = random.randint(0, viewport['width'])
                y = random.randint(0, viewport['height'])
                await page.mouse.move(x, y)
                await asyncio.sleep(random.uniform(0.1, 0.3))


# Header Manager
class HeaderManager:
    """Manages headers and rotation"""
    def __init__(self, header_list: List[Dict]):
        self.headers = header_list
        self.current_index = 0

    @staticmethod
    def filter(func):
        """Filters headers based on some user specified filter routine"""
        @wraps(func)
        def wrapper_filter(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return wrapper_filter
    

    def get_random_header(self, custom_headers: Optional[List[Dict]] = None):
        """Returns a random header from header pool"""
        if (not custom_headers) and (self.headers):
            custom_headers = self.headers

        custom_headers = random.sample(custom_headers, len(custom_headers))
        header = custom_headers[self.current_index]
        self.current_index = (self.current_index + 1)%len(custom_headers)
        # logger.info("Successfully returned a header")
        return header

# Proxy Configuration
class Proxy:
    """Basic Proxy Configuration"""
    def __init__(self, host: str, port: str, username: Optional[str] = None, password: Optional[str] = None, protocol: Optional[str] = "http"):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.protocol = protocol
        self._key = f"{host}:{port}"

    @property
    def _url(self):
        """Dynamically computes the proxy url"""
        if self.username and self.password:
            return f"{self.protocol}://{self.username}:{self.password}@{self.host}:{self.port}"
        return f"{self.protocol}://{self.host}:{self.port}"
      
    @property
    def _auth(self):
        """A boolean flag indicating whether the proxy authentication is enabled or not"""
        return (self.username and self.password)
    
    @property
    def playwright_format(self):
        """
        Returns the Proxy in playwright format\n
        ```python
        config = {"server": "proxy_server", "username": "lorem", "password": "mypass@123"}
        ```
        """
        proxy_config = {
            "server": self._url,
        }
        ## Check if authentication needed ...
        if self.username and self.password:
            proxy_config["username"] = self.username
            proxy_config["password"] = self.password
        return proxy_config

# Proxy Manager
class ProxyManager:
    """Manages proxy rotation and health checking"""
    def __init__(self, proxies: List[Dict], cooldown_period: float = 120, timeout: float = 10):
        self.proxies: List[Proxy] = []
        self.cooldown_period = cooldown_period
        self.current_index = 0
        self.failed_proxies = set()
        self.proxy_usage_count = {}
        self.last_used = {}
        self.timeout = timeout
        # Initialize proxies
        self._initialize_proxies(proxies)

    def _initialize_proxies(self, proxies: List[Dict[str, Any]]):
        """Initializes proxies based on the input proxy pool"""
        try:
            for proxy in proxies:
                # Get the mandatory credentials
                host = proxy['host']
                port = proxy['port']
                # Optional Parameters
                username = proxy.get("username", None)
                password = proxy.get("password", None)
                protocol = proxy.get("protocol", "http")
                # Create proxy object
                p = Proxy(
                    host = host,
                    port = port,
                    username = username,
                    password = password,
                    protocol = protocol
                )
                self.proxies.append(p)
        except Exception as e:
            # logger.error(e)
            raise Exception
        
        # Check for the proxies instantiated
        if not self.proxies:
            # logger.critical(f"{Fore.RED}No proxies fetched, check the proxy_pool and other configs for reference ...")
            raise ValueError("Invalid Proxy Pool !")
        # logger.info(f"{Fore.LIGHTBLUE_EX}Initialized {len(self.proxies)} proxies successfully")
    
    async def get_next_proxy(self)-> Proxy:
        """Returns next available proxy"""
        max_attempts = len(self.proxies) * 2
        attempts = 0

        while attempts < max_attempts:
            proxy_list = random.sample(self.proxies, k = len(self.proxies))
            proxy = proxy_list[self.current_index]

            ## Check if proxy is not in failed list and hasn't been overused (max 50 requests per proxy)
            if(proxy._key not in self.failed_proxies and self.proxy_usage_count.get(proxy._key, 0) < 50):

                ### Check cooldown period
                last_used = self.last_used.get(proxy._key, datetime.min)
                if datetime.now() - last_used > timedelta(seconds=self.cooldown_period):
                    self.proxy_usage_count[proxy._key] = self.proxy_usage_count.get(proxy._key, 0) + 1
                    self.last_used[proxy._key] = datetime.now()
                    self.current_index = (self.current_index + 1) % len(self.proxies)
                    return proxy
            self.current_index = (self.current_index + 1) % len(self.proxies)
            attempts += 1

            ## If all the proxies are exhausted, reset counters
            self.proxy_usage_count.clear()
            self.failed_proxies.clear()
            return self.proxies[0]  

    def mark_proxy_failed(self, proxy: Proxy):
        """Mark proxy as failed"""
        self.failed_proxies.add(proxy._key)

    async def test_proxy(self, proxy: Proxy):
        """Test if the proxy is in working state or not"""
        try:

            tcp_connector = aiohttp.TCPConnector(
                use_dns_cache=True,
                ttl_dns_cache=300, # 5 minutes for Time To Live
                limit = 100, # 100 concurrent connections | 20 effectively
                limit_per_host=5, # 10 concurrent connections per host
                force_close=True,
                enable_cleanup_closed=True # some SSL servers do not properly complete SSL shutdown process, in that case asyncio leaks ssl connections. If this parameter is set to True, aiohttp additionally aborts underlining transport after 2 seconds. It is off by default.
            )

            client_timeout = aiohttp.ClientTimeout(total=self.timeout)

            if proxy._auth:
                auth = aiohttp.BasicAuth(proxy.username, proxy.password)
            else: auth = None

            async with aiohttp.ClientSession(
                connector = tcp_connector,
                timeout= client_timeout,
                trust_env=True # Imp for proxy authentication
            ) as session:
                async with session.get(
                    url = "https://httpbin.org/ip",
                    proxy = proxy._url,
                    auth = auth,
                    allow_redirects=True,
                ) as response:
                    return response.status == 200
        except Exception as e:
            return False
    
    # Read Proxies from .csv file
def read_proxies(file_path: TextIO, protocol: str = "http")->List[Dict[str, Any]]:
    """Reads proxies from ```proxies.csv``` file and returns a dict of proxies"""

    proxy_collection = []

    try:
        with open(file_path, newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if all(k in ('Host', 'Port', 'User', 'Pass') for k in row):
                    # Generate proxy record ...
                    host = row['Host'].strip()
                    port = row['Port'].strip()
                    user = row['User'].strip()
                    password = row['Pass'].strip()
                    proxy_url = f"{protocol}://{user}:{password}@{host}:{port}"
                    proxy_record = {
                        'url': proxy_url,
                        'host': host,
                        'port': int(port),
                        'username': user,
                        'password': password,
                        'protocol': protocol,
                    }
                    proxy_collection.append(proxy_record)
        # logger.info(f"{Fore.BLUE}Read {len(proxy_collection)} proxy records successfully")
        return proxy_collection
    
    except Exception as e:
        # logger.critical(f"{Fore.RED}Failed to initialize proxies => Error {str(e)} | Quiting ...")
        sys.exit(1)

class WebScraper:
    """Web scraper that supports bulk processing"""
    def __init__(self, config: SearchConfig):
        # For search configuration
        self.config = config
        # For proxy management
        self.proxy_manager = ProxyManager(proxy_list=config.proxy_list)
        # For database management
        self.db_manager = DatabaseManager()
        # For results queue
        self.results_queue = Queue()
        # Unique sesion id
        self.session_id = datetime.now().strftime("%Y%m%d_%I%M%S")
        # Optimized Header
        self.headers = httpx.Headers(self.get_valid_header())
    
    def get_valid_header(self, api_key = SCRAPEOPS_API_KEY):
        """Returns a valid header"""
        # Get a list of valid headers
        response = httpx.get('http://headers.scrapeops.io/v1/browser-headers?api_key=' + api_key, follow_redirects=True)
        json_response = response.json()
        headers =  json_response.get('result', [])
        # Select a random header from the list
        random_index = random.randint(0, len(headers) - 1)
        return headers[random_index]


    def search_google(self, query: str, page: int = 0):
        """Search google results page by page"""
        try:
            # By default, Google search results are set to display 10 results per page for most users.
            start_index = page * 10
            target_url = f"https://www.google.com/search?q={quote_plus(query)}&start={start_index}&num=10"

            # Proxy management
            if self.config.proxies_enabled and self.config.proxy_list:
                proxy = self.fetch_proxy(self.config.proxy_list)
                proxy_settings = {
                    'http://': f'http://{proxy}',
                    'https://': 'https://{proxy}',
                }
        except Exception as e:
            pass