"""
Website Content Ingester for SEO Content Knowledge Graph System.

This module provides comprehensive website content scraping capabilities including:
- Sitemap parsing and crawling
- Robots.txt compliance checking
- Rate limiting and respectful crawling
- Content extraction and cleaning
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
import aiohttp
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse, parse_qs, urlencode
from urllib.robotparser import RobotFileParser
import re
from pathlib import Path

from .base_ingester import BaseIngester, RawContent, ContentSource
from ..database.supabase_client import supabase_client

logger = logging.getLogger(__name__)


class WebsiteIngestConfig(Dict):
    """Configuration for website ingestion."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setdefault('max_pages', 100)
        self.setdefault('max_depth', 3)
        self.setdefault('delay_between_requests', 1.0)
        self.setdefault('respect_robots_txt', True)
        self.setdefault('include_images', False)
        self.setdefault('include_pdfs', True)
        self.setdefault('user_agent', 'SEO-ContentGraph-Bot/1.0')
        self.setdefault('timeout', 30)
        self.setdefault('follow_redirects', True)
        self.setdefault('allowed_domains', [])
        self.setdefault('excluded_paths', [])
        self.setdefault('content_selectors', {})


class RobotsTxtParser:
    """Parser for robots.txt files."""
    
    def __init__(self, robots_url: str, user_agent: str = '*'):
        self.robots_url = robots_url
        self.user_agent = user_agent
        self.robot_parser = RobotFileParser()
        self.robot_parser.set_url(robots_url)
        self.crawl_delay = 1.0
        self.sitemaps = []
        self.loaded = False
    
    async def load(self, session: aiohttp.ClientSession):
        """Load and parse robots.txt."""
        try:
            async with session.get(self.robots_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    robots_content = await response.text()
                    
                    # Parse manually to extract additional info
                    await self._parse_robots_content(robots_content)
                    
                    # Use urllib's parser for can_fetch checks
                    self.robot_parser.read()
                    self.loaded = True
                    
                    logger.info(f"Loaded robots.txt from {self.robots_url}")
                    
        except Exception as e:
            logger.warning(f"Failed to load robots.txt from {self.robots_url}: {e}")
            self.loaded = False
    
    async def _parse_robots_content(self, content: str):
        """Parse robots.txt content to extract crawl delay and sitemaps."""
        lines = content.split('\n')
        current_user_agent = None
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            if line.lower().startswith('user-agent:'):
                current_user_agent = line.split(':', 1)[1].strip()
            
            elif line.lower().startswith('crawl-delay:'):
                if self._applies_to_user_agent(current_user_agent):
                    try:
                        self.crawl_delay = max(float(line.split(':', 1)[1].strip()), 0.5)
                    except ValueError:
                        pass
            
            elif line.lower().startswith('sitemap:'):
                sitemap_url = line.split(':', 1)[1].strip()
                if sitemap_url not in self.sitemaps:
                    self.sitemaps.append(sitemap_url)
    
    def _applies_to_user_agent(self, robot_user_agent: str) -> bool:
        """Check if robots.txt rule applies to our user agent."""
        if not robot_user_agent:
            return False
        
        robot_user_agent = robot_user_agent.lower()
        return (robot_user_agent == '*' or 
                robot_user_agent in self.user_agent.lower() or
                'seo' in robot_user_agent or 
                'content' in robot_user_agent)
    
    def can_fetch(self, url: str) -> bool:
        """Check if URL can be fetched according to robots.txt."""
        if not self.loaded:
            return True  # Allow if robots.txt couldn't be loaded
        
        try:
            return self.robot_parser.can_fetch(self.user_agent, url)
        except Exception as e:
            logger.warning(f"Error checking robots.txt for {url}: {e}")
            return True


class SitemapParser:
    """Parser for XML sitemaps."""
    
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
    
    async def parse_sitemap(self, sitemap_url: str) -> List[str]:
        """Parse a sitemap and return list of URLs."""
        try:
            async with self.session.get(sitemap_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch sitemap {sitemap_url}: {response.status}")
                    return []
                
                content = await response.text()
                return await self._parse_sitemap_content(content)
                
        except Exception as e:
            logger.error(f"Failed to parse sitemap {sitemap_url}: {e}")
            return []
    
    async def _parse_sitemap_content(self, content: str) -> List[str]:
        """Parse sitemap XML content."""
        urls = []
        
        try:
            root = ET.fromstring(content)
            
            # Handle sitemap index files
            if 'sitemapindex' in root.tag:
                for sitemap in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap'):
                    loc_elem = sitemap.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                    if loc_elem is not None and loc_elem.text:
                        nested_urls = await self.parse_sitemap(loc_elem.text)
                        urls.extend(nested_urls)
            
            # Handle regular sitemaps
            else:
                for url_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                    loc_elem = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                    if loc_elem is not None and loc_elem.text:
                        
                        # Extract additional metadata
                        lastmod_elem = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}lastmod')
                        changefreq_elem = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}changefreq')
                        priority_elem = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}priority')
                        
                        url_data = {
                            'url': loc_elem.text,
                            'lastmod': lastmod_elem.text if lastmod_elem is not None else None,
                            'changefreq': changefreq_elem.text if changefreq_elem is not None else None,
                            'priority': priority_elem.text if priority_elem is not None else None
                        }
                        
                        urls.append(url_data['url'])
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse sitemap XML: {e}")
        except Exception as e:
            logger.error(f"Unexpected error parsing sitemap: {e}")
        
        return urls


class WebsiteIngester(BaseIngester):
    """
    Website content ingester with comprehensive crawling capabilities.
    
    Features:
    - Sitemap-based crawling
    - Robots.txt compliance
    - Rate limiting and respectful crawling
    - Content extraction and cleaning
    - Link discovery and following
    - Duplicate detection and avoidance
    """
    
    def __init__(self, organization_id: str):
        super().__init__("website", organization_id)
        self.session = None
        self.robots_parser = None
        self.sitemap_parser = None
        self.visited_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()
    
    async def __aenter__(self):
        """Async context manager entry."""
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        self.session = aiohttp.ClientSession(connector=connector)
        self.sitemap_parser = SitemapParser(self.session)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def validate_source(self, source_config: Dict[str, Any]) -> bool:
        """Validate that the website is accessible."""
        website_url = source_config.get('website_url')
        if not website_url:
            return False
        
        try:
            if not self.session:
                await self.__aenter__()
            
            async with self.session.get(website_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                return response.status in [200, 301, 302]
                
        except Exception as e:
            self.logger.error(f"Failed to validate website {website_url}: {e}")
            return False
    
    async def extract_content(self, source_config: Dict[str, Any]) -> List[RawContent]:
        """Extract content from website."""
        website_url = source_config['website_url']
        config = WebsiteIngestConfig(**source_config.get('config', {}))
        
        if not self.session:
            await self.__aenter__()
        
        # Initialize robots.txt parser
        await self._initialize_robots_parser(website_url, config['user_agent'])
        
        # Discover URLs to crawl
        urls_to_crawl = await self._discover_urls(website_url, config)
        
        # Filter and limit URLs
        urls_to_crawl = self._filter_urls(urls_to_crawl, config)
        urls_to_crawl = urls_to_crawl[:config['max_pages']]
        
        self.logger.info(f"Found {len(urls_to_crawl)} URLs to crawl for {website_url}")
        
        # Extract content from each URL
        raw_contents = []
        
        for i, url in enumerate(urls_to_crawl):
            if url in self.visited_urls or url in self.failed_urls:
                continue
            
            try:
                # Respect crawl delay
                if i > 0:
                    delay = max(config['delay_between_requests'], self.robots_parser.crawl_delay if self.robots_parser else 1.0)
                    await asyncio.sleep(delay)
                
                # Extract content from URL
                content = await self._extract_url_content(url, config)
                if content:
                    raw_contents.append(content)
                    self.visited_urls.add(url)
                
            except Exception as e:
                self.logger.error(f"Failed to extract content from {url}: {e}")
                self.failed_urls.add(url)
        
        self.logger.info(f"Successfully extracted content from {len(raw_contents)} URLs")
        return raw_contents
    
    async def _initialize_robots_parser(self, website_url: str, user_agent: str):
        """Initialize robots.txt parser."""
        try:
            parsed_url = urlparse(website_url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
            
            self.robots_parser = RobotsTxtParser(robots_url, user_agent)
            await self.robots_parser.load(self.session)
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize robots.txt parser: {e}")
            self.robots_parser = None
    
    async def _discover_urls(self, website_url: str, config: WebsiteIngestConfig) -> List[str]:
        """Discover URLs to crawl from sitemaps and homepage."""
        urls = set()
        
        # Add the main URL
        urls.add(website_url)
        
        # Discover from sitemaps
        sitemap_urls = await self._discover_from_sitemaps(website_url)
        urls.update(sitemap_urls)
        
        # Discover from homepage links if needed
        if len(urls) < config['max_pages'] // 2:
            homepage_urls = await self._discover_from_homepage(website_url, config)
            urls.update(homepage_urls)
        
        return list(urls)
    
    async def _discover_from_sitemaps(self, website_url: str) -> List[str]:
        """Discover URLs from sitemaps."""
        urls = []
        sitemap_urls = []
        
        # Get sitemap URLs from robots.txt
        if self.robots_parser and self.robots_parser.sitemaps:
            sitemap_urls.extend(self.robots_parser.sitemaps)
        
        # Try common sitemap locations
        parsed_url = urlparse(website_url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        common_sitemaps = [
            f"{base_url}/sitemap.xml",
            f"{base_url}/sitemap_index.xml",
            f"{base_url}/sitemaps.xml"
        ]
        
        sitemap_urls.extend(common_sitemaps)
        
        # Parse each sitemap
        for sitemap_url in set(sitemap_urls):
            try:
                sitemap_urls_list = await self.sitemap_parser.parse_sitemap(sitemap_url)
                urls.extend(sitemap_urls_list)
                self.logger.info(f"Found {len(sitemap_urls_list)} URLs in sitemap {sitemap_url}")
                
            except Exception as e:
                self.logger.warning(f"Failed to parse sitemap {sitemap_url}: {e}")
        
        return urls
    
    async def _discover_from_homepage(self, website_url: str, config: WebsiteIngestConfig) -> List[str]:
        """Discover URLs by crawling links from homepage."""
        urls = set()
        
        try:
            async with self.session.get(website_url, timeout=aiohttp.ClientTimeout(total=config['timeout'])) as response:
                if response.status == 200:
                    html_content = await response.text()
                    
                    # Extract links from HTML
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        full_url = urljoin(website_url, href)
                        
                        # Basic URL validation
                        if self._is_valid_url(full_url, config):
                            urls.add(full_url)
                    
                    self.logger.info(f"Found {len(urls)} links on homepage {website_url}")
                    
        except Exception as e:
            self.logger.warning(f"Failed to discover URLs from homepage {website_url}: {e}")
        
        return list(urls)
    
    def _filter_urls(self, urls: List[str], config: WebsiteIngestConfig) -> List[str]:
        """Filter URLs based on configuration."""
        filtered_urls = []
        
        for url in urls:
            if self._should_crawl_url(url, config):
                filtered_urls.append(url)
        
        # Sort URLs to prioritize important pages
        filtered_urls.sort(key=lambda x: self._calculate_url_priority(x))
        
        return filtered_urls
    
    def _should_crawl_url(self, url: str, config: WebsiteIngestConfig) -> bool:
        """Check if URL should be crawled."""
        # Check robots.txt
        if config['respect_robots_txt'] and self.robots_parser and not self.robots_parser.can_fetch(url):
            return False
        
        # Check allowed domains
        if config['allowed_domains']:
            parsed_url = urlparse(url)
            if parsed_url.netloc not in config['allowed_domains']:
                return False
        
        # Check excluded paths
        for excluded_path in config['excluded_paths']:
            if excluded_path in url:
                return False
        
        # Check file extensions
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        
        # Include HTML pages
        if not path or path.endswith('/') or path.endswith('.html') or path.endswith('.htm'):
            return True
        
        # Include PDFs if configured
        if config['include_pdfs'] and path.endswith('.pdf'):
            return True
        
        # Exclude other file types
        excluded_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.css', '.js', '.ico', '.svg', '.woff', '.ttf']
        if any(path.endswith(ext) for ext in excluded_extensions):
            return False
        
        return True
    
    def _is_valid_url(self, url: str, config: WebsiteIngestConfig) -> bool:
        """Check if URL is valid for crawling."""
        try:
            parsed_url = urlparse(url)
            return (parsed_url.scheme in ['http', 'https'] and 
                    parsed_url.netloc and
                    not url.startswith('mailto:') and
                    not url.startswith('tel:'))
        except Exception:
            return False
    
    def _calculate_url_priority(self, url: str) -> int:
        """Calculate URL priority for crawling order."""
        priority = 0
        url_lower = url.lower()
        
        # Prioritize important pages
        if any(keyword in url_lower for keyword in ['home', 'about', 'service', 'product', 'blog']):
            priority += 100
        
        # Prioritize shorter URLs (likely more important)
        priority += max(0, 200 - len(url))
        
        # Prioritize URLs with fewer query parameters
        parsed_url = urlparse(url)
        query_params = len(parse_qs(parsed_url.query))
        priority -= query_params * 10
        
        return priority
    
    async def _extract_url_content(self, url: str, config: WebsiteIngestConfig) -> Optional[RawContent]:
        """Extract content from a single URL."""
        try:
            headers = {
                'User-Agent': config['user_agent'],
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            async with self.session.get(url, headers=headers, 
                                      timeout=aiohttp.ClientTimeout(total=config['timeout']),
                                      allow_redirects=config['follow_redirects']) as response:
                
                if response.status != 200:
                    self.logger.warning(f"Failed to fetch {url}: HTTP {response.status}")
                    return None
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                
                if 'text/html' in content_type:
                    return await self._extract_html_content(url, response, config)
                elif 'application/pdf' in content_type and config['include_pdfs']:
                    return await self._extract_pdf_content(url, response)
                else:
                    self.logger.info(f"Skipping unsupported content type {content_type} for {url}")
                    return None
                
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout extracting content from {url}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to extract content from {url}: {e}")
            return None
    
    async def _extract_html_content(self, url: str, response: aiohttp.ClientResponse, 
                                  config: WebsiteIngestConfig) -> Optional[RawContent]:
        """Extract content from HTML page."""
        try:
            html_content = await response.text()
            
            # Extract title
            title = self._extract_title_from_html(html_content)
            
            # Extract main content
            text_content = self._extract_text_from_html(html_content)
            
            if not text_content or len(text_content.strip()) < 100:
                self.logger.info(f"Insufficient content extracted from {url}")
                return None
            
            # Generate content ID
            content_id = self._generate_content_id("website", url)
            
            # Calculate metadata
            word_count = self._count_words(text_content)
            content_hash = self._calculate_content_hash(text_content)
            
            return RawContent(
                content_id=content_id,
                source_id="website",  # Will be updated with actual source ID
                raw_text=text_content,
                content_type="webpage",
                title=title,
                url=url,
                metadata={
                    "url": url,
                    "content_type": "text/html",
                    "response_status": response.status,
                    "content_length": len(html_content),
                    "headers": dict(response.headers),
                    "extraction_method": "html_parsing"
                },
                content_hash=content_hash,
                word_count=word_count,
                file_size=len(html_content.encode('utf-8'))
            )
            
        except Exception as e:
            self.logger.error(f"Failed to extract HTML content from {url}: {e}")
            return None
    
    async def _extract_pdf_content(self, url: str, response: aiohttp.ClientResponse) -> Optional[RawContent]:
        """Extract content from PDF file."""
        try:
            pdf_content = await response.read()
            
            # Extract text from PDF (simplified - would use PyPDF2 or similar)
            # For now, we'll store the PDF metadata and indicate it needs processing
            
            content_id = self._generate_content_id("website", url)
            
            return RawContent(
                content_id=content_id,
                source_id="website",
                raw_text="[PDF content requires processing]",
                content_type="pdf",
                title=f"PDF Document: {Path(urlparse(url).path).name}",
                url=url,
                metadata={
                    "url": url,
                    "content_type": "application/pdf",
                    "response_status": response.status,
                    "pdf_size": len(pdf_content),
                    "requires_pdf_processing": True
                },
                content_hash=self._calculate_content_hash(str(pdf_content)),
                word_count=0,  # Will be updated after PDF processing
                file_size=len(pdf_content)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to extract PDF content from {url}: {e}")
            return None
    
    # Utility methods for better content extraction
    
    def _extract_main_content(self, html_content: str, config: WebsiteIngestConfig) -> str:
        """Extract main content using custom selectors or readability."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Try custom content selectors first
            content_selectors = config.get('content_selectors', {})
            
            if content_selectors.get('main_content'):
                main_content = soup.select(content_selectors['main_content'])
                if main_content:
                    return ' '.join(elem.get_text(strip=True) for elem in main_content)
            
            # Try common content selectors
            content_selectors = [
                'main', 'article', '.content', '#content', 
                '.post-content', '.entry-content', '.article-content'
            ]
            
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content_text = ' '.join(elem.get_text(strip=True) for elem in elements)
                    if len(content_text) > 200:  # Minimum content length
                        return content_text
            
            # Fallback to full text extraction
            return self._extract_text_from_html(html_content)
            
        except Exception as e:
            self.logger.warning(f"Failed to extract main content with custom selectors: {e}")
            return self._extract_text_from_html(html_content)
    
    async def create_website_source(self, website_url: str, source_config: Dict[str, Any]) -> ContentSource:
        """Create a new website content source."""
        try:
            source_id = f"website_{hashlib.md5(website_url.encode()).hexdigest()[:16]}"
            
            source = ContentSource(
                source_id=source_id,
                source_type="website",
                source_url=website_url,
                source_metadata={
                    "config": source_config,
                    "last_crawl_status": "not_started",
                    "total_pages_discovered": 0,
                    "total_pages_processed": 0
                },
                organization_id=self.organization_id
            )
            
            # Store in database
            await self._store_content_source(source)
            
            return source
            
        except Exception as e:
            self.logger.error(f"Failed to create website source: {e}")
            raise
    
    async def _store_content_source(self, source: ContentSource):
        """Store content source in database."""
        try:
            data = {
                "source_id": source.source_id,
                "source_type": source.source_type,
                "source_url": source.source_url,
                "source_metadata": json.dumps(source.source_metadata),
                "organization_id": source.organization_id,
                "created_at": source.created_at.isoformat(),
                "last_crawled": source.last_crawled.isoformat() if source.last_crawled else None,
                "is_active": source.is_active
            }
            
            supabase_client.client.table("content_sources").insert(data).execute()
            
        except Exception as e:
            self.logger.error(f"Failed to store content source: {e}")
            raise