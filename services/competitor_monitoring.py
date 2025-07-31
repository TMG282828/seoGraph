"""
Competitor Monitoring System for the SEO Content Knowledge Graph System.

This module provides automated competitor content monitoring with scheduled crawling,
rate limiting, robots.txt compliance, and change detection capabilities.
"""

import asyncio
import hashlib
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import aiohttp
import structlog
from aiohttp import ClientSession, ClientTimeout
from bs4 import BeautifulSoup
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential

from database.neo4j_client import Neo4jClient
from database.qdrant_client import QdrantClient
from services.embedding_service import EmbeddingService
from models.seo_models import CompetitorContentAnalysis, CompetitorData, KeywordData
from models.content_models import ContentItem, ContentType
from config.settings import get_settings

logger = structlog.get_logger(__name__)


class CompetitorMonitoringError(Exception):
    """Raised when competitor monitoring operations fail."""
    pass


class RobotsTxtError(CompetitorMonitoringError):
    """Raised when robots.txt prevents crawling."""
    pass


class RateLimitError(CompetitorMonitoringError):
    """Raised when rate limits are exceeded."""
    pass


class CompetitorSite:
    """Represents a competitor website with crawling configuration."""
    
    def __init__(self, 
                 domain: str,
                 name: Optional[str] = None,
                 crawl_frequency: int = 24,  # hours
                 max_pages: int = 100,
                 respect_robots_txt: bool = True,
                 user_agent: str = "SEO-ContentKG-Bot/1.0"):
        self.domain = domain
        self.name = name or domain
        self.crawl_frequency = crawl_frequency
        self.max_pages = max_pages
        self.respect_robots_txt = respect_robots_txt
        self.user_agent = user_agent
        
        # Crawling state
        self.last_crawled: Optional[datetime] = None
        self.robots_txt_cache: Optional[RobotFileParser] = None
        self.robots_txt_checked: Optional[datetime] = None
        self.crawl_errors: List[str] = []
        
        # Rate limiting
        self.request_count = 0
        self.last_request_time = 0.0
        self.rate_limit_delay = 1.0  # seconds between requests
        
        # Content tracking
        self.discovered_urls: Set[str] = set()
        self.content_hashes: Dict[str, str] = {}
        self.last_content_check: Dict[str, datetime] = {}


class CompetitorMonitoringService:
    """
    Competitor monitoring service with scheduled crawling and change detection.
    
    Provides comprehensive competitor content monitoring including:
    - Respectful web crawling with robots.txt compliance
    - Rate limiting and politeness policies
    - Content change detection and analysis
    - Scheduled monitoring with configurable intervals
    - Content extraction and semantic analysis
    """
    
    def __init__(self,
                 neo4j_client: Neo4jClient,
                 qdrant_client: QdrantClient,
                 embedding_service: EmbeddingService,
                 max_concurrent_requests: int = 5,
                 default_rate_limit: float = 1.0):
        self.neo4j_client = neo4j_client
        self.qdrant_client = qdrant_client
        self.embedding_service = embedding_service
        
        # Configuration
        self.max_concurrent_requests = max_concurrent_requests
        self.default_rate_limit = default_rate_limit
        self.settings = get_settings()
        
        # Competitor sites registry
        self.competitor_sites: Dict[str, CompetitorSite] = {}
        
        # Session management
        self.session: Optional[ClientSession] = None
        self.session_timeout = ClientTimeout(total=30, connect=10)
        
        # Caching
        self.content_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour
        self.robots_cache = TTLCache(maxsize=100, ttl=86400)   # 24 hours
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.monitoring_stats = {
            'total_crawls': 0,
            'successful_crawls': 0,
            'failed_crawls': 0,
            'content_changes_detected': 0,
            'last_monitoring_run': None
        }
        
        logger.info("Competitor monitoring service initialized")
    
    async def initialize(self) -> None:
        """Initialize the monitoring service."""
        try:
            # Create HTTP session
            connector = aiohttp.TCPConnector(
                limit=self.max_concurrent_requests,
                limit_per_host=2,
                ttl_dns_cache=300,
                use_dns_cache=True
            )
            
            self.session = ClientSession(
                connector=connector,
                timeout=self.session_timeout,
                headers={
                    'User-Agent': 'SEO-ContentKG-Bot/1.0 (Competitor Analysis)',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive'
                }
            )
            
            logger.info("Competitor monitoring service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize competitor monitoring service: {e}")
            raise CompetitorMonitoringError(f"Initialization failed: {e}")
    
    async def close(self) -> None:
        """Close the monitoring service."""
        try:
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            if self.session:
                await self.session.close()
            
            logger.info("Competitor monitoring service closed")
            
        except Exception as e:
            logger.error(f"Error closing competitor monitoring service: {e}")
    
    async def add_competitor(self, 
                           domain: str,
                           name: Optional[str] = None,
                           crawl_frequency: int = 24,
                           max_pages: int = 100,
                           tenant_id: str = "default") -> bool:
        """
        Add a competitor site for monitoring.
        
        Args:
            domain: Competitor domain to monitor
            name: Optional name for the competitor
            crawl_frequency: Crawling frequency in hours
            max_pages: Maximum pages to crawl
            tenant_id: Tenant identifier
            
        Returns:
            True if competitor was added successfully
        """
        try:
            # Validate domain
            if not self._validate_domain(domain):
                raise CompetitorMonitoringError(f"Invalid domain: {domain}")
            
            # Check if already exists
            if domain in self.competitor_sites:
                logger.warning(f"Competitor {domain} already exists")
                return False
            
            # Create competitor site
            competitor_site = CompetitorSite(
                domain=domain,
                name=name,
                crawl_frequency=crawl_frequency,
                max_pages=max_pages
            )
            
            # Test robots.txt compliance
            if competitor_site.respect_robots_txt:
                robots_allowed = await self._check_robots_txt(domain)
                if not robots_allowed:
                    logger.warning(f"Robots.txt disallows crawling for {domain}")
                    # Continue but mark as restricted
                    competitor_site.crawl_errors.append("robots_txt_restricted")
            
            # Add to registry
            self.competitor_sites[domain] = competitor_site
            
            # Store in database
            await self._store_competitor_config(competitor_site, tenant_id)
            
            logger.info(f"Added competitor {domain} for monitoring")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add competitor {domain}: {e}")
            return False
    
    async def remove_competitor(self, domain: str) -> bool:
        """Remove a competitor from monitoring."""
        try:
            if domain in self.competitor_sites:
                del self.competitor_sites[domain]
                logger.info(f"Removed competitor {domain} from monitoring")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove competitor {domain}: {e}")
            return False
    
    async def crawl_competitor(self, 
                             domain: str,
                             tenant_id: str,
                             force_crawl: bool = False) -> Optional[CompetitorContentAnalysis]:
        """
        Crawl a specific competitor site.
        
        Args:
            domain: Competitor domain to crawl
            tenant_id: Tenant identifier
            force_crawl: Force crawl even if recently crawled
            
        Returns:
            CompetitorContentAnalysis if successful
        """
        try:
            if not self.session:
                await self.initialize()
            
            competitor_site = self.competitor_sites.get(domain)
            if not competitor_site:
                raise CompetitorMonitoringError(f"Competitor {domain} not found")
            
            # Check if crawling is needed
            if not force_crawl and competitor_site.last_crawled:
                time_since_crawl = datetime.now(timezone.utc) - competitor_site.last_crawled
                if time_since_crawl.total_seconds() < competitor_site.crawl_frequency * 3600:
                    logger.info(f"Competitor {domain} crawled recently, skipping")
                    return None
            
            logger.info(f"Starting crawl of competitor {domain}")
            
            # Discover URLs to crawl
            urls_to_crawl = await self._discover_urls(competitor_site)
            
            # Crawl pages
            crawled_content = []
            for url in urls_to_crawl[:competitor_site.max_pages]:
                try:
                    content = await self._crawl_page(url, competitor_site)
                    if content:
                        crawled_content.append(content)
                    
                    # Rate limiting
                    await asyncio.sleep(competitor_site.rate_limit_delay)
                    
                except Exception as e:
                    logger.warning(f"Failed to crawl {url}: {e}")
                    competitor_site.crawl_errors.append(f"crawl_error_{url}")
                    continue
            
            # Analyze content
            analysis = await self._analyze_competitor_content(
                competitor_site, crawled_content, tenant_id
            )
            
            # Update crawl status
            competitor_site.last_crawled = datetime.now(timezone.utc)
            
            # Store analysis
            await self._store_competitor_analysis(analysis, tenant_id)
            
            # Update statistics
            self.monitoring_stats['total_crawls'] += 1
            self.monitoring_stats['successful_crawls'] += 1
            
            logger.info(
                f"Completed crawl of competitor {domain}",
                pages_crawled=len(crawled_content),
                analysis_id=analysis.analyzed_at
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to crawl competitor {domain}: {e}")
            self.monitoring_stats['failed_crawls'] += 1
            
            if domain in self.competitor_sites:
                self.competitor_sites[domain].crawl_errors.append(str(e))
            
            raise CompetitorMonitoringError(f"Crawl failed: {e}")
    
    async def _discover_urls(self, competitor_site: CompetitorSite) -> List[str]:
        """Discover URLs to crawl for a competitor site."""
        try:
            base_url = f"https://{competitor_site.domain}"
            urls_to_crawl = set()
            
            # Start with homepage
            urls_to_crawl.add(base_url)
            
            # Common content pages
            common_paths = [
                '/blog',
                '/news',
                '/articles',
                '/resources',
                '/guides',
                '/case-studies',
                '/whitepapers',
                '/insights',
                '/content'
            ]
            
            for path in common_paths:
                urls_to_crawl.add(urljoin(base_url, path))
            
            # Crawl sitemap if available
            sitemap_urls = await self._crawl_sitemap(base_url)
            urls_to_crawl.update(sitemap_urls)
            
            # Discover links from homepage
            homepage_links = await self._extract_links_from_page(base_url, competitor_site)
            urls_to_crawl.update(homepage_links)
            
            # Filter and prioritize URLs
            filtered_urls = self._filter_urls(list(urls_to_crawl), competitor_site)
            
            return filtered_urls
            
        except Exception as e:
            logger.error(f"Failed to discover URLs for {competitor_site.domain}: {e}")
            return [f"https://{competitor_site.domain}"]
    
    async def _crawl_sitemap(self, base_url: str) -> List[str]:
        """Crawl sitemap.xml to discover URLs."""
        try:
            sitemap_urls = [
                urljoin(base_url, '/sitemap.xml'),
                urljoin(base_url, '/sitemap_index.xml'),
                urljoin(base_url, '/robots.txt')  # Check for sitemap reference
            ]
            
            discovered_urls = []
            
            for sitemap_url in sitemap_urls:
                try:
                    async with self.session.get(sitemap_url) as response:
                        if response.status == 200:
                            content = await response.text()
                            
                            if 'xml' in response.headers.get('content-type', ''):
                                # Parse XML sitemap
                                urls = self._parse_sitemap_xml(content)
                                discovered_urls.extend(urls)
                            elif 'robots.txt' in sitemap_url:
                                # Extract sitemap URLs from robots.txt
                                sitemap_lines = [line for line in content.split('\n') 
                                               if line.lower().startswith('sitemap:')]
                                for line in sitemap_lines:
                                    sitemap_ref = line.split(':', 1)[1].strip()
                                    async with self.session.get(sitemap_ref) as sitemap_response:
                                        if sitemap_response.status == 200:
                                            sitemap_content = await sitemap_response.text()
                                            urls = self._parse_sitemap_xml(sitemap_content)
                                            discovered_urls.extend(urls)
                    
                    await asyncio.sleep(0.5)  # Be polite
                    
                except Exception as e:
                    logger.debug(f"Failed to crawl sitemap {sitemap_url}: {e}")
                    continue
            
            return discovered_urls[:500]  # Limit to avoid overwhelming
            
        except Exception as e:
            logger.error(f"Failed to crawl sitemaps for {base_url}: {e}")
            return []
    
    def _parse_sitemap_xml(self, xml_content: str) -> List[str]:
        """Parse XML sitemap content."""
        try:
            soup = BeautifulSoup(xml_content, 'xml')
            urls = []
            
            # Standard sitemap format
            for url_tag in soup.find_all('url'):
                loc_tag = url_tag.find('loc')
                if loc_tag:
                    urls.append(loc_tag.get_text().strip())
            
            # Sitemap index format
            for sitemap_tag in soup.find_all('sitemap'):
                loc_tag = sitemap_tag.find('loc')
                if loc_tag:
                    urls.append(loc_tag.get_text().strip())
            
            return urls
            
        except Exception as e:
            logger.error(f"Failed to parse sitemap XML: {e}")
            return []
    
    async def _extract_links_from_page(self, url: str, competitor_site: CompetitorSite) -> List[str]:
        """Extract links from a web page."""
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return []
                
                content = await response.text()
                soup = BeautifulSoup(content, 'html.parser')
                
                links = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    
                    # Convert relative URLs to absolute
                    if href.startswith('/'):
                        href = urljoin(url, href)
                    
                    # Only include URLs from the same domain
                    if competitor_site.domain in href:
                        links.append(href)
                
                return links[:100]  # Limit to avoid overwhelming
                
        except Exception as e:
            logger.error(f"Failed to extract links from {url}: {e}")
            return []
    
    def _filter_urls(self, urls: List[str], competitor_site: CompetitorSite) -> List[str]:
        """Filter and prioritize URLs for crawling."""
        filtered_urls = []
        
        # Content type priorities
        content_indicators = {
            'blog': 10,
            'article': 9,
            'news': 8,
            'guide': 8,
            'resource': 7,
            'case-study': 7,
            'whitepaper': 6,
            'insight': 6,
            'content': 5
        }
        
        # Score URLs based on content indicators
        url_scores = []
        for url in urls:
            score = 0
            url_lower = url.lower()
            
            for indicator, weight in content_indicators.items():
                if indicator in url_lower:
                    score += weight
            
            # Boost for certain paths
            if '/blog/' in url_lower or '/articles/' in url_lower:
                score += 5
            
            # Penalize for certain patterns
            if any(pattern in url_lower for pattern in ['admin', 'login', 'register', 'cart']):
                score -= 10
            
            url_scores.append((url, score))
        
        # Sort by score and take top URLs
        url_scores.sort(key=lambda x: x[1], reverse=True)
        filtered_urls = [url for url, score in url_scores if score >= 0]
        
        return filtered_urls
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _crawl_page(self, url: str, competitor_site: CompetitorSite) -> Optional[Dict[str, Any]]:
        """Crawl a single page and extract content."""
        try:
            # Check robots.txt if enabled
            if competitor_site.respect_robots_txt:
                if not await self._can_crawl_url(url, competitor_site):
                    logger.debug(f"Robots.txt disallows crawling {url}")
                    return None
            
            # Rate limiting
            current_time = time.time()
            if current_time - competitor_site.last_request_time < competitor_site.rate_limit_delay:
                sleep_time = competitor_site.rate_limit_delay - (current_time - competitor_site.last_request_time)
                await asyncio.sleep(sleep_time)
            
            # Make request
            async with self.session.get(url) as response:
                competitor_site.last_request_time = time.time()
                competitor_site.request_count += 1
                
                if response.status != 200:
                    logger.warning(f"HTTP {response.status} for {url}")
                    return None
                
                # Check content type
                content_type = response.headers.get('content-type', '')
                if 'text/html' not in content_type:
                    logger.debug(f"Skipping non-HTML content: {url}")
                    return None
                
                content = await response.text()
                
                # Extract content
                page_content = self._extract_page_content(content, url)
                
                if page_content:
                    # Generate content hash for change detection
                    content_hash = hashlib.sha256(
                        (page_content['title'] + page_content['content']).encode()
                    ).hexdigest()
                    
                    # Check for changes
                    previous_hash = competitor_site.content_hashes.get(url)
                    if previous_hash and previous_hash != content_hash:
                        self.monitoring_stats['content_changes_detected'] += 1
                        page_content['content_changed'] = True
                        page_content['previous_hash'] = previous_hash
                    
                    # Update hash
                    competitor_site.content_hashes[url] = content_hash
                    competitor_site.last_content_check[url] = datetime.now(timezone.utc)
                    
                    page_content['url'] = url
                    page_content['crawled_at'] = datetime.now(timezone.utc)
                    page_content['content_hash'] = content_hash
                    
                    return page_content
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to crawl page {url}: {e}")
            return None
    
    def _extract_page_content(self, html_content: str, url: str) -> Optional[Dict[str, Any]]:
        """Extract structured content from HTML."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract title
            title = ""
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text().strip()
            else:
                # Try h1 as fallback
                h1_tag = soup.find('h1')
                if h1_tag:
                    title = h1_tag.get_text().strip()
            
            # Extract meta description
            meta_description = ""
            meta_desc_tag = soup.find('meta', attrs={'name': 'description'})
            if meta_desc_tag:
                meta_description = meta_desc_tag.get('content', '')
            
            # Extract main content
            content = ""
            
            # Try to find main content area
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|main|article|post'))
            
            if main_content:
                content = main_content.get_text()
            else:
                # Fallback to body
                body = soup.find('body')
                if body:
                    content = body.get_text()
            
            # Clean up text
            content = re.sub(r'\s+', ' ', content).strip()
            
            # Extract headings
            headings = []
            for heading_tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                heading_text = heading_tag.get_text().strip()
                if heading_text:
                    headings.append({
                        'level': int(heading_tag.name[1]),
                        'text': heading_text
                    })
            
            # Extract links
            links = []
            for link in soup.find_all('a', href=True):
                link_text = link.get_text().strip()
                if link_text:
                    links.append({
                        'url': link['href'],
                        'text': link_text
                    })
            
            # Extract images
            images = []
            for img in soup.find_all('img', src=True):
                alt_text = img.get('alt', '')
                images.append({
                    'src': img['src'],
                    'alt': alt_text
                })
            
            # Basic content metrics
            word_count = len(content.split())
            
            if word_count < 50:  # Skip very short content
                return None
            
            return {
                'title': title,
                'content': content,
                'meta_description': meta_description,
                'headings': headings,
                'links': links[:20],  # Limit to avoid overwhelming
                'images': images[:10],  # Limit to avoid overwhelming
                'word_count': word_count,
                'url': url
            }
            
        except Exception as e:
            logger.error(f"Failed to extract content from {url}: {e}")
            return None
    
    async def _check_robots_txt(self, domain: str) -> bool:
        """Check if crawling is allowed by robots.txt."""
        try:
            cache_key = f"robots_{domain}"
            if cache_key in self.robots_cache:
                return self.robots_cache[cache_key]
            
            robots_url = f"https://{domain}/robots.txt"
            
            async with self.session.get(robots_url) as response:
                if response.status == 200:
                    robots_content = await response.text()
                    
                    # Parse robots.txt
                    rp = RobotFileParser()
                    rp.set_url(robots_url)
                    rp.set_content(robots_content)
                    
                    # Check if our user agent can crawl
                    user_agent = 'SEO-ContentKG-Bot'
                    can_crawl = rp.can_fetch(user_agent, f"https://{domain}/")
                    
                    # Cache result
                    self.robots_cache[cache_key] = can_crawl
                    
                    return can_crawl
                else:
                    # No robots.txt found, assume crawling is allowed
                    self.robots_cache[cache_key] = True
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to check robots.txt for {domain}: {e}")
            # Assume crawling is allowed if we can't check
            return True
    
    async def _can_crawl_url(self, url: str, competitor_site: CompetitorSite) -> bool:
        """Check if a specific URL can be crawled."""
        try:
            # Check robots.txt cache first
            if competitor_site.robots_txt_cache:
                return competitor_site.robots_txt_cache.can_fetch(
                    competitor_site.user_agent, url
                )
            
            # Load robots.txt if not cached
            domain = competitor_site.domain
            robots_url = f"https://{domain}/robots.txt"
            
            try:
                async with self.session.get(robots_url) as response:
                    if response.status == 200:
                        robots_content = await response.text()
                        
                        rp = RobotFileParser()
                        rp.set_url(robots_url)
                        rp.set_content(robots_content)
                        
                        competitor_site.robots_txt_cache = rp
                        competitor_site.robots_txt_checked = datetime.now(timezone.utc)
                        
                        return rp.can_fetch(competitor_site.user_agent, url)
            except:
                pass
            
            # Default to allowing crawling
            return True
            
        except Exception as e:
            logger.error(f"Failed to check if URL can be crawled: {url}: {e}")
            return True
    
    async def _analyze_competitor_content(self,
                                        competitor_site: CompetitorSite,
                                        crawled_content: List[Dict[str, Any]],
                                        tenant_id: str) -> CompetitorContentAnalysis:
        """Analyze crawled competitor content."""
        try:
            # Extract topics and keywords
            all_content = " ".join([item['content'] for item in crawled_content])
            topics = await self._extract_topics_from_content(all_content)
            
            # Calculate content metrics
            total_word_count = sum(item['word_count'] for item in crawled_content)
            avg_word_count = total_word_count // len(crawled_content) if crawled_content else 0
            
            # Analyze content types
            content_types = self._analyze_content_types(crawled_content)
            
            # Extract keywords
            keywords = await self._extract_keywords_from_content(all_content)
            
            # Calculate publishing frequency
            content_frequency = await self._calculate_content_frequency(crawled_content)
            
            # Create analysis
            analysis = CompetitorContentAnalysis(
                competitor_domain=competitor_site.domain,
                competitor_name=competitor_site.name,
                total_content_pieces=len(crawled_content),
                content_frequency=content_frequency,
                average_word_count=avg_word_count,
                topics_covered=topics,
                topic_distribution=content_types,
                keyword_rankings=keywords,
                tenant_id=tenant_id
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze competitor content: {e}")
            raise CompetitorMonitoringError(f"Content analysis failed: {e}")
    
    async def _extract_topics_from_content(self, content: str) -> List[str]:
        """Extract topics from content using simple keyword extraction."""
        try:
            # Simple topic extraction (in production, use more sophisticated NLP)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
            
            # Remove common stop words
            stop_words = {
                'the', 'and', 'are', 'was', 'will', 'been', 'have', 'has', 'had',
                'this', 'that', 'these', 'those', 'with', 'for', 'from', 'not',
                'but', 'can', 'could', 'would', 'should', 'may', 'might', 'must'
            }
            
            # Count word frequency
            word_freq = {}
            for word in words:
                if word not in stop_words and len(word) > 3:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top topics
            topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [topic[0] for topic in topics[:50]]
            
        except Exception as e:
            logger.error(f"Failed to extract topics: {e}")
            return []
    
    async def _extract_keywords_from_content(self, content: str) -> Dict[str, int]:
        """Extract keywords from content."""
        try:
            # Simple keyword extraction
            words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
            
            # Generate 2-3 word phrases
            phrases = []
            for i in range(len(words) - 1):
                phrase = ' '.join(words[i:i+2])
                phrases.append(phrase)
            
            for i in range(len(words) - 2):
                phrase = ' '.join(words[i:i+3])
                phrases.append(phrase)
            
            # Count phrase frequency
            phrase_freq = {}
            for phrase in phrases:
                if len(phrase) > 6:  # Minimum length
                    phrase_freq[phrase] = phrase_freq.get(phrase, 0) + 1
            
            # Get top keywords
            keywords = sorted(phrase_freq.items(), key=lambda x: x[1], reverse=True)
            return dict(keywords[:20])
            
        except Exception as e:
            logger.error(f"Failed to extract keywords: {e}")
            return {}
    
    def _analyze_content_types(self, crawled_content: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze content types from crawled content."""
        content_types = {}
        
        for item in crawled_content:
            url = item.get('url', '')
            title = item.get('title', '').lower()
            
            # Simple content type classification
            if 'blog' in url or 'blog' in title:
                content_type = 'blog'
            elif 'news' in url or 'news' in title:
                content_type = 'news'
            elif 'guide' in url or 'guide' in title:
                content_type = 'guide'
            elif 'case-study' in url or 'case study' in title:
                content_type = 'case_study'
            elif 'whitepaper' in url or 'whitepaper' in title:
                content_type = 'whitepaper'
            else:
                content_type = 'article'
            
            content_types[content_type] = content_types.get(content_type, 0) + 1
        
        return content_types
    
    async def _calculate_content_frequency(self, crawled_content: List[Dict[str, Any]]) -> float:
        """Calculate content publishing frequency."""
        if not crawled_content:
            return 0.0
        
        # Simple estimate based on content count
        # In production, would extract publish dates from content
        return len(crawled_content) / 30.0  # Assume content spread over 30 days
    
    async def _store_competitor_config(self, competitor_site: CompetitorSite, tenant_id: str) -> None:
        """Store competitor configuration in database."""
        try:
            query = """
            MERGE (c:Competitor {domain: $domain, tenant_id: $tenant_id})
            SET c.name = $name,
                c.crawl_frequency = $crawl_frequency,
                c.max_pages = $max_pages,
                c.created_at = datetime(),
                c.updated_at = datetime()
            """
            
            await self.neo4j_client.run_query(query, {
                'domain': competitor_site.domain,
                'name': competitor_site.name,
                'crawl_frequency': competitor_site.crawl_frequency,
                'max_pages': competitor_site.max_pages,
                'tenant_id': tenant_id
            })
            
        except Exception as e:
            logger.error(f"Failed to store competitor config: {e}")
    
    async def _store_competitor_analysis(self, analysis: CompetitorContentAnalysis, tenant_id: str) -> None:
        """Store competitor analysis in database."""
        try:
            query = """
            MATCH (c:Competitor {domain: $domain, tenant_id: $tenant_id})
            CREATE (a:CompetitorAnalysis {
                analysis_id: $analysis_id,
                competitor_domain: $domain,
                total_content_pieces: $total_content_pieces,
                content_frequency: $content_frequency,
                average_word_count: $average_word_count,
                topics_covered: $topics_covered,
                topic_distribution: $topic_distribution,
                analyzed_at: datetime(),
                tenant_id: $tenant_id
            })
            CREATE (c)-[:HAS_ANALYSIS]->(a)
            """
            
            await self.neo4j_client.run_query(query, {
                'analysis_id': str(analysis.analyzed_at.timestamp()),
                'domain': analysis.competitor_domain,
                'total_content_pieces': analysis.total_content_pieces,
                'content_frequency': analysis.content_frequency,
                'average_word_count': analysis.average_word_count,
                'topics_covered': analysis.topics_covered,
                'topic_distribution': analysis.topic_distribution,
                'tenant_id': tenant_id
            })
            
        except Exception as e:
            logger.error(f"Failed to store competitor analysis: {e}")
    
    def _validate_domain(self, domain: str) -> bool:
        """Validate domain format."""
        try:
            # Simple domain validation
            pattern = r'^[a-zA-Z0-9][a-zA-Z0-9-]{1,61}[a-zA-Z0-9]\.[a-zA-Z]{2,}$'
            return re.match(pattern, domain) is not None
        except:
            return False
    
    async def start_monitoring(self, interval_hours: int = 24) -> None:
        """Start automated monitoring of all competitors."""
        if self.is_monitoring:
            logger.warning("Monitoring already running")
            return
        
        self.is_monitoring = True
        
        async def monitoring_loop():
            try:
                while self.is_monitoring:
                    logger.info("Starting monitoring cycle")
                    
                    for domain, competitor_site in self.competitor_sites.items():
                        try:
                            # Check if crawling is due
                            if competitor_site.last_crawled:
                                time_since_crawl = datetime.now(timezone.utc) - competitor_site.last_crawled
                                if time_since_crawl.total_seconds() < competitor_site.crawl_frequency * 3600:
                                    continue
                            
                            # Crawl competitor
                            await self.crawl_competitor(domain, "default")
                            
                            # Delay between competitors
                            await asyncio.sleep(30)
                            
                        except Exception as e:
                            logger.error(f"Error monitoring competitor {domain}: {e}")
                            continue
                    
                    self.monitoring_stats['last_monitoring_run'] = datetime.now(timezone.utc)
                    
                    # Wait for next cycle
                    await asyncio.sleep(interval_hours * 3600)
                    
            except asyncio.CancelledError:
                logger.info("Monitoring loop cancelled")
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
        
        self.monitoring_task = asyncio.create_task(monitoring_loop())
        logger.info(f"Started competitor monitoring with {interval_hours}h interval")
    
    async def stop_monitoring(self) -> None:
        """Stop automated monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped competitor monitoring")
    
    async def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return {
            'is_monitoring': self.is_monitoring,
            'competitors_count': len(self.competitor_sites),
            'monitoring_stats': self.monitoring_stats,
            'competitor_sites': {
                domain: {
                    'name': site.name,
                    'last_crawled': site.last_crawled,
                    'crawl_errors': len(site.crawl_errors),
                    'request_count': site.request_count,
                    'discovered_urls': len(site.discovered_urls),
                    'content_hashes': len(site.content_hashes)
                }
                for domain, site in self.competitor_sites.items()
            }
        }


# =============================================================================
# Utility Functions
# =============================================================================

async def monitor_competitor_simple(domain: str, tenant_id: str = "default") -> Optional[CompetitorContentAnalysis]:
    """
    Simple function to monitor a single competitor.
    
    Args:
        domain: Competitor domain
        tenant_id: Tenant identifier
        
    Returns:
        CompetitorContentAnalysis if successful
    """
    # Initialize required services
    settings = get_settings()
    
    neo4j_client = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_username,
        password=settings.neo4j_password
    )
    
    qdrant_client = QdrantClient(settings.qdrant_url)
    embedding_service = EmbeddingService()
    
    # Create monitoring service
    monitoring_service = CompetitorMonitoringService(
        neo4j_client=neo4j_client,
        qdrant_client=qdrant_client,
        embedding_service=embedding_service
    )
    
    try:
        await monitoring_service.initialize()
        
        # Add competitor
        await monitoring_service.add_competitor(domain, tenant_id=tenant_id)
        
        # Crawl competitor
        analysis = await monitoring_service.crawl_competitor(domain, tenant_id)
        
        return analysis
        
    finally:
        await monitoring_service.close()


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        # Test competitor monitoring
        analysis = await monitor_competitor_simple(
            domain="example.com",
            tenant_id="test-tenant"
        )
        
        if analysis:
            print(f"Competitor analysis completed: {analysis.competitor_domain}")
            print(f"Content pieces: {analysis.total_content_pieces}")
            print(f"Topics covered: {len(analysis.topics_covered)}")
            print(f"Average word count: {analysis.average_word_count}")
            print(f"Content frequency: {analysis.content_frequency:.2f}")
        else:
            print("No analysis generated")

    asyncio.run(main())