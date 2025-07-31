"""
SearXNG search service for the SEO Content Knowledge Graph System.

This module provides search capabilities using SearXNG with rate limiting,
retry logic, and trend analysis functionality.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlencode, urlparse

import httpx
import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from config.settings import get_settings
from models.seo_models import KeywordData, SearchIntent, TrendDirection

logger = structlog.get_logger(__name__)


class SearXNGError(Exception):
    """Raised when SearXNG API calls fail."""
    pass


class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, max_requests: int = 60, time_window: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    async def acquire(self) -> None:
        """Acquire permission to make a request."""
        current_time = time.time()
        
        # Remove old requests outside the time window
        self.requests = [
            req_time for req_time in self.requests
            if current_time - req_time < self.time_window
        ]
        
        # Check if we can make a request
        if len(self.requests) >= self.max_requests:
            # Calculate wait time
            oldest_request = min(self.requests)
            wait_time = self.time_window - (current_time - oldest_request)
            
            if wait_time > 0:
                logger.info(f"Rate limit hit, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        
        # Record the request
        self.requests.append(current_time)


class SearXNGService:
    """
    Service for searching and analyzing content using SearXNG.
    
    Provides search functionality with rate limiting, caching,
    and trend analysis capabilities.
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_requests_per_minute: int = 30,
        timeout: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 2.0
    ):
        """
        Initialize SearXNG service.
        
        Args:
            base_url: SearXNG instance URL
            api_key: API key if required
            max_requests_per_minute: Rate limit for requests
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            backoff_factor: Exponential backoff factor
        """
        settings = get_settings()
        
        self.base_url = (base_url or settings.searxng_url).rstrip('/')
        self.api_key = api_key or settings.searxng_api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            max_requests=max_requests_per_minute,
            time_window=60
        )
        
        # Initialize HTTP client
        headers = {
            "User-Agent": "SEO-Content-KnowledgeGraph/1.0",
            "Accept": "application/json",
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        self.client = httpx.AsyncClient(
            timeout=self.timeout,
            headers=headers,
            follow_redirects=True
        )
        
        # Cache for search results
        self._cache: Dict[str, Tuple[Dict[str, Any], float]] = {}
        self._cache_ttl = 3600  # 1 hour
        self._max_cache_size = 1000
        
        logger.info(
            "SearXNG service initialized",
            base_url=self.base_url,
            timeout=self.timeout,
            rate_limit=max_requests_per_minute
        )
    
    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()
    
    def _generate_cache_key(self, query: str, **params) -> str:
        """Generate cache key for search query and parameters."""
        key_data = {"query": query, **params}
        key_str = json.dumps(key_data, sort_keys=True)
        return f"search:{hash(key_str)}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get result from cache if not expired."""
        if cache_key in self._cache:
            result, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                logger.debug("Cache hit", cache_key=cache_key)
                return result
            else:
                # Remove expired entry
                del self._cache[cache_key]
        return None
    
    def _store_in_cache(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Store result in cache."""
        # Remove old entries if cache is full
        if len(self._cache) >= self._max_cache_size:
            # Remove oldest entry
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
        
        self._cache[cache_key] = (result, time.time())
        logger.debug("Cached search result", cache_key=cache_key)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException)),
    )
    async def _make_request(
        self,
        endpoint: str,
        params: Dict[str, Any],
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Make HTTP request to SearXNG API.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            use_cache: Whether to use caching
            
        Returns:
            API response data
        """
        # Check cache first
        if use_cache:
            cache_key = self._generate_cache_key(endpoint, **params)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return cached_result
        
        # Apply rate limiting
        await self.rate_limiter.acquire()
        
        # Make request
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            # Parse response
            if response.headers.get("content-type", "").startswith("application/json"):
                result = response.json()
            else:
                # Try to parse as JSON anyway
                try:
                    result = response.json()
                except json.JSONDecodeError:
                    raise SearXNGError(f"Invalid response format: {response.text[:100]}")
            
            # Store in cache
            if use_cache:
                self._store_in_cache(cache_key, result)
            
            logger.debug(
                "SearXNG request successful",
                endpoint=endpoint,
                status_code=response.status_code,
                results_count=len(result.get("results", []))
            )
            
            return result
            
        except httpx.HTTPStatusError as e:
            logger.error(
                "SearXNG HTTP error",
                endpoint=endpoint,
                status_code=e.response.status_code,
                response_text=e.response.text[:200]
            )
            raise SearXNGError(f"HTTP {e.response.status_code}: {e.response.text[:100]}")
        
        except Exception as e:
            logger.error("SearXNG request failed", endpoint=endpoint, error=str(e))
            raise SearXNGError(f"Request failed: {e}") from e
    
    async def search(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        engines: Optional[List[str]] = None,
        language: str = "en",
        safe_search: int = 1,
        time_range: Optional[str] = None,
        max_results: int = 20,
        format: str = "json"
    ) -> Dict[str, Any]:
        """
        Perform search query.
        
        Args:
            query: Search query
            categories: Search categories (general, images, videos, etc.)
            engines: Specific search engines to use
            language: Search language
            safe_search: Safe search level (0=off, 1=moderate, 2=strict)
            time_range: Time range filter (day, week, month, year)
            max_results: Maximum results to return
            format: Response format
            
        Returns:
            Search results
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")
        
        params = {
            "q": query.strip(),
            "format": format,
            "language": language,
            "safesearch": safe_search,
        }
        
        if categories:
            params["categories"] = ",".join(categories)
        
        if engines:
            params["engines"] = ",".join(engines)
        
        if time_range:
            params["time_range"] = time_range
        
        # SearXNG doesn't have a direct max_results parameter
        # We'll limit results in post-processing
        
        try:
            result = await self._make_request("search", params)
            
            # Limit results if needed
            if "results" in result and len(result["results"]) > max_results:
                result["results"] = result["results"][:max_results]
            
            logger.info(
                "Search completed",
                query=query,
                results_count=len(result.get("results", [])),
                engines_used=result.get("engines", [])
            )
            
            return result
            
        except Exception as e:
            logger.error("Search failed", query=query, error=str(e))
            raise
    
    async def search_trends(
        self,
        topic: str,
        time_periods: List[str] = None,
        languages: List[str] = None
    ) -> Dict[str, Any]:
        """
        Search for trending information about a topic.
        
        Args:
            topic: Topic to analyze trends for
            time_periods: Time periods to check (day, week, month)
            languages: Languages to search in
            
        Returns:
            Trend analysis data
        """
        time_periods = time_periods or ["day", "week", "month"]
        languages = languages or ["en"]
        
        trends_data = {
            "topic": topic,
            "time_periods": {},
            "trend_direction": TrendDirection.STABLE,
            "confidence": 0.5
        }
        
        for period in time_periods:
            period_results = []
            
            for language in languages:
                try:
                    # Search for recent content about the topic
                    recent_query = f"{topic} news trends {period}"
                    results = await self.search(
                        query=recent_query,
                        language=language,
                        time_range=period,
                        max_results=10
                    )
                    
                    period_results.extend(results.get("results", []))
                    
                except Exception as e:
                    logger.warning(
                        "Trend search failed for period",
                        topic=topic,
                        period=period,
                        language=language,
                        error=str(e)
                    )
            
            trends_data["time_periods"][period] = {
                "results_count": len(period_results),
                "results": period_results[:5],  # Keep top 5 for analysis
                "search_volume_indicator": min(len(period_results) / 10, 1.0)
            }
        
        # Analyze trend direction based on result counts
        result_counts = [
            trends_data["time_periods"][period]["results_count"]
            for period in time_periods
        ]
        
        if len(result_counts) >= 2:
            if result_counts[-1] > result_counts[0] * 1.5:
                trends_data["trend_direction"] = TrendDirection.RISING
                trends_data["confidence"] = 0.7
            elif result_counts[-1] < result_counts[0] * 0.5:
                trends_data["trend_direction"] = TrendDirection.DECLINING
                trends_data["confidence"] = 0.7
        
        logger.info(
            "Trend analysis completed",
            topic=topic,
            trend_direction=trends_data["trend_direction"],
            confidence=trends_data["confidence"]
        )
        
        return trends_data
    
    async def extract_keywords_from_results(
        self,
        search_results: List[Dict[str, Any]],
        min_frequency: int = 2,
        exclude_common_words: bool = True
    ) -> List[str]:
        """
        Extract keywords from search results.
        
        Args:
            search_results: List of search result dictionaries
            min_frequency: Minimum frequency for keyword inclusion
            exclude_common_words: Whether to exclude common words
            
        Returns:
            List of extracted keywords
        """
        # Common words to exclude
        common_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
            "been", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can", "this", "that", "these",
            "those", "i", "you", "he", "she", "it", "we", "they", "me", "him",
            "her", "us", "them", "my", "your", "his", "its", "our", "their",
            "what", "where", "when", "why", "how", "which", "who", "whom"
        }
        
        word_frequency = {}
        
        for result in search_results:
            # Extract text from title and content
            text_fields = [
                result.get("title", ""),
                result.get("content", ""),
                result.get("snippet", "")
            ]
            
            for text in text_fields:
                if not text:
                    continue
                
                # Simple word extraction (could be improved with NLP)
                import re
                words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
                
                for word in words:
                    if exclude_common_words and word in common_words:
                        continue
                    
                    word_frequency[word] = word_frequency.get(word, 0) + 1
        
        # Filter by frequency and return sorted list
        keywords = [
            word for word, freq in word_frequency.items()
            if freq >= min_frequency
        ]
        
        # Sort by frequency (descending)
        keywords.sort(key=lambda w: word_frequency[w], reverse=True)
        
        logger.debug(
            "Keywords extracted from search results",
            total_keywords=len(keywords),
            min_frequency=min_frequency
        )
        
        return keywords
    
    async def analyze_competitor_content(
        self,
        competitor_domain: str,
        topic_keywords: List[str],
        max_results_per_keyword: int = 5
    ) -> Dict[str, Any]:
        """
        Analyze competitor content for specific topics.
        
        Args:
            competitor_domain: Competitor domain to analyze
            topic_keywords: Keywords/topics to search for
            max_results_per_keyword: Max results per keyword
            
        Returns:
            Competitor content analysis
        """
        analysis = {
            "competitor_domain": competitor_domain,
            "topics_analyzed": topic_keywords,
            "content_found": {},
            "content_gaps": [],
            "total_content_pieces": 0
        }
        
        for keyword in topic_keywords:
            # Search for content from competitor domain
            site_query = f"site:{competitor_domain} {keyword}"
            
            try:
                results = await self.search(
                    query=site_query,
                    max_results=max_results_per_keyword
                )
                
                competitor_results = []
                for result in results.get("results", []):
                    # Verify the result is actually from the competitor domain
                    url = result.get("url", "")
                    if competitor_domain in url:
                        competitor_results.append(result)
                
                analysis["content_found"][keyword] = competitor_results
                analysis["total_content_pieces"] += len(competitor_results)
                
                # If no content found for this keyword, it's a potential gap
                if not competitor_results:
                    analysis["content_gaps"].append(keyword)
                
            except Exception as e:
                logger.warning(
                    "Competitor analysis failed for keyword",
                    competitor_domain=competitor_domain,
                    keyword=keyword,
                    error=str(e)
                )
                analysis["content_found"][keyword] = []
                analysis["content_gaps"].append(keyword)
        
        logger.info(
            "Competitor analysis completed",
            competitor_domain=competitor_domain,
            total_content=analysis["total_content_pieces"],
            content_gaps=len(analysis["content_gaps"])
        )
        
        return analysis
    
    async def get_related_searches(
        self,
        query: str,
        max_suggestions: int = 10
    ) -> List[str]:
        """
        Get related search suggestions for a query.
        
        Args:
            query: Base search query
            max_suggestions: Maximum suggestions to return
            
        Returns:
            List of related search queries
        """
        related_queries = set()
        
        # Try different approaches to get related searches
        search_variations = [
            f"{query} tips",
            f"{query} guide",
            f"{query} tutorial",
            f"{query} best practices",
            f"how to {query}",
            f"{query} examples",
            f"{query} strategies",
            f"{query} techniques"
        ]
        
        for variation in search_variations:
            try:
                results = await self.search(
                    query=variation,
                    max_results=5
                )
                
                # Extract keywords from titles
                for result in results.get("results", []):
                    title = result.get("title", "")
                    if title:
                        # Simple extraction of potential related queries
                        import re
                        phrases = re.findall(r'\b[a-zA-Z\s]{3,30}\b', title.lower())
                        for phrase in phrases:
                            phrase = phrase.strip()
                            if (len(phrase.split()) >= 2 and 
                                phrase not in related_queries and
                                query.lower() not in phrase):
                                related_queries.add(phrase)
            
            except Exception as e:
                logger.warning(
                    "Related search extraction failed",
                    variation=variation,
                    error=str(e)
                )
        
        # Convert to list and limit results
        related_list = list(related_queries)[:max_suggestions]
        
        logger.debug(
            "Related searches extracted",
            base_query=query,
            related_count=len(related_list)
        )
        
        return related_list
    
    def get_service_stats(self) -> Dict[str, Any]:
        """
        Get service statistics.
        
        Returns:
            Dictionary containing service statistics
        """
        return {
            "base_url": self.base_url,
            "cache_size": len(self._cache),
            "cache_max_size": self._max_cache_size,
            "cache_ttl": self._cache_ttl,
            "rate_limit_requests": len(self.rate_limiter.requests),
            "rate_limit_max": self.rate_limiter.max_requests,
            "timeout": self.timeout,
            "max_retries": self.max_retries
        }


# =============================================================================
# Utility Functions
# =============================================================================

async def get_searxng_service(**kwargs) -> SearXNGService:
    """
    Get configured SearXNG service instance.
    
    Args:
        **kwargs: Additional service parameters
        
    Returns:
        SearXNGService instance
    """
    return SearXNGService(**kwargs)


async def search_simple(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Simple search function.
    
    Args:
        query: Search query
        max_results: Maximum results to return
        
    Returns:
        List of search results
    """
    service = SearXNGService()
    try:
        result = await service.search(query, max_results=max_results)
        return result.get("results", [])
    finally:
        await service.close()


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        service = SearXNGService()
        
        try:
            # Test basic search
            query = "SEO content optimization techniques"
            results = await service.search(query, max_results=5)
            print(f"Search for '{query}' returned {len(results.get('results', []))} results")
            
            # Test trend analysis
            trends = await service.search_trends("artificial intelligence")
            print(f"Trend analysis: {trends['trend_direction']}")
            
            # Test keyword extraction
            if results.get("results"):
                keywords = await service.extract_keywords_from_results(results["results"])
                print(f"Extracted keywords: {keywords[:5]}")
            
            # Test related searches
            related = await service.get_related_searches("content marketing")
            print(f"Related searches: {related[:3]}")
            
            # Service stats
            stats = service.get_service_stats()
            print(f"Service stats: {stats}")
            
        finally:
            await service.close()

    asyncio.run(main())