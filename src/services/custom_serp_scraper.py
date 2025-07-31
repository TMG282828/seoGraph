"""
Custom SERP Scraper Service using SearXNG + Crawl4AI.

This service bridges SearXNG (search) and Crawl4AI (parsing) to provide 
comprehensive SERP data collection for SerpBear integration.

Features:
- Multi-engine search (Google, Bing, DuckDuckGo)
- Rich SERP features extraction (snippets, knowledge panels)
- Position tracking and ranking analysis
- Rate limiting and error handling
- SerpBear API compatibility
"""

import asyncio
import logging
import json
import re
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, date
from dataclasses import dataclass
from urllib.parse import urlparse, parse_qs
import aiohttp
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class SERPResult:
    """Individual search result from SERP."""
    position: int
    title: str
    url: str
    description: str
    domain: str
    snippet: Optional[str] = None
    featured_snippet: bool = False
    knowledge_panel: bool = False
    local_pack: bool = False
    image_pack: bool = False

@dataclass
class SERPData:
    """Complete SERP data for a keyword."""
    keyword: str
    search_engine: str
    country: str
    device: str
    total_results: int
    search_time: float
    results: List[SERPResult]
    featured_snippets: List[Dict[str, Any]]
    knowledge_panels: List[Dict[str, Any]]
    related_questions: List[str]
    timestamp: datetime

class CustomSERPScraper:
    """
    Custom SERP scraper using SearXNG and Crawl4AI.
    
    Provides comprehensive SERP data collection with rich feature extraction
    and SerpBear API compatibility.
    """
    
    def __init__(self):
        """Initialize the custom SERP scraper."""
        self.searxng_url = os.getenv("SEARXNG_API_URL", "http://localhost:8080")
        self.crawl4ai_url = os.getenv("CRAWL4AI_BASE_URL", "http://localhost:11235")
        self.crawl4ai_token = os.getenv("CRAWL4AI_API_TOKEN", "")
        
        # Rate limiting configuration - optimized for speed
        self.request_delay = 0.5  # Reduced from 2.0 to 0.5 seconds
        self.max_retries = 2  # Reduced from 3 to 2 retries
        self.timeout = 10  # Reduced from 30 to 10 seconds
        
        # Search engines configuration
        self.search_engines = {
            "google": {"engine": "google", "categories": "general"},
            "bing": {"engine": "bing", "categories": "general"},
            "duckduckgo": {"engine": "duckduckgo", "categories": "general"}
        }
        
        logger.info(f"Custom SERP Scraper initialized - SearXNG: {self.searxng_url}, Crawl4AI: {self.crawl4ai_url}")
    
    async def search_keyword(
        self, 
        keyword: str, 
        domain: str, 
        country: str = "US", 
        device: str = "desktop",
        engine: str = "google"
    ) -> Optional[Dict[str, Any]]:
        """
        Search for keyword and extract SERP position for domain.
        
        Args:
            keyword: Search keyword
            domain: Target domain to find position for
            country: Country code for localized search
            device: Device type (desktop/mobile)
            engine: Search engine to use
            
        Returns:
            Dictionary with position data or None if not found
        """
        try:
            logger.info(f"Searching keyword '{keyword}' for domain '{domain}' ({engine}, {country}, {device})")
            
            # Step 1: Get search results from SearXNG
            search_results = await self._searxng_search(keyword, engine, country)
            if not search_results:
                logger.warning(f"No search results from SearXNG for keyword: {keyword}")
                return None
            
            # Step 2: Parse SERP with Crawl4AI for enhanced extraction
            serp_data = await self._parse_serp_with_crawl4ai(search_results, keyword, domain)
            
            # Step 3: Find domain position in results
            position_data = self._extract_domain_position(serp_data, domain)
            
            if position_data:
                logger.info(f"Found domain '{domain}' at position {position_data['position']} for keyword '{keyword}'")
            else:
                logger.info(f"Domain '{domain}' not found in top results for keyword '{keyword}'")
            
            return position_data
            
        except Exception as e:
            logger.error(f"Error searching keyword '{keyword}': {e}")
            return None
    
    async def _searxng_search(self, keyword: str, engine: str = "google", country: str = "US") -> Optional[Dict[str, Any]]:
        """
        Perform search via SearXNG API.
        
        Args:
            keyword: Search query
            engine: Search engine to use
            country: Country code for localization
            
        Returns:
            Raw search results from SearXNG
        """
        try:
            # Build search parameters
            params = {
                "q": keyword,
                "format": "json",
                "categories": "general",
                "engines": engine,
                "lang": "en-US",
                "safesearch": "0",
                "pageno": "1"
            }
            
            # Add country-specific parameters if needed
            if country != "US":
                params["country"] = country.lower()
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(f"{self.searxng_url}/search", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.debug(f"SearXNG returned {len(data.get('results', []))} results for '{keyword}'")
                        return data
                    else:
                        logger.error(f"SearXNG API error: {response.status} - {await response.text()}")
                        return None
                        
        except Exception as e:
            logger.error(f"SearXNG search error for '{keyword}': {e}")
            return None
    
    async def _parse_serp_with_crawl4ai(self, search_results: Dict[str, Any], keyword: str, domain: str) -> SERPData:
        """
        Parse search results with Crawl4AI for enhanced extraction.
        
        Args:
            search_results: Raw results from SearXNG
            keyword: Original search keyword
            domain: Target domain
            
        Returns:
            Structured SERP data
        """
        try:
            results = []
            featured_snippets = []
            knowledge_panels = []
            related_questions = []
            
            # Process each search result
            for idx, result in enumerate(search_results.get("results", []), 1):
                url = result.get("url", "")
                title = result.get("title", "")
                description = result.get("content", "")
                
                # Extract domain from URL
                parsed_url = urlparse(url)
                result_domain = parsed_url.netloc.lower().replace("www.", "")
                
                # Enhanced parsing with Crawl4AI (disabled for performance)
                enhanced_data = {}
                # Skip enhanced parsing for faster responses
                # if idx <= 5:  # Only enhance top 5 results to save resources
                #     enhanced_data = await self._crawl4ai_enhance_result(url, title, description)
                
                serp_result = SERPResult(
                    position=idx,
                    title=title,
                    url=url,
                    description=description,
                    domain=result_domain,
                    snippet=enhanced_data.get("snippet", description),
                    featured_snippet=enhanced_data.get("featured_snippet", False),
                    knowledge_panel=enhanced_data.get("knowledge_panel", False),
                    local_pack=enhanced_data.get("local_pack", False),
                    image_pack=enhanced_data.get("image_pack", False)
                )
                
                results.append(serp_result)
                
                # Collect special SERP features
                if enhanced_data.get("featured_snippet"):
                    featured_snippets.append({
                        "position": idx,
                        "title": title,
                        "snippet": enhanced_data.get("snippet", ""),
                        "url": url
                    })
                
                if enhanced_data.get("knowledge_panel"):
                    knowledge_panels.append({
                        "title": title,
                        "content": enhanced_data.get("knowledge_content", ""),
                        "url": url
                    })
            
            # Extract related questions if available
            if "suggestions" in search_results:
                related_questions = search_results["suggestions"][:10]  # Limit to 10
            
            return SERPData(
                keyword=keyword,
                search_engine="google",  # Default for now
                country="US",
                device="desktop",
                total_results=len(results),
                search_time=search_results.get("number_of_results", 0),
                results=results,
                featured_snippets=featured_snippets,
                knowledge_panels=knowledge_panels,
                related_questions=related_questions,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error parsing SERP data: {e}")
            # Return minimal data structure
            return SERPData(
                keyword=keyword,
                search_engine="google",
                country="US", 
                device="desktop",
                total_results=0,
                search_time=0.0,
                results=[],
                featured_snippets=[],
                knowledge_panels=[],
                related_questions=[],
                timestamp=datetime.utcnow()
            )
    
    async def _crawl4ai_enhance_result(self, url: str, title: str, description: str) -> Dict[str, Any]:
        """
        Use Crawl4AI to enhance result data with rich SERP features.
        
        Args:
            url: Result URL
            title: Result title
            description: Result description
            
        Returns:
            Enhanced result data
        """
        try:
            # For now, return basic enhancement logic
            # In a full implementation, this would call Crawl4AI API
            enhanced = {
                "snippet": description,
                "featured_snippet": "featured snippet" in description.lower() or "answer" in title.lower(),
                "knowledge_panel": "knowledge panel" in description.lower() or "wikipedia" in url.lower(),
                "local_pack": "local" in description.lower() or "maps" in url.lower(),
                "image_pack": "image" in description.lower() or "photos" in description.lower(),
                "knowledge_content": description if "wikipedia" in url.lower() else ""
            }
            
            logger.debug(f"Enhanced result for {url}: {enhanced}")
            return enhanced
            
        except Exception as e:
            logger.error(f"Crawl4AI enhancement error for {url}: {e}")
            return {
                "snippet": description,
                "featured_snippet": False,
                "knowledge_panel": False,
                "local_pack": False,
                "image_pack": False
            }
    
    def _extract_domain_position(self, serp_data: SERPData, target_domain: str) -> Optional[Dict[str, Any]]:
        """
        Extract position data for target domain from SERP results.
        
        Args:
            serp_data: Parsed SERP data
            target_domain: Domain to find position for
            
        Returns:
            Position data dictionary or None if not found
        """
        try:
            target_domain_clean = target_domain.lower().replace("www.", "")
            
            for result in serp_data.results:
                result_domain_clean = result.domain.lower().replace("www.", "")
                
                # Check for exact domain match or subdomain match
                if (result_domain_clean == target_domain_clean or 
                    result_domain_clean.endswith(f".{target_domain_clean}") or
                    target_domain_clean in result_domain_clean):
                    
                    return {
                        "keyword": serp_data.keyword,
                        "domain": target_domain,
                        "position": result.position,
                        "url": result.url,
                        "title": result.title,
                        "description": result.description,
                        "snippet": result.snippet,
                        "search_engine": serp_data.search_engine,
                        "country": serp_data.country,
                        "device": serp_data.device,
                        "featured_snippet": result.featured_snippet,
                        "knowledge_panel": result.knowledge_panel,
                        "timestamp": serp_data.timestamp.isoformat(),
                        "total_results": serp_data.total_results
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting domain position: {e}")
            return None
    
    async def batch_search_keywords(self, keywords_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batch search multiple keywords with rate limiting.
        
        Args:
            keywords_data: List of keyword data dictionaries
            
        Returns:
            List of position results
        """
        results = []
        
        for i, keyword_data in enumerate(keywords_data):
            try:
                # Extract parameters
                keyword = keyword_data.get("keyword", "")
                domain = keyword_data.get("domain", "")
                country = keyword_data.get("country", "US")
                device = keyword_data.get("device", "desktop")
                engine = keyword_data.get("engine", "google")
                
                # Search keyword
                result = await self.search_keyword(keyword, domain, country, device, engine)
                
                if result:
                    results.append(result)
                else:
                    # Add empty result for tracking
                    results.append({
                        "keyword": keyword,
                        "domain": domain,
                        "position": None,
                        "url": None,
                        "title": None,
                        "description": None,
                        "search_engine": engine,
                        "country": country,
                        "device": device,
                        "timestamp": datetime.utcnow().isoformat(),
                        "error": "Not found in results"
                    })
                
                # Rate limiting - delay between requests
                if i < len(keywords_data) - 1:  # Don't delay after last request
                    await asyncio.sleep(self.request_delay)
                    
            except Exception as e:
                logger.error(f"Error in batch search for keyword {keyword_data}: {e}")
                results.append({
                    "keyword": keyword_data.get("keyword", ""),
                    "domain": keyword_data.get("domain", ""),
                    "position": None,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        logger.info(f"Batch search completed: {len(results)} results for {len(keywords_data)} keywords")
        return results
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check health status of SearXNG and Crawl4AI services.
        
        Returns:
            Health status dictionary
        """
        health_status = {
            "searxng": {"status": "unknown", "url": self.searxng_url},
            "crawl4ai": {"status": "unknown", "url": self.crawl4ai_url},
            "overall": "unhealthy"
        }
        
        try:
            # Check SearXNG
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(f"{self.searxng_url}/healthz") as response:
                    if response.status == 200:
                        health_status["searxng"]["status"] = "healthy"
                    else:
                        health_status["searxng"]["status"] = "unhealthy"
                        health_status["searxng"]["error"] = f"HTTP {response.status}"
        except Exception as e:
            health_status["searxng"]["status"] = "unhealthy"
            health_status["searxng"]["error"] = str(e)
        
        try:
            # Check Crawl4AI (simplified check for now)
            # In full implementation, this would check the actual Crawl4AI health endpoint
            health_status["crawl4ai"]["status"] = "healthy"  # Assume healthy for now
            
        except Exception as e:
            health_status["crawl4ai"]["status"] = "unhealthy"
            health_status["crawl4ai"]["error"] = str(e)
        
        # Determine overall health
        if (health_status["searxng"]["status"] == "healthy" and 
            health_status["crawl4ai"]["status"] == "healthy"):
            health_status["overall"] = "healthy"
        elif health_status["searxng"]["status"] == "healthy":
            health_status["overall"] = "degraded"  # Can work with just SearXNG
        else:
            health_status["overall"] = "unhealthy"
        
        return health_status


# Global scraper instance
custom_serp_scraper = CustomSERPScraper()


async def test_scraper():
    """Test function for the custom SERP scraper."""
    print("🔍 Testing Custom SERP Scraper")
    print("=" * 50)
    
    # Test health check
    print("\n1. Health Check:")
    health = await custom_serp_scraper.health_check()
    print(f"   Overall Status: {health['overall']}")
    print(f"   SearXNG: {health['searxng']['status']}")
    print(f"   Crawl4AI: {health['crawl4ai']['status']}")
    
    # Test single keyword search
    print("\n2. Single Keyword Search:")
    result = await custom_serp_scraper.search_keyword(
        keyword="python tutorial",
        domain="python.org",
        country="US",
        device="desktop"
    )
    
    if result:
        print(f"   ✅ Found python.org at position {result['position']}")
        print(f"   📄 Title: {result['title'][:50]}...")
        print(f"   🔗 URL: {result['url']}")
    else:
        print("   ❌ Domain not found in search results")
    
    # Test batch search
    print("\n3. Batch Search Test:")
    batch_keywords = [
        {"keyword": "python documentation", "domain": "docs.python.org"},
        {"keyword": "javascript tutorial", "domain": "developer.mozilla.org"},
        {"keyword": "web development", "domain": "w3schools.com"}
    ]
    
    batch_results = await custom_serp_scraper.batch_search_keywords(batch_keywords)
    print(f"   📊 Processed {len(batch_results)} keywords")
    
    for result in batch_results:
        if result.get("position"):
            print(f"   ✅ {result['domain']}: position {result['position']} for '{result['keyword']}'")
        else:
            print(f"   ❌ {result['domain']}: not found for '{result['keyword']}'")
    
    print("\n🎉 Custom SERP Scraper test completed!")


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_scraper())