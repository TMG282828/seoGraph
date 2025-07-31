"""
SerpBear Bridge Service.

This service acts as a bridge between our custom SERP scraper and SerpBear,
translating API calls and data formats to make SerpBear use our local scraper
instead of third-party services.

Features:
- SerpBear API compatibility
- Custom scraper integration
- Automated keyword position updates
- Bulk scraping support
- Error handling and fallbacks
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, Field
import json
import os
import hashlib
import time

from .custom_serp_scraper import custom_serp_scraper
from .serpbear_client import serpbear_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/serp-bridge", tags=["serp-bridge"])


class SerpBearKeywordRequest(BaseModel):
    """Request model for SerpBear keyword scraping."""
    keyword: str
    domain: str
    country: str = "US"
    device: str = "desktop"
    engine: str = "google"


class SerpBearBatchRequest(BaseModel):
    """Request model for batch keyword scraping."""
    keywords: List[SerpBearKeywordRequest]


class SerpBearScrapingResponse(BaseModel):
    """Response model matching SerpBear's expected format."""
    success: bool
    keyword: str
    domain: str
    position: Optional[int] = None
    url: Optional[str] = None
    title: Optional[str] = None
    snippet: Optional[str] = None
    country: str = "US"
    device: str = "desktop"
    search_engine: str = "google"
    timestamp: str
    error: Optional[str] = None


class SerpBearBridge:
    """
    Bridge service between custom SERP scraper and SerpBear.
    
    Provides SerpBear-compatible API endpoints while using our local
    scraping infrastructure instead of third-party services.
    """
    
    def __init__(self):
        """Initialize the SerpBear bridge."""
        self.scraper = custom_serp_scraper
        self.is_enabled = True
        self.daily_limit = 10000  # Generous limit for local scraping
        self.daily_usage = 0
        self.last_reset = datetime.utcnow().date()
        
        # Performance optimization - simple cache
        self.cache = {}
        self.cache_ttl = int(os.getenv("BRIDGE_CACHE_TTL", "300"))  # 5 minutes default
        self.fast_mode = False  # Disabled - only use real SERP data
        
        logger.info("SerpBear Bridge initialized - real SERP data only, no mock data")
    
    def _reset_daily_usage_if_needed(self):
        """Reset daily usage counter if it's a new day."""
        today = datetime.utcnow().date()
        if today != self.last_reset:
            self.daily_usage = 0
            self.last_reset = today
            logger.info("Daily usage counter reset")
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        self._reset_daily_usage_if_needed()
        return self.daily_usage < self.daily_limit
    
    def _get_cache_key(self, request: SerpBearKeywordRequest) -> str:
        """Generate cache key for request."""
        key_data = f"{request.keyword}|{request.domain}|{request.country}|{request.device}|{request.engine}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[SerpBearScrapingResponse]:
        """Get cached result if available and fresh."""
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_data
            else:
                # Remove expired cache entry
                del self.cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: SerpBearScrapingResponse):
        """Cache the result."""
        self.cache[cache_key] = (result, time.time())
        logger.debug(f"Cached result for key: {cache_key}")
        
        # Simple cache cleanup - remove old entries if cache gets too large
        if len(self.cache) > 100:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
    
    
    async def scrape_keyword(self, request: SerpBearKeywordRequest) -> SerpBearScrapingResponse:
        """
        Scrape a single keyword using our custom scraper.
        
        Args:
            request: Keyword scraping request
            
        Returns:
            SerpBear-compatible response
        """
        try:
            # Check cache first for performance
            cache_key = self._get_cache_key(request)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            # Check rate limits
            if not self._check_rate_limit():
                return SerpBearScrapingResponse(
                    success=False,
                    keyword=request.keyword,
                    domain=request.domain,
                    country=request.country,
                    device=request.device,
                    search_engine=request.engine,
                    timestamp=datetime.utcnow().isoformat(),
                    error="Daily rate limit exceeded"
                )
            
            # Always perform real search using custom scraper - no mock data
            result = await self.scraper.search_keyword(
                keyword=request.keyword,
                domain=request.domain,
                country=request.country,
                device=request.device,
                engine=request.engine
            )
            
            # Update usage counter
            self.daily_usage += 1
            
            # Create response
            if result:
                response = SerpBearScrapingResponse(
                    success=True,
                    keyword=result["keyword"],
                    domain=result["domain"],
                    position=result["position"],
                    url=result.get("url"),
                    title=result.get("title"),
                    snippet=result.get("snippet", result.get("description")),
                    country=result["country"],
                    device=result["device"],
                    search_engine=result["search_engine"],
                    timestamp=result["timestamp"]
                )
            else:
                response = SerpBearScrapingResponse(
                    success=True,  # Still successful, just no position found
                    keyword=request.keyword,
                    domain=request.domain,
                    position=None,
                    country=request.country,
                    device=request.device,
                    search_engine=request.engine,
                    timestamp=datetime.utcnow().isoformat(),
                    error="Domain not found in search results"
                )
            
            # Cache the response
            self._cache_result(cache_key, response)
            return response
                
        except Exception as e:
            logger.error(f"Error scraping keyword '{request.keyword}': {e}")
            return SerpBearScrapingResponse(
                success=False,
                keyword=request.keyword,
                domain=request.domain,
                country=request.country,
                device=request.device,
                search_engine=request.engine,
                timestamp=datetime.utcnow().isoformat(),
                error=str(e)
            )
    
    async def batch_scrape_keywords(self, request: SerpBearBatchRequest) -> List[SerpBearScrapingResponse]:
        """
        Batch scrape multiple keywords.
        
        Args:
            request: Batch scraping request
            
        Returns:
            List of SerpBear-compatible responses
        """
        try:
            logger.info(f"Starting batch scrape for {len(request.keywords)} keywords")
            
            # Convert to custom scraper format
            keywords_data = [
                {
                    "keyword": kw.keyword,
                    "domain": kw.domain,
                    "country": kw.country,
                    "device": kw.device,
                    "engine": kw.engine
                }
                for kw in request.keywords
            ]
            
            # Check if we have enough quota for batch request
            if not self._check_rate_limit() or (self.daily_usage + len(keywords_data)) > self.daily_limit:
                logger.warning(f"Batch request would exceed daily limit. Usage: {self.daily_usage}/{self.daily_limit}")
                return [
                    SerpBearScrapingResponse(
                        success=False,
                        keyword=kw.keyword,
                        domain=kw.domain,
                        country=kw.country,
                        device=kw.device,
                        search_engine=kw.engine,
                        timestamp=datetime.utcnow().isoformat(),
                        error="Daily rate limit would be exceeded"
                    )
                    for kw in request.keywords
                ]
            
            # Perform batch search
            results = await self.scraper.batch_search_keywords(keywords_data)
            
            # Update usage counter
            self.daily_usage += len(keywords_data)
            
            # Convert results to SerpBear format
            responses = []
            for result in results:
                if result.get("position") is not None:
                    responses.append(SerpBearScrapingResponse(
                        success=True,
                        keyword=result["keyword"],
                        domain=result["domain"],
                        position=result["position"],
                        url=result.get("url"),
                        title=result.get("title"),
                        snippet=result.get("snippet", result.get("description")),
                        country=result.get("country", "US"),
                        device=result.get("device", "desktop"),
                        search_engine=result.get("search_engine", "google"),
                        timestamp=result["timestamp"]
                    ))
                else:
                    responses.append(SerpBearScrapingResponse(
                        success=True,
                        keyword=result["keyword"],
                        domain=result["domain"],
                        position=None,
                        country=result.get("country", "US"),
                        device=result.get("device", "desktop"),
                        search_engine=result.get("search_engine", "google"),
                        timestamp=result["timestamp"],
                        error=result.get("error", "Domain not found in search results")
                    ))
            
            logger.info(f"Batch scrape completed: {len(responses)} results")
            return responses
            
        except Exception as e:
            logger.error(f"Error in batch scrape: {e}")
            return [
                SerpBearScrapingResponse(
                    success=False,
                    keyword=kw.keyword,
                    domain=kw.domain,
                    country=kw.country,
                    device=kw.device,
                    search_engine=kw.engine,
                    timestamp=datetime.utcnow().isoformat(),
                    error=str(e)
                )
                for kw in request.keywords
            ]
    
    async def sync_serpbear_keywords(self) -> Dict[str, Any]:
        """
        Sync with SerpBear to get all keywords and update their positions.
        
        Returns:
            Sync status and results
        """
        try:
            logger.info("Starting SerpBear keyword sync")
            
            # Get all domains from SerpBear
            async with serpbear_client as client:
                domains = await client.get_domains()
            
            if not domains:
                return {"success": False, "error": "No domains found in SerpBear"}
            
            all_keywords = []
            
            # Get keywords for each domain
            for domain in domains:
                async with serpbear_client as client:
                    keywords = await client.get_keywords(domain.domain)
                
                for keyword in keywords:
                    all_keywords.append(SerpBearKeywordRequest(
                        keyword=keyword.keyword,
                        domain=domain.domain,
                        country=keyword.country,
                        device=keyword.device
                    ))
            
            if not all_keywords:
                return {"success": False, "error": "No keywords found in SerpBear"}
            
            logger.info(f"Found {len(all_keywords)} keywords to sync")
            
            # Batch scrape all keywords
            batch_request = SerpBearBatchRequest(keywords=all_keywords)
            results = await self.batch_scrape_keywords(batch_request)
            
            # Count successful updates
            successful_updates = sum(1 for r in results if r.success)
            failed_updates = len(results) - successful_updates
            
            return {
                "success": True,
                "total_keywords": len(all_keywords),
                "successful_updates": successful_updates,
                "failed_updates": failed_updates,
                "domains_processed": len(domains),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error syncing SerpBear keywords: {e}")
            return {"success": False, "error": str(e)}
    
    def get_bridge_status(self) -> Dict[str, Any]:
        """
        Get current bridge status and usage statistics.
        
        Returns:
            Bridge status information
        """
        self._reset_daily_usage_if_needed()
        
        return {
            "enabled": self.is_enabled,
            "daily_usage": self.daily_usage,
            "daily_limit": self.daily_limit,
            "usage_percentage": (self.daily_usage / self.daily_limit) * 100,
            "last_reset": self.last_reset.isoformat(),
            "scraper_health": "healthy" if self.scraper else "unhealthy"
        }


# Global bridge instance
serpbear_bridge = SerpBearBridge()


# API Endpoints
@router.post("/scrape", response_model=SerpBearScrapingResponse)
async def scrape_single_keyword(request: SerpBearKeywordRequest):
    """
    Scrape a single keyword for SERP position with timeout.
    
    This endpoint mimics SerpBear's scraping API format while using
    our custom local scraper infrastructure.
    """
    try:
        result = await asyncio.wait_for(
            serpbear_bridge.scrape_keyword(request),
            timeout=15.0
        )
        return result
    except asyncio.TimeoutError:
        logger.error(f"Timeout scraping keyword: {request.keyword}")
        return SerpBearScrapingResponse(
            success=False,
            keyword=request.keyword,
            domain=request.domain,
            country=request.country,
            device=request.device,
            search_engine=request.engine,
            timestamp=datetime.utcnow().isoformat(),
            error="Request timeout"
        )


# Add root endpoint that SerpBear might be calling
@router.post("/", response_model=SerpBearScrapingResponse)
async def scrape_keyword_root(request: SerpBearKeywordRequest):
    """
    Root endpoint for SerpBear keyword scraping with timeout handling.
    
    SerpBear might be calling the root URL of the custom scraper.
    """
    logger.info(f"🎯 Root endpoint called for keyword: {request.keyword}")
    
    try:
        # Add timeout to prevent SerpBear from waiting too long
        result = await asyncio.wait_for(
            serpbear_bridge.scrape_keyword(request),
            timeout=15.0  # 15-second timeout
        )
        return result
    except asyncio.TimeoutError:
        logger.error(f"Timeout scraping keyword: {request.keyword}")
        return SerpBearScrapingResponse(
            success=False,
            keyword=request.keyword,
            domain=request.domain,
            country=request.country,
            device=request.device,
            search_engine=request.engine,
            timestamp=datetime.utcnow().isoformat(),
            error="Request timeout"
        )


# Add alternative endpoint patterns that SerpBear might expect
@router.post("/api/scrape")
async def scrape_keyword_api(request: SerpBearKeywordRequest):
    """Alternative API endpoint pattern."""
    logger.info(f"🎯 API scrape endpoint called for keyword: {request.keyword}")
    return await serpbear_bridge.scrape_keyword(request)


@router.post("/batch-scrape", response_model=List[SerpBearScrapingResponse])
async def batch_scrape_keywords(request: SerpBearBatchRequest):
    """
    Batch scrape multiple keywords for SERP positions.
    
    Efficient bulk processing with rate limiting and error handling.
    """
    return await serpbear_bridge.batch_scrape_keywords(request)


@router.post("/sync-serpbear")
async def sync_with_serpbear(background_tasks: BackgroundTasks):
    """
    Sync all keywords from SerpBear and update their positions.
    
    This is the main endpoint that SerpBear can call to refresh
    all keyword positions using our local scraper.
    """
    # Run sync in background to avoid timeout
    background_tasks.add_task(serpbear_bridge.sync_serpbear_keywords)
    
    return {
        "success": True,
        "message": "Keyword sync started in background",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/status")
async def get_bridge_status():
    """Get current bridge status and health information."""
    bridge_status = serpbear_bridge.get_bridge_status()
    scraper_health = await custom_serp_scraper.health_check()
    
    return {
        "bridge": bridge_status,
        "scraper": scraper_health,
        "overall_health": "healthy" if (
            bridge_status["enabled"] and 
            scraper_health["overall"] == "healthy"
        ) else "degraded"
    }


@router.get("/health")
async def health_check():
    """Simple health check endpoint."""
    try:
        scraper_health = await custom_serp_scraper.health_check()
        
        if scraper_health["overall"] == "healthy":
            return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
        else:
            return {"status": "degraded", "details": scraper_health}
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


# Add catch-all endpoint to see what SerpBear is actually calling
@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def catch_all_requests(path: str, request: Request):
    """Catch all requests to see what SerpBear is calling."""
    method = request.method
    headers = dict(request.headers)
    query_params = dict(request.query_params)
    
    try:
        body = await request.body()
        body_text = body.decode() if body else ""
    except:
        body_text = ""
    
    logger.info(f"🔍 SerpBear Request: {method} /{path}")
    logger.info(f"   Query: {query_params}")
    logger.info(f"   Headers: {headers}")
    logger.info(f"   Body: {body_text}")
    
    # Return a basic response so SerpBear doesn't get errors
    return {
        "status": "received",
        "method": method,
        "path": path,
        "message": f"Bridge received {method} request to /{path}"
    }


# Background task for automated syncing
async def automated_keyword_sync():
    """
    Automated background task for syncing SerpBear keywords.
    
    This can be called by a scheduler (like the ranking_scheduler)
    to automatically update all keyword positions.
    """
    try:
        logger.info("Starting automated keyword sync")
        result = await serpbear_bridge.sync_serpbear_keywords()
        
        if result["success"]:
            logger.info(f"Automated sync successful: {result['successful_updates']} keywords updated")
        else:
            logger.error(f"Automated sync failed: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Automated sync error: {e}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    # Test the bridge service
    async def test_bridge():
        print("🌉 Testing SerpBear Bridge Service")
        print("=" * 50)
        
        # Test single keyword scrape
        print("\n1. Single Keyword Scrape:")
        request = SerpBearKeywordRequest(
            keyword="python tutorial",
            domain="python.org",
            country="US",
            device="desktop",
            engine="google"
        )
        
        result = await serpbear_bridge.scrape_keyword(request)
        print(f"   Success: {result.success}")
        if result.position:
            print(f"   Position: {result.position}")
            print(f"   URL: {result.url}")
        else:
            print(f"   Error: {result.error}")
        
        # Test bridge status
        print("\n2. Bridge Status:")
        status = serpbear_bridge.get_bridge_status()
        print(f"   Enabled: {status['enabled']}")
        print(f"   Daily Usage: {status['daily_usage']}/{status['daily_limit']}")
        print(f"   Usage: {status['usage_percentage']:.1f}%")
        
        print("\n🎉 SerpBear Bridge test completed!")
    
    asyncio.run(test_bridge())