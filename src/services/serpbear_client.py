"""
SerpBear API Client for SEO Rank Tracking Integration.

This module provides a comprehensive interface to the SerpBear API for:
- Keyword registration and management
- Rank position tracking (desktop + mobile)
- Historical ranking data retrieval
- Domain management

Integrates with our SEO Knowledge Graph to provide real-time ranking insights.
"""

import logging
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, date
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class SerpBearKeyword(BaseModel):
    """Model for SerpBear keyword data."""
    id: int
    keyword: str
    device: str = Field(description="desktop or mobile")
    country: str = Field(default="US")
    position: Optional[int] = Field(default=None, description="Current SERP position")
    url: Optional[str] = Field(default=None, description="Ranking URL")
    domain: str
    history: Dict[str, int] = Field(default_factory=dict, description="Historical positions by date")
    lastResult: Optional[Dict[str, Any]] = Field(default=None)
    lastUpdated: Optional[str] = Field(default=None)


class SerpBearDomain(BaseModel):
    """Model for SerpBear domain data."""
    domain: str
    keywordCount: int = 0
    lastUpdated: Optional[str] = None
    notifications: bool = True


class RankingUpdate(BaseModel):
    """Model for ranking position updates."""
    keyword_id: int
    keyword: str
    domain: str
    device: str
    country: str
    position: Optional[int]
    previous_position: Optional[int]
    url: Optional[str]
    date: str
    change: int = 0  # Position change from previous


class SerpBearClient:
    """
    SerpBear API client for rank tracking integration.
    
    Provides methods to interact with SerpBear API for keyword management
    and rank tracking data retrieval.
    """
    
    def __init__(self, base_url: str = None, api_key: str = None):
        """
        Initialize SerpBear client.
        
        Args:
            base_url: SerpBear instance URL (from env if not provided)
            api_key: API key for authentication (from env if not provided)
        """
        self.base_url = base_url or os.getenv("SERPBEAR_BASE_URL", "http://localhost:3001")
        self.api_key = api_key or os.getenv("SERPBEAR_API_KEY")
        
        if not self.api_key:
            logger.warning("SerpBear API key not configured - some features may not work")
        
        self.session = None
        logger.info(f"SerpBear client initialized for {self.base_url}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def start_session(self):
        """Start aiohttp client session."""
        if not self.session:
            self.session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
    
    async def close_session(self):
        """Close aiohttp client session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make HTTP request to SerpBear API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            **kwargs: Additional request parameters
            
        Returns:
            Response data as dictionary
            
        Raises:
            Exception: If API request fails
        """
        if not self.session:
            await self.start_session()
        
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        try:
            async with self.session.request(method, url, **kwargs) as response:
                response_data = await response.json()
                
                if response.status != 200:
                    logger.error(f"SerpBear API error: {response.status} - {response_data}")
                    raise Exception(f"API request failed: {response.status}")
                
                logger.debug(f"SerpBear API {method} {endpoint} -> {response.status}")
                return response_data
                
        except Exception as e:
            logger.error(f"SerpBear API request failed: {e}")
            raise
    
    async def get_domains(self) -> List[SerpBearDomain]:
        """
        Get all tracked domains.
        
        Returns:
            List of domain objects
        """
        try:
            response = await self._make_request("GET", "/api/domains")
            domains = []
            
            for domain_data in response.get("domains", []):
                domains.append(SerpBearDomain(**domain_data))
            
            logger.info(f"Retrieved {len(domains)} domains from SerpBear")
            return domains
            
        except Exception as e:
            logger.error(f"Failed to get domains: {e}")
            return []
    
    async def get_keywords(self, domain: str) -> List[SerpBearKeyword]:
        """
        Get all keywords for a specific domain.
        
        Args:
            domain: Domain to get keywords for
            
        Returns:
            List of keyword objects
        """
        try:
            response = await self._make_request("GET", "/api/keywords", params={"domain": domain})
            keywords = []
            
            for keyword_data in response.get("keywords", []):
                keyword_obj = SerpBearKeyword(**keyword_data)
                keywords.append(keyword_obj)
            
            logger.info(f"Retrieved {len(keywords)} keywords for {domain}")
            return keywords
            
        except Exception as e:
            logger.error(f"Failed to get keywords for {domain}: {e}")
            return []
    
    async def get_keyword(self, keyword_id: int) -> Optional[SerpBearKeyword]:
        """
        Get specific keyword data by ID.
        
        Args:
            keyword_id: Keyword ID to retrieve
            
        Returns:
            Keyword object or None if not found
        """
        try:
            response = await self._make_request("GET", "/api/keyword", params={"id": keyword_id})
            
            if "keyword" in response:
                keyword_obj = SerpBearKeyword(**response["keyword"])
                logger.info(f"Retrieved keyword {keyword_id}: {keyword_obj.keyword}")
                return keyword_obj
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get keyword {keyword_id}: {e}")
            return None
    
    async def refresh_keywords(self, keyword_ids: List[int]) -> bool:
        """
        Refresh SERP positions for specific keywords.
        
        Args:
            keyword_ids: List of keyword IDs to refresh
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await self._make_request(
                "POST", 
                "/api/refresh", 
                json={"keyword_ids": keyword_ids}
            )
            
            logger.info(f"Refreshed {len(keyword_ids)} keywords")
            return True
            
        except Exception as e:
            logger.error(f"Failed to refresh keywords: {e}")
            return False
    
    async def refresh_all_keywords(self) -> bool:
        """
        Trigger immediate scraping for all keywords.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            await self._make_request("POST", "/api/cron")
            logger.info("Triggered full keyword refresh")
            return True
            
        except Exception as e:
            logger.error(f"Failed to refresh all keywords: {e}")
            return False
    
    async def get_ranking_updates(self, domain: str, days: int = 7) -> List[RankingUpdate]:
        """
        Get ranking position changes for a domain over specified days.
        
        Args:
            domain: Domain to analyze
            days: Number of days to look back
            
        Returns:
            List of ranking updates with position changes
        """
        try:
            keywords = await self.get_keywords(domain)
            updates = []
            
            today = date.today()
            
            for keyword in keywords:
                if not keyword.history:
                    continue
                
                # Get recent history entries
                recent_dates = sorted([
                    date.fromisoformat(d) for d in keyword.history.keys()
                    if (today - date.fromisoformat(d)).days <= days
                ])
                
                if len(recent_dates) < 2:
                    continue
                
                # Calculate change from previous position
                latest_date = recent_dates[-1]
                previous_date = recent_dates[-2] if len(recent_dates) > 1 else None
                
                current_pos = keyword.history.get(str(latest_date))
                previous_pos = keyword.history.get(str(previous_date)) if previous_date else None
                
                change = 0
                if current_pos and previous_pos:
                    change = previous_pos - current_pos  # Positive = moved up
                
                update = RankingUpdate(
                    keyword_id=keyword.id,
                    keyword=keyword.keyword,
                    domain=domain,
                    device=keyword.device,
                    country=keyword.country,
                    position=current_pos,
                    previous_position=previous_pos,
                    url=keyword.url,
                    date=str(latest_date),
                    change=change
                )
                
                updates.append(update)
            
            logger.info(f"Generated {len(updates)} ranking updates for {domain}")
            return updates
            
        except Exception as e:
            logger.error(f"Failed to get ranking updates: {e}")
            return []
    
    async def add_domain(self, domain: str) -> bool:
        """
        Add a domain to SerpBear for tracking.
        
        Args:
            domain: Domain name to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await self._make_request(
                "POST",
                "/api/domains",
                json={"domain": domain}
            )
            
            logger.info(f"Added domain {domain} to SerpBear")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add domain {domain}: {e}")
            return False
    
    async def add_keyword(self, keyword: str, domain: str, device: str = "desktop", 
                         country: str = "US", tags: List[str] = None) -> Optional[int]:
        """
        Add a keyword to SerpBear for tracking.
        
        Args:
            keyword: Keyword to track
            domain: Domain to track keyword for  
            device: Device type (desktop/mobile)
            country: Country code for search
            tags: Optional tags for keyword
            
        Returns:
            Keyword ID if successful, None otherwise
        """
        try:
            # Ensure domain exists first
            domains = await self.get_domains()
            domain_exists = any(d.domain == domain for d in domains)
            
            if not domain_exists:
                await self.add_domain(domain)
            
            # Add keyword
            response = await self._make_request(
                "POST",
                "/api/keywords",
                json={
                    "keyword": keyword,
                    "domain": domain,
                    "device": device,
                    "country": country,
                    "tags": tags or []
                }
            )
            
            keyword_id = response.get("id")
            logger.info(f"Added keyword '{keyword}' for {domain} (ID: {keyword_id})")
            return keyword_id
            
        except Exception as e:
            logger.error(f"Failed to add keyword '{keyword}' for {domain}: {e}")
            return None
    
    async def bulk_add_keywords(self, keywords: List[Dict[str, Any]]) -> List[int]:
        """
        Bulk add multiple keywords to SerpBear.
        
        Args:
            keywords: List of keyword dicts with keys: keyword, domain, device, country, tags
            
        Returns:
            List of successfully added keyword IDs
        """
        added_ids = []
        
        for kw_data in keywords:
            keyword_id = await self.add_keyword(
                keyword=kw_data["keyword"],
                domain=kw_data["domain"],
                device=kw_data.get("device", "desktop"),
                country=kw_data.get("country", "US"),
                tags=kw_data.get("tags", [])
            )
            
            if keyword_id:
                added_ids.append(keyword_id)
                
            # Add small delay to avoid overwhelming SerpBear
            await asyncio.sleep(0.1)
        
        logger.info(f"Bulk added {len(added_ids)}/{len(keywords)} keywords to SerpBear")
        return added_ids
    
    async def remove_keyword(self, keyword_id: int) -> bool:
        """
        Remove a keyword from SerpBear tracking.
        
        Args:
            keyword_id: ID of keyword to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await self._make_request(
                "DELETE",
                f"/api/keywords/{keyword_id}"
            )
            
            logger.info(f"Removed keyword ID {keyword_id} from SerpBear")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove keyword ID {keyword_id}: {e}")
            return False
    
    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get connection status and configuration info.
        
        Returns:
            Status information dictionary
        """
        return {
            "base_url": self.base_url,
            "api_key_configured": bool(self.api_key),
            "session_active": self.session is not None
        }


# Global SerpBear client instance
serpbear_client = SerpBearClient()


async def test_serpbear_connection() -> bool:
    """
    Test SerpBear API connection.
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        async with serpbear_client as client:
            domains = await client.get_domains()
            logger.info(f"SerpBear connection test successful - {len(domains)} domains found")
            return True
    except Exception as e:
        logger.error(f"SerpBear connection test failed: {e}")
        return False


if __name__ == "__main__":
    # Test the SerpBear client
    async def main():
        print("Testing SerpBear connection...")
        success = await test_serpbear_connection()
        print(f"Connection test: {'✅ Success' if success else '❌ Failed'}")
    
    asyncio.run(main())