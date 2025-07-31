"""
Unified SEO Data Service.

This service combines data from multiple sources to provide comprehensive
keyword intelligence:
- SerpBear: Real ranking positions and trends (via direct database access)
- Google Ads: Search volume, competition, CPC data
- Google Search Console: Real traffic, impressions, CTR data
- Tracked Keywords: Our monitoring metadata

Bypasses SerpBear's buggy stats API by directly accessing the database.
"""

import logging
import sqlite3
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.database.models import TrackedKeyword
from src.services.google_ads_service import google_ads_service
from src.services.serpbear_client import serpbear_client

logger = logging.getLogger(__name__)


class UnifiedKeywordData(BaseModel):
    """Unified keyword data from all sources."""
    keyword: str
    domain: str
    
    # SerpBear ranking data
    position: Optional[int] = None
    previous_position: Optional[int] = None
    position_change: int = 0
    ranking_url: Optional[str] = None
    last_updated: Optional[str] = None
    historical_positions: Dict[str, int] = {}
    
    # Google Ads research data
    search_volume: Optional[int] = None
    competition: Optional[str] = None
    competition_score: Optional[float] = None
    cpc: Optional[float] = None
    difficulty: Optional[int] = None
    
    # Google Search Console traffic data
    clicks: Optional[int] = None
    impressions: Optional[int] = None
    ctr: Optional[float] = None
    avg_position_gsc: Optional[float] = None
    
    # Metadata
    data_sources: List[str] = []
    confidence_score: float = 0.0
    tracked_keyword_id: Optional[int] = None


class DomainSummary(BaseModel):
    """Domain-level SEO summary."""
    domain: str
    total_keywords: int = 0
    avg_position: float = 0.0
    best_position: int = 0
    worst_position: int = 0
    top_3_count: int = 0
    top_10_count: int = 0
    total_traffic: int = 0
    total_impressions: int = 0
    avg_ctr: float = 0.0
    visibility_score: float = 0.0
    data_sources: List[str] = []


class UnifiedSEODataService:
    """
    Service for combining SEO data from multiple sources.
    
    Provides unified keyword intelligence by merging:
    - SerpBear database (direct access)
    - Google Ads API
    - Google Search Console API
    - Tracked keywords database
    """
    
    def __init__(self):
        """Initialize the unified SEO data service."""
        self.serpbear_db_path = "/Users/kitan/Desktop/apps/Context-Engineering-Intro/serpbear_db.sqlite"
        logger.info("Unified SEO data service initialized")
    
    def _get_serpbear_data_direct(self, domain: str = None) -> List[Dict[str, Any]]:
        """
        Get SerpBear data directly from database, bypassing buggy API.
        
        Args:
            domain: Optional domain filter
            
        Returns:
            List of keyword data dictionaries
        """
        try:
            # Copy latest database from SerpBear container
            import subprocess
            subprocess.run([
                "docker", "cp", "seo-serpbear:/app/data/database.sqlite", 
                self.serpbear_db_path
            ], check=True, capture_output=True)
            
            # Connect to database
            conn = sqlite3.connect(self.serpbear_db_path)
            conn.row_factory = sqlite3.Row  # Access columns by name
            cursor = conn.cursor()
            
            # Query keywords
            if domain:
                cursor.execute("""
                    SELECT k.*, d.domain as domain_name 
                    FROM keyword k 
                    JOIN domain d ON k.domain = d.domain 
                    WHERE d.domain = ?
                    ORDER BY k.keyword
                """, (domain,))
            else:
                cursor.execute("""
                    SELECT k.*, d.domain as domain_name 
                    FROM keyword k 
                    JOIN domain d ON k.domain = d.domain 
                    ORDER BY d.domain, k.keyword
                """)
            
            keywords = []
            for row in cursor.fetchall():
                # Parse JSON fields safely
                try:
                    history = json.loads(row['history']) if row['history'] else {}
                except (json.JSONDecodeError, TypeError):
                    history = {}
                
                try:
                    last_result = json.loads(row['lastResult']) if row['lastResult'] else {}
                except (json.JSONDecodeError, TypeError):
                    last_result = {}
                
                keyword_data = {
                    'id': row['ID'],
                    'keyword': row['keyword'],
                    'domain': row['domain_name'],
                    'device': row['device'],
                    'country': row['country'],
                    'position': row['position'] if row['position'] else None,
                    'history': history,
                    'last_result': last_result,
                    'last_updated': row['lastUpdated'],
                    'url': last_result.get('url') if last_result else None,
                    'title': last_result.get('title') if last_result else None
                }
                keywords.append(keyword_data)
            
            conn.close()
            logger.info(f"Retrieved {len(keywords)} keywords from SerpBear database")
            return keywords
            
        except Exception as e:
            logger.error(f"Failed to get SerpBear data directly: {e}")
            return []
    
    async def get_unified_keyword_data(
        self, 
        keyword: str, 
        domain: str,
        db: Session = None
    ) -> Optional[UnifiedKeywordData]:
        """
        Get unified data for a single keyword from all sources.
        
        Args:
            keyword: Keyword to analyze
            domain: Domain to analyze for
            db: Database session for tracked keywords
            
        Returns:
            UnifiedKeywordData or None if not found
        """
        try:
            unified_data = UnifiedKeywordData(keyword=keyword, domain=domain)
            data_sources = []
            
            # 1. Get SerpBear ranking data (direct database access)
            serpbear_keywords = self._get_serpbear_data_direct(domain)
            serpbear_keyword = next(
                (kw for kw in serpbear_keywords 
                 if kw['keyword'].lower() == keyword.lower()),
                None
            )
            
            if serpbear_keyword:
                unified_data.position = serpbear_keyword['position']
                unified_data.ranking_url = serpbear_keyword['url']
                unified_data.last_updated = serpbear_keyword['last_updated']
                unified_data.historical_positions = serpbear_keyword['history']
                
                # Calculate position change from history
                if unified_data.historical_positions:
                    sorted_dates = sorted(unified_data.historical_positions.keys(), reverse=True)
                    if len(sorted_dates) >= 2:
                        current_pos = unified_data.historical_positions.get(sorted_dates[0], 0)
                        previous_pos = unified_data.historical_positions.get(sorted_dates[1], 0)
                        if current_pos and previous_pos:
                            unified_data.previous_position = previous_pos
                            unified_data.position_change = previous_pos - current_pos  # Positive = improved
                
                data_sources.append("serpbear")
                unified_data.confidence_score += 0.4
            
            # 2. Get Google Ads research data
            try:
                ads_data = await google_ads_service.get_keyword_ideas([keyword])
                if ads_data.get("keywords"):
                    kw_data = ads_data["keywords"][0]
                    unified_data.search_volume = kw_data.get("volume")
                    unified_data.competition = kw_data.get("competition")
                    unified_data.competition_score = kw_data.get("competition_score")
                    unified_data.cpc = kw_data.get("cpc")
                    unified_data.difficulty = kw_data.get("difficulty")
                    
                    data_sources.append("google_ads")
                    unified_data.confidence_score += 0.3
            except Exception as ads_error:
                logger.debug(f"Google Ads data not available for '{keyword}': {ads_error}")
            
            # 3. Get tracked keyword metadata
            if db:
                try:
                    tracked_kw = db.query(TrackedKeyword).filter(
                        TrackedKeyword.keyword == keyword.lower(),
                        TrackedKeyword.domain == domain,
                        TrackedKeyword.is_active == True
                    ).first()
                    
                    if tracked_kw:
                        unified_data.tracked_keyword_id = tracked_kw.id
                        # Override with tracked keyword data if available
                        if tracked_kw.search_volume:
                            unified_data.search_volume = tracked_kw.search_volume
                        if tracked_kw.difficulty:
                            unified_data.difficulty = tracked_kw.difficulty
                        if tracked_kw.cpc:
                            unified_data.cpc = tracked_kw.cpc
                        
                        data_sources.append("tracked_keywords")
                        unified_data.confidence_score += 0.2
                except Exception as tracked_error:
                    logger.debug(f"Tracked keyword data not available: {tracked_error}")
            
            # 4. TODO: Add Google Search Console data when connected
            # This would add clicks, impressions, CTR, avg_position_gsc
            
            unified_data.data_sources = data_sources
            
            if unified_data.confidence_score > 0:
                logger.debug(f"Unified data for '{keyword}': position={unified_data.position}, volume={unified_data.search_volume}, sources={data_sources}")
                return unified_data
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to get unified data for '{keyword}': {e}")
            return None
    
    async def get_domain_summary(
        self, 
        domain: str,
        db: Session = None
    ) -> DomainSummary:
        """
        Get unified domain summary from all sources.
        
        Args:
            domain: Domain to analyze
            db: Database session for tracked keywords
            
        Returns:
            DomainSummary with aggregated metrics
        """
        try:
            summary = DomainSummary(domain=domain)
            data_sources = []
            
            # Get SerpBear data for domain
            serpbear_keywords = self._get_serpbear_data_direct(domain)
            
            if serpbear_keywords:
                positions = [kw['position'] for kw in serpbear_keywords if kw['position']]
                
                summary.total_keywords = len(serpbear_keywords)
                
                if positions:
                    summary.avg_position = sum(positions) / len(positions)
                    summary.best_position = min(positions)
                    summary.worst_position = max(positions)
                    summary.top_3_count = len([p for p in positions if p <= 3])
                    summary.top_10_count = len([p for p in positions if p <= 10])
                    
                    # Calculate visibility score (percentage of keywords in top 10)
                    summary.visibility_score = (summary.top_10_count / summary.total_keywords) * 100
                
                data_sources.append("serpbear")
            
            # Get Google Ads data for tracked keywords
            if db:
                try:
                    tracked_keywords = db.query(TrackedKeyword).filter(
                        TrackedKeyword.domain == domain,
                        TrackedKeyword.is_active == True
                    ).all()
                    
                    if tracked_keywords:
                        volumes = [kw.search_volume for kw in tracked_keywords if kw.search_volume]
                        if volumes:
                            # Estimate total potential traffic based on search volumes
                            estimated_traffic = sum(volumes) * 0.1  # Rough 10% CTR estimation
                            summary.total_impressions = sum(volumes)
                            
                        data_sources.append("tracked_keywords")
                except Exception as tracked_error:
                    logger.debug(f"Tracked keywords data not available: {tracked_error}")
            
            # TODO: Add Google Search Console aggregations when connected
            # This would add real traffic, impressions, CTR data
            
            summary.data_sources = data_sources
            
            logger.info(f"Domain summary for {domain}: {summary.total_keywords} keywords, avg pos {summary.avg_position:.1f}, {summary.top_10_count} in top 10")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get domain summary for {domain}: {e}")
            return DomainSummary(domain=domain)
    
    async def get_all_unified_keywords(
        self, 
        domain: str = None,
        limit: int = 100,
        db: Session = None
    ) -> List[UnifiedKeywordData]:
        """
        Get unified data for all keywords.
        
        Args:
            domain: Optional domain filter
            limit: Maximum keywords to return
            db: Database session
            
        Returns:
            List of UnifiedKeywordData
        """
        try:
            # Get SerpBear keywords as base
            serpbear_keywords = self._get_serpbear_data_direct(domain)
            
            if not serpbear_keywords:
                logger.warning("No SerpBear keywords found")
                return []
            
            # Limit results
            serpbear_keywords = serpbear_keywords[:limit]
            
            # Get unified data for each keyword
            unified_keywords = []
            
            # Batch get Google Ads data for efficiency
            all_keyword_terms = [kw['keyword'] for kw in serpbear_keywords]
            ads_data_map = {}
            
            try:
                ads_response = await google_ads_service.get_keyword_ideas(all_keyword_terms)
                if ads_response.get("keywords"):
                    ads_data_map = {
                        kw_data["term"]: kw_data 
                        for kw_data in ads_response["keywords"]
                    }
            except Exception as ads_error:
                logger.debug(f"Batch Google Ads data not available: {ads_error}")
            
            # Build unified data
            for serpbear_kw in serpbear_keywords:
                try:
                    unified_data = UnifiedKeywordData(
                        keyword=serpbear_kw['keyword'],
                        domain=serpbear_kw['domain']
                    )
                    
                    # Add SerpBear data
                    unified_data.position = serpbear_kw['position']
                    unified_data.ranking_url = serpbear_kw['url']
                    unified_data.last_updated = serpbear_kw['last_updated']
                    unified_data.historical_positions = serpbear_kw['history']
                    unified_data.data_sources.append("serpbear")
                    unified_data.confidence_score += 0.4
                    
                    # Calculate position change
                    if unified_data.historical_positions:
                        sorted_dates = sorted(unified_data.historical_positions.keys(), reverse=True)
                        if len(sorted_dates) >= 2:
                            current_pos = unified_data.historical_positions.get(sorted_dates[0], 0)
                            previous_pos = unified_data.historical_positions.get(sorted_dates[1], 0)
                            if current_pos and previous_pos:
                                unified_data.previous_position = previous_pos
                                unified_data.position_change = previous_pos - current_pos
                    
                    # Add Google Ads data if available
                    ads_data = ads_data_map.get(serpbear_kw['keyword'])
                    if ads_data:
                        unified_data.search_volume = ads_data.get("volume")
                        unified_data.competition = ads_data.get("competition")
                        unified_data.cpc = ads_data.get("cpc")
                        unified_data.difficulty = ads_data.get("difficulty")
                        unified_data.data_sources.append("google_ads")
                        unified_data.confidence_score += 0.3
                    
                    # Add tracked keyword data if available
                    if db:
                        tracked_kw = db.query(TrackedKeyword).filter(
                            TrackedKeyword.keyword == serpbear_kw['keyword'].lower(),
                            TrackedKeyword.domain == serpbear_kw['domain'],
                            TrackedKeyword.is_active == True
                        ).first()
                        
                        if tracked_kw:
                            unified_data.tracked_keyword_id = tracked_kw.id
                            unified_data.data_sources.append("tracked_keywords")
                            unified_data.confidence_score += 0.2
                    
                    unified_keywords.append(unified_data)
                    
                except Exception as keyword_error:
                    logger.debug(f"Failed to process keyword {serpbear_kw['keyword']}: {keyword_error}")
                    continue
            
            logger.info(f"Generated unified data for {len(unified_keywords)} keywords")
            return unified_keywords
            
        except Exception as e:
            logger.error(f"Failed to get unified keywords: {e}")
            return []
    
    async def get_all_domain_keywords(self, domain: str, organization_id: Optional[str] = None) -> List[UnifiedKeywordData]:
        """
        Get all keywords for a domain with multi-tenant support.
        
        Args:
            domain: Domain to get keywords for
            organization_id: Organization ID for multi-tenant filtering
            
        Returns:
            List of unified keyword data for the domain
        """
        try:
            logger.info(f"Getting all keywords for domain: {domain} (org: {organization_id})")
            
            # Get unified keywords for the domain
            unified_keywords = await self.get_unified_keywords(domain=domain)
            
            # Filter by organization if specified (multi-tenant support)
            if organization_id:
                # Note: Currently SerpBear doesn't have org filtering, 
                # but we maintain the interface for future multi-tenant support
                logger.debug(f"Organization filtering not yet implemented in SerpBear, returning all keywords for domain")
            
            logger.info(f"Retrieved {len(unified_keywords)} keywords for domain {domain}")
            return unified_keywords
            
        except Exception as e:
            logger.error(f"Failed to get all domain keywords for {domain}: {e}")
            return []


# Global unified SEO data service instance
unified_seo_data_service = UnifiedSEODataService()