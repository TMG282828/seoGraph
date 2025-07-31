"""
Keyword Sync Service for TrackedKeywords ↔ SerpBear Integration.

This service manages the bidirectional sync between our tracked keywords
and SerpBear's ranking system, ensuring keywords are automatically 
added to SerpBear for position tracking.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from sqlalchemy.orm import Session
from pydantic import BaseModel

from src.database.database import get_db
from src.database.models import TrackedKeyword
from .serpbear_client import serpbear_client, SerpBearKeyword

logger = logging.getLogger(__name__)


class KeywordSyncResult(BaseModel):
    """Results from keyword sync operation."""
    success: bool
    added_to_serpbear: int = 0
    removed_from_serpbear: int = 0
    already_synced: int = 0
    failed_syncs: int = 0
    errors: List[str] = []


class KeywordSyncService:
    """
    Service for syncing tracked keywords with SerpBear.
    
    Handles:
    - Adding new tracked keywords to SerpBear
    - Removing deactivated keywords from SerpBear
    - Bulk sync operations
    - Sync status tracking
    """
    
    def __init__(self):
        """Initialize the keyword sync service."""
        self.sync_in_progress = False
        logger.info("Keyword sync service initialized")
    
    async def sync_keyword_to_serpbear(
        self, 
        tracked_keyword: TrackedKeyword, 
        db: Session
    ) -> bool:
        """
        Sync a single tracked keyword to SerpBear.
        
        Args:
            tracked_keyword: TrackedKeyword instance to sync
            db: Database session
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not tracked_keyword.is_active:
                logger.debug(f"Skipping inactive keyword: {tracked_keyword.keyword}")
                return True
            
            # Extract domain from target_url if available, otherwise use domain field
            domain = tracked_keyword.domain
            if not domain and tracked_keyword.target_url:
                from urllib.parse import urlparse
                parsed = urlparse(tracked_keyword.target_url)
                domain = parsed.netloc.replace('www.', '')
            
            if not domain:
                logger.warning(f"No domain found for keyword: {tracked_keyword.keyword}")
                return False
            
            # Check if keyword already exists in SerpBear
            async with serpbear_client as client:
                # Get existing keywords for domain
                existing_keywords = await client.get_keywords(domain)
                keyword_exists = any(
                    kw.keyword.lower() == tracked_keyword.keyword.lower() 
                    for kw in existing_keywords
                )
                
                if keyword_exists:
                    logger.debug(f"Keyword '{tracked_keyword.keyword}' already exists in SerpBear")
                    return True
                
                # Add keyword to SerpBear
                keyword_id = await client.add_keyword(
                    keyword=tracked_keyword.keyword,
                    domain=domain,
                    device="desktop",  # Default to desktop tracking
                    country="US",      # Default to US
                    tags=["tracked-keyword"]  # Tag to identify synced keywords
                )
                
                if keyword_id:
                    # Update tracked keyword with SerpBear metadata
                    tracked_keyword.notes = (tracked_keyword.notes or "") + f" [SerpBear ID: {keyword_id}]"
                    tracked_keyword.updated_at = datetime.now()
                    db.commit()
                    
                    logger.info(f"✅ Synced '{tracked_keyword.keyword}' to SerpBear (ID: {keyword_id})")
                    return True
                else:
                    logger.error(f"❌ Failed to add '{tracked_keyword.keyword}' to SerpBear")
                    return False
        
        except Exception as e:
            logger.error(f"Failed to sync keyword '{tracked_keyword.keyword}': {e}")
            return False
    
    async def bulk_sync_to_serpbear(self, db: Session) -> KeywordSyncResult:
        """
        Bulk sync all active tracked keywords to SerpBear.
        
        Args:
            db: Database session
            
        Returns:
            KeywordSyncResult with sync statistics
        """
        if self.sync_in_progress:
            return KeywordSyncResult(
                success=False,
                errors=["Sync already in progress"]
            )
        
        self.sync_in_progress = True
        logger.info("🔄 Starting bulk keyword sync to SerpBear")
        
        try:
            # Get all active tracked keywords
            tracked_keywords = db.query(TrackedKeyword).filter(
                TrackedKeyword.is_active == True
            ).all()
            
            if not tracked_keywords:
                logger.info("No active tracked keywords to sync")
                return KeywordSyncResult(success=True, already_synced=0)
            
            logger.info(f"Found {len(tracked_keywords)} active tracked keywords to sync")
            
            # Group keywords by domain for efficient processing
            keywords_by_domain = {}
            for kw in tracked_keywords:
                domain = kw.domain
                if not domain and kw.target_url:
                    from urllib.parse import urlparse
                    parsed = urlparse(kw.target_url)
                    domain = parsed.netloc.replace('www.', '')
                
                if domain:
                    if domain not in keywords_by_domain:
                        keywords_by_domain[domain] = []
                    keywords_by_domain[domain].append(kw)
            
            result = KeywordSyncResult(success=True)
            
            # Process each domain
            for domain, domain_keywords in keywords_by_domain.items():
                try:
                    logger.info(f"Processing {len(domain_keywords)} keywords for {domain}")
                    
                    # Get existing SerpBear keywords for this domain
                    async with serpbear_client as client:
                        existing_keywords = await client.get_keywords(domain)
                        existing_keyword_set = {
                            kw.keyword.lower() for kw in existing_keywords
                        }
                    
                    # Determine which keywords need to be added
                    keywords_to_add = []
                    for kw in domain_keywords:
                        if kw.keyword.lower() not in existing_keyword_set:
                            keywords_to_add.append({
                                "keyword": kw.keyword,
                                "domain": domain,
                                "device": "desktop",
                                "country": "US",
                                "tags": ["tracked-keyword"]
                            })
                        else:
                            result.already_synced += 1
                    
                    if keywords_to_add:
                        # Bulk add keywords
                        async with serpbear_client as client:
                            added_ids = await client.bulk_add_keywords(keywords_to_add)
                            result.added_to_serpbear += len(added_ids)
                            result.failed_syncs += len(keywords_to_add) - len(added_ids)
                            
                            # Update tracked keywords with SerpBear IDs
                            for i, kw_data in enumerate(keywords_to_add):
                                if i < len(added_ids):
                                    # Find corresponding tracked keyword
                                    tracked_kw = next(
                                        (tk for tk in domain_keywords 
                                         if tk.keyword == kw_data["keyword"]), 
                                        None
                                    )
                                    if tracked_kw:
                                        tracked_kw.notes = (tracked_kw.notes or "") + f" [SerpBear ID: {added_ids[i]}]"
                                        tracked_kw.updated_at = datetime.now()
                        
                        db.commit()
                        logger.info(f"✅ Added {len(added_ids)} keywords to SerpBear for {domain}")
                    
                except Exception as domain_error:
                    error_msg = f"Failed to sync keywords for {domain}: {domain_error}"
                    logger.error(error_msg)
                    result.errors.append(error_msg)
                    result.failed_syncs += len(domain_keywords)
            
            logger.info(f"🎉 Bulk sync completed: {result.added_to_serpbear} added, {result.already_synced} already synced, {result.failed_syncs} failed")
            return result
            
        except Exception as e:
            error_msg = f"Bulk sync failed: {e}"
            logger.error(error_msg)
            return KeywordSyncResult(
                success=False,
                errors=[error_msg]
            )
        finally:
            self.sync_in_progress = False
    
    async def remove_from_serpbear(self, keyword: str, domain: str) -> bool:
        """
        Remove a keyword from SerpBear tracking.
        
        Args:
            keyword: Keyword to remove
            domain: Domain the keyword was tracked for
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with serpbear_client as client:
                # Find the keyword in SerpBear
                keywords = await client.get_keywords(domain)
                target_keyword = next(
                    (kw for kw in keywords if kw.keyword.lower() == keyword.lower()),
                    None
                )
                
                if target_keyword:
                    success = await client.remove_keyword(target_keyword.id)
                    if success:
                        logger.info(f"✅ Removed '{keyword}' from SerpBear")
                        return True
                    else:
                        logger.error(f"❌ Failed to remove '{keyword}' from SerpBear")
                        return False
                else:
                    logger.debug(f"Keyword '{keyword}' not found in SerpBear for {domain}")
                    return True  # Not found = effectively removed
                    
        except Exception as e:
            logger.error(f"Failed to remove keyword '{keyword}' from SerpBear: {e}")
            return False
    
    async def get_sync_status(self, db: Session) -> Dict[str, Any]:
        """
        Get current sync status between tracked keywords and SerpBear.
        
        Args:
            db: Database session
            
        Returns:
            Status information dictionary
        """
        try:
            # Get tracked keywords count
            total_tracked = db.query(TrackedKeyword).filter(
                TrackedKeyword.is_active == True
            ).count()
            
            # Get SerpBear keywords count
            total_serpbear = 0
            serpbear_domains = []
            
            async with serpbear_client as client:
                domains = await client.get_domains()
                for domain in domains:
                    keywords = await client.get_keywords(domain.domain)
                    total_serpbear += len(keywords)
                    serpbear_domains.append({
                        "domain": domain.domain,
                        "keyword_count": len(keywords)
                    })
            
            return {
                "sync_in_progress": self.sync_in_progress,
                "total_tracked_keywords": total_tracked,
                "total_serpbear_keywords": total_serpbear,
                "serpbear_domains": serpbear_domains,
                "last_checked": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get sync status: {e}")
            return {
                "sync_in_progress": self.sync_in_progress,
                "error": str(e),
                "last_checked": datetime.now().isoformat()
            }


# Global keyword sync service instance
keyword_sync_service = KeywordSyncService()