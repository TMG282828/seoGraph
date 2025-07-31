"""
SerpBear Database Service

Direct SQLite database access for SerpBear domain and keyword management.
Provides the missing API functionality that SerpBear doesn't expose.

Database Schema:
- domain: ID, domain, slug, keywordCount, lastUpdated, added, tags, notification settings
- keyword: ID, keyword, device, country, domain, position, history, volume, url, etc.
"""

import sqlite3
import logging
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import asyncio
import aiosqlite

logger = logging.getLogger(__name__)


@dataclass
class SerpBearDomain:
    """SerpBear domain model."""
    domain: str
    slug: str
    keyword_count: int = 0
    last_updated: Optional[str] = None
    added: Optional[str] = None
    tags: List[str] = None
    notification: bool = True
    notification_interval: str = "daily"
    notification_emails: str = ""
    search_console: Optional[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.added is None:
            self.added = datetime.utcnow().isoformat()
        if self.last_updated is None:
            self.last_updated = datetime.utcnow().isoformat()


@dataclass 
class SerpBearKeyword:
    """SerpBear keyword model."""
    keyword: str
    domain: str
    device: str = "desktop"
    country: str = "US"
    city: str = ""
    latlong: str = ""
    position: int = 0
    volume: int = 0
    history: List[Dict] = None
    url: List[str] = None
    tags: List[str] = None
    last_result: List[Dict] = None
    sticky: bool = True
    updating: bool = False
    last_update_error: str = "0"
    settings: Optional[str] = None
    added: Optional[str] = None
    last_updated: Optional[str] = None

    def __post_init__(self):
        if self.history is None:
            self.history = []
        if self.url is None:
            self.url = []
        if self.tags is None:
            self.tags = []
        if self.last_result is None:
            self.last_result = []
        if self.added is None:
            self.added = datetime.utcnow().isoformat()
        if self.last_updated is None:
            self.last_updated = datetime.utcnow().isoformat()


class SerpBearDatabase:
    """
    SerpBear database service for direct SQLite access.
    
    Provides domain and keyword management functionality that's missing
    from SerpBear's API.
    """
    
    def __init__(self, database_path: str = "/app/data/database.sqlite"):
        """Initialize database service."""
        self.database_path = database_path
        self.local_database_path = "./serpbear_database.sqlite"  # Local copy for development
        
        logger.info(f"SerpBear database service initialized - Path: {database_path}")
    
    def _create_domain_slug(self, domain: str) -> str:
        """Create URL-safe slug from domain name."""
        # Remove protocol if present
        clean_domain = domain.replace("https://", "").replace("http://", "")
        # Remove www. if present
        clean_domain = clean_domain.replace("www.", "")
        # Remove trailing slash
        clean_domain = clean_domain.rstrip("/")
        # Replace dots and special chars with dashes
        slug = re.sub(r'[^a-zA-Z0-9]', '-', clean_domain)
        # Remove multiple consecutive dashes
        slug = re.sub(r'-+', '-', slug)
        # Remove leading/trailing dashes
        slug = slug.strip('-').lower()
        
        return slug
    
    def _get_sync_connection(self) -> sqlite3.Connection:
        """Get synchronous database connection for compatibility."""
        try:
            # Try to connect to Docker volume first
            if Path(self.database_path).exists():
                return sqlite3.connect(self.database_path)
            # Fallback to local copy for development
            elif Path(self.local_database_path).exists():
                logger.warning(f"Using local database copy: {self.local_database_path}")
                return sqlite3.connect(self.local_database_path)
            else:
                raise FileNotFoundError(f"SerpBear database not found at {self.database_path} or {self.local_database_path}")
        except Exception as e:
            logger.error(f"Failed to connect to SerpBear database: {e}")
            raise
    
    async def _get_connection(self) -> aiosqlite.Connection:
        """Get async database connection."""
        try:
            # Try to connect to Docker volume first
            if Path(self.database_path).exists():
                return await aiosqlite.connect(self.database_path)
            # Fallback to local copy for development
            elif Path(self.local_database_path).exists():
                logger.warning(f"Using local database copy: {self.local_database_path}")
                return await aiosqlite.connect(self.local_database_path)
            else:
                raise FileNotFoundError(f"SerpBear database not found at {self.database_path} or {self.local_database_path}")
        except Exception as e:
            logger.error(f"Failed to connect to SerpBear database: {e}")
            raise
    
    def domain_exists_sync(self, domain: str) -> bool:
        """Check if domain already exists in database (synchronous)."""
        try:
            conn = self._get_sync_connection()
            cursor = conn.execute(
                "SELECT COUNT(*) FROM domain WHERE domain = ?",
                (domain,)
            )
            count = cursor.fetchone()
            conn.close()
            return count[0] > 0
        except Exception as e:
            logger.error(f"Error checking domain existence: {e}")
            return False
    
    async def domain_exists(self, domain: str) -> bool:
        """Check if domain already exists in database."""
        try:
            async with await self._get_connection() as conn:
                cursor = await conn.execute(
                    "SELECT COUNT(*) FROM domain WHERE domain = ?",
                    (domain,)
                )
                count = await cursor.fetchone()
                return count[0] > 0
        except Exception as e:
            logger.error(f"Error checking domain existence: {e}")
            return False
    
    async def create_domain(self, domain_obj: SerpBearDomain) -> bool:
        """
        Create a new domain in SerpBear database.
        
        Args:
            domain_obj: Domain object with all required fields
            
        Returns:
            True if domain created successfully
        """
        try:
            # Check if domain already exists
            if await self.domain_exists(domain_obj.domain):
                logger.warning(f"Domain {domain_obj.domain} already exists")
                return True
            
            async with await self._get_connection() as conn:
                await conn.execute("""
                    INSERT INTO domain (
                        domain, slug, keywordCount, lastUpdated, added, tags,
                        notification, notification_interval, notification_emails, search_console
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    domain_obj.domain,
                    domain_obj.slug,
                    domain_obj.keyword_count,
                    domain_obj.last_updated,
                    domain_obj.added,
                    json.dumps(domain_obj.tags),
                    1 if domain_obj.notification else 0,
                    domain_obj.notification_interval,
                    domain_obj.notification_emails,
                    domain_obj.search_console
                ))
                
                await conn.commit()
                logger.info(f"✅ Created domain: {domain_obj.domain} (slug: {domain_obj.slug})")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create domain {domain_obj.domain}: {e}")
            return False
    
    async def add_keyword(self, keyword_obj: SerpBearKeyword) -> bool:
        """
        Add a keyword to a domain in SerpBear database.
        
        Args:
            keyword_obj: Keyword object with all required fields
            
        Returns:
            True if keyword added successfully
        """
        try:
            # Check if domain exists
            if not await self.domain_exists(keyword_obj.domain):
                logger.error(f"Cannot add keyword - domain {keyword_obj.domain} does not exist")
                return False
            
            async with await self._get_connection() as conn:
                # Check if keyword already exists for this domain
                cursor = await conn.execute(
                    "SELECT COUNT(*) FROM keyword WHERE keyword = ? AND domain = ? AND device = ?",
                    (keyword_obj.keyword, keyword_obj.domain, keyword_obj.device)
                )
                count = await cursor.fetchone()
                
                if count[0] > 0:
                    logger.warning(f"Keyword '{keyword_obj.keyword}' already exists for {keyword_obj.domain} ({keyword_obj.device})")
                    return True
                
                # Insert keyword
                await conn.execute("""
                    INSERT INTO keyword (
                        keyword, device, country, city, latlong, domain, lastUpdated, added,
                        position, history, volume, url, tags, lastResult, sticky, updating,
                        lastUpdateError, settings
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    keyword_obj.keyword,
                    keyword_obj.device,
                    keyword_obj.country,
                    keyword_obj.city,
                    keyword_obj.latlong,
                    keyword_obj.domain,
                    keyword_obj.last_updated,
                    keyword_obj.added,
                    keyword_obj.position,
                    json.dumps(keyword_obj.history),
                    keyword_obj.volume,
                    json.dumps(keyword_obj.url),
                    json.dumps(keyword_obj.tags),
                    json.dumps(keyword_obj.last_result),
                    1 if keyword_obj.sticky else 0,
                    1 if keyword_obj.updating else 0,
                    keyword_obj.last_update_error,
                    keyword_obj.settings
                ))
                
                # Update domain keyword count
                await conn.execute("""
                    UPDATE domain 
                    SET keywordCount = (
                        SELECT COUNT(*) FROM keyword WHERE domain = ?
                    ), lastUpdated = ?
                    WHERE domain = ?
                """, (keyword_obj.domain, datetime.utcnow().isoformat(), keyword_obj.domain))
                
                await conn.commit()
                logger.info(f"✅ Added keyword: '{keyword_obj.keyword}' to {keyword_obj.domain} ({keyword_obj.device})")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add keyword '{keyword_obj.keyword}': {e}")
            return False
    
    async def bulk_add_keywords(self, domain: str, keywords: List[str], device: str = "desktop", country: str = "US") -> Dict[str, Any]:
        """
        Add multiple keywords to a domain efficiently.
        
        Args:
            domain: Target domain
            keywords: List of keyword strings
            device: Device type (desktop/mobile)
            country: Country code
            
        Returns:
            Results summary
        """
        try:
            if not await self.domain_exists(domain):
                logger.error(f"Cannot add keywords - domain {domain} does not exist")
                return {"success": False, "error": "Domain does not exist"}
            
            successful = 0
            failed = 0
            skipped = 0
            
            for keyword_str in keywords:
                keyword_obj = SerpBearKeyword(
                    keyword=keyword_str.strip(),
                    domain=domain,
                    device=device,
                    country=country
                )
                
                # Check if keyword already exists
                async with await self._get_connection() as conn:
                    cursor = await conn.execute(
                        "SELECT COUNT(*) FROM keyword WHERE keyword = ? AND domain = ? AND device = ?",
                        (keyword_obj.keyword, domain, device)
                    )
                    count = await cursor.fetchone()
                    
                    if count[0] > 0:
                        skipped += 1
                        continue
                
                # Add keyword
                if await self.add_keyword(keyword_obj):
                    successful += 1
                else:
                    failed += 1
            
            logger.info(f"Bulk keyword addition completed - Success: {successful}, Failed: {failed}, Skipped: {skipped}")
            
            return {
                "success": True,
                "domain": domain,
                "total_keywords": len(keywords),
                "successful": successful,
                "failed": failed,
                "skipped": skipped,
                "results": f"{successful}/{len(keywords)} keywords added successfully"
            }
            
        except Exception as e:
            logger.error(f"Bulk keyword addition failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_domain_info(self, domain: str) -> Optional[Dict[str, Any]]:
        """Get domain information including keyword count."""
        try:
            async with await self._get_connection() as conn:
                cursor = await conn.execute("""
                    SELECT domain, slug, keywordCount, lastUpdated, added, tags,
                           notification, notification_interval, notification_emails
                    FROM domain WHERE domain = ?
                """, (domain,))
                
                row = await cursor.fetchone()
                if not row:
                    return None
                
                return {
                    "domain": row[0],
                    "slug": row[1],
                    "keywordCount": row[2],
                    "lastUpdated": row[3],
                    "added": row[4],
                    "tags": json.loads(row[5]) if row[5] else [],
                    "notification": bool(row[6]),
                    "notification_interval": row[7],
                    "notification_emails": row[8]
                }
                
        except Exception as e:
            logger.error(f"Failed to get domain info for {domain}: {e}")
            return None
    
    async def get_domain_keywords(self, domain: str) -> List[Dict[str, Any]]:
        """Get all keywords for a domain."""
        try:
            async with await self._get_connection() as conn:
                cursor = await conn.execute("""
                    SELECT keyword, device, country, position, volume, lastUpdated, added
                    FROM keyword WHERE domain = ?
                    ORDER BY added DESC
                """, (domain,))
                
                rows = await cursor.fetchall()
                
                return [
                    {
                        "keyword": row[0],
                        "device": row[1],
                        "country": row[2],
                        "position": row[3],
                        "volume": row[4],
                        "lastUpdated": row[5],
                        "added": row[6]
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Failed to get keywords for domain {domain}: {e}")
            return []
    
    async def list_all_domains(self) -> List[Dict[str, Any]]:
        """List all domains in the database."""
        try:
            async with await self._get_connection() as conn:
                cursor = await conn.execute("""
                    SELECT domain, slug, keywordCount, lastUpdated, added
                    FROM domain ORDER BY added DESC
                """)
                
                rows = await cursor.fetchall()
                
                return [
                    {
                        "domain": row[0],
                        "slug": row[1],
                        "keywordCount": row[2],
                        "lastUpdated": row[3],
                        "added": row[4]
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Failed to list domains: {e}")
            return []
    
    async def delete_domain(self, domain: str) -> bool:
        """
        Delete a domain and all its keywords.
        
        Args:
            domain: Domain to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            async with await self._get_connection() as conn:
                # Delete keywords first
                await conn.execute("DELETE FROM keyword WHERE domain = ?", (domain,))
                
                # Delete domain
                cursor = await conn.execute("DELETE FROM domain WHERE domain = ?", (domain,))
                
                await conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"✅ Deleted domain: {domain}")
                    return True
                else:
                    logger.warning(f"Domain {domain} not found for deletion")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to delete domain {domain}: {e}")
            return False
    
    def health_check_sync(self) -> Dict[str, Any]:
        """Check database health and connectivity (synchronous)."""
        try:
            conn = self._get_sync_connection()
            
            # Test basic query
            cursor = conn.execute("SELECT COUNT(*) FROM domain")
            domain_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM keyword")
            keyword_count = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "status": "healthy",
                "database_path": self.database_path,
                "accessible": True,
                "domain_count": domain_count,
                "keyword_count": keyword_count,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "database_path": self.database_path,
                "accessible": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def health_check(self) -> Dict[str, Any]:
        """Check database health and connectivity."""
        try:
            async with await self._get_connection() as conn:
                # Test basic query
                cursor = await conn.execute("SELECT COUNT(*) FROM domain")
                domain_count = await cursor.fetchone()
                
                cursor = await conn.execute("SELECT COUNT(*) FROM keyword")
                keyword_count = await cursor.fetchone()
                
                return {
                    "status": "healthy",
                    "database_path": self.database_path,
                    "accessible": True,
                    "domain_count": domain_count[0],
                    "keyword_count": keyword_count[0],
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "database_path": self.database_path,
                "accessible": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


# Global database service instance
serpbear_database = SerpBearDatabase()


# Helper functions for easy domain/keyword creation
async def create_domain_with_keywords(
    domain: str, 
    keywords: List[str] = None,
    device: str = "desktop",
    country: str = "US",
    notification: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to create domain and add keywords in one call.
    
    Args:
        domain: Domain name
        keywords: List of keywords to add
        device: Device type
        country: Country code
        notification: Enable notifications
        
    Returns:
        Creation results
    """
    try:
        # Create domain object
        domain_obj = SerpBearDomain(
            domain=domain,
            slug=serpbear_database._create_domain_slug(domain),
            notification=notification
        )
        
        # Create domain
        domain_created = await serpbear_database.create_domain(domain_obj)
        
        if not domain_created:
            return {"success": False, "error": "Failed to create domain"}
        
        # Add keywords if provided
        keyword_results = {"success": True, "total_keywords": 0}
        if keywords:
            keyword_results = await serpbear_database.bulk_add_keywords(
                domain, keywords, device, country
            )
        
        return {
            "success": True,
            "domain_created": domain_created,
            "domain": domain,
            "slug": domain_obj.slug,
            "keywords": keyword_results,
            "message": f"Domain '{domain}' created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create domain with keywords: {e}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    # Test the database service
    async def test_database_service():
        print("🗄️ Testing SerpBear Database Service")
        print("=" * 50)
        
        # Health check
        print("\n1. Database Health Check:")
        health = await serpbear_database.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Domains: {health.get('domain_count', 0)}")
        print(f"   Keywords: {health.get('keyword_count', 0)}")
        
        # Test domain creation
        print("\n2. Testing Domain Creation:")
        test_domain = "example.com"
        result = await create_domain_with_keywords(
            domain=test_domain,
            keywords=["test keyword", "seo tools", "web development"],
            device="desktop",
            country="US"
        )
        print(f"   Result: {'✅ Success' if result['success'] else '❌ Failed'}")
        if result['success']:
            print(f"   Domain: {result['domain']}")
            print(f"   Keywords: {result['keywords']['successful']}/{result['keywords']['total_keywords']}")
        
        # List domains
        print("\n3. Listing All Domains:")
        domains = await serpbear_database.list_all_domains()
        for domain in domains:
            print(f"   - {domain['domain']} ({domain['keywordCount']} keywords)")
        
        print("\n🎉 Database service test completed!")
    
    asyncio.run(test_database_service())