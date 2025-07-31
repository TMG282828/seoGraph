"""
SerpBear Domain Management API Routes

Custom API endpoints for SerpBear domain and keyword creation.
Fills the gap in SerpBear's API by providing domain creation functionality.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from pydantic import BaseModel, Field

from .content_auth import get_current_user_safe
from utils.tenant_mapper import TenantOrgMapper
from src.services.serpbear_database import (
    serpbear_database, 
    SerpBearDomain, 
    SerpBearKeyword,
    create_domain_with_keywords
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/serpbear-domains", tags=["serpbear-domains"])


class DomainCreateRequest(BaseModel):
    """Request model for domain creation."""
    domain: str = Field(..., description="Domain name (e.g., example.com)")
    keywords: List[str] = Field(default=[], description="Initial keywords to add")
    device: str = Field(default="desktop", description="Device type (desktop/mobile)")
    country: str = Field(default="US", description="Country code")
    notification: bool = Field(default=True, description="Enable notifications")


class KeywordAddRequest(BaseModel):
    """Request model for adding keywords to domain."""
    domain: str = Field(..., description="Target domain")
    keywords: List[str] = Field(..., description="Keywords to add")
    device: str = Field(default="desktop", description="Device type")
    country: str = Field(default="US", description="Country code")


class DomainResponse(BaseModel):
    """Response model for domain operations."""
    success: bool
    domain: str
    message: str
    details: Dict[str, Any] = None


@router.post("/create", response_model=DomainResponse)
async def create_domain(
    request: DomainCreateRequest,
    http_request: Request
):
    """
    Create a new domain in SerpBear with optional keywords.
    
    This endpoint provides the missing domain creation functionality
    that SerpBear's API doesn't expose.
    """
    try:
        user = await get_current_user_safe(http_request)
        organization_id = user["organization_id"]
        
        logger.info(f"Creating SerpBear domain {request.domain} for organization {organization_id}")
        
        # Validate domain format
        domain_clean = request.domain.lower().strip()
        if not domain_clean:
            raise HTTPException(status_code=400, detail="Domain name is required")
        
        # Remove protocol if present
        domain_clean = domain_clean.replace("https://", "").replace("http://", "")
        domain_clean = domain_clean.replace("www.", "").rstrip("/")
        
        # Create domain with keywords
        result = await create_domain_with_keywords(
            domain=domain_clean,
            keywords=request.keywords,
            device=request.device,
            country=request.country,
            notification=request.notification
        )
        
        if not result["success"]:
            logger.error(f"Failed to create domain {domain_clean}: {result.get('error')}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to create domain: {result.get('error', 'Unknown error')}"
            )
        
        # Note: SerpBear restart will be handled by background task
        
        logger.info(f"✅ Created domain {domain_clean} with {len(request.keywords)} keywords")
        
        return DomainResponse(
            success=True,
            domain=domain_clean,
            message=f"Domain '{domain_clean}' created successfully",
            details={
                "domain": domain_clean,
                "slug": result.get("slug"),
                "keywords_added": result.get("keywords", {}).get("successful", 0),
                "total_keywords": len(request.keywords),
                "device": request.device,
                "country": request.country,
                "restart_required": "SerpBear will be restarted to show changes"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Domain creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Domain creation failed: {str(e)}")


@router.post("/add-keywords", response_model=DomainResponse)
async def add_keywords_to_domain(
    request: KeywordAddRequest,
    http_request: Request
):
    """
    Add keywords to an existing domain in SerpBear.
    
    Allows bulk addition of keywords to domains that already exist.
    """
    try:
        user = await get_current_user_safe(http_request)
        organization_id = user["organization_id"]
        
        logger.info(f"Adding {len(request.keywords)} keywords to {request.domain} for organization {organization_id}")
        
        # Validate inputs
        if not request.domain.strip():
            raise HTTPException(status_code=400, detail="Domain name is required")
        
        if not request.keywords:
            raise HTTPException(status_code=400, detail="At least one keyword is required")
        
        # Clean domain name
        domain_clean = request.domain.lower().strip()
        domain_clean = domain_clean.replace("https://", "").replace("http://", "")
        domain_clean = domain_clean.replace("www.", "").rstrip("/")
        
        # Check if domain exists
        if not await serpbear_database.domain_exists(domain_clean):
            raise HTTPException(
                status_code=404, 
                detail=f"Domain '{domain_clean}' does not exist. Create it first."
            )
        
        # Add keywords in bulk
        result = await serpbear_database.bulk_add_keywords(
            domain=domain_clean,
            keywords=request.keywords,
            device=request.device,
            country=request.country
        )
        
        if not result["success"]:
            logger.error(f"Failed to add keywords to {domain_clean}: {result.get('error')}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to add keywords: {result.get('error', 'Unknown error')}"
            )
        
        logger.info(f"✅ Added {result['successful']} keywords to {domain_clean}")
        
        return DomainResponse(
            success=True,
            domain=domain_clean,
            message=f"Added {result['successful']} keywords to '{domain_clean}'",
            details={
                "domain": domain_clean,
                "keywords_added": result["successful"],
                "keywords_failed": result["failed"],
                "keywords_skipped": result["skipped"],
                "total_requested": len(request.keywords),
                "device": request.device,
                "country": request.country
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Keyword addition failed: {e}")
        raise HTTPException(status_code=500, detail=f"Keyword addition failed: {str(e)}")


@router.get("/list")
async def list_domains(http_request: Request):
    """
    List all domains in SerpBear database.
    
    Provides visibility into all domains and their keyword counts.
    """
    try:
        user = await get_current_user_safe(http_request)
        organization_id = user["organization_id"]
        
        logger.info(f"Listing SerpBear domains for organization {organization_id}")
        
        # Get all domains
        domains = await serpbear_database.list_all_domains()
        
        return {
            "success": True,
            "domains": domains,
            "total_domains": len(domains),
            "organization_id": organization_id
        }
        
    except Exception as e:
        logger.error(f"Failed to list domains: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list domains: {str(e)}")


@router.get("/info/{domain}")
async def get_domain_info(domain: str, http_request: Request):
    """
    Get detailed information about a specific domain.
    
    Returns domain metadata and associated keywords.
    """
    try:
        user = await get_current_user_safe(http_request)
        organization_id = user["organization_id"]
        
        logger.info(f"Getting info for domain {domain} (organization {organization_id})")
        
        # Clean domain name
        domain_clean = domain.lower().strip()
        domain_clean = domain_clean.replace("https://", "").replace("http://", "")
        domain_clean = domain_clean.replace("www.", "").rstrip("/")
        
        # Get domain info
        domain_info = await serpbear_database.get_domain_info(domain_clean)
        if not domain_info:
            raise HTTPException(status_code=404, detail=f"Domain '{domain_clean}' not found")
        
        # Get domain keywords
        keywords = await serpbear_database.get_domain_keywords(domain_clean)
        
        return {
            "success": True,
            "domain": domain_info,
            "keywords": keywords,
            "keyword_count": len(keywords)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get domain info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get domain info: {str(e)}")


@router.delete("/delete/{domain}")
async def delete_domain(domain: str, http_request: Request):
    """
    Delete a domain and all its keywords from SerpBear.
    
    WARNING: This is destructive and cannot be undone.
    """
    try:
        user = await get_current_user_safe(http_request)
        organization_id = user["organization_id"]
        
        logger.warning(f"DELETING domain {domain} for organization {organization_id}")
        
        # Clean domain name
        domain_clean = domain.lower().strip()
        domain_clean = domain_clean.replace("https://", "").replace("http://", "")
        domain_clean = domain_clean.replace("www.", "").rstrip("/")
        
        # Delete domain
        success = await serpbear_database.delete_domain(domain_clean)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Domain '{domain_clean}' not found")
        
        logger.info(f"✅ Deleted domain {domain_clean}")
        
        return DomainResponse(
            success=True,
            domain=domain_clean,
            message=f"Domain '{domain_clean}' deleted successfully",
            details={
                "domain": domain_clean,
                "action": "deleted",
                "warning": "This action cannot be undone"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete domain: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete domain: {str(e)}")


@router.get("/health")
async def domain_service_health():
    """
    Health check for the domain management service.
    
    Returns database connectivity and service status.
    """
    try:
        # Check database health (use sync version for API compatibility)
        health = serpbear_database.health_check_sync()
        
        return {
            "service": "SerpBear Domain Management",
            "status": health["status"],
            "database": {
                "accessible": health["accessible"],
                "domain_count": health["domain_count"],
                "keyword_count": health["keyword_count"]
            },
            "endpoints": [
                "POST /api/serpbear-domains/create",
                "POST /api/serpbear-domains/add-keywords", 
                "GET /api/serpbear-domains/list",
                "GET /api/serpbear-domains/info/{domain}",
                "DELETE /api/serpbear-domains/delete/{domain}",
                "GET /api/serpbear-domains/health"
            ],
            "timestamp": health["timestamp"]
        }
        
    except Exception as e:
        logger.error(f"Domain service health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


# Background task functions
async def restart_serpbear_container():
    """
    Restart SerpBear container to pick up database changes.
    
    This is needed because SerpBear caches domain/keyword data and doesn't
    automatically reload when the database is modified externally.
    """
    try:
        import subprocess
        import asyncio
        
        logger.info("Restarting SerpBear container to pick up database changes...")
        
        # Run docker restart in background
        process = await asyncio.create_subprocess_exec(
            "docker", "restart", "seo-serpbear",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            logger.info("✅ SerpBear container restarted successfully")
        else:
            logger.error(f"❌ Failed to restart SerpBear: {stderr.decode()}")
            
    except Exception as e:
        logger.error(f"Failed to restart SerpBear container: {e}")


# Auto-domain creation helper
async def auto_create_domain_from_settings(
    domain: str, 
    organization_id: str,
    keywords: List[str] = None
) -> Dict[str, Any]:
    """
    Helper function to automatically create domain from settings.
    
    This can be called from the settings save logic to automatically
    create domains when users configure SerpBear settings.
    """
    try:
        logger.info(f"Auto-creating domain {domain} for organization {organization_id}")
        
        # Create domain with default settings
        result = await create_domain_with_keywords(
            domain=domain,
            keywords=keywords or [],
            device="desktop",
            country="US",
            notification=True
        )
        
        if result["success"]:
            # Schedule SerpBear restart (non-blocking)
            asyncio.create_task(restart_serpbear_container())
            
            logger.info(f"✅ Auto-created domain {domain}")
        
        return result
        
    except Exception as e:
        logger.error(f"Auto-domain creation failed: {e}")
        return {"success": False, "error": str(e)}