"""
SerpBear Settings API Routes.

Handles SerpBear configuration management with multi-tenant support.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field

from .content_auth import get_current_user_safe
from utils.tenant_mapper import TenantOrgMapper
from src.services.serpbear_config import serpbear_configurator, initialize_configurator
from src.services.serpbear_client import serpbear_client, test_serpbear_connection

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/serpbear-settings", tags=["serpbear-settings"])



class SerpBearConnectionSettings(BaseModel):
    """SerpBear connection configuration."""
    base_url: str = Field("http://localhost:3001", description="SerpBear instance URL")
    api_key: str = Field(..., description="SerpBear API key")
    primary_domain: str = Field(..., description="Primary domain to track")


class GoogleSearchConsoleSettings(BaseModel):
    """Google Search Console configuration."""
    enabled: bool = Field(False, description="Enable GSC integration")
    client_email: str = Field("", description="Service account client email")
    private_key: str = Field("", description="Service account private key")


class GoogleAdsSettings(BaseModel):
    """Google Ads configuration."""
    enabled: bool = Field(False, description="Enable Google Ads integration")
    client_id: str = Field("", description="Google Ads client ID")
    client_secret: str = Field("", description="Google Ads client secret")
    refresh_token: str = Field("", description="Google Ads refresh token")
    developer_token: str = Field("", description="Google Ads developer token")
    customer_id: str = Field("", description="Google Ads customer ID")


class SerpBearSettingsRequest(BaseModel):
    """Complete SerpBear settings request."""
    connection: SerpBearConnectionSettings
    search_console: GoogleSearchConsoleSettings
    google_ads: GoogleAdsSettings


class SerpBearSettingsResponse(BaseModel):
    """SerpBear settings response."""
    success: bool
    settings: Dict[str, Any]
    connection_status: str
    last_updated: Optional[str] = None


@router.get("/current", response_model=SerpBearSettingsResponse)
async def get_current_serpbear_settings(request: Request):
    """
    Get current SerpBear settings for the user's organization.
    
    Returns:
        Current SerpBear configuration
    """
    try:
        user = await get_current_user_safe(request)
        organization_id = user["organization_id"]
        
        logger.info(f"Getting SerpBear settings for organization {organization_id}")
        
        # Get settings from organization
        settings = await TenantOrgMapper.get_serpbear_settings(organization_id)
        
        # Test connection if settings exist
        connection_status = "disconnected"
        if settings.get("connection", {}).get("api_key"):
            try:
                connected = await test_serpbear_connection()
                connection_status = "connected" if connected else "error"
            except Exception as e:
                logger.warning(f"Connection test failed: {e}")
                connection_status = "error"
        
        return SerpBearSettingsResponse(
            success=True,
            settings=settings,
            connection_status=connection_status,
            last_updated=settings.get("last_updated")
        )
        
    except Exception as e:
        logger.error(f"Failed to get SerpBear settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get settings: {str(e)}")


@router.post("/save", response_model=SerpBearSettingsResponse)
async def save_serpbear_settings(
    settings_request: SerpBearSettingsRequest,
    request: Request
):
    """
    Save SerpBear settings for the user's organization.
    
    Args:
        settings_request: SerpBear configuration to save
        
    Returns:
        Save operation result
    """
    try:
        user = await get_current_user_safe(request)
        organization_id = user["organization_id"]
        
        logger.info(f"Saving SerpBear settings for organization {organization_id}")
        
        # Convert to dictionary for storage
        settings_dict = {
            "connection": settings_request.connection.model_dump(),
            "search_console": settings_request.search_console.model_dump(),
            "google_ads": settings_request.google_ads.model_dump(),
            "last_updated": str(datetime.now(dt.timezone.utc))
        }
        
        # Save to organization settings
        success = await TenantOrgMapper.update_serpbear_settings(
            organization_id, 
            settings_dict
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save settings")
        
        # Configure SerpBear with tenant_id (mapped from org_id)
        tenant_id = await TenantOrgMapper.get_tenant_from_org(organization_id)
        
        # Initialize configurator if needed
        if not serpbear_configurator:
            initialize_configurator()
        
        # Configure SerpBear for custom scraper
        scraper_configured = False
        if serpbear_configurator:
            scraper_configured = await configure_serpbear_for_tenant(tenant_id, settings_dict)
        
        # Auto-create domain in SerpBear database
        domain_created = False
        primary_domain = settings_dict['connection']['primary_domain']
        
        try:
            from web.api.serpbear_domain_routes import auto_create_domain_from_settings
            domain_result = await auto_create_domain_from_settings(
                domain=primary_domain,
                organization_id=organization_id,
                keywords=[]  # Start with no keywords, user can add them later
            )
            domain_created = domain_result.get("success", False)
            
            if domain_created:
                logger.info(f"✅ Auto-created domain {primary_domain} in SerpBear")
            else:
                logger.warning(f"⚠️ Failed to auto-create domain {primary_domain}: {domain_result.get('error')}")
                
        except Exception as e:
            logger.error(f"Auto-domain creation failed: {e}")
            domain_created = False
        
        # Update setup guidance based on auto-creation result
        settings_dict["domain_setup_required"] = not domain_created
        settings_dict["domain_created_automatically"] = domain_created
        
        if domain_created:
            settings_dict["domain_setup_instructions"] = {
                "step1": f"✅ Domain {primary_domain} created automatically",
                "step2": "✅ Custom scraper configured",
                "step3": "Add keywords through SerpBear UI or API",
                "step4": "System ready for ranking tracking",
                "note": "Domain was created automatically via API"
            }
        else:
            settings_dict["domain_setup_instructions"] = {
                "step1": f"Open SerpBear at {settings_dict['connection']['base_url']}",
                "step2": f"Manually add domain: {primary_domain}",
                "step3": "Add initial keywords for the domain",
                "step4": "The system will use your custom scraper automatically",
                "note": "Manual domain setup required - auto-creation failed"
            }
        
        settings_dict["scraper_configured"] = scraper_configured
        
        return SerpBearSettingsResponse(
            success=True,
            settings=settings_dict,
            connection_status="saved",
            last_updated=settings_dict["last_updated"]
        )
        
    except Exception as e:
        logger.error(f"Failed to save SerpBear settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save settings: {str(e)}")


@router.post("/test-connection")
async def test_serpbear_connection_endpoint(request: Request):
    """
    Test SerpBear connection with current settings.
    
    Returns:
        Connection test results
    """
    try:
        user = await get_current_user_safe(request)
        organization_id = user["organization_id"]
        
        logger.info(f"Testing SerpBear connection for organization {organization_id}")
        
        # Get current settings
        settings = await TenantOrgMapper.get_serpbear_settings(organization_id)
        connection_settings = settings.get("connection", {})
        
        if not connection_settings.get("api_key"):
            return {
                "success": False,
                "connected": False,
                "error": "API key not configured"
            }
        
        # Update serpbear client configuration
        serpbear_client.base_url = connection_settings.get("base_url", "http://localhost:3001")
        serpbear_client.api_key = connection_settings.get("api_key")
        
        # Test connection
        connected = await test_serpbear_connection()
        
        if connected:
            # Get basic stats
            async with serpbear_client as client:
                domains = await client.get_domains()
                total_keywords = 0
                for domain in domains:
                    keywords = await client.get_keywords(domain.domain)
                    total_keywords += len(keywords)
                
                return {
                    "success": True,
                    "connected": True,
                    "stats": {
                        "domains_tracked": len(domains),
                        "total_keywords": total_keywords,
                        "domains": [d.domain for d in domains]
                    },
                    "message": "Connection successful"
                }
        else:
            return {
                "success": False,
                "connected": False,
                "error": "Connection failed - check URL and API key"
            }
        
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return {
            "success": False,
            "connected": False,
            "error": str(e)
        }


@router.get("/configuration-status")
async def get_serpbear_configuration_status(request: Request):
    """
    Get SerpBear configuration status and domain setup guidance.
    
    Returns:
        Configuration status and setup instructions
    """
    try:
        user = await get_current_user_safe(request)
        organization_id = user["organization_id"]
        
        logger.info(f"Getting SerpBear configuration status for organization {organization_id}")
        
        # Get current settings
        settings = await TenantOrgMapper.get_serpbear_settings(organization_id)
        connection_settings = settings.get("connection", {})
        
        # Initialize configurator if needed
        if not serpbear_configurator:
            initialize_configurator()
        
        # Check SerpBear configuration status
        config_status = "not_configured"
        scraper_configured = False
        domain_in_serpbear = False
        
        if serpbear_configurator:
            config = serpbear_configurator.get_current_config()
            config_status = config.get("status", "unknown")
            scraper_configured = config_status == "configured_for_local_scraper"
        
        # Check if domain exists in SerpBear
        domain_count = 0
        if connection_settings.get("api_key") and connection_settings.get("primary_domain"):
            try:
                # Update client configuration
                serpbear_client.base_url = connection_settings.get("base_url", "http://localhost:3001")
                serpbear_client.api_key = connection_settings.get("api_key")
                
                # Check for domain in SerpBear
                async with serpbear_client as client:
                    domains = await client.get_domains()
                    domain_count = len(domains)
                    primary_domain = connection_settings.get("primary_domain", "").replace("www.", "")
                    
                    for domain in domains:
                        domain_clean = domain.domain.replace("www.", "")
                        if domain_clean == primary_domain:
                            domain_in_serpbear = True
                            break
                            
            except Exception as e:
                logger.warning(f"Could not check domains in SerpBear: {e}")
        
        # Build setup instructions
        setup_instructions = {
            "step1": {
                "title": "Open SerpBear Dashboard",
                "description": f"Navigate to {connection_settings.get('base_url', 'http://localhost:3001')}",
                "completed": bool(connection_settings.get("api_key"))
            },
            "step2": {
                "title": "Add Your Domain",
                "description": f"Manually add domain: {connection_settings.get('primary_domain', 'your-domain.com')}",
                "completed": domain_in_serpbear
            },
            "step3": {
                "title": "Configure Keywords",
                "description": "Add initial keywords for tracking in SerpBear",
                "completed": False  # We can't check this easily
            },
            "step4": {
                "title": "Custom Scraper Active",
                "description": "System will automatically use your custom scraper",
                "completed": scraper_configured
            }
        }
        
        return {
            "success": True,
            "configuration_status": config_status,
            "scraper_configured": scraper_configured,
            "domain_in_serpbear": domain_in_serpbear,
            "domains_count": domain_count,
            "primary_domain": connection_settings.get("primary_domain", ""),
            "serpbear_url": connection_settings.get("base_url", "http://localhost:3001"),
            "setup_instructions": setup_instructions,
            "next_steps": [
                "Manual domain setup required - SerpBear API doesn't support domain creation",
                "Add domain and keywords through SerpBear web interface",
                "Custom scraper will automatically handle ranking updates"
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to get configuration status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


async def configure_serpbear_for_tenant(tenant_id: str, settings: Dict[str, Any]) -> bool:
    """
    Configure SerpBear for a specific tenant to use custom scraper.
    
    Args:
        tenant_id: Tenant identifier
        settings: SerpBear settings dictionary
        
    Returns:
        True if configuration successful
    """
    try:
        if not serpbear_configurator:
            logger.error("SerpBear configurator not initialized")
            return False
        
        logger.info(f"Configuring SerpBear for tenant {tenant_id} with custom scraper bridge")
        
        # Configure SerpBear to use our custom scraper bridge
        bridge_url = "http://seo-app:8000/api/serp-bridge"
        success = serpbear_configurator.configure_local_scraper(bridge_url)
        
        if success:
            logger.info(f"✅ SerpBear configured for tenant {tenant_id} - custom scraper enabled")
            logger.info(f"🔗 Bridge URL: {bridge_url}")
            
            # Log configuration status
            config_status = serpbear_configurator.get_current_config()
            logger.info(f"📋 Config status: {config_status.get('status', 'unknown')}")
        else:
            logger.error(f"❌ Failed to configure SerpBear for tenant {tenant_id}")
        
        return success
        
    except Exception as e:
        logger.error(f"Error configuring SerpBear for tenant {tenant_id}: {e}")
        return False


# Add missing import
from datetime import datetime
import datetime as dt