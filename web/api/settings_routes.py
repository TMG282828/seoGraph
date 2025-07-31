"""
General Settings API Routes.

Handles application settings, API keys, system configuration, and user preferences
with multi-tenant support.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field

from .content_auth import get_current_user_safe
from utils.tenant_mapper import TenantOrgMapper

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/settings", tags=["settings"])


class ApplicationSettings(BaseModel):
    """General application settings."""
    checkin_frequency: str = Field("medium", description="Content check-in frequency: low, medium, high")
    require_approval: bool = Field(True, description="Require approval for content generation")
    notify_low_confidence: bool = Field(True, description="Notify when content confidence is low")


class ApiKeysSettings(BaseModel):
    """API keys configuration."""
    openai: str = Field("", description="OpenAI API key")
    langfuse_public: str = Field("", description="Langfuse public key")
    langfuse_secret: str = Field("", description="Langfuse secret key")


class SystemConfigSettings(BaseModel):
    """System configuration settings."""
    max_content_length: int = Field(50000, description="Maximum content length in characters")
    batch_size: int = Field(10, description="Batch processing size")
    enable_caching: bool = Field(True, description="Enable system caching")


class NotificationSettings(BaseModel):
    """Notification preferences."""
    email: bool = Field(True, description="Enable email notifications")
    browser: bool = Field(False, description="Enable browser notifications")
    email_address: str = Field("", description="Email address for notifications")
    report_frequency: str = Field("weekly", description="Report frequency: daily, weekly, monthly")


class AllSettingsRequest(BaseModel):
    """Complete settings request model."""
    settings: ApplicationSettings
    api_keys: ApiKeysSettings
    system_config: SystemConfigSettings
    notifications: NotificationSettings


class AllSettingsResponse(BaseModel):
    """Complete settings response model."""
    success: bool
    settings: ApplicationSettings
    api_keys: ApiKeysSettings
    system_config: SystemConfigSettings
    notifications: NotificationSettings
    last_updated: Optional[str] = None


@router.get("/", response_model=AllSettingsResponse)
async def get_all_settings(request: Request):
    """
    Get all application settings for the user's organization.
    
    Returns:
        Complete settings configuration
    """
    try:
        user = await get_current_user_safe(request)
        organization_id = user["organization_id"]
        
        logger.info(f"Getting application settings for organization {organization_id}")
        
        # Get settings from organization storage
        all_settings = await TenantOrgMapper.get_app_settings(organization_id)
        
        # Set defaults if no settings exist
        if not all_settings:
            all_settings = {
                "settings": {
                    "checkin_frequency": "medium",
                    "require_approval": True,
                    "notify_low_confidence": True
                },
                "api_keys": {
                    "openai": "",
                    "langfuse_public": "",
                    "langfuse_secret": ""
                },
                "system_config": {
                    "max_content_length": 50000,
                    "batch_size": 10,
                    "enable_caching": True
                },
                "notifications": {
                    "email": True,
                    "browser": False,
                    "email_address": "",
                    "report_frequency": "weekly"
                }
            }
        
        return AllSettingsResponse(
            success=True,
            settings=ApplicationSettings(**all_settings.get("settings", {})),
            api_keys=ApiKeysSettings(**all_settings.get("api_keys", {})),
            system_config=SystemConfigSettings(**all_settings.get("system_config", {})),
            notifications=NotificationSettings(**all_settings.get("notifications", {})),
            last_updated=all_settings.get("last_updated")
        )
        
    except Exception as e:
        logger.error(f"Failed to get application settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get settings: {str(e)}")


@router.post("/", response_model=AllSettingsResponse)
async def save_all_settings(
    settings_request: AllSettingsRequest,
    request: Request
):
    """
    Save all application settings for the user's organization.
    
    Args:
        settings_request: Complete settings configuration
        
    Returns:
        Save operation result
    """
    try:
        user = await get_current_user_safe(request)
        organization_id = user["organization_id"]
        
        logger.info(f"Saving application settings for organization {organization_id}")
        
        # Convert to dictionary for storage
        from datetime import datetime, timezone
        settings_dict = {
            "settings": settings_request.settings.model_dump(),
            "api_keys": settings_request.api_keys.model_dump(),
            "system_config": settings_request.system_config.model_dump(),
            "notifications": settings_request.notifications.model_dump(),
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        
        # Save to organization settings
        success = await TenantOrgMapper.update_app_settings(
            organization_id, 
            settings_dict
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save settings")
        
        return AllSettingsResponse(
            success=True,
            settings=settings_request.settings,
            api_keys=settings_request.api_keys,
            system_config=settings_request.system_config,
            notifications=settings_request.notifications,
            last_updated=settings_dict["last_updated"]
        )
        
    except Exception as e:
        logger.error(f"Failed to save application settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save settings: {str(e)}")


@router.get("/api-keys")
async def get_api_keys(request: Request):
    """
    Get API keys configuration (with sensitive data masked).
    
    Returns:
        API keys configuration with masked sensitive values
    """
    try:
        user = await get_current_user_safe(request)
        organization_id = user["organization_id"]
        
        logger.info(f"Getting API keys for organization {organization_id}")
        
        # Get settings
        all_settings = await TenantOrgMapper.get_app_settings(organization_id)
        api_keys = all_settings.get("api_keys", {}) if all_settings else {}
        
        # Mask sensitive values for display
        masked_keys = {}
        for key, value in api_keys.items():
            if value and len(value) > 8:
                # Show first 4 and last 4 characters, mask the middle
                masked_keys[key] = f"{value[:4]}...{value[-4:]}"
            elif value:
                masked_keys[key] = "***"
            else:
                masked_keys[key] = ""
        
        return {
            "success": True,
            "api_keys": masked_keys,
            "has_openai": bool(api_keys.get("openai")),
            "has_langfuse": bool(api_keys.get("langfuse_public") and api_keys.get("langfuse_secret"))
        }
        
    except Exception as e:
        logger.error(f"Failed to get API keys: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get API keys: {str(e)}")


@router.post("/api-keys")
async def update_api_keys(
    api_keys: ApiKeysSettings,
    request: Request
):
    """
    Update API keys configuration.
    
    Args:
        api_keys: API keys to update
        
    Returns:
        Update operation result
    """
    try:
        user = await get_current_user_safe(request)
        organization_id = user["organization_id"]
        
        logger.info(f"Updating API keys for organization {organization_id}")
        
        # Get current settings
        all_settings = await TenantOrgMapper.get_app_settings(organization_id)
        if not all_settings:
            all_settings = {}
        
        # Update API keys
        all_settings["api_keys"] = api_keys.model_dump()
        
        from datetime import datetime, timezone
        all_settings["last_updated"] = datetime.now(timezone.utc).isoformat()
        
        # Save updated settings
        success = await TenantOrgMapper.update_app_settings(
            organization_id, 
            all_settings
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update API keys")
        
        return {
            "success": True,
            "message": "API keys updated successfully",
            "last_updated": all_settings["last_updated"]
        }
        
    except Exception as e:
        logger.error(f"Failed to update API keys: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update API keys: {str(e)}")


@router.post("/test-configuration")
async def test_configuration(request: Request):
    """
    Test API keys and system configuration.
    
    Returns:
        Configuration test results
    """
    try:
        user = await get_current_user_safe(request)
        organization_id = user["organization_id"]
        
        logger.info(f"Testing configuration for organization {organization_id}")
        
        # Get settings
        all_settings = await TenantOrgMapper.get_app_settings(organization_id)
        api_keys = all_settings.get("api_keys", {}) if all_settings else {}
        
        test_results = {
            "success": True,
            "tests": []
        }
        
        # Test OpenAI API key
        openai_key = api_keys.get("openai", "")
        if openai_key:
            try:
                # Simple validation - just check if it looks like a valid key
                if openai_key.startswith(("sk-", "pk-")) and len(openai_key) > 20:
                    test_results["tests"].append({
                        "service": "OpenAI",
                        "status": "valid_format",
                        "message": "API key format is valid"
                    })
                else:
                    test_results["tests"].append({
                        "service": "OpenAI", 
                        "status": "invalid_format",
                        "message": "API key format appears invalid"
                    })
            except Exception as e:
                test_results["tests"].append({
                    "service": "OpenAI",
                    "status": "error",
                    "message": f"Error testing OpenAI key: {str(e)}"
                })
        else:
            test_results["tests"].append({
                "service": "OpenAI",
                "status": "not_configured",
                "message": "OpenAI API key not configured"
            })
        
        # Test Langfuse keys
        langfuse_pub = api_keys.get("langfuse_public", "")
        langfuse_sec = api_keys.get("langfuse_secret", "")
        
        if langfuse_pub and langfuse_sec:
            try:
                # Basic format validation
                if langfuse_pub.startswith("pk-") and langfuse_sec.startswith("sk-"):
                    test_results["tests"].append({
                        "service": "Langfuse",
                        "status": "valid_format", 
                        "message": "API keys format is valid"
                    })
                else:
                    test_results["tests"].append({
                        "service": "Langfuse",
                        "status": "invalid_format",
                        "message": "API keys format appears invalid"
                    })
            except Exception as e:
                test_results["tests"].append({
                    "service": "Langfuse",
                    "status": "error",
                    "message": f"Error testing Langfuse keys: {str(e)}"
                })
        else:
            test_results["tests"].append({
                "service": "Langfuse",
                "status": "not_configured",
                "message": "Langfuse API keys not configured"
            })
        
        # Check if any tests failed
        failed_tests = [t for t in test_results["tests"] if t["status"] in ["error", "invalid_format"]]
        test_results["success"] = len(failed_tests) == 0
        
        return test_results
        
    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "tests": []
        }


@router.post("/reset-defaults")
async def reset_to_defaults(request: Request):
    """
    Reset all settings to default values.
    
    Returns:
        Reset operation result
    """
    try:
        user = await get_current_user_safe(request)
        organization_id = user["organization_id"]
        
        logger.info(f"Resetting settings to defaults for organization {organization_id}")
        
        # Default settings
        from datetime import datetime, timezone
        default_settings = {
            "settings": {
                "checkin_frequency": "medium",
                "require_approval": True,
                "notify_low_confidence": True
            },
            "api_keys": {
                "openai": "",
                "langfuse_public": "",
                "langfuse_secret": ""
            },
            "system_config": {
                "max_content_length": 50000,
                "batch_size": 10,
                "enable_caching": True
            },
            "notifications": {
                "email": True,
                "browser": False,
                "email_address": "",
                "report_frequency": "weekly"
            },
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        
        # Save default settings
        success = await TenantOrgMapper.update_app_settings(
            organization_id, 
            default_settings
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to reset settings")
        
        return {
            "success": True,
            "message": "Settings reset to defaults successfully",
            "last_updated": default_settings["last_updated"]
        }
        
    except Exception as e:
        logger.error(f"Failed to reset settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset settings: {str(e)}")