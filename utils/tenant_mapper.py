"""
Tenant-Organization Mapping Utility.

Provides mapping between tenant_id (used by modules) and organization_id (used by Supabase)
without breaking existing code. Supports future multi-tenant expansion.
"""

import logging
from typing import Optional, Dict, Any
from database.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


class TenantOrgMapper:
    """
    Maps tenant_id to organization_id without breaking existing code.
    
    Current implementation: Simple 1:1 mapping (tenant_id = organization_id)
    Future: Support multiple tenants per organization
    """
    
    @staticmethod
    async def get_org_from_tenant(tenant_id: str) -> str:
        """
        Get organization_id from tenant_id.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Organization identifier
        """
        # For now: simple 1:1 mapping
        # Future: lookup from tenants table
        return tenant_id
    
    @staticmethod 
    async def get_tenant_from_org(organization_id: str) -> str:
        """
        Get default tenant_id from organization_id.
        
        Args:
            organization_id: Organization identifier
            
        Returns:
            Default tenant identifier for the organization
        """
        # For now: simple 1:1 mapping
        # Future: get primary tenant for org
        return organization_id
    
    @staticmethod
    async def get_org_settings(organization_id: str) -> Dict[str, Any]:
        """
        Get organization settings from Supabase.
        
        Args:
            organization_id: Organization identifier
            
        Returns:
            Organization settings dictionary
        """
        try:
            # For now, use a temporary file-based storage until Supabase is fixed
            import json
            import os
            
            settings_file = f"/tmp/org_settings_{organization_id}.json"
            if os.path.exists(settings_file):
                with open(settings_file, 'r') as f:
                    return json.load(f)
            
            logger.warning(f"Organization {organization_id} settings file not found, returning empty")
            return {}
            
            # Original Supabase code (commented out until auth is fixed):
            # client = get_supabase_client()
            # service_client = client._get_service_client()
            # response = service_client.table("organizations").select("settings").eq("id", organization_id).execute()
            # if response.data:
            #     return response.data[0].get("settings", {})
            # logger.warning(f"Organization {organization_id} not found")
            # return {}
            
        except Exception as e:
            logger.error(f"Failed to get organization settings: {e}")
            return {}
    
    @staticmethod
    async def update_org_settings(organization_id: str, settings_update: Dict[str, Any]) -> bool:
        """
        Update organization settings in Supabase.
        
        Args:
            organization_id: Organization identifier
            settings_update: Settings to update (partial update)
            
        Returns:
            True if update successful
        """
        try:
            # For now, use temporary file-based storage until Supabase is fixed
            import json
            import os
            
            # Get current settings
            current_settings = await TenantOrgMapper.get_org_settings(organization_id)
            
            # Merge with new settings
            updated_settings = {**current_settings, **settings_update}
            
            # Save to temporary file
            settings_file = f"/tmp/org_settings_{organization_id}.json"
            os.makedirs("/tmp", exist_ok=True)
            with open(settings_file, 'w') as f:
                json.dump(updated_settings, f, indent=2)
            
            logger.info(f"Updated settings for organization {organization_id} (temporary file storage)")
            return True
            
            # Original Supabase code (commented out until auth is fixed):
            # client = get_supabase_client()
            # service_client = client._get_service_client()
            # response = service_client.table("organizations").update(
            #     {"settings": updated_settings}
            # ).eq("id", organization_id).execute()
            # success = bool(response.data)
            # if success:
            #     logger.info(f"Updated settings for organization {organization_id}")
            # else:
            #     logger.error(f"Failed to update settings for organization {organization_id}")
            # return success
        
        except Exception as e:
            logger.error(f"Failed to update organization settings: {e}")
            return False
    
    @staticmethod
    async def get_serpbear_settings(organization_id: str) -> Dict[str, Any]:
        """
        Get SerpBear settings for an organization.
        
        Args:
            organization_id: Organization identifier
            
        Returns:
            SerpBear settings dictionary
        """
        settings = await TenantOrgMapper.get_org_settings(organization_id)
        return settings.get("serpbear", {})
    
    @staticmethod
    async def update_serpbear_settings(organization_id: str, serpbear_settings: Dict[str, Any]) -> bool:
        """
        Update SerpBear settings for an organization.
        
        Args:
            organization_id: Organization identifier
            serpbear_settings: SerpBear configuration
            
        Returns:
            True if update successful
        """
        return await TenantOrgMapper.update_org_settings(
            organization_id, 
            {"serpbear": serpbear_settings}
        )
    
    @staticmethod
    async def get_app_settings(organization_id: str) -> Dict[str, Any]:
        """
        Get general application settings for an organization.
        
        Args:
            organization_id: Organization identifier
            
        Returns:
            Application settings dictionary
        """
        settings = await TenantOrgMapper.get_org_settings(organization_id)
        return settings.get("app_settings", {})
    
    @staticmethod
    async def update_app_settings(organization_id: str, app_settings: Dict[str, Any]) -> bool:
        """
        Update general application settings for an organization.
        
        Args:
            organization_id: Organization identifier
            app_settings: Application configuration
            
        Returns:
            True if update successful
        """
        return await TenantOrgMapper.update_org_settings(
            organization_id, 
            {"app_settings": app_settings}
        )


# Convenience functions for backward compatibility
async def get_tenant_id_from_org(organization_id: str) -> str:
    """Get tenant_id from organization_id (backward compatibility)."""
    return await TenantOrgMapper.get_tenant_from_org(organization_id)


async def get_org_id_from_tenant(tenant_id: str) -> str:
    """Get organization_id from tenant_id (backward compatibility)."""
    return await TenantOrgMapper.get_org_from_tenant(tenant_id)