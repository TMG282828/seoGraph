"""
Workspace Settings API Routes.

Handles workspace configuration, member management, and usage tracking
for multi-user workspaces.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field

from src.auth.auth_middleware import get_current_user
from models.user_models import (
    WorkspaceSettings, 
    WorkspaceSettingsUpdateRequest, 
    WorkspaceSettingsResponse,
    TenantRole,
    create_tenant,
    create_user_tenant_association
)
from utils.tenant_mapper import TenantOrgMapper

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/workspaces", tags=["workspace-settings"])


# Import Supabase client for database persistence
from src.database.supabase_client import supabase_client

# In-memory storage for workspace settings (replace with database in production)
workspace_settings_store: Dict[str, WorkspaceSettings] = {}


def get_default_workspace_settings(workspace_id: str, workspace_name: str) -> WorkspaceSettings:
    """Get default workspace settings for a workspace."""
    return WorkspaceSettings(
        workspace_name=workspace_name,
        description=f"Workspace for {workspace_name}",
        avatar_url=None,
        seat_limit=5,
        default_member_role=TenantRole.MEMBER,
        auto_approve_invites=False,
        current_seats_used=1,  # Owner counts as 1
        storage_used_gb=0.0
    )


def calculate_usage_stats(workspace_id: str, settings: WorkspaceSettings) -> Dict[str, Any]:
    """Calculate current usage statistics for workspace."""
    # This would connect to actual usage tracking in production
    return {
        "seats": {
            "used": settings.current_seats_used,
            "limit": settings.seat_limit,
            "percentage": settings.get_seat_usage_percentage(),
            "available": settings.get_available_seats()
        },
        "storage": {
            "used_gb": settings.storage_used_gb,
            "limit_gb": 10.0,  # Default limit
            "percentage": min((settings.storage_used_gb / 10.0) * 100, 100.0)
        },
        "api_calls": {
            "used_this_month": 150,  # Mock data
            "limit_monthly": 10000,
            "percentage": 1.5
        }
    }


@router.get("/{workspace_id}/settings", response_model=WorkspaceSettingsResponse)
async def get_workspace_settings(
    workspace_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get workspace settings for the specified workspace.
    
    Args:
        workspace_id: Workspace identifier
        
    Returns:
        Complete workspace settings and usage statistics
    """
    try:
        logger.info(f"Getting workspace settings for workspace {workspace_id}")
        
        # Check if user has access to this workspace
        # In production, verify user's role in workspace
        
        # Get settings from database
        settings_data = await supabase_client.get_workspace_settings(workspace_id)
        
        if settings_data:
            # Convert database data to WorkspaceSettings model
            settings = WorkspaceSettings(
                workspace_name=settings_data.get('workspace_name', f"Workspace {workspace_id[-8:]}"),
                description=settings_data.get('description', ''),
                avatar_url=settings_data.get('avatar_url'),
                seat_limit=settings_data.get('seat_limit', 5),
                default_member_role=TenantRole(settings_data.get('default_member_role', 'member')),
                auto_approve_invites=settings_data.get('auto_approve_invites', False),
                current_seats_used=settings_data.get('current_seats_used', 1),
                storage_used_gb=settings_data.get('storage_used_gb', 0.0),
                created_at=datetime.fromisoformat(settings_data.get('created_at', datetime.now(timezone.utc).isoformat())),
                updated_at=datetime.fromisoformat(settings_data.get('updated_at')) if settings_data.get('updated_at') else None
            )
        else:
            # Get workspace info and create default settings
            workspace_info = await supabase_client.get_workspace_by_id(workspace_id)
            workspace_name = workspace_info.get('name', f"Workspace {workspace_id[-8:]}") if workspace_info else f"Workspace {workspace_id[-8:]}"
            
            settings = get_default_workspace_settings(workspace_id, workspace_name)
            
            # Save default settings to database
            await supabase_client.update_workspace_settings(workspace_id, {
                'workspace_name': settings.workspace_name,
                'description': settings.description,
                'avatar_url': settings.avatar_url,
                'seat_limit': settings.seat_limit,
                'default_member_role': settings.default_member_role.value,
                'auto_approve_invites': settings.auto_approve_invites,
                'current_seats_used': settings.current_seats_used,
                'storage_used_gb': settings.storage_used_gb,
                'created_at': settings.created_at.isoformat(),
                'updated_at': datetime.now(timezone.utc).isoformat()
            })
        usage_stats = calculate_usage_stats(workspace_id, settings)
        
        return WorkspaceSettingsResponse(
            success=True,
            settings=settings,
            usage_stats=usage_stats,
            last_updated=settings.updated_at.isoformat() if settings.updated_at else None
        )
        
    except Exception as e:
        logger.error(f"Failed to get workspace settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get workspace settings: {str(e)}")


@router.put("/{workspace_id}/settings", response_model=WorkspaceSettingsResponse)
async def update_workspace_settings(
    workspace_id: str,
    settings_request: WorkspaceSettingsUpdateRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Update workspace settings.
    
    Args:
        workspace_id: Workspace identifier
        settings_request: Settings to update
        
    Returns:
        Updated workspace settings
    """
    try:
        logger.info(f"🔄 Updating workspace settings for workspace {workspace_id}")
        logger.info(f"📥 Request data: {settings_request.model_dump()}")
        logger.info(f"👤 User: {user.get('email', 'Unknown')} (ID: {user.get('id', 'Unknown')})")
        
        # Check if user has admin permissions for this workspace
        # In production, verify user's role allows settings modification
        
        # Get current settings from database
        settings_data = await supabase_client.get_workspace_settings(workspace_id)
        
        if settings_data:
            current_settings = WorkspaceSettings(
                workspace_name=settings_data.get('workspace_name', f"Workspace {workspace_id[-8:]}"),
                description=settings_data.get('description', ''),
                avatar_url=settings_data.get('avatar_url'),
                seat_limit=settings_data.get('seat_limit', 5),
                default_member_role=TenantRole(settings_data.get('default_member_role', 'member')),
                auto_approve_invites=settings_data.get('auto_approve_invites', False),
                current_seats_used=settings_data.get('current_seats_used', 1),
                storage_used_gb=settings_data.get('storage_used_gb', 0.0),
                created_at=datetime.fromisoformat(settings_data.get('created_at', datetime.now(timezone.utc).isoformat())),
                updated_at=datetime.fromisoformat(settings_data.get('updated_at')) if settings_data.get('updated_at') else None
            )
        else:
            # Create default settings if none exist
            workspace_info = await supabase_client.get_workspace_by_id(workspace_id)
            workspace_name = workspace_info.get('name', f"Workspace {workspace_id[-8:]}") if workspace_info else f"Workspace {workspace_id[-8:]}"
            current_settings = get_default_workspace_settings(workspace_id, workspace_name)
        
        # Update only provided fields
        update_data = settings_request.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            if hasattr(current_settings, field):
                setattr(current_settings, field, value)
        
        # Update timestamp
        current_settings.updated_at = datetime.now(timezone.utc)
        
        # Save updated settings to database
        update_dict = {
            'workspace_name': current_settings.workspace_name,
            'description': current_settings.description,
            'avatar_url': current_settings.avatar_url,
            'seat_limit': current_settings.seat_limit,
            'default_member_role': current_settings.default_member_role.value,
            'auto_approve_invites': current_settings.auto_approve_invites,
            'current_seats_used': current_settings.current_seats_used,
            'storage_used_gb': current_settings.storage_used_gb,
            'updated_at': current_settings.updated_at.isoformat()
        }
        
        logger.info(f"💾 Sending to database: {update_dict}")
        success = await supabase_client.update_workspace_settings(workspace_id, update_dict)
        logger.info(f"📊 Database update result: {success}")
        
        if not success:
            logger.error(f"❌ Database update failed for workspace {workspace_id}")
            raise HTTPException(status_code=500, detail="Failed to save workspace settings")
        
        # Calculate updated usage stats
        usage_stats = calculate_usage_stats(workspace_id, current_settings)
        
        return WorkspaceSettingsResponse(
            success=True,
            settings=current_settings,
            usage_stats=usage_stats,
            last_updated=current_settings.updated_at.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to update workspace settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update workspace settings: {str(e)}")


@router.get("/{workspace_id}/settings/profile")
async def get_workspace_profile(
    workspace_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get workspace profile information.
    
    Returns:
        Workspace profile data
    """
    try:
        logger.info(f"Getting workspace profile for workspace {workspace_id}")
        
        # Get settings
        if workspace_id not in workspace_settings_store:
            workspace_name = f"Workspace {workspace_id[-8:]}"
            workspace_settings_store[workspace_id] = get_default_workspace_settings(
                workspace_id, workspace_name
            )
        
        settings = workspace_settings_store[workspace_id]
        
        return {
            "success": True,
            "profile": {
                "workspace_name": settings.workspace_name,
                "description": settings.description,
                "avatar_url": settings.avatar_url,
                "created_at": settings.created_at.isoformat(),
                "updated_at": settings.updated_at.isoformat() if settings.updated_at else None
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get workspace profile: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get workspace profile: {str(e)}")


@router.get("/{workspace_id}/settings/usage")
async def get_workspace_usage(
    workspace_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get workspace usage statistics.
    
    Returns:
        Current usage data and limits
    """
    try:
        logger.info(f"Getting workspace usage for workspace {workspace_id}")
        
        # Get settings
        if workspace_id not in workspace_settings_store:
            workspace_name = f"Workspace {workspace_id[-8:]}"
            workspace_settings_store[workspace_id] = get_default_workspace_settings(
                workspace_id, workspace_name
            )
        
        settings = workspace_settings_store[workspace_id]
        usage_stats = calculate_usage_stats(workspace_id, settings)
        
        return {
            "success": True,
            "usage": usage_stats,
            "limits": {
                "seat_limit": settings.seat_limit,
                "can_add_member": settings.can_add_member(),
                "seats_available": settings.get_available_seats()
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get workspace usage: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get workspace usage: {str(e)}")


class MemberSettingsUpdateRequest(BaseModel):
    """Request model for updating member management settings."""
    
    seat_limit: Optional[int] = Field(None, ge=1, le=1000, description="Seat limit")
    default_member_role: Optional[TenantRole] = Field(None, description="Default member role")
    auto_approve_invites: Optional[bool] = Field(None, description="Auto-approve invites")


@router.get("/{workspace_id}/settings/members")
async def get_member_settings(
    workspace_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get member management settings.
    
    Returns:
        Member management configuration
    """
    try:
        logger.info(f"Getting member settings for workspace {workspace_id}")
        
        # Get settings
        if workspace_id not in workspace_settings_store:
            workspace_name = f"Workspace {workspace_id[-8:]}"
            workspace_settings_store[workspace_id] = get_default_workspace_settings(
                workspace_id, workspace_name
            )
        
        settings = workspace_settings_store[workspace_id]
        
        return {
            "success": True,
            "member_settings": {
                "seat_limit": settings.seat_limit,
                "default_member_role": settings.default_member_role.value,
                "auto_approve_invites": settings.auto_approve_invites,
                "current_seats_used": settings.current_seats_used,
                "seats_available": settings.get_available_seats(),
                "can_add_member": settings.can_add_member()
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get member settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get member settings: {str(e)}")


@router.put("/{workspace_id}/settings/members")
async def update_member_settings(
    workspace_id: str,
    member_settings: MemberSettingsUpdateRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Update member management settings.
    
    Args:
        workspace_id: Workspace identifier
        member_settings: Member settings to update
        
    Returns:
        Updated member settings
    """
    try:
        logger.info(f"Updating member settings for workspace {workspace_id}")
        
        # Get current settings
        if workspace_id not in workspace_settings_store:
            workspace_name = f"Workspace {workspace_id[-8:]}"
            workspace_settings_store[workspace_id] = get_default_workspace_settings(
                workspace_id, workspace_name
            )
        
        current_settings = workspace_settings_store[workspace_id]
        
        # Update only provided fields
        update_data = member_settings.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            if hasattr(current_settings, field):
                setattr(current_settings, field, value)
        
        # Update timestamp
        current_settings.updated_at = datetime.now(timezone.utc)
        
        # Save updated settings
        workspace_settings_store[workspace_id] = current_settings
        
        return {
            "success": True,
            "message": "Member settings updated successfully",
            "member_settings": {
                "seat_limit": current_settings.seat_limit,
                "default_member_role": current_settings.default_member_role.value,
                "auto_approve_invites": current_settings.auto_approve_invites,
                "current_seats_used": current_settings.current_seats_used,
                "seats_available": current_settings.get_available_seats(),
                "can_add_member": current_settings.can_add_member()
            },
            "last_updated": current_settings.updated_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to update member settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update member settings: {str(e)}")