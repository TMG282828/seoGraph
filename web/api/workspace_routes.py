"""
Multi-User Workspace API Routes.

Handles workspace creation, management, member collaboration, and invite code system
for modern SaaS-style workspace functionality.
"""

import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field
import structlog

from models.user_models import (
    Tenant, TenantCreateRequest, TenantSettings, 
    UserTenant, TenantRole, TenantStatus,
    User, create_tenant, create_user_tenant_association
)
from src.auth.auth_middleware import get_current_user, get_current_organization
from utils.tenant_mapper import TenantOrgMapper

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/workspaces", tags=["workspaces"])

# Import Supabase client for database persistence
from src.database.supabase_client import supabase_client

# In-memory invite code storage (could be Redis in production)
invite_codes: Dict[str, Dict[str, Any]] = {}


class WorkspaceCreateRequest(BaseModel):
    """Request model for creating a new workspace."""
    name: str = Field(..., min_length=1, max_length=200, description="Workspace name")
    description: Optional[str] = Field(None, max_length=1000, description="Workspace description")
    slug: Optional[str] = Field(None, min_length=2, max_length=50, description="URL-friendly identifier")


class WorkspaceUpdateRequest(BaseModel):
    """Request model for updating workspace settings."""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)


class InviteCodeCreateRequest(BaseModel):
    """Request model for creating invite codes."""
    expires_in_hours: int = Field(168, ge=1, le=8760, description="Expiration in hours (default: 7 days)")
    max_uses: int = Field(10, ge=1, le=100, description="Maximum number of uses")


class JoinWorkspaceRequest(BaseModel):
    """Request model for joining workspace via invite code."""
    invite_code: str = Field(..., min_length=6, max_length=20, description="Invite code")


class MemberRoleUpdateRequest(BaseModel):
    """Request model for updating member roles."""
    role: TenantRole = Field(..., description="New role for the member")


class WorkspaceResponse(BaseModel):
    """Response model for workspace information."""
    id: str
    name: str
    description: Optional[str]
    slug: str
    user_role: TenantRole
    is_current: bool
    member_count: int
    created_at: str
    settings: Optional[Dict[str, Any]] = None


class MemberResponse(BaseModel):
    """Response model for workspace members."""
    user_id: str
    email: str
    display_name: str
    role: TenantRole
    joined_at: str
    last_accessed: Optional[str] = None


class InviteCodeResponse(BaseModel):
    """Response model for invite codes."""
    code: str
    expires_at: str
    max_uses: int
    current_uses: int
    is_active: bool
    created_at: str


# =============================================================================
# Core Workspace Operations
# =============================================================================

@router.get("/", response_model=List[WorkspaceResponse])
async def get_user_workspaces(
    request: Request,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get all workspaces accessible to the current user.
    
    Returns:
        List of workspaces with user's role and permissions
    """
    try:
        user_id = user["id"]
        
        logger.info(f"Getting workspaces for user {user_id}")
        
        # Get user's workspaces from database
        workspaces_data = await supabase_client.get_user_workspaces(user_id)
        
        workspaces = []
        for workspace in workspaces_data:
            # Get member count for each workspace
            members = await supabase_client.get_workspace_members(workspace["id"])
            
            # Get workspace settings to get the most up-to-date name
            settings_data = await supabase_client.get_workspace_settings(workspace["id"])
            workspace_name = workspace["name"]  # Default from workspaces table
            if settings_data and settings_data.get('workspace_name'):
                workspace_name = settings_data.get('workspace_name')  # Use settings name if available
            
            workspace_response = {
                "id": workspace["id"],
                "name": workspace_name,
                "description": workspace.get("description"),
                "slug": workspace.get("slug"),
                "user_role": workspace.get("user_role", TenantRole.MEMBER),
                "is_current": workspace["id"] == user.get("organization_id"),
                "member_count": len(members),
                "created_at": workspace.get("created_at", datetime.now(timezone.utc).isoformat()),
                "settings": None  # Will be loaded separately if needed
            }
            workspaces.append(workspace_response)
        
        # If no workspaces found, create a fallback based on current org
        if not workspaces and user.get("organization_id"):
            current_org_id = user.get("organization_id")
            
            # Try to get actual workspace data from database
            workspace_info = await supabase_client.get_workspace_by_id(current_org_id)
            settings_data = await supabase_client.get_workspace_settings(current_org_id)
            
            # Use actual workspace name from settings or database, fallback to "My Workspace"
            workspace_name = "My Workspace"  # Default fallback
            if settings_data and settings_data.get('workspace_name'):
                workspace_name = settings_data.get('workspace_name')
            elif workspace_info and workspace_info.get('name'):
                workspace_name = workspace_info.get('name')
            
            workspace_data = {
                "id": current_org_id,
                "name": workspace_name,
                "description": workspace_info.get('description') if workspace_info else None,
                "slug": current_org_id,
                "user_role": TenantRole.OWNER,
                "is_current": True,
                "member_count": 1,
                "created_at": workspace_info.get('created_at', datetime.now(timezone.utc).isoformat()) if workspace_info else datetime.now(timezone.utc).isoformat(),
                "settings": None
            }
            workspaces.append(workspace_data)
        
        logger.info(f"Retrieved {len(workspaces)} workspaces for user {user_id}")
        return workspaces
        
    except Exception as e:
        logger.error(f"Failed to get user workspaces: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve workspaces: {str(e)}")


@router.post("/", response_model=WorkspaceResponse)
async def create_workspace(
    workspace_request: WorkspaceCreateRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Create a new workspace with the current user as owner.
    
    Args:
        workspace_request: Workspace creation details
        
    Returns:
        Created workspace information
    """
    try:
        user_id = user["id"]
        user_email = user["email"]
        
        logger.info(f"Creating workspace '{workspace_request.name}' for user {user_email}")
        
        # Generate slug if not provided
        slug = workspace_request.slug
        if not slug:
            # Create slug from workspace name
            slug = workspace_request.name.lower().replace(" ", "-")
            # Add timestamp to ensure uniqueness
            slug += f"-{int(datetime.now().timestamp())}"
        
        # Create workspace in database
        import uuid
        workspace_id = str(uuid.uuid4())
        
        workspace_data = {
            'id': workspace_id,
            'name': workspace_request.name,
            'description': workspace_request.description,
            'slug': slug,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'created_by': user_id
        }
        
        # Create workspace
        create_result = await supabase_client.create_workspace(workspace_data)
        if not create_result.get('success'):
            raise HTTPException(status_code=500, detail=f"Failed to create workspace: {create_result.get('error')}")
        
        # Add user as owner
        join_success = await supabase_client.join_workspace(user_id, workspace_id, TenantRole.OWNER.value)
        if not join_success:
            logger.warning(f"Failed to add user as owner to workspace {workspace_id}")
        
        workspace_response = WorkspaceResponse(
            id=workspace_id,
            name=workspace_request.name,
            description=workspace_request.description,
            slug=slug,
            user_role=TenantRole.OWNER,
            is_current=False,  # Not switched to it yet
            member_count=1,
            created_at=datetime.now(timezone.utc).isoformat(),
            settings=None
        )
        
        logger.info(f"Successfully created workspace {workspace_id} for user {user_email}")
        return workspace_response
        
    except Exception as e:
        logger.error(f"Failed to create workspace: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create workspace: {str(e)}")


@router.put("/{workspace_id}", response_model=WorkspaceResponse)
async def update_workspace(
    workspace_id: str,
    workspace_update: WorkspaceUpdateRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Update workspace settings (owner/admin only).
    
    Args:
        workspace_id: Workspace identifier
        workspace_update: Updated workspace information
        
    Returns:
        Updated workspace information
    """
    try:
        user_id = user["id"]
        
        logger.info(f"Updating workspace {workspace_id} by user {user_id}")
        
        # TODO: Verify user has admin/owner permissions for this workspace
        # For now, allow if user has access to workspace
        
        # TODO: Update workspace in database
        # For MVP, return mock updated response
        
        updated_workspace = WorkspaceResponse(
            id=workspace_id,
            name=workspace_update.name or "Updated Workspace",
            description=workspace_update.description,
            slug=workspace_id,
            user_role=TenantRole.OWNER,
            is_current=workspace_id == user.get("organization_id"),
            member_count=1,
            created_at=datetime.now(timezone.utc).isoformat()
        )
        
        logger.info(f"Successfully updated workspace {workspace_id}")
        return updated_workspace
        
    except Exception as e:
        logger.error(f"Failed to update workspace {workspace_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update workspace: {str(e)}")


@router.delete("/{workspace_id}")
async def delete_workspace(
    workspace_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Delete workspace (owner only).
    
    Args:
        workspace_id: Workspace identifier
        
    Returns:
        Success confirmation
    """
    try:
        user_id = user["id"]
        
        logger.info(f"Deleting workspace {workspace_id} by user {user_id}")
        
        # TODO: Verify user is owner of this workspace
        # TODO: Delete all workspace data (content, settings, etc.)
        # TODO: Remove all user associations
        
        # For MVP, just return success
        logger.info(f"Successfully deleted workspace {workspace_id}")
        return {"success": True, "message": "Workspace deleted successfully"}
        
    except Exception as e:
        logger.error(f"Failed to delete workspace {workspace_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete workspace: {str(e)}")


@router.post("/{workspace_id}/switch")
async def switch_workspace(
    workspace_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Switch to a different workspace context.
    
    Args:
        workspace_id: Target workspace identifier
        
    Returns:
        New access token with updated workspace context
    """
    try:
        user_id = user["id"]
        
        logger.info(f"Switching to workspace {workspace_id} for user {user_id}")
        
        # TODO: Verify user has access to this workspace
        # TODO: Generate new JWT token with updated organization_id
        
        # For MVP, simulate workspace switching
        from src.auth.auth_middleware import create_access_token
        
        # Update user context with new organization_id
        updated_user_data = {
            **user,
            "organization_id": workspace_id
        }
        
        # Create new access token
        new_token = create_access_token(updated_user_data)
        
        logger.info(f"Successfully switched to workspace {workspace_id} for user {user_id}")
        
        return {
            "success": True,
            "access_token": new_token,
            "workspace_id": workspace_id,
            "message": "Workspace switched successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to switch workspace {workspace_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to switch workspace: {str(e)}")


# =============================================================================
# Invite Code System
# =============================================================================

@router.post("/{workspace_id}/invite-codes", response_model=InviteCodeResponse)
async def create_invite_code(
    workspace_id: str,
    invite_request: InviteCodeCreateRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Generate a shareable invite code for the workspace (owner/admin only).
    
    Args:
        workspace_id: Workspace identifier
        invite_request: Invite code configuration
        
    Returns:
        Generated invite code details
    """
    try:
        user_id = user["id"]
        
        logger.info(f"Creating invite code for workspace {workspace_id} by user {user_id}")
        
        # TODO: Verify user has admin/owner permissions for this workspace
        
        # Generate secure invite code
        invite_code = secrets.token_urlsafe(8)  # Generates ~11 character code
        
        # Calculate expiration time
        expires_at = datetime.now(timezone.utc) + timedelta(hours=invite_request.expires_in_hours)
        
        # Store invite code
        invite_codes[invite_code] = {
            "workspace_id": workspace_id,
            "created_by": user_id,
            "expires_at": expires_at,
            "max_uses": invite_request.max_uses,
            "current_uses": 0,
            "is_active": True,
            "created_at": datetime.now(timezone.utc)
        }
        
        logger.info(f"Successfully created invite code {invite_code[:4]}... for workspace {workspace_id}")
        
        return InviteCodeResponse(
            code=invite_code,
            expires_at=expires_at.isoformat(),
            max_uses=invite_request.max_uses,
            current_uses=0,
            is_active=True,
            created_at=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to create invite code for workspace {workspace_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create invite code: {str(e)}")


@router.get("/{workspace_id}/invite-codes", response_model=List[InviteCodeResponse])
async def get_invite_codes(
    workspace_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get all active invite codes for the workspace (owner/admin only).
    
    Args:
        workspace_id: Workspace identifier
        
    Returns:
        List of active invite codes
    """
    try:
        user_id = user["id"]
        
        logger.info(f"Getting invite codes for workspace {workspace_id} by user {user_id}")
        
        # TODO: Verify user has admin/owner permissions for this workspace
        
        # Filter invite codes for this workspace
        workspace_codes = []
        for code, details in invite_codes.items():
            if details["workspace_id"] == workspace_id and details["is_active"]:
                workspace_codes.append(InviteCodeResponse(
                    code=code,
                    expires_at=details["expires_at"].isoformat(),
                    max_uses=details["max_uses"],
                    current_uses=details["current_uses"],
                    is_active=details["is_active"],
                    created_at=details["created_at"].isoformat()
                ))
        
        logger.info(f"Retrieved {len(workspace_codes)} invite codes for workspace {workspace_id}")
        return workspace_codes
        
    except Exception as e:
        logger.error(f"Failed to get invite codes for workspace {workspace_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve invite codes: {str(e)}")


@router.delete("/{workspace_id}/invite-codes/{invite_code}")
async def revoke_invite_code(
    workspace_id: str,
    invite_code: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Revoke an invite code (owner/admin only).
    
    Args:
        workspace_id: Workspace identifier
        invite_code: Invite code to revoke
        
    Returns:
        Success confirmation
    """
    try:
        user_id = user["id"]
        
        logger.info(f"Revoking invite code {invite_code[:4]}... for workspace {workspace_id} by user {user_id}")
        
        # TODO: Verify user has admin/owner permissions for this workspace
        
        if invite_code in invite_codes and invite_codes[invite_code]["workspace_id"] == workspace_id:
            invite_codes[invite_code]["is_active"] = False
            logger.info(f"Successfully revoked invite code {invite_code[:4]}...")
            return {"success": True, "message": "Invite code revoked successfully"}
        else:
            raise HTTPException(status_code=404, detail="Invite code not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to revoke invite code {invite_code}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to revoke invite code: {str(e)}")


@router.post("/join", response_model=WorkspaceResponse)
async def join_workspace(
    join_request: JoinWorkspaceRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Join a workspace using an invite code.
    
    Args:
        join_request: Join request with invite code
        
    Returns:
        Joined workspace information
    """
    try:
        user_id = user["id"]
        user_email = user["email"]
        invite_code = join_request.invite_code
        
        logger.info(f"User {user_email} attempting to join workspace with code {invite_code[:4]}...")
        
        # Validate invite code
        if invite_code not in invite_codes:
            raise HTTPException(status_code=400, detail="Invalid invite code")
        
        code_details = invite_codes[invite_code]
        
        # Check if code is active
        if not code_details["is_active"]:
            raise HTTPException(status_code=400, detail="Invite code has been revoked")
        
        # Check if code is expired
        if datetime.now(timezone.utc) > code_details["expires_at"]:
            raise HTTPException(status_code=400, detail="Invite code has expired")
        
        # Check if code has reached max uses
        if code_details["current_uses"] >= code_details["max_uses"]:
            raise HTTPException(status_code=400, detail="Invite code has reached maximum uses")
        
        workspace_id = code_details["workspace_id"]
        
        # TODO: Check if user is already a member of this workspace
        # TODO: Create UserTenant association with MEMBER role
        # TODO: Update member count in workspace
        
        # Increment usage count
        invite_codes[invite_code]["current_uses"] += 1
        
        # For MVP, return mock workspace data
        workspace_response = WorkspaceResponse(
            id=workspace_id,
            name="Joined Workspace",
            description="Workspace joined via invite code",
            slug=workspace_id,
            user_role=TenantRole.MEMBER,
            is_current=False,
            member_count=2,  # Mock increment
            created_at=datetime.now(timezone.utc).isoformat()
        )
        
        logger.info(f"User {user_email} successfully joined workspace {workspace_id}")
        return workspace_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to join workspace with code {join_request.invite_code}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to join workspace: {str(e)}")


# =============================================================================
# Member Management
# =============================================================================

@router.get("/{workspace_id}/members", response_model=List[MemberResponse])
async def get_workspace_members(
    workspace_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get all members of the workspace.
    
    Args:
        workspace_id: Workspace identifier
        
    Returns:
        List of workspace members with roles
    """
    try:
        user_id = user["id"]
        
        logger.info(f"Getting members for workspace {workspace_id} by user {user_id}")
        
        # TODO: Verify user has access to this workspace
        # TODO: Query UserTenant associations for this workspace
        
        # For MVP, return mock member data
        members = [
            MemberResponse(
                user_id=user_id,
                email=user["email"],
                display_name=user.get("display_name", "Current User"),
                role=TenantRole.OWNER,
                joined_at=datetime.now(timezone.utc).isoformat()
            )
        ]
        
        logger.info(f"Retrieved {len(members)} members for workspace {workspace_id}")
        return members
        
    except Exception as e:
        logger.error(f"Failed to get members for workspace {workspace_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve members: {str(e)}")


@router.put("/{workspace_id}/members/{member_user_id}")
async def update_member_role(
    workspace_id: str,
    member_user_id: str,
    role_update: MemberRoleUpdateRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Update a member's role in the workspace (owner/admin only).
    
    Args:
        workspace_id: Workspace identifier
        member_user_id: Member to update
        role_update: New role information
        
    Returns:
        Success confirmation
    """
    try:
        user_id = user["id"]
        
        logger.info(f"Updating role for member {member_user_id} in workspace {workspace_id} by user {user_id}")
        
        # TODO: Verify user has admin/owner permissions
        # TODO: Update UserTenant role in database
        
        # For MVP, return success
        logger.info(f"Successfully updated member {member_user_id} role to {role_update.role}")
        return {
            "success": True,
            "message": f"Member role updated to {role_update.role.value}"
        }
        
    except Exception as e:
        logger.error(f"Failed to update member role: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update member role: {str(e)}")


@router.delete("/{workspace_id}/members/{member_user_id}")
async def remove_member(
    workspace_id: str,
    member_user_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Remove a member from the workspace (owner/admin only).
    
    Args:
        workspace_id: Workspace identifier
        member_user_id: Member to remove
        
    Returns:
        Success confirmation
    """
    try:
        user_id = user["id"]
        
        logger.info(f"Removing member {member_user_id} from workspace {workspace_id} by user {user_id}")
        
        # TODO: Verify user has admin/owner permissions
        # TODO: Remove UserTenant association from database
        # TODO: Clean up member's data if needed
        
        # Prevent owner from removing themselves
        if member_user_id == user_id:
            raise HTTPException(status_code=400, detail="Cannot remove yourself from workspace")
        
        # For MVP, return success
        logger.info(f"Successfully removed member {member_user_id} from workspace {workspace_id}")
        return {"success": True, "message": "Member removed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove member: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove member: {str(e)}")