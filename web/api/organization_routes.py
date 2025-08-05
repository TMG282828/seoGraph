"""
Organization API routes for multi-tenant system.

Handles organization creation, management, and settings during onboarding and normal operations.
"""

import logging
from datetime import timedelta
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Response
from pydantic import BaseModel, Field
import structlog

from src.database.supabase_client import supabase_client  
from src.auth.auth_middleware import get_current_user, create_access_token
from utils.tenant_mapper import TenantOrgMapper

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/organizations", tags=["organizations"])

# Request models
class OrganizationCreateRequest(BaseModel):
    """Request model for creating organization during onboarding."""
    name: str = Field(..., min_length=1, max_length=200, description="Organization name")
    slug: str = Field(..., min_length=2, max_length=50, description="URL-friendly identifier")
    admin_email: str = Field(..., description="Admin user email")
    admin_name: str = Field(..., description="Admin user name")
    
    # Additional onboarding fields
    industry: Optional[str] = Field(None, description="Industry/business type")
    website: Optional[str] = Field(None, description="Company website")
    content_goals: Optional[str] = Field(None, description="Content marketing goals")
    target_audience: Optional[str] = Field(None, description="Target audience description")

class OrganizationUpdateRequest(BaseModel):
    """Request model for updating organization settings."""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    website: Optional[str] = Field(None)
    industry: Optional[str] = Field(None)
    content_goals: Optional[str] = Field(None)
    target_audience: Optional[str] = Field(None)
    settings: Optional[Dict[str, Any]] = Field(None)

# Response models  
class OrganizationResponse(BaseModel):
    """Response model for organization data."""
    success: bool
    organization: Dict[str, Any]
    message: str

class OrganizationListResponse(BaseModel):
    """Response model for organization list."""
    success: bool
    organizations: list[Dict[str, Any]]

class MessageResponse(BaseModel):
    """Response model for simple messages."""
    success: bool
    message: str


@router.post("", response_model=OrganizationResponse)
async def create_organization(
    request: OrganizationCreateRequest, 
    response: Response,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Create a new organization during onboarding.
    
    This endpoint is called by the onboarding flow after user registration.
    It creates the organization and updates the user's JWT token with organization context.
    """
    try:
        logger.info(f"Creating organization '{request.name}' for user: {user.get('email')}")
        
        # Check if user already has an organization
        if user.get('organization_id'):
            logger.warning(f"User {user.get('email')} already has organization: {user.get('organization_id')}")
            raise HTTPException(status_code=400, detail="User already belongs to an organization")
        
        # Create organization in Supabase
        org_result = await supabase_client.create_organization(
            name=request.name,
            slug=request.slug,
            admin_email=request.admin_email,
            admin_name=request.admin_name
        )
        
        if not org_result.get('success'):
            error_msg = org_result.get('error', 'Failed to create organization')
            logger.error(f"Organization creation failed: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        organization_id = org_result.get('organization_id')
        logger.info(f"Organization created successfully: {organization_id}")
        
        # Update user record with organization_id
        user_update_success = await supabase_client.update_user(
            user_id=user.get('id'),
            updates={
                'organization_id': organization_id,
                'role': 'admin',  # User who creates org becomes admin
                'onboarding_completed': True
            }
        )
        
        if not user_update_success:
            logger.warning(f"Failed to update user {user.get('id')} with organization_id")
            # Don't fail the request, just log the warning
        
        # Create updated JWT token with organization context
        jwt_payload = {
            'id': user.get('id'),
            'email': user.get('email'),
            'display_name': user.get('display_name'),
            'role': 'admin',
            'organization_id': organization_id
        }
        
        # Create new access token with organization context
        access_token = create_access_token(jwt_payload, expires_delta=timedelta(days=7))
        
        # Update cookie with new token
        response.set_cookie(
            key="access_token",
            value=access_token,
            max_age=7 * 24 * 60 * 60,  # 7 days
            httponly=True,
            secure=False,  # Set to True in production with HTTPS
            samesite="lax"
        )
        
        logger.info(f"Organization setup completed for {user.get('email')}")
        
        return OrganizationResponse(
            success=True,
            organization={
                'id': organization_id,
                'name': request.name,
                'slug': request.slug,
                'user_role': 'admin',
                'settings': {
                    'industry': request.industry,
                    'website': request.website,
                    'content_goals': request.content_goals,
                    'target_audience': request.target_audience,
                    'configuration_completed': True
                }
            },
            message="Organization created successfully!"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Organization creation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to create organization. Please try again.")


@router.get("/current", response_model=OrganizationResponse)
async def get_current_organization(user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get current user's organization details.
    """
    try:
        organization_id = user.get('organization_id')
        
        if not organization_id:
            raise HTTPException(status_code=404, detail="User is not part of any organization")
        
        # Get organization data from Supabase
        org_data = await supabase_client.get_organization(organization_id)
        
        if not org_data:
            logger.warning(f"Organization {organization_id} not found for user {user.get('email')}")
            raise HTTPException(status_code=404, detail="Organization not found")
        
        return OrganizationResponse(
            success=True,
            organization={
                'id': org_data.get('id'),
                'name': org_data.get('name'),
                'slug': org_data.get('slug'),
                'user_role': user.get('role', 'member'),
                'settings': org_data.get('settings', {}),
                'created_at': org_data.get('created_at'),
                'configuration_completed': org_data.get('configuration_completed', False)
            },
            message="Organization retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get organization error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get organization details")


@router.put("/current", response_model=OrganizationResponse)
async def update_current_organization(
    request: OrganizationUpdateRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Update current user's organization settings.
    Only admins can update organization details.
    """
    try:
        organization_id = user.get('organization_id')
        user_role = user.get('role', 'member')
        
        if not organization_id:
            raise HTTPException(status_code=404, detail="User is not part of any organization")
        
        if user_role != 'admin':
            raise HTTPException(status_code=403, detail="Only organization admins can update organization settings")
        
        # Prepare update data
        updates = {}
        if request.name:
            updates['name'] = request.name
        if request.website:
            updates['website'] = request.website
        if request.industry or request.content_goals or request.target_audience or request.settings:
            # Update settings field
            current_settings = {}
            org_data = await supabase_client.get_organization(organization_id)
            if org_data and org_data.get('settings'):
                current_settings = org_data.get('settings', {})
            
            if request.industry:
                current_settings['industry'] = request.industry
            if request.content_goals:
                current_settings['content_goals'] = request.content_goals
            if request.target_audience:
                current_settings['target_audience'] = request.target_audience
            if request.settings:
                current_settings.update(request.settings)
            
            updates['settings'] = current_settings
        
        if not updates:
            raise HTTPException(status_code=400, detail="No valid updates provided")
        
        # Update organization
        update_success = await supabase_client.update_organization(organization_id, updates)
        
        if not update_success:
            raise HTTPException(status_code=500, detail="Failed to update organization")
        
        # Get updated organization data
        updated_org = await supabase_client.get_organization(organization_id)
        
        logger.info(f"Organization {organization_id} updated by {user.get('email')}")
        
        return OrganizationResponse(
            success=True,
            organization={
                'id': updated_org.get('id'),
                'name': updated_org.get('name'),
                'slug': updated_org.get('slug'),
                'user_role': user_role,
                'settings': updated_org.get('settings', {}),
                'created_at': updated_org.get('created_at'),
                'configuration_completed': updated_org.get('configuration_completed', False)
            },
            message="Organization updated successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Organization update error: {e}")
        raise HTTPException(status_code=500, detail="Failed to update organization")


@router.get("/members")
async def get_organization_members(user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get all members of the current organization.
    """
    try:
        organization_id = user.get('organization_id')
        
        if not organization_id:
            raise HTTPException(status_code=404, detail="User is not part of any organization")
        
        # Get organization members
        members = await supabase_client.get_organization_users(organization_id)
        
        return {
            "success": True,
            "members": members,
            "total": len(members)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get organization members error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get organization members")


@router.get("/settings")
async def get_organization_settings(user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get organization settings using the TenantOrgMapper.
    This is used by various services that need organization configuration.
    """
    try:
        organization_id = user.get('organization_id')
        
        if not organization_id:
            raise HTTPException(status_code=404, detail="User is not part of any organization")
        
        # Get organization settings via mapper
        settings = await TenantOrgMapper.get_org_settings(organization_id)
        
        return {
            "success": True,
            "settings": settings,
            "organization_id": organization_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get organization settings error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get organization settings")


@router.put("/settings")
async def update_organization_settings(
    settings_update: Dict[str, Any],
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Update organization settings using the TenantOrgMapper.
    Only admins can update settings.
    """
    try:
        organization_id = user.get('organization_id')
        user_role = user.get('role', 'member')
        
        if not organization_id:
            raise HTTPException(status_code=404, detail="User is not part of any organization")
        
        if user_role != 'admin':
            raise HTTPException(status_code=403, detail="Only organization admins can update settings")
        
        # Update settings via mapper
        success = await TenantOrgMapper.update_org_settings(organization_id, settings_update)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update organization settings")
        
        logger.info(f"Organization settings updated for {organization_id} by {user.get('email')}")
        
        return {
            "success": True,
            "message": "Organization settings updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update organization settings error: {e}")
        raise HTTPException(status_code=500, detail="Failed to update organization settings")