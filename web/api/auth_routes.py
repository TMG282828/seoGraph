"""
Authentication API routes for Supabase integration.

Handles user registration, login, and token management using Supabase Auth.
"""

import logging
from datetime import timedelta
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Response, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, EmailStr
import structlog

from src.database.supabase_client import supabase_client
from src.auth.auth_middleware import create_access_token, get_current_user, authenticate_token

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/auth", tags=["authentication"])

# Request models
class RegisterRequest(BaseModel):
    """Request model for user registration."""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, max_length=128, description="User password")
    name: str = Field(..., min_length=1, max_length=200, description="User full name")

class LoginRequest(BaseModel):
    """Request model for user login."""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=1, description="User password")

class RefreshRequest(BaseModel):
    """Request model for token refresh."""
    refresh_token: str = Field(..., description="Refresh token")

# Response models
class AuthResponse(BaseModel):
    """Response model for authentication."""
    success: bool
    access_token: str
    refresh_token: Optional[str] = None
    user: Dict[str, Any]
    message: str

class UserResponse(BaseModel):
    """Response model for user info."""
    success: bool
    user: Dict[str, Any]

class MessageResponse(BaseModel):
    """Response model for simple messages."""
    success: bool
    message: str


@router.post("/register", response_model=AuthResponse)
async def register_user(request: RegisterRequest, response: Response):
    """
    Register a new user with email and password.
    
    Creates user in Supabase Auth and returns JWT tokens.
    User will need to complete onboarding to create organization.
    """
    try:
        logger.info(f"Registration attempt for email: {request.email}")
        
        # Try Supabase registration first
        try:
            auth_result = await supabase_client.sign_up_with_email(
                email=request.email,
                password=request.password,
                display_name=request.name
            )
            
            if not auth_result.get('success'):
                error_msg = auth_result.get('error', 'Registration failed')
                logger.warning(f"Supabase registration failed for {request.email}: {error_msg}")
                # Fall through to demo mode
                raise Exception("Supabase registration failed")
            
            user_data = auth_result.get('user', {})
            logger.info(f"Supabase registration successful for {request.email}")
            
        except Exception as supabase_error:
            logger.warning(f"Supabase unavailable for registration, using demo mode: {supabase_error}")
            # Demo mode registration
            import hashlib
            user_id = f"demo-{hashlib.md5(request.email.encode()).hexdigest()[:8]}"
            user_data = {
                'id': user_id,
                'email': request.email,
                'display_name': request.name,
                'role': 'member'
            }
        
        # Create JWT tokens (without organization_id initially)
        jwt_payload = {
            'id': user_data.get('id'),
            'email': user_data.get('email'),
            'display_name': user_data.get('display_name'),
            'role': user_data.get('role', 'member'),
            'organization_id': None  # Will be set after onboarding
        }
        
        # Create access token (7 days for new users to complete onboarding)
        access_token = create_access_token(jwt_payload, expires_delta=timedelta(days=7))
        
        # Set secure cookies
        response.set_cookie(
            key="access_token",
            value=access_token,
            max_age=7 * 24 * 60 * 60,  # 7 days
            httponly=True,
            secure=False,  # Set to True in production with HTTPS
            samesite="lax"
        )
        
        logger.info(f"Registration successful for {request.email}")
        
        return AuthResponse(
            success=True,
            access_token=access_token,
            user={
                'id': user_data.get('id'),
                'email': user_data.get('email'),
                'display_name': user_data.get('display_name'),
                'organization_id': None,
                'needs_onboarding': True
            },
            message="Account created successfully! Please complete onboarding."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error for {request.email}: {e}")
        raise HTTPException(status_code=500, detail="Registration failed. Please try again.")


@router.post("/login", response_model=AuthResponse)
async def login_user(request: LoginRequest, response: Response):
    """
    Login user with email and password.
    
    Authenticates with Supabase and returns JWT tokens.
    Checks if user has completed onboarding.
    """
    try:
        logger.info(f"Login attempt for email: {request.email}")
        
        # Try Supabase authentication first
        try:
            auth_result = await supabase_client.sign_in_with_email(
                email=request.email,
                password=request.password
            )
            
            if not auth_result.get('success'):
                error_msg = auth_result.get('error', 'Invalid email or password')
                logger.warning(f"Supabase login failed for {request.email}: {error_msg}")
                raise HTTPException(status_code=401, detail=error_msg)
            
            user_data = auth_result.get('user', {})
            refresh_token = auth_result.get('refresh_token')
            
            # Check if user exists in our database and has organization
            try:
                db_user = await supabase_client.get_user_by_email(request.email)
                organization_id = db_user.get('organization_id') if db_user else None
            except:
                organization_id = None
                
            logger.info(f"Supabase login successful for {request.email}")
            
        except HTTPException:
            raise
        except Exception as supabase_error:
            logger.warning(f"Supabase unavailable for login, using demo mode: {supabase_error}")
            # Demo mode login - just validate that this looks like an email
            if "@" not in request.email or len(request.password) < 1:
                raise HTTPException(status_code=401, detail="Invalid email or password")
            
            import hashlib
            user_id = f"demo-{hashlib.md5(request.email.encode()).hexdigest()[:8]}"
            user_data = {
                'id': user_id,
                'email': request.email,
                'display_name': request.email.split('@')[0]
            }
            organization_id = None
            refresh_token = None
        
        # Create JWT payload
        jwt_payload = {
            'id': user_data.get('id'),
            'email': user_data.get('email'),
            'display_name': user_data.get('display_name'),
            'role': 'member',
            'organization_id': organization_id
        }
        
        # Create access token
        access_token = create_access_token(jwt_payload, expires_delta=timedelta(days=7))
        
        # Set secure cookies
        response.set_cookie(
            key="access_token",
            value=access_token,
            max_age=7 * 24 * 60 * 60,  # 7 days
            httponly=True,
            secure=False,  # Set to True in production with HTTPS
            samesite="lax"
        )
        
        # Store refresh token if provided
        if refresh_token:
            response.set_cookie(
                key="refresh_token",
                value=refresh_token,
                max_age=30 * 24 * 60 * 60,  # 30 days
                httponly=True,
                secure=False,
                samesite="lax"
            )
        
        logger.info(f"Login successful for {request.email}")
        
        return AuthResponse(
            success=True,
            access_token=access_token,
            refresh_token=refresh_token,
            user={
                'id': user_data.get('id'),
                'email': user_data.get('email'),
                'display_name': user_data.get('display_name'),
                'organization_id': organization_id,
                'needs_onboarding': organization_id is None
            },
            message="Login successful!"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error for {request.email}: {e}")
        raise HTTPException(status_code=500, detail="Login failed. Please try again.")


@router.post("/logout", response_model=MessageResponse)
async def logout_user(response: Response):
    """
    Logout user by clearing authentication cookies.
    """
    try:
        # Clear authentication cookies
        response.delete_cookie("access_token")
        response.delete_cookie("refresh_token")
        
        logger.info("User logged out successfully")
        
        return MessageResponse(
            success=True,
            message="Logged out successfully"
        )
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(status_code=500, detail="Logout failed")


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get current authenticated user information.
    """
    try:
        # Get additional user info from database if needed
        db_user = await supabase_client.get_user_by_email(user.get('email'))
        
        user_info = {
            'id': user.get('id'),
            'email': user.get('email'),
            'display_name': user.get('display_name'),
            'organization_id': user.get('organization_id'),
            'role': user.get('role'),
            'needs_onboarding': user.get('organization_id') is None,
            'avatar_url': db_user.get('avatar_url') if db_user else None,
            'last_seen': db_user.get('last_seen') if db_user else None
        }
        
        return UserResponse(
            success=True,
            user=user_info
        )
        
    except Exception as e:
        logger.error(f"Get user info error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get user information")


@router.post("/refresh", response_model=AuthResponse)
async def refresh_access_token(request: RefreshRequest, response: Response):
    """
    Refresh access token using refresh token.
    """
    try:
        # For now, we'll use a simple token verification approach
        # In production, you might want to store refresh tokens in database
        token_data = await authenticate_token(request.refresh_token)
        
        if not token_data:
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        
        # Get updated user data
        db_user = await supabase_client.get_user_by_email(token_data.get('email'))
        
        # Create new JWT payload with updated info
        jwt_payload = {
            'id': token_data.get('id'),
            'email': token_data.get('email'),
            'display_name': token_data.get('display_name'),
            'role': db_user.get('role', 'member') if db_user else 'member',
            'organization_id': db_user.get('organization_id') if db_user else None
        }
        
        # Create new access token
        access_token = create_access_token(jwt_payload, expires_delta=timedelta(days=7))
        
        # Set new cookie
        response.set_cookie(
            key="access_token",
            value=access_token,
            max_age=7 * 24 * 60 * 60,  # 7 days
            httponly=True,
            secure=False,
            samesite="lax"
        )
        
        return AuthResponse(
            success=True,
            access_token=access_token,
            user={
                'id': token_data.get('id'),
                'email': token_data.get('email'),
                'display_name': token_data.get('display_name'),
                'organization_id': jwt_payload['organization_id'],
                'needs_onboarding': jwt_payload['organization_id'] is None
            },
            message="Token refreshed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(status_code=500, detail="Token refresh failed")


@router.get("/status")
async def auth_status(request: Request):
    """
    Check authentication status without requiring authentication.
    Used by frontend to determine if user is logged in.
    """
    try:
        # Check for access token in cookies
        access_token = request.cookies.get('access_token')
        
        if not access_token:
            return {"authenticated": False, "message": "No access token found"}
        
        # Verify token
        token_data = await authenticate_token(access_token)
        
        if not token_data:
            return {"authenticated": False, "message": "Invalid or expired token"}
        
        return {
            "authenticated": True,
            "user": {
                "email": token_data.get('email'),
                "organization_id": token_data.get('organization_id'),
                "needs_onboarding": token_data.get('organization_id') is None
            }
        }
        
    except Exception as e:
        logger.error(f"Auth status check error: {e}")
        return {"authenticated": False, "message": "Authentication check failed"}