"""
Google OAuth authentication service for SEO Content Knowledge Graph System.
Handles Google Sign-In integration with JWT token generation.
"""

import os
import json
import logging
import httpx
from typing import Dict, Optional, Any
from urllib.parse import urlencode
from fastapi import HTTPException
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load production environment variables
load_dotenv('.env.production')

from .auth_middleware import create_access_token
from ..database.supabase_client import supabase_client

logger = logging.getLogger(__name__)

class GoogleOAuthService:
    """Service for handling Google OAuth authentication."""
    
    def __init__(self):
        self.client_id = os.getenv("GOOGLE_CLIENT_ID")
        self.client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        self.redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/api/auth/google/callback")
        self.scopes = [
            "openid",
            "email", 
            "profile",
            "https://www.googleapis.com/auth/drive.readonly"
        ]
        
        # Validate required configuration
        if not self.client_id or not self.client_secret:
            logger.error("Google OAuth credentials not configured. Please set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET environment variables.")
            raise ValueError("Google OAuth credentials not configured")
    
    def get_authorization_url(self, state: Optional[str] = None) -> str:
        """
        Generate Google OAuth authorization URL.
        
        Args:
            state: Optional state parameter for security
            
        Returns:
            Authorization URL string
        """
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(self.scopes),
            "response_type": "code",
            "access_type": "offline",
            "prompt": "consent"
        }
        
        if state:
            params["state"] = state
        
        auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)
        return auth_url
    
    async def exchange_code_for_tokens(self, code: str) -> Dict[str, Any]:
        """
        Exchange authorization code for access and refresh tokens.
        
        Args:
            code: Authorization code from Google
            
        Returns:
            Token response dictionary
            
        Raises:
            HTTPException: If token exchange fails
        """
        token_data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": self.redirect_uri
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://oauth2.googleapis.com/token",
                    data=token_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                
                if response.status_code != 200:
                    logger.error(f"Token exchange failed: {response.text}")
                    raise HTTPException(
                        status_code=400,
                        detail="Failed to exchange code for tokens"
                    )
                
                return response.json()
                
        except Exception as e:
            logger.error(f"Error during token exchange: {e}")
            raise HTTPException(
                status_code=500,
                detail="Authentication service error"
            )
    
    async def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """
        Get user information from Google using access token.
        
        Args:
            access_token: Google access token
            
        Returns:
            User information dictionary
            
        Raises:
            HTTPException: If user info retrieval fails
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://www.googleapis.com/oauth2/v2/userinfo",
                    headers={"Authorization": f"Bearer {access_token}"}
                )
                
                if response.status_code != 200:
                    logger.error(f"Failed to get user info: {response.text}")
                    raise HTTPException(
                        status_code=400,
                        detail="Failed to retrieve user information"
                    )
                
                return response.json()
                
        except Exception as e:
            logger.error(f"Error getting user info: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve user information"
            )
    
    async def authenticate_user(self, code: str) -> Dict[str, Any]:
        """
        Complete Google OAuth authentication flow.
        
        Args:
            code: Authorization code from Google
            
        Returns:
            Authentication result with JWT tokens and user info
            
        Raises:
            HTTPException: If authentication fails
        """
        try:
            # Exchange code for tokens
            token_response = await self.exchange_code_for_tokens(code)
            access_token = token_response.get("access_token")
            refresh_token = token_response.get("refresh_token")
            
            if not access_token:
                raise HTTPException(
                    status_code=400,
                    detail="No access token received from Google"
                )
            
            # Get user information
            user_info = await self.get_user_info(access_token)
            
            # Check if user exists in our system
            user_data = await self._get_or_create_user(user_info, refresh_token)
            
            # Don't assign fallback organization_id - let user go through onboarding
            if not user_data.get('organization_id'):
                logger.info(f"User {user_data.get('email')} has no organization - will need onboarding")
            
            # Generate our JWT tokens
            logger.info(f"Creating JWT for user: email={user_data.get('email')}, org_id={user_data.get('organization_id')}")
            access_jwt = create_access_token(
                user_data,
                expires_delta=timedelta(hours=24)
            )
            
            refresh_jwt = create_access_token(
                {"user_id": user_data["id"], "type": "refresh"},
                expires_delta=timedelta(days=30)
            )
            
            return {
                "access_token": access_jwt,
                "refresh_token": refresh_jwt,
                "token_type": "bearer",
                "expires_in": 86400,  # 24 hours
                "user": {
                    "id": user_data["id"],
                    "email": user_data["email"],
                    "display_name": user_data["display_name"],
                    "avatar_url": user_data.get("avatar_url"),
                    "organization_id": user_data.get("organization_id"),
                    "role": user_data["role"]
                }
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise HTTPException(
                status_code=500,
                detail="Authentication failed"
            )
    
    async def _get_or_create_user(self, google_user: Dict[str, Any], 
                                 refresh_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Get existing user or create new user from Google profile.
        
        Args:
            google_user: Google user profile data
            refresh_token: Optional Google refresh token
            
        Returns:
            User data dictionary
        """
        email = google_user.get("email")
        if not email:
            raise HTTPException(
                status_code=400,
                detail="No email address provided by Google"
            )
        
        # Check if user exists
        try:
            existing_user = await supabase_client.get_user_by_email(email)
            if existing_user:
                # Update Google tokens if provided
                if refresh_token:
                    await supabase_client.update_user_google_tokens(
                        existing_user["id"],
                        refresh_token
                    )
                return existing_user
        except Exception as e:
            logger.warning(f"Error checking existing user: {e}")
        
        # Create new user
        user_data = {
            "email": email,
            "display_name": google_user.get("name", email.split("@")[0]),
            "avatar_url": google_user.get("picture"),
            "google_id": google_user.get("id"),
            "google_refresh_token": refresh_token,
            "email_verified": google_user.get("verified_email", True),
            "role": "member",  # Default role
            "created_at": datetime.utcnow().isoformat(),
            "last_login": datetime.utcnow().isoformat()
        }
        
        try:
            # Create user in Supabase
            created_user = await supabase_client.create_user(user_data)
            
            # Check if user needs to create/join an organization
            if not created_user.get("organization_id"):
                # For now, create a personal organization for the user
                org_name = f"{created_user['display_name']}'s Organization"
                org_slug = created_user['email'].split('@')[0].lower().replace('.', '-')
                admin_email = created_user['email']
                admin_name = created_user['display_name']
                
                organization = await supabase_client.create_organization(
                    org_name, org_slug, admin_email, admin_name
                )
                
                # Update user with organization
                org_id = organization.get("organization_id")
                created_user["organization_id"] = org_id
                await supabase_client.update_user(
                    created_user["id"],
                    {"organization_id": org_id}
                )
            
            return created_user
            
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to create user account"
            )
    
    async def refresh_access_token(self, google_refresh_token: str) -> Optional[str]:
        """
        Refresh Google access token using refresh token.
        
        Args:
            google_refresh_token: Google refresh token
            
        Returns:
            New access token or None if refresh failed
        """
        token_data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": google_refresh_token,
            "grant_type": "refresh_token"
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://oauth2.googleapis.com/token",
                    data=token_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                
                if response.status_code == 200:
                    token_response = response.json()
                    return token_response.get("access_token")
                else:
                    logger.warning(f"Token refresh failed: {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error refreshing token: {e}")
            return None


# Global instance
google_oauth_service = GoogleOAuthService()