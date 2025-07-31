"""
Authentication middleware for FastAPI application.
Handles JWT token validation and user context.
"""

import jwt
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse
from src.database.supabase_client import supabase_client
import os
from dotenv import load_dotenv

# Load production environment variables
load_dotenv('.env.production')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

security = HTTPBearer(auto_error=False)

class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware for protecting routes."""
    
    def __init__(self, app, excluded_paths=None):
        super().__init__(app)
        self.excluded_paths = excluded_paths or [
            '/login',
            '/auth/',
            '/api/auth/',
            '/api/gsc/auth/',  # Allow GSC OAuth flow
            '/api/content/',   # Allow content API for demo
            '/onboarding',
            '/static/',
            '/docs',
            '/openapi.json',
            '/health'
        ]
    
    async def dispatch(self, request: Request, call_next):
        """Process request through authentication middleware."""
        path = request.url.path
        
        # Skip authentication for excluded paths
        excluded = any(path.startswith(excluded) for excluded in self.excluded_paths)
        if excluded:
            logger.debug(f"Path {path} excluded from auth, proceeding")
            return await call_next(request)
        
        logger.debug(f"Path {path} requires auth check")
        
        # Check for authentication token in headers or cookies
        auth_header = request.headers.get('Authorization')
        access_token = None
        
        if auth_header and auth_header.startswith('Bearer '):
            access_token = auth_header.split(' ')[1]
        else:
            # Check for token in cookies
            access_token = request.cookies.get('access_token')
        
        logger.debug(f"Auth check for {path}: token={'***' if access_token else None}, cookies={list(request.cookies.keys())}")
        
        if not access_token:
            # Redirect to login for web requests (including HEAD requests from browsers)
            if (request.headers.get('accept', '').startswith('text/html') or 
                request.method == 'HEAD'):
                logger.debug(f"No token found, redirecting to login for {path}")
                return RedirectResponse(url='/login')
            else:
                raise HTTPException(status_code=401, detail="Authentication required")
        
        try:
            user_data = await authenticate_token(access_token)
            
            if not user_data:
                logger.debug(f"Token validation failed for {path}")
                if (request.headers.get('accept', '').startswith('text/html') or 
                    request.method == 'HEAD'):
                    return RedirectResponse(url='/login')
                else:
                    raise HTTPException(status_code=401, detail="Invalid or expired token")
            
            logger.debug(f"Authentication successful for {path}: user={user_data.get('email')}")
            
            # Add user context to request state
            request.state.user = user_data
            request.state.organization_id = user_data.get('organization_id')
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            if (request.headers.get('accept', '').startswith('text/html') or 
                request.method == 'HEAD'):
                return RedirectResponse(url='/login')
            else:
                raise HTTPException(status_code=401, detail="Authentication failed")
        
        return await call_next(request)


async def authenticate_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Authenticate JWT token and return user data.
    
    Args:
        token: JWT token string
        
    Returns:
        User data dict if valid, None otherwise
    """
    try:
        # Production authentication only
        
        # Decode and validate JWT token
        secret_key = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
            
        try:
            payload = jwt.decode(token, secret_key, algorithms=["HS256"])
            
            # Extract user data from JWT payload
            user_data = {
                'id': payload.get('id'),
                'email': payload.get('email'),
                'display_name': payload.get('display_name'),
                'organization_id': payload.get('organization_id'),
                'role': payload.get('role', 'member'),
                'avatar_url': payload.get('avatar_url')
            }
            
            # Debug logging
            logger.info(f"JWT payload decoded: organization_id={payload.get('organization_id')}, email={payload.get('email')}")
            
            return user_data
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
        
    except Exception as e:
        logger.error(f"Token authentication failed: {e}")
        return None


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """
    Dependency to get current authenticated user.
    
    Args:
        request: FastAPI request object
        credentials: HTTP authorization credentials
        
    Returns:
        User data dictionary
        
    Raises:
        HTTPException: If authentication fails
    """
    access_token = None
    
    # Check Authorization header first
    if credentials:
        access_token = credentials.credentials
    else:
        # Check cookies if no Authorization header
        access_token = request.cookies.get('access_token')
    
    if not access_token:
        logger.debug(f"No token found in request to {request.url.path}")
        raise HTTPException(status_code=401, detail="Authentication required")
    
    user_data = await authenticate_token(access_token)
    if not user_data:
        logger.debug(f"Token validation failed for {request.url.path}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    logger.debug(f"Authentication successful for {request.url.path}: user={user_data.get('email')}")
    return user_data


async def get_current_organization(user: Dict[str, Any] = Depends(get_current_user)) -> str:
    """
    Dependency to get current user's organization ID.
    
    Args:
        user: Current authenticated user
        
    Returns:
        Organization ID string
        
    Raises:
        HTTPException: If no organization context
    """
    organization_id = user.get('organization_id')
    if not organization_id:
        raise HTTPException(status_code=403, detail="No organization context")
    
    return organization_id


def create_access_token(user_data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token for user.
    
    Args:
        user_data: User information to encode
        expires_delta: Token expiration time
        
    Returns:
        JWT token string
    """
    to_encode = user_data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=7)
    
    to_encode.update({"exp": expire})
    
    # Use a secret key from environment or default
    secret_key = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
    
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm="HS256")
    return encoded_jwt


def verify_access_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify and decode JWT access token.
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded token data if valid, None otherwise
    """
    try:
        secret_key = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
        payload = jwt.decode(token, secret_key, algorithms=["HS256"])
        return payload
    except jwt.PyJWTError as e:
        logger.error(f"JWT verification failed: {e}")
        return None


class RequireRole:
    """Dependency class to require specific user roles."""
    
    def __init__(self, allowed_roles: list):
        self.allowed_roles = allowed_roles
    
    def __call__(self, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        """
        Check if user has required role.
        
        Args:
            user: Current authenticated user
            
        Returns:
            User data if authorized
            
        Raises:
            HTTPException: If user doesn't have required role
        """
        user_role = user.get('role', 'viewer')
        if user_role not in self.allowed_roles:
            raise HTTPException(
                status_code=403, 
                detail=f"Insufficient permissions. Required: {self.allowed_roles}, Got: {user_role}"
            )
        return user


# Common role dependencies
require_admin = RequireRole(['admin'])
require_member = RequireRole(['admin', 'member'])
require_viewer = RequireRole(['admin', 'member', 'viewer'])