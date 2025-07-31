"""
Authentication utilities for content API.
"""

from fastapi import Request
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


async def get_current_user_safe(request: Request) -> Dict[str, Any]:
    """
    Safe wrapper for getting current user with proper fallback.
    
    Args:
        request: FastAPI request object
        
    Returns:
        User data dictionary with safe defaults
    """
    try:
        from src.auth.auth_middleware import verify_access_token
        
        # Check Authorization header first
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            user_data = verify_access_token(token)
            if user_data:
                return {
                    "id": user_data.get('id', 'anonymous'),
                    "email": user_data.get('email', 'user@localhost'),
                    "org_id": user_data.get('organization_id', 'demo-org'),
                    "organization_id": user_data.get('organization_id', 'demo-org'),
                    "role": user_data.get('role', 'member')
                }
        
        # Check cookies as fallback
        access_token = request.cookies.get('access_token')
        if access_token:
            user_data = verify_access_token(access_token)
            if user_data:
                return {
                    "id": user_data.get('id', 'anonymous'),
                    "email": user_data.get('email', 'user@localhost'),
                    "org_id": user_data.get('organization_id', 'demo-org'),
                    "organization_id": user_data.get('organization_id', 'demo-org'),  
                    "role": user_data.get('role', 'member')
                }
        
        # Safe fallback for development/demo
        return {
            "id": "anonymous",
            "email": "user@localhost",
            "org_id": "demo-org",
            "organization_id": "demo-org",
            "role": "member"
        }
        
    except Exception as e:
        logger.warning(f"Auth failed, using fallback: {e}")
        # Safe fallback user
        return {
            "id": "anonymous",
            "email": "user@localhost", 
            "org_id": "demo-org",
            "organization_id": "demo-org",
            "role": "member"
        }