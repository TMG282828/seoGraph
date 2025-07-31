"""
Authentication service for user management and Google OAuth integration.
"""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple, Any
import json
import jwt
from passlib.context import CryptContext
from google.oauth2 import id_token
from google.auth.transport import requests
from google_auth_oauthlib.flow import Flow
import httpx

from config.settings import Settings
from models.user_models import User, UserSession, UserRegistrationRequest, UserLoginRequest
import structlog

logger = structlog.get_logger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    """Authentication service for user management and OAuth."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.google_flow = None
        self._init_google_oauth()
    
    def _init_google_oauth(self):
        """Initialize Google OAuth flow."""
        if self.settings.google_client_id and self.settings.google_client_secret:
            client_config = {
                "web": {
                    "client_id": self.settings.google_client_id,
                    "client_secret": self.settings.google_client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": [self.settings.google_redirect_uri]
                }
            }
            
            self.google_flow = Flow.from_client_config(
                client_config,
                scopes=[
                    "openid",
                    "email",
                    "profile",
                    "https://www.googleapis.com/auth/drive.readonly",
                    "https://www.googleapis.com/auth/drive.file"
                ]
            )
            self.google_flow.redirect_uri = self.settings.google_redirect_uri
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def generate_tokens(self, user: User) -> Tuple[str, str]:
        """Generate access and refresh tokens for a user."""
        # Access token (expires in 30 minutes)
        access_payload = {
            "user_id": user.id,
            "email": user.email,
            "exp": datetime.utcnow() + timedelta(minutes=self.settings.access_token_expire_minutes),
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        # Refresh token (expires in 30 days)
        refresh_payload = {
            "user_id": user.id,
            "email": user.email,
            "exp": datetime.utcnow() + timedelta(days=self.settings.refresh_token_expire_days),
            "iat": datetime.utcnow(),
            "type": "refresh"
        }
        
        access_token = jwt.encode(access_payload, self.settings.secret_key, algorithm="HS256")
        refresh_token = jwt.encode(refresh_payload, self.settings.secret_key, algorithm="HS256")
        
        return access_token, refresh_token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, self.settings.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Generate a new access token using refresh token."""
        payload = self.verify_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            return None
        
        # Generate new access token
        access_payload = {
            "user_id": payload["user_id"],
            "email": payload["email"],
            "exp": datetime.utcnow() + timedelta(minutes=self.settings.access_token_expire_minutes),
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        return jwt.encode(access_payload, self.settings.secret_key, algorithm="HS256")
    
    def get_google_auth_url(self, state: Optional[str] = None) -> str:
        """Get Google OAuth authorization URL."""
        if not self.google_flow:
            raise ValueError("Google OAuth not configured")
        
        auth_url, _ = self.google_flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            state=state
        )
        return auth_url
    
    async def handle_google_callback(self, code: str, state: Optional[str] = None) -> Dict[str, Any]:
        """Handle Google OAuth callback."""
        if not self.google_flow:
            raise ValueError("Google OAuth not configured")
        
        try:
            # Exchange code for tokens
            self.google_flow.fetch_token(code=code)
            credentials = self.google_flow.credentials
            
            # Get user info from Google
            user_info = await self._get_google_user_info(credentials.token)
            
            return {
                "access_token": credentials.token,
                "refresh_token": credentials.refresh_token,
                "user_info": user_info,
                "credentials": credentials
            }
        except Exception as e:
            logger.error(f"Google OAuth callback error: {str(e)}")
            raise
    
    async def _get_google_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get user information from Google API."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            response.raise_for_status()
            return response.json()
    
    def create_session(self, user: User, ip_address: Optional[str] = None, user_agent: Optional[str] = None) -> UserSession:
        """Create a new user session."""
        session = UserSession(
            user_id=user.id,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=3600)  # 1 hour
        )
        return session
    
    def validate_session(self, session: UserSession) -> bool:
        """Validate if a session is still active."""
        if not session.is_active:
            return False
        
        if session.is_expired():
            return False
        
        return True
    
    def generate_password_reset_token(self, user: User) -> str:
        """Generate password reset token."""
        payload = {
            "user_id": user.id,
            "email": user.email,
            "exp": datetime.utcnow() + timedelta(hours=1),  # 1 hour expiry
            "iat": datetime.utcnow(),
            "type": "password_reset"
        }
        return jwt.encode(payload, self.settings.secret_key, algorithm="HS256")
    
    def verify_password_reset_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify password reset token."""
        payload = self.verify_token(token)
        if not payload or payload.get("type") != "password_reset":
            return None
        return payload
    
    def generate_email_verification_token(self, user: User) -> str:
        """Generate email verification token."""
        payload = {
            "user_id": user.id,
            "email": user.email,
            "exp": datetime.utcnow() + timedelta(days=7),  # 7 days expiry
            "iat": datetime.utcnow(),
            "type": "email_verification"
        }
        return jwt.encode(payload, self.settings.secret_key, algorithm="HS256")
    
    def verify_email_verification_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify email verification token."""
        payload = self.verify_token(token)
        if not payload or payload.get("type") != "email_verification":
            return None
        return payload


class GoogleDriveService:
    """Service for Google Drive integration."""
    
    def __init__(self, credentials):
        self.credentials = credentials
    
    async def list_files(self, page_token: Optional[str] = None, page_size: int = 10) -> Dict[str, Any]:
        """List files from Google Drive."""
        params = {
            "pageSize": page_size,
            "fields": "nextPageToken, files(id, name, mimeType, size, modifiedTime, webViewLink, thumbnailLink, parents)"
        }
        
        if page_token:
            params["pageToken"] = page_token
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://www.googleapis.com/drive/v3/files",
                headers={"Authorization": f"Bearer {self.credentials.token}"},
                params=params
            )
            response.raise_for_status()
            return response.json()
    
    async def get_file_content(self, file_id: str) -> bytes:
        """Get file content from Google Drive."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media",
                headers={"Authorization": f"Bearer {self.credentials.token}"}
            )
            response.raise_for_status()
            return response.content
    
    async def get_file_metadata(self, file_id: str) -> Dict[str, Any]:
        """Get file metadata from Google Drive."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://www.googleapis.com/drive/v3/files/{file_id}",
                headers={"Authorization": f"Bearer {self.credentials.token}"},
                params={"fields": "id, name, mimeType, size, modifiedTime, webViewLink, thumbnailLink, parents"}
            )
            response.raise_for_status()
            return response.json()


# In-memory storage for demo purposes
# In production, use a proper database
class InMemoryUserStore:
    """In-memory user storage for demo purposes."""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, UserSession] = {}
        self.email_to_user_id: Dict[str, str] = {}
        
        # Create a demo user
        demo_user = User(
            id="demo-user-123",
            email="demo@example.com",
            first_name="Demo",
            last_name="User",
            status="active",
            email_verified=True
        )
        self.users[demo_user.id] = demo_user
        self.email_to_user_id[demo_user.email] = demo_user.id
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        user_id = self.email_to_user_id.get(email.lower())
        if user_id:
            return self.users.get(user_id)
        return None
    
    def create_user(self, user: User) -> User:
        """Create a new user."""
        self.users[user.id] = user
        self.email_to_user_id[user.email.lower()] = user.id
        return user
    
    def update_user(self, user: User) -> User:
        """Update an existing user."""
        if user.id in self.users:
            self.users[user.id] = user
            self.email_to_user_id[user.email.lower()] = user.id
        return user
    
    def create_session(self, session: UserSession) -> UserSession:
        """Create a new session."""
        self.sessions[session.session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get session by ID."""
        return self.sessions.get(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False


# Global user store instance
user_store = InMemoryUserStore()