"""
Integration Management API for the SEO Content Knowledge Graph System.

This module provides RESTful integration management endpoints with OAuth flows,
third-party service connections, and integration lifecycle management.
"""

import asyncio
import json
import secrets
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlencode, parse_qs
import hashlib
import base64
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, Query, Request, BackgroundTasks
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel, Field, validator, AnyUrl
import structlog
import httpx
from cryptography.fernet import Fernet

from database.neo4j_client import Neo4jClient
from src.services.google_drive_service import GoogleDriveService
from src.services.competitor_monitoring import CompetitorMonitoringService
from config.settings import get_settings
from auth.dependencies import get_current_user, get_current_tenant, require_admin_role
from utils.cache import cache_response
from utils.rate_limiting import rate_limit

logger = structlog.get_logger(__name__)

# Create router
integrations_router = APIRouter(prefix="/integrations", tags=["integrations"])

# Supported integration types
SUPPORTED_INTEGRATIONS = {
    "google_drive": {
        "name": "Google Drive",
        "description": "Sync content briefs from Google Drive",
        "oauth_enabled": True,
        "required_scopes": ["https://www.googleapis.com/auth/drive.readonly"],
        "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token"
    },
    "google_analytics": {
        "name": "Google Analytics",
        "description": "Import website analytics data",
        "oauth_enabled": True,
        "required_scopes": ["https://www.googleapis.com/auth/analytics.readonly"],
        "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token"
    },
    "google_search_console": {
        "name": "Google Search Console",
        "description": "Import search performance data",
        "oauth_enabled": True,
        "required_scopes": ["https://www.googleapis.com/auth/webmasters.readonly"],
        "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token"
    },
    "slack": {
        "name": "Slack",
        "description": "Send notifications to Slack channels",
        "oauth_enabled": True,
        "required_scopes": ["chat:write", "channels:read"],
        "auth_url": "https://slack.com/oauth/v2/authorize",
        "token_url": "https://slack.com/api/oauth.v2.access"
    },
    "webhook": {
        "name": "Webhook",
        "description": "Send HTTP webhooks for events",
        "oauth_enabled": False,
        "required_fields": ["url", "secret"]
    },
    "api_key": {
        "name": "API Key",
        "description": "Third-party API key authentication",
        "oauth_enabled": False,
        "required_fields": ["api_key", "base_url"]
    }
}


# =============================================================================
# Models
# =============================================================================

class IntegrationStatus(str, Enum):
    """Integration status enum."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    EXPIRED = "expired"
    ERROR = "error"


class IntegrationConfig(BaseModel):
    """Integration configuration model."""
    
    integration_id: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    name: str = Field(..., description="Integration name")
    integration_type: str = Field(..., description="Type of integration")
    
    # OAuth configuration
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    redirect_uri: Optional[str] = None
    scopes: List[str] = Field(default_factory=list)
    
    # API configuration
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    
    # Webhook configuration
    webhook_url: Optional[str] = None
    webhook_secret: Optional[str] = None
    
    # Custom configuration
    custom_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Status and metadata
    status: IntegrationStatus = IntegrationStatus.INACTIVE
    enabled: bool = True
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None
    last_sync: Optional[datetime] = None
    
    # Tenant information
    tenant_id: str = Field(..., description="Tenant identifier")
    created_by: str = Field(..., description="User who created integration")
    
    def is_oauth_type(self) -> bool:
        """Check if integration uses OAuth."""
        return SUPPORTED_INTEGRATIONS.get(self.integration_type, {}).get("oauth_enabled", False)


class OAuthToken(BaseModel):
    """OAuth token model."""
    
    token_id: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    integration_id: str = Field(..., description="Associated integration ID")
    
    # Token data
    access_token: str = Field(..., description="Access token")
    refresh_token: Optional[str] = None
    token_type: str = "Bearer"
    expires_in: Optional[int] = None
    scope: Optional[str] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    
    # Tenant information
    tenant_id: str = Field(..., description="Tenant identifier")
    
    def is_expired(self) -> bool:
        """Check if token is expired."""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) >= self.expires_at


class IntegrationResponse(BaseModel):
    """Integration response model."""
    
    integration_id: str
    name: str
    integration_type: str
    status: IntegrationStatus
    enabled: bool
    oauth_enabled: bool
    last_sync: Optional[datetime] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    config_valid: bool
    error_message: Optional[str] = None


class OAuthAuthorizationResponse(BaseModel):
    """OAuth authorization response model."""
    
    authorization_url: str
    state: str
    code_verifier: Optional[str] = None  # For PKCE
    expires_at: datetime


class IntegrationSyncResult(BaseModel):
    """Integration sync result model."""
    
    integration_id: str
    sync_id: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    records_processed: int = 0
    records_created: int = 0
    records_updated: int = 0
    records_failed: int = 0
    error_message: Optional[str] = None


class IntegrationWebhookEvent(BaseModel):
    """Integration webhook event model."""
    
    event_id: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    integration_id: str
    event_type: str
    payload: Dict[str, Any]
    headers: Dict[str, str]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    processed: bool = False
    tenant_id: str


# =============================================================================
# Utility Functions
# =============================================================================

def get_encryption_key() -> bytes:
    """Get encryption key for sensitive data."""
    settings = get_settings()
    key = settings.encryption_key or Fernet.generate_key()
    return key.encode() if isinstance(key, str) else key


def encrypt_sensitive_data(data: str) -> str:
    """Encrypt sensitive data."""
    fernet = Fernet(get_encryption_key())
    return fernet.encrypt(data.encode()).decode()


def decrypt_sensitive_data(encrypted_data: str) -> str:
    """Decrypt sensitive data."""
    fernet = Fernet(get_encryption_key())
    return fernet.decrypt(encrypted_data.encode()).decode()


def generate_pkce_challenge() -> tuple[str, str]:
    """Generate PKCE code verifier and challenge."""
    code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip('=')
    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode()).digest()
    ).decode().rstrip('=')
    return code_verifier, code_challenge


# =============================================================================
# Dependencies
# =============================================================================

async def get_neo4j_client() -> Neo4jClient:
    """Get Neo4j client."""
    settings = get_settings()
    return Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)


# =============================================================================
# Integration Management Endpoints
# =============================================================================

@integrations_router.get("/types", response_model=Dict[str, Any])
@cache_response(ttl=3600)  # Cache for 1 hour
async def get_integration_types():
    """Get supported integration types."""
    return {
        "supported_integrations": SUPPORTED_INTEGRATIONS
    }


@integrations_router.get("/", response_model=List[IntegrationResponse])
@cache_response(ttl=300)  # Cache for 5 minutes
async def get_integrations(
    integration_type: Optional[str] = Query(None, description="Filter by integration type"),
    status: Optional[IntegrationStatus] = Query(None, description="Filter by status"),
    neo4j_client: Neo4jClient = Depends(get_neo4j_client),
    current_tenant: str = Depends(get_current_tenant)
):
    """Get integrations for tenant."""
    try:
        # Build query
        query = """
        MATCH (i:Integration)
        WHERE i.tenant_id = $tenant_id
        """
        
        params = {"tenant_id": current_tenant}
        
        if integration_type:
            query += " AND i.integration_type = $integration_type"
            params["integration_type"] = integration_type
        
        if status:
            query += " AND i.status = $status"
            params["status"] = status.value
        
        query += " RETURN i ORDER BY i.created_at DESC"
        
        # Execute query
        result = await neo4j_client.execute_query(query, **params)
        
        integrations = []
        for record in result:
            integration_data = record['i']
            
            # Check if config is valid
            config_valid = await _validate_integration_config(integration_data)
            
            integrations.append(IntegrationResponse(
                integration_id=integration_data['integration_id'],
                name=integration_data['name'],
                integration_type=integration_data['integration_type'],
                status=IntegrationStatus(integration_data['status']),
                enabled=integration_data.get('enabled', True),
                oauth_enabled=SUPPORTED_INTEGRATIONS.get(
                    integration_data['integration_type'], {}
                ).get('oauth_enabled', False),
                last_sync=integration_data.get('last_sync'),
                created_at=integration_data['created_at'],
                updated_at=integration_data.get('updated_at'),
                config_valid=config_valid
            ))
        
        return integrations
        
    except Exception as e:
        logger.error(f"Failed to get integrations: {e}")
        raise HTTPException(status_code=500, detail="Failed to get integrations")


@integrations_router.post("/", response_model=Dict[str, str])
@rate_limit(requests_per_minute=10)
async def create_integration(
    integration_data: Dict[str, Any],
    neo4j_client: Neo4jClient = Depends(get_neo4j_client),
    current_user: Dict[str, Any] = Depends(get_current_user),
    current_tenant: str = Depends(get_current_tenant)
):
    """Create new integration."""
    try:
        # Validate integration type
        integration_type = integration_data.get('integration_type')
        if integration_type not in SUPPORTED_INTEGRATIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported integration type: {integration_type}"
            )
        
        # Create integration config
        config = IntegrationConfig(
            tenant_id=current_tenant,
            created_by=current_user["user_id"],
            **integration_data
        )
        
        # Encrypt sensitive data
        encrypted_config = config.dict()
        if config.client_secret:
            encrypted_config['client_secret'] = encrypt_sensitive_data(config.client_secret)
        if config.api_key:
            encrypted_config['api_key'] = encrypt_sensitive_data(config.api_key)
        if config.webhook_secret:
            encrypted_config['webhook_secret'] = encrypt_sensitive_data(config.webhook_secret)
        
        # Save to Neo4j
        query = """
        CREATE (i:Integration)
        SET i += $config
        RETURN i.integration_id as integration_id
        """
        
        result = await neo4j_client.execute_query(query, config=encrypted_config)
        integration_id = result[0]['integration_id']
        
        return {
            "message": "Integration created successfully",
            "integration_id": integration_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create integration: {e}")
        raise HTTPException(status_code=500, detail="Failed to create integration")


@integrations_router.get("/{integration_id}", response_model=Dict[str, Any])
@cache_response(ttl=60)  # Cache for 1 minute
async def get_integration(
    integration_id: str,
    neo4j_client: Neo4jClient = Depends(get_neo4j_client),
    current_tenant: str = Depends(get_current_tenant)
):
    """Get specific integration."""
    try:
        query = """
        MATCH (i:Integration)
        WHERE i.integration_id = $integration_id AND i.tenant_id = $tenant_id
        RETURN i
        """
        
        result = await neo4j_client.execute_query(
            query,
            integration_id=integration_id,
            tenant_id=current_tenant
        )
        
        if not result:
            raise HTTPException(status_code=404, detail="Integration not found")
        
        integration_data = result[0]['i']
        
        # Decrypt sensitive data for display (mask secrets)
        if integration_data.get('client_secret'):
            integration_data['client_secret'] = "***masked***"
        if integration_data.get('api_key'):
            integration_data['api_key'] = "***masked***"
        if integration_data.get('webhook_secret'):
            integration_data['webhook_secret'] = "***masked***"
        
        return integration_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get integration: {e}")
        raise HTTPException(status_code=500, detail="Failed to get integration")


@integrations_router.put("/{integration_id}")
@rate_limit(requests_per_minute=20)
async def update_integration(
    integration_id: str,
    update_data: Dict[str, Any],
    neo4j_client: Neo4jClient = Depends(get_neo4j_client),
    current_user: Dict[str, Any] = Depends(get_current_user),
    current_tenant: str = Depends(get_current_tenant)
):
    """Update integration."""
    try:
        # Get existing integration
        query = """
        MATCH (i:Integration)
        WHERE i.integration_id = $integration_id AND i.tenant_id = $tenant_id
        RETURN i
        """
        
        result = await neo4j_client.execute_query(
            query,
            integration_id=integration_id,
            tenant_id=current_tenant
        )
        
        if not result:
            raise HTTPException(status_code=404, detail="Integration not found")
        
        # Encrypt sensitive data if provided
        if 'client_secret' in update_data and update_data['client_secret']:
            update_data['client_secret'] = encrypt_sensitive_data(update_data['client_secret'])
        if 'api_key' in update_data and update_data['api_key']:
            update_data['api_key'] = encrypt_sensitive_data(update_data['api_key'])
        if 'webhook_secret' in update_data and update_data['webhook_secret']:
            update_data['webhook_secret'] = encrypt_sensitive_data(update_data['webhook_secret'])
        
        # Add update timestamp
        update_data['updated_at'] = datetime.now(timezone.utc)
        
        # Update integration
        update_query = """
        MATCH (i:Integration)
        WHERE i.integration_id = $integration_id AND i.tenant_id = $tenant_id
        SET i += $update_data
        RETURN i
        """
        
        await neo4j_client.execute_query(
            update_query,
            integration_id=integration_id,
            tenant_id=current_tenant,
            update_data=update_data
        )
        
        return {"message": "Integration updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update integration: {e}")
        raise HTTPException(status_code=500, detail="Failed to update integration")


@integrations_router.delete("/{integration_id}")
@rate_limit(requests_per_minute=10)
async def delete_integration(
    integration_id: str,
    neo4j_client: Neo4jClient = Depends(get_neo4j_client),
    current_user: Dict[str, Any] = Depends(get_current_user),
    current_tenant: str = Depends(get_current_tenant)
):
    """Delete integration."""
    try:
        # Delete integration and related tokens
        query = """
        MATCH (i:Integration)
        WHERE i.integration_id = $integration_id AND i.tenant_id = $tenant_id
        OPTIONAL MATCH (i)-[:HAS_TOKEN]->(t:OAuthToken)
        DETACH DELETE i, t
        RETURN count(i) as deleted_count
        """
        
        result = await neo4j_client.execute_query(
            query,
            integration_id=integration_id,
            tenant_id=current_tenant
        )
        
        if result[0]['deleted_count'] == 0:
            raise HTTPException(status_code=404, detail="Integration not found")
        
        return {"message": "Integration deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete integration: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete integration")


# =============================================================================
# OAuth Flow Endpoints
# =============================================================================

@integrations_router.post("/{integration_id}/oauth/authorize", response_model=OAuthAuthorizationResponse)
@rate_limit(requests_per_minute=30)
async def start_oauth_flow(
    integration_id: str,
    request: Request,
    neo4j_client: Neo4jClient = Depends(get_neo4j_client),
    current_user: Dict[str, Any] = Depends(get_current_user),
    current_tenant: str = Depends(get_current_tenant)
):
    """Start OAuth authorization flow."""
    try:
        # Get integration
        query = """
        MATCH (i:Integration)
        WHERE i.integration_id = $integration_id AND i.tenant_id = $tenant_id
        RETURN i
        """
        
        result = await neo4j_client.execute_query(
            query,
            integration_id=integration_id,
            tenant_id=current_tenant
        )
        
        if not result:
            raise HTTPException(status_code=404, detail="Integration not found")
        
        integration_data = result[0]['i']
        integration_type = integration_data['integration_type']
        
        # Check if integration supports OAuth
        if not SUPPORTED_INTEGRATIONS.get(integration_type, {}).get('oauth_enabled'):
            raise HTTPException(
                status_code=400,
                detail="Integration does not support OAuth"
            )
        
        # Get OAuth configuration
        oauth_config = SUPPORTED_INTEGRATIONS[integration_type]
        
        # Generate state parameter
        state = secrets.token_urlsafe(32)
        
        # Generate PKCE challenge for enhanced security
        code_verifier, code_challenge = generate_pkce_challenge()
        
        # Build authorization URL
        auth_params = {
            'client_id': decrypt_sensitive_data(integration_data['client_id']),
            'redirect_uri': integration_data['redirect_uri'],
            'scope': ' '.join(integration_data.get('scopes', oauth_config['required_scopes'])),
            'response_type': 'code',
            'state': state,
            'access_type': 'offline',  # For refresh tokens
            'prompt': 'consent'
        }
        
        # Add PKCE parameters
        auth_params.update({
            'code_challenge': code_challenge,
            'code_challenge_method': 'S256'
        })
        
        authorization_url = f"{oauth_config['auth_url']}?{urlencode(auth_params)}"
        
        # Store state and code verifier
        oauth_state = {
            'state': state,
            'code_verifier': code_verifier,
            'integration_id': integration_id,
            'user_id': current_user['user_id'],
            'tenant_id': current_tenant,
            'created_at': datetime.now(timezone.utc),
            'expires_at': datetime.now(timezone.utc) + timedelta(minutes=10)
        }
        
        # Store OAuth state
        state_query = """
        CREATE (s:OAuthState)
        SET s += $oauth_state
        """
        
        await neo4j_client.execute_query(state_query, oauth_state=oauth_state)
        
        return OAuthAuthorizationResponse(
            authorization_url=authorization_url,
            state=state,
            code_verifier=code_verifier,
            expires_at=oauth_state['expires_at']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start OAuth flow: {e}")
        raise HTTPException(status_code=500, detail="Failed to start OAuth flow")


@integrations_router.get("/oauth/callback")
@rate_limit(requests_per_minute=30)
async def oauth_callback(
    code: str = Query(..., description="Authorization code"),
    state: str = Query(..., description="State parameter"),
    error: Optional[str] = Query(None, description="OAuth error"),
    neo4j_client: Neo4jClient = Depends(get_neo4j_client)
):
    """Handle OAuth callback."""
    try:
        # Check for OAuth errors
        if error:
            logger.error(f"OAuth error: {error}")
            raise HTTPException(status_code=400, detail=f"OAuth error: {error}")
        
        # Get stored OAuth state
        state_query = """
        MATCH (s:OAuthState)
        WHERE s.state = $state
        RETURN s
        """
        
        result = await neo4j_client.execute_query(state_query, state=state)
        
        if not result:
            raise HTTPException(status_code=400, detail="Invalid state parameter")
        
        oauth_state = result[0]['s']
        
        # Check if state is expired
        if datetime.now(timezone.utc) > oauth_state['expires_at']:
            raise HTTPException(status_code=400, detail="OAuth state expired")
        
        # Get integration
        integration_query = """
        MATCH (i:Integration)
        WHERE i.integration_id = $integration_id
        RETURN i
        """
        
        result = await neo4j_client.execute_query(
            integration_query,
            integration_id=oauth_state['integration_id']
        )
        
        if not result:
            raise HTTPException(status_code=404, detail="Integration not found")
        
        integration_data = result[0]['i']
        integration_type = integration_data['integration_type']
        oauth_config = SUPPORTED_INTEGRATIONS[integration_type]
        
        # Exchange code for tokens
        token_data = {
            'client_id': decrypt_sensitive_data(integration_data['client_id']),
            'client_secret': decrypt_sensitive_data(integration_data['client_secret']),
            'code': code,
            'grant_type': 'authorization_code',
            'redirect_uri': integration_data['redirect_uri'],
            'code_verifier': oauth_state['code_verifier']
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                oauth_config['token_url'],
                data=token_data,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )
            
            if response.status_code != 200:
                logger.error(f"Token exchange failed: {response.text}")
                raise HTTPException(status_code=400, detail="Token exchange failed")
            
            token_response = response.json()
        
        # Create OAuth token
        oauth_token = OAuthToken(
            integration_id=oauth_state['integration_id'],
            access_token=token_response['access_token'],
            refresh_token=token_response.get('refresh_token'),
            token_type=token_response.get('token_type', 'Bearer'),
            expires_in=token_response.get('expires_in'),
            scope=token_response.get('scope'),
            tenant_id=oauth_state['tenant_id']
        )
        
        # Calculate expiration time
        if oauth_token.expires_in:
            oauth_token.expires_at = datetime.now(timezone.utc) + timedelta(seconds=oauth_token.expires_in)
        
        # Encrypt tokens
        token_data_encrypted = oauth_token.dict()
        token_data_encrypted['access_token'] = encrypt_sensitive_data(oauth_token.access_token)
        if oauth_token.refresh_token:
            token_data_encrypted['refresh_token'] = encrypt_sensitive_data(oauth_token.refresh_token)
        
        # Save token and update integration status
        save_token_query = """
        MATCH (i:Integration)
        WHERE i.integration_id = $integration_id
        CREATE (t:OAuthToken)
        SET t += $token_data
        CREATE (i)-[:HAS_TOKEN]->(t)
        SET i.status = 'active', i.updated_at = datetime()
        """
        
        await neo4j_client.execute_query(
            save_token_query,
            integration_id=oauth_state['integration_id'],
            token_data=token_data_encrypted
        )
        
        # Clean up OAuth state
        cleanup_query = """
        MATCH (s:OAuthState)
        WHERE s.state = $state
        DELETE s
        """
        
        await neo4j_client.execute_query(cleanup_query, state=state)
        
        # Return success response
        return JSONResponse(
            content={
                "message": "OAuth authorization successful",
                "integration_id": oauth_state['integration_id'],
                "status": "active"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OAuth callback failed: {e}")
        raise HTTPException(status_code=500, detail="OAuth callback failed")


@integrations_router.post("/{integration_id}/oauth/refresh")
@rate_limit(requests_per_minute=30)
async def refresh_oauth_token(
    integration_id: str,
    neo4j_client: Neo4jClient = Depends(get_neo4j_client),
    current_tenant: str = Depends(get_current_tenant)
):
    """Refresh OAuth token."""
    try:
        # Get integration and token
        query = """
        MATCH (i:Integration)-[:HAS_TOKEN]->(t:OAuthToken)
        WHERE i.integration_id = $integration_id AND i.tenant_id = $tenant_id
        RETURN i, t
        """
        
        result = await neo4j_client.execute_query(
            query,
            integration_id=integration_id,
            tenant_id=current_tenant
        )
        
        if not result:
            raise HTTPException(status_code=404, detail="Integration or token not found")
        
        integration_data = result[0]['i']
        token_data = result[0]['t']
        
        # Check if refresh token exists
        if not token_data.get('refresh_token'):
            raise HTTPException(status_code=400, detail="No refresh token available")
        
        # Refresh token
        integration_type = integration_data['integration_type']
        oauth_config = SUPPORTED_INTEGRATIONS[integration_type]
        
        refresh_data = {
            'client_id': decrypt_sensitive_data(integration_data['client_id']),
            'client_secret': decrypt_sensitive_data(integration_data['client_secret']),
            'refresh_token': decrypt_sensitive_data(token_data['refresh_token']),
            'grant_type': 'refresh_token'
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                oauth_config['token_url'],
                data=refresh_data,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )
            
            if response.status_code != 200:
                logger.error(f"Token refresh failed: {response.text}")
                # Mark integration as expired
                await neo4j_client.execute_query(
                    "MATCH (i:Integration) WHERE i.integration_id = $integration_id SET i.status = 'expired'",
                    integration_id=integration_id
                )
                raise HTTPException(status_code=400, detail="Token refresh failed")
            
            token_response = response.json()
        
        # Update token
        update_data = {
            'access_token': encrypt_sensitive_data(token_response['access_token']),
            'token_type': token_response.get('token_type', 'Bearer'),
            'expires_in': token_response.get('expires_in'),
            'scope': token_response.get('scope'),
            'updated_at': datetime.now(timezone.utc)
        }
        
        # Update refresh token if provided
        if token_response.get('refresh_token'):
            update_data['refresh_token'] = encrypt_sensitive_data(token_response['refresh_token'])
        
        # Calculate new expiration
        if token_response.get('expires_in'):
            update_data['expires_at'] = datetime.now(timezone.utc) + timedelta(seconds=token_response['expires_in'])
        
        # Update token in database
        update_query = """
        MATCH (i:Integration)-[:HAS_TOKEN]->(t:OAuthToken)
        WHERE i.integration_id = $integration_id AND i.tenant_id = $tenant_id
        SET t += $update_data, i.status = 'active', i.updated_at = datetime()
        """
        
        await neo4j_client.execute_query(
            update_query,
            integration_id=integration_id,
            tenant_id=current_tenant,
            update_data=update_data
        )
        
        return {"message": "Token refreshed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to refresh token: {e}")
        raise HTTPException(status_code=500, detail="Failed to refresh token")


# =============================================================================
# Integration Sync Endpoints
# =============================================================================

@integrations_router.post("/{integration_id}/sync")
@rate_limit(requests_per_minute=10)
async def sync_integration(
    integration_id: str,
    background_tasks: BackgroundTasks,
    sync_options: Dict[str, Any] = {},
    neo4j_client: Neo4jClient = Depends(get_neo4j_client),
    current_user: Dict[str, Any] = Depends(get_current_user),
    current_tenant: str = Depends(get_current_tenant)
):
    """Sync integration data."""
    try:
        # Get integration
        query = """
        MATCH (i:Integration)
        WHERE i.integration_id = $integration_id AND i.tenant_id = $tenant_id
        RETURN i
        """
        
        result = await neo4j_client.execute_query(
            query,
            integration_id=integration_id,
            tenant_id=current_tenant
        )
        
        if not result:
            raise HTTPException(status_code=404, detail="Integration not found")
        
        integration_data = result[0]['i']
        
        # Check if integration is active
        if integration_data['status'] != 'active':
            raise HTTPException(
                status_code=400,
                detail="Integration is not active"
            )
        
        # Create sync record
        sync_result = IntegrationSyncResult(
            integration_id=integration_id,
            sync_id=secrets.token_urlsafe(32),
            status="started",
            started_at=datetime.now(timezone.utc)
        )
        
        # Start sync in background
        background_tasks.add_task(
            _perform_integration_sync,
            integration_data,
            sync_result,
            sync_options,
            current_user["user_id"],
            current_tenant
        )
        
        return {
            "message": "Integration sync started",
            "sync_id": sync_result.sync_id,
            "status": "started"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start integration sync: {e}")
        raise HTTPException(status_code=500, detail="Failed to start integration sync")


@integrations_router.get("/{integration_id}/sync/status")
@cache_response(ttl=30)  # Cache for 30 seconds
async def get_sync_status(
    integration_id: str,
    neo4j_client: Neo4jClient = Depends(get_neo4j_client),
    current_tenant: str = Depends(get_current_tenant)
):
    """Get integration sync status."""
    try:
        # Get recent sync records
        query = """
        MATCH (s:IntegrationSync)
        WHERE s.integration_id = $integration_id AND s.tenant_id = $tenant_id
        RETURN s
        ORDER BY s.started_at DESC
        LIMIT 10
        """
        
        result = await neo4j_client.execute_query(
            query,
            integration_id=integration_id,
            tenant_id=current_tenant
        )
        
        sync_records = [record['s'] for record in result]
        
        return {
            "integration_id": integration_id,
            "sync_records": sync_records,
            "last_sync": sync_records[0] if sync_records else None
        }
        
    except Exception as e:
        logger.error(f"Failed to get sync status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get sync status")


# =============================================================================
# Webhook Endpoints
# =============================================================================

@integrations_router.post("/webhooks/{integration_id}")
@rate_limit(requests_per_minute=100)
async def receive_webhook(
    integration_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    neo4j_client: Neo4jClient = Depends(get_neo4j_client)
):
    """Receive webhook from integration."""
    try:
        # Get integration
        query = """
        MATCH (i:Integration)
        WHERE i.integration_id = $integration_id
        RETURN i
        """
        
        result = await neo4j_client.execute_query(query, integration_id=integration_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Integration not found")
        
        integration_data = result[0]['i']
        
        # Verify webhook signature if configured
        if integration_data.get('webhook_secret'):
            signature = request.headers.get('X-Signature')
            if not signature:
                raise HTTPException(status_code=400, detail="Missing webhook signature")
            
            # Verify signature (implementation depends on service)
            webhook_secret = decrypt_sensitive_data(integration_data['webhook_secret'])
            body = await request.body()
            
            # Basic HMAC verification (adjust for specific services)
            import hmac
            expected_signature = hmac.new(
                webhook_secret.encode(),
                body,
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(signature, expected_signature):
                raise HTTPException(status_code=400, detail="Invalid webhook signature")
        
        # Parse webhook payload
        payload = await request.json()
        
        # Create webhook event
        webhook_event = IntegrationWebhookEvent(
            integration_id=integration_id,
            event_type=payload.get('event_type', 'unknown'),
            payload=payload,
            headers=dict(request.headers),
            tenant_id=integration_data['tenant_id']
        )
        
        # Process webhook in background
        background_tasks.add_task(
            _process_webhook_event,
            webhook_event,
            integration_data
        )
        
        return {"message": "Webhook received successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to receive webhook: {e}")
        raise HTTPException(status_code=500, detail="Failed to receive webhook")


# =============================================================================
# Helper Functions
# =============================================================================

async def _validate_integration_config(integration_data: Dict[str, Any]) -> bool:
    """Validate integration configuration."""
    try:
        integration_type = integration_data.get('integration_type')
        
        if integration_type not in SUPPORTED_INTEGRATIONS:
            return False
        
        integration_spec = SUPPORTED_INTEGRATIONS[integration_type]
        
        # Check OAuth configuration
        if integration_spec.get('oauth_enabled'):
            required_fields = ['client_id', 'client_secret', 'redirect_uri']
            for field in required_fields:
                if not integration_data.get(field):
                    return False
        
        # Check required fields
        required_fields = integration_spec.get('required_fields', [])
        for field in required_fields:
            if not integration_data.get(field):
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to validate integration config: {e}")
        return False


async def _perform_integration_sync(
    integration_data: Dict[str, Any],
    sync_result: IntegrationSyncResult,
    sync_options: Dict[str, Any],
    user_id: str,
    tenant_id: str
):
    """Perform integration sync."""
    try:
        integration_type = integration_data['integration_type']
        
        # Perform sync based on integration type
        if integration_type == 'google_drive':
            await _sync_google_drive(integration_data, sync_result, sync_options, tenant_id)
        elif integration_type == 'google_analytics':
            await _sync_google_analytics(integration_data, sync_result, sync_options, tenant_id)
        elif integration_type == 'google_search_console':
            await _sync_google_search_console(integration_data, sync_result, sync_options, tenant_id)
        else:
            raise ValueError(f"Unsupported integration type: {integration_type}")
        
        # Update sync result
        sync_result.status = "completed"
        sync_result.completed_at = datetime.now(timezone.utc)
        
    except Exception as e:
        logger.error(f"Integration sync failed: {e}")
        sync_result.status = "failed"
        sync_result.error_message = str(e)
        sync_result.completed_at = datetime.now(timezone.utc)
    
    finally:
        # Save sync result
        await _save_sync_result(sync_result, tenant_id)


async def _sync_google_drive(
    integration_data: Dict[str, Any],
    sync_result: IntegrationSyncResult,
    sync_options: Dict[str, Any],
    tenant_id: str
):
    """Sync Google Drive integration."""
    # This would integrate with the GoogleDriveService
    # For now, return placeholder implementation
    sync_result.records_processed = 10
    sync_result.records_created = 5
    sync_result.records_updated = 3
    sync_result.records_failed = 2


async def _sync_google_analytics(
    integration_data: Dict[str, Any],
    sync_result: IntegrationSyncResult,
    sync_options: Dict[str, Any],
    tenant_id: str
):
    """Sync Google Analytics integration."""
    # Placeholder implementation
    sync_result.records_processed = 20
    sync_result.records_created = 15
    sync_result.records_updated = 5
    sync_result.records_failed = 0


async def _sync_google_search_console(
    integration_data: Dict[str, Any],
    sync_result: IntegrationSyncResult,
    sync_options: Dict[str, Any],
    tenant_id: str
):
    """Sync Google Search Console integration."""
    # Placeholder implementation
    sync_result.records_processed = 30
    sync_result.records_created = 25
    sync_result.records_updated = 5
    sync_result.records_failed = 0


async def _save_sync_result(sync_result: IntegrationSyncResult, tenant_id: str):
    """Save sync result to database."""
    try:
        settings = get_settings()
        neo4j_client = Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
        
        sync_data = sync_result.dict()
        sync_data['tenant_id'] = tenant_id
        
        query = """
        CREATE (s:IntegrationSync)
        SET s += $sync_data
        """
        
        await neo4j_client.execute_query(query, sync_data=sync_data)
        
    except Exception as e:
        logger.error(f"Failed to save sync result: {e}")


async def _process_webhook_event(
    webhook_event: IntegrationWebhookEvent,
    integration_data: Dict[str, Any]
):
    """Process webhook event."""
    try:
        # Process webhook based on integration type and event type
        integration_type = integration_data['integration_type']
        event_type = webhook_event.event_type
        
        logger.info(f"Processing webhook event: {integration_type} - {event_type}")
        
        # Mark as processed
        webhook_event.processed = True
        
        # Save webhook event
        await _save_webhook_event(webhook_event)
        
    except Exception as e:
        logger.error(f"Failed to process webhook event: {e}")


async def _save_webhook_event(webhook_event: IntegrationWebhookEvent):
    """Save webhook event to database."""
    try:
        settings = get_settings()
        neo4j_client = Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
        
        event_data = webhook_event.dict()
        
        query = """
        CREATE (e:WebhookEvent)
        SET e += $event_data
        """
        
        await neo4j_client.execute_query(query, event_data=event_data)
        
    except Exception as e:
        logger.error(f"Failed to save webhook event: {e}")


# =============================================================================
# Health Check Endpoint
# =============================================================================

@integrations_router.get("/health")
async def health_check():
    """Health check for integrations service."""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "supported_integrations": len(SUPPORTED_INTEGRATIONS)
        }
        
    except Exception as e:
        logger.error(f"Integration service health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e)
        }


# Export the router
__all__ = ["integrations_router"]