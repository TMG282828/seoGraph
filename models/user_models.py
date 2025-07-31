"""
User and tenant-related Pydantic models for the SEO Content Knowledge Graph System.

This module defines data models for user management, authentication,
tenant management, and multi-tenant operations.
"""

import uuid
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator, model_validator, EmailStr
import structlog

logger = structlog.get_logger(__name__)


class UserRole(str, Enum):
    """User roles within the system."""
    
    SUPER_ADMIN = "super_admin"
    TENANT_ADMIN = "tenant_admin"
    CONTENT_MANAGER = "content_manager"
    CONTENT_CREATOR = "content_creator"
    SEO_ANALYST = "seo_analyst"
    EDITOR = "editor"
    REVIEWER = "reviewer"
    VIEWER = "viewer"


class TenantRole(str, Enum):
    """User roles within a specific tenant."""
    
    OWNER = "owner"
    ADMIN = "admin"
    MANAGER = "manager"
    MEMBER = "member"
    VIEWER = "viewer"
    GUEST = "guest"


class UserStatus(str, Enum):
    """User account status."""
    
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"
    DELETED = "deleted"


class TenantStatus(str, Enum):
    """Tenant account status."""
    
    ACTIVE = "active"
    TRIAL = "trial"
    SUSPENDED = "suspended"
    CANCELLED = "cancelled"
    DELETED = "deleted"


class SubscriptionTier(str, Enum):
    """Subscription tier levels."""
    
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class UserPreferences(BaseModel):
    """User preferences and settings."""
    
    language: str = Field("en", description="Preferred language")
    timezone: str = Field("UTC", description="User timezone")
    theme: str = Field("light", description="UI theme preference")
    
    # Notification preferences
    email_notifications: bool = Field(True, description="Enable email notifications")
    push_notifications: bool = Field(True, description="Enable push notifications")
    weekly_reports: bool = Field(True, description="Enable weekly reports")
    
    # Dashboard preferences
    default_tenant: Optional[str] = Field(None, description="Default tenant ID")
    dashboard_widgets: List[str] = Field(default_factory=list, description="Enabled dashboard widgets")
    
    # Content preferences
    default_content_type: Optional[str] = Field(None, description="Default content type")
    auto_save_interval: int = Field(300, ge=60, le=3600, description="Auto-save interval in seconds")
    
    # Analysis preferences
    default_analysis_depth: str = Field("standard", description="Default analysis depth")
    include_competitors: bool = Field(False, description="Include competitors in analysis by default")


class User(BaseModel):
    """User model for authentication and profile management."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique user identifier")
    email: EmailStr = Field(..., description="User email address")
    
    # Personal information
    first_name: str = Field(..., min_length=1, max_length=100, description="First name")
    last_name: str = Field(..., min_length=1, max_length=100, description="Last name")
    display_name: Optional[str] = Field(None, max_length=200, description="Display name")
    avatar_url: Optional[str] = Field(None, description="Avatar image URL")
    
    # Account information
    role: UserRole = Field(UserRole.CONTENT_CREATOR, description="System role")
    status: UserStatus = Field(UserStatus.PENDING_VERIFICATION, description="Account status")
    
    # Preferences and settings
    preferences: UserPreferences = Field(default_factory=UserPreferences, description="User preferences")
    
    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    last_active: Optional[datetime] = Field(None, description="Last activity timestamp")
    
    # Security
    email_verified: bool = Field(False, description="Email verification status")
    two_factor_enabled: bool = Field(False, description="Two-factor authentication enabled")
    
    # Usage tracking
    login_count: int = Field(0, ge=0, description="Total login count")
    content_created_count: int = Field(0, ge=0, description="Total content created")
    
    @validator('display_name')
    def set_display_name(cls, v, values):
        """Set display name from first and last name if not provided."""
        if not v and 'first_name' in values and 'last_name' in values:
            return f"{values['first_name']} {values['last_name']}"
        return v
    
    @validator('email')
    def validate_email(cls, v):
        """Validate email format and normalize."""
        return v.lower().strip()
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_active = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
    
    def can_access_tenant(self, tenant_id: str, user_tenants: List['UserTenant']) -> bool:
        """Check if user can access a specific tenant."""
        return any(ut.tenant_id == tenant_id for ut in user_tenants)
    
    def get_tenant_role(self, tenant_id: str, user_tenants: List['UserTenant']) -> Optional[TenantRole]:
        """Get user role for a specific tenant."""
        for ut in user_tenants:
            if ut.tenant_id == tenant_id:
                return ut.role
        return None


class TenantSettings(BaseModel):
    """Tenant-specific settings and configuration."""
    
    # Branding
    company_name: str = Field(..., min_length=1, max_length=200, description="Company name")
    logo_url: Optional[str] = Field(None, description="Company logo URL")
    brand_colors: Dict[str, str] = Field(default_factory=dict, description="Brand color scheme")
    
    # Content settings
    default_language: str = Field("en", description="Default content language")
    content_approval_required: bool = Field(False, description="Require content approval")
    auto_publish_enabled: bool = Field(False, description="Enable auto-publishing")
    
    # SEO settings
    default_meta_author: Optional[str] = Field(None, description="Default meta author")
    google_analytics_id: Optional[str] = Field(None, description="Google Analytics tracking ID")
    google_search_console_domain: Optional[str] = Field(None, description="Google Search Console domain")
    
    # API settings
    api_rate_limit: int = Field(1000, ge=1, description="API rate limit per hour")
    webhook_urls: List[str] = Field(default_factory=list, description="Webhook URLs")
    
    # Integration settings
    integrations: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Third-party integrations")
    
    # Security settings
    ip_whitelist: List[str] = Field(default_factory=list, description="IP address whitelist")
    session_timeout: int = Field(3600, ge=300, le=86400, description="Session timeout in seconds")
    
    # Notification settings
    admin_email: Optional[EmailStr] = Field(None, description="Admin notification email")
    notification_channels: List[str] = Field(default_factory=list, description="Notification channels")


class WorkspaceSettings(BaseModel):
    """Workspace-specific settings for team collaboration and management."""
    
    # Workspace Profile
    workspace_name: str = Field(..., min_length=1, max_length=200, description="Workspace display name")
    description: Optional[str] = Field(None, max_length=1000, description="Workspace description")
    avatar_url: Optional[str] = Field(None, description="Workspace avatar URL")
    
    # Member Management
    seat_limit: int = Field(5, ge=1, le=1000, description="Maximum number of workspace members")
    default_member_role: TenantRole = Field(TenantRole.MEMBER, description="Default role for new members")
    auto_approve_invites: bool = Field(False, description="Automatically approve invite codes")
    
    # Current Usage (read-only tracking)
    current_seats_used: int = Field(0, ge=0, description="Current number of workspace members")
    storage_used_gb: float = Field(0.0, ge=0.0, description="Current storage usage in GB")
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = Field(None, description="Last settings update")
    
    def get_seat_usage_percentage(self) -> float:
        """Get seat usage as percentage."""
        if self.seat_limit == 0:
            return 0.0
        return min((self.current_seats_used / self.seat_limit) * 100, 100.0)
    
    def can_add_member(self) -> bool:
        """Check if workspace can add more members."""
        return self.current_seats_used < self.seat_limit
    
    def get_available_seats(self) -> int:
        """Get number of available seats."""
        return max(0, self.seat_limit - self.current_seats_used)


class Tenant(BaseModel):
    """Tenant model for multi-tenant architecture."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique tenant identifier")
    name: str = Field(..., min_length=1, max_length=200, description="Tenant name")
    slug: str = Field(..., min_length=2, max_length=50, description="URL-friendly tenant identifier")
    
    # Organization information
    description: Optional[str] = Field(None, max_length=1000, description="Tenant description")
    website: Optional[str] = Field(None, description="Company website")
    industry: Optional[str] = Field(None, description="Industry classification")
    
    # Account information
    status: TenantStatus = Field(TenantStatus.TRIAL, description="Tenant status")
    subscription_tier: SubscriptionTier = Field(SubscriptionTier.FREE, description="Subscription tier")
    
    # Owner information
    owner_user_id: str = Field(..., description="Owner user ID")
    
    # Settings
    settings: TenantSettings = Field(default_factory=TenantSettings, description="Tenant settings")
    
    # Limits and quotas
    max_users: int = Field(5, ge=1, description="Maximum number of users")
    max_content_items: int = Field(100, ge=1, description="Maximum content items")
    max_api_calls_per_month: int = Field(10000, ge=1000, description="Maximum API calls per month")
    storage_limit_gb: float = Field(1.0, ge=0.1, description="Storage limit in GB")
    
    # Usage tracking
    current_users: int = Field(0, ge=0, description="Current number of users")
    current_content_items: int = Field(0, ge=0, description="Current content items")
    api_calls_this_month: int = Field(0, ge=0, description="API calls this month")
    storage_used_gb: float = Field(0.0, ge=0.0, description="Storage used in GB")
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    trial_ends_at: Optional[datetime] = Field(None, description="Trial end date")
    subscription_ends_at: Optional[datetime] = Field(None, description="Subscription end date")
    
    @validator('slug')
    def validate_slug(cls, v):
        """Validate tenant slug format."""
        import re
        v = v.lower().strip()
        if not re.match(r'^[a-z0-9-]+$', v):
            raise ValueError("Slug must contain only lowercase letters, numbers, and hyphens")
        if v.startswith('-') or v.endswith('-'):
            raise ValueError("Slug cannot start or end with a hyphen")
        return v
    
    @validator('website')
    def validate_website(cls, v):
        """Validate website URL format."""
        if v:
            if not v.startswith(('http://', 'https://')):
                v = f"https://{v}"
        return v
    
    def is_active(self) -> bool:
        """Check if tenant is active."""
        return self.status in [TenantStatus.ACTIVE, TenantStatus.TRIAL]
    
    def is_trial_expired(self) -> bool:
        """Check if trial period has expired."""
        if self.status == TenantStatus.TRIAL and self.trial_ends_at:
            return datetime.now(timezone.utc) > self.trial_ends_at
        return False
    
    def can_add_user(self) -> bool:
        """Check if tenant can add more users."""
        return self.current_users < self.max_users
    
    def can_create_content(self) -> bool:
        """Check if tenant can create more content."""
        return self.current_content_items < self.max_content_items
    
    def get_usage_percentage(self, resource: str) -> float:
        """Get usage percentage for a resource."""
        if resource == "users":
            return (self.current_users / self.max_users) * 100
        elif resource == "content":
            return (self.current_content_items / self.max_content_items) * 100
        elif resource == "storage":
            return (self.storage_used_gb / self.storage_limit_gb) * 100
        elif resource == "api_calls":
            return (self.api_calls_this_month / self.max_api_calls_per_month) * 100
        return 0.0


class UserTenant(BaseModel):
    """Association between users and tenants."""
    
    user_id: str = Field(..., description="User identifier")
    tenant_id: str = Field(..., description="Tenant identifier")
    role: TenantRole = Field(..., description="User role in tenant")
    
    # Permissions
    permissions: List[str] = Field(default_factory=list, description="Specific permissions")
    
    # Status
    status: str = Field("active", description="Association status")
    invited_by: Optional[str] = Field(None, description="User ID who sent invitation")
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    joined_at: Optional[datetime] = Field(None, description="When user joined tenant")
    last_accessed: Optional[datetime] = Field(None, description="Last access to tenant")
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        # Role-based permissions
        role_permissions = {
            TenantRole.OWNER: ["*"],  # All permissions
            TenantRole.ADMIN: [
                "users.manage", "content.manage", "settings.manage",
                "analytics.view", "integrations.manage"
            ],
            TenantRole.MANAGER: [
                "content.manage", "content.create", "content.edit",
                "analytics.view", "users.view"
            ],
            TenantRole.MEMBER: [
                "content.create", "content.edit", "content.view",
                "analytics.view"
            ],
            TenantRole.VIEWER: ["content.view", "analytics.view"],
            TenantRole.GUEST: ["content.view"]
        }
        
        # Check role permissions
        if self.role in role_permissions:
            if "*" in role_permissions[self.role] or permission in role_permissions[self.role]:
                return True
        
        # Check specific permissions
        return permission in self.permissions


class UserInvitation(BaseModel):
    """User invitation to join a tenant."""
    
    invitation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: EmailStr = Field(..., description="Invited user email")
    tenant_id: str = Field(..., description="Tenant identifier")
    role: TenantRole = Field(..., description="Invited role")
    
    # Invitation details
    invited_by: str = Field(..., description="User ID who sent invitation")
    message: Optional[str] = Field(None, max_length=500, description="Invitation message")
    
    # Status and expiration
    status: str = Field("pending", description="Invitation status")
    expires_at: datetime = Field(..., description="Invitation expiration")
    
    # Tracking
    sent_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    accepted_at: Optional[datetime] = Field(None, description="Acceptance timestamp")
    declined_at: Optional[datetime] = Field(None, description="Decline timestamp")
    
    @validator('expires_at')
    def set_default_expiration(cls, v):
        """Set default expiration if not provided."""
        if not v:
            # Default to 7 days from now
            return datetime.now(timezone.utc).replace(hour=23, minute=59, second=59) + timedelta(days=7)
        return v
    
    def is_expired(self) -> bool:
        """Check if invitation has expired."""
        return datetime.now(timezone.utc) > self.expires_at
    
    def is_pending(self) -> bool:
        """Check if invitation is still pending."""
        return self.status == "pending" and not self.is_expired()


class UserSession(BaseModel):
    """User session tracking."""
    
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(..., description="User identifier")
    tenant_id: Optional[str] = Field(None, description="Active tenant identifier")
    
    # Session details
    ip_address: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, description="Client user agent")
    device_type: Optional[str] = Field(None, description="Device type")
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = Field(..., description="Session expiration")
    
    # Status
    is_active: bool = Field(True, description="Session active status")
    
    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.now(timezone.utc) > self.expires_at
    
    def extend_session(self, seconds: int = 3600) -> None:
        """Extend session expiration."""
        self.expires_at = datetime.now(timezone.utc) + timedelta(seconds=seconds)
        self.last_activity = datetime.now(timezone.utc)


# =============================================================================
# Request/Response Models
# =============================================================================

class UserRegistrationRequest(BaseModel):
    """User registration request."""
    
    email: EmailStr = Field(..., description="User email")
    password: str = Field(..., min_length=8, description="User password")
    first_name: str = Field(..., min_length=1, max_length=100, description="First name")
    last_name: str = Field(..., min_length=1, max_length=100, description="Last name")
    
    # Optional tenant creation
    create_tenant: bool = Field(False, description="Create new tenant")
    tenant_name: Optional[str] = Field(None, description="Tenant name if creating")
    tenant_slug: Optional[str] = Field(None, description="Tenant slug if creating")
    
    @model_validator(mode='before')
    @classmethod
    def validate_tenant_creation(cls, values):
        """Validate tenant creation fields."""
        if values.get('create_tenant'):
            if not values.get('tenant_name'):
                raise ValueError("tenant_name required when create_tenant is True")
            if not values.get('tenant_slug'):
                raise ValueError("tenant_slug required when create_tenant is True")
        return values


class UserLoginRequest(BaseModel):
    """User login request."""
    
    email: EmailStr = Field(..., description="User email")
    password: str = Field(..., description="User password")
    remember_me: bool = Field(False, description="Remember login")
    tenant_slug: Optional[str] = Field(None, description="Tenant to login to")


class UserUpdateRequest(BaseModel):
    """User profile update request."""
    
    first_name: Optional[str] = Field(None, min_length=1, max_length=100)
    last_name: Optional[str] = Field(None, min_length=1, max_length=100)
    display_name: Optional[str] = Field(None, max_length=200)
    avatar_url: Optional[str] = None
    preferences: Optional[UserPreferences] = None


class TenantCreateRequest(BaseModel):
    """Tenant creation request."""
    
    name: str = Field(..., min_length=1, max_length=200, description="Tenant name")
    slug: str = Field(..., min_length=2, max_length=50, description="Tenant slug")
    description: Optional[str] = Field(None, max_length=1000)
    website: Optional[str] = None
    industry: Optional[str] = None
    settings: Optional[TenantSettings] = None


class TenantUpdateRequest(BaseModel):
    """Tenant update request."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    website: Optional[str] = None
    industry: Optional[str] = None
    settings: Optional[TenantSettings] = None


class UserInviteRequest(BaseModel):
    """User invitation request."""
    
    email: EmailStr = Field(..., description="Email to invite")
    role: TenantRole = Field(..., description="Role to assign")
    message: Optional[str] = Field(None, max_length=500, description="Invitation message")


class AuthResponse(BaseModel):
    """Authentication response."""
    
    user: User
    access_token: str
    refresh_token: Optional[str] = None
    expires_in: int = Field(3600, description="Token expiration in seconds")
    
    # Tenant information
    tenants: List[Dict[str, Any]] = Field(default_factory=list, description="User's tenants")
    current_tenant: Optional[Dict[str, Any]] = None


class WorkspaceSettingsUpdateRequest(BaseModel):
    """Request model for updating workspace settings."""
    
    workspace_name: Optional[str] = Field(None, min_length=1, max_length=200, description="Workspace name")
    description: Optional[str] = Field(None, max_length=1000, description="Workspace description")
    avatar_url: Optional[str] = Field(None, description="Workspace avatar URL")
    seat_limit: Optional[int] = Field(None, ge=1, le=1000, description="Seat limit")
    default_member_role: Optional[TenantRole] = Field(None, description="Default member role")
    auto_approve_invites: Optional[bool] = Field(None, description="Auto-approve invites")


class WorkspaceSettingsResponse(BaseModel):
    """Response model for workspace settings."""
    
    success: bool = True
    settings: WorkspaceSettings
    usage_stats: Dict[str, Any] = Field(default_factory=dict, description="Current usage statistics")
    last_updated: Optional[str] = Field(None, description="Last update timestamp")


# =============================================================================
# Utility Functions
# =============================================================================

def create_user(
    email: str,
    first_name: str,
    last_name: str,
    role: UserRole = UserRole.CONTENT_CREATOR
) -> User:
    """
    Create a new user with default settings.
    
    Args:
        email: User email
        first_name: First name
        last_name: Last name
        role: User role
        
    Returns:
        User instance
    """
    return User(
        email=email,
        first_name=first_name,
        last_name=last_name,
        role=role
    )


def create_tenant(
    name: str,
    slug: str,
    owner_user_id: str,
    subscription_tier: SubscriptionTier = SubscriptionTier.FREE
) -> Tenant:
    """
    Create a new tenant with default settings.
    
    Args:
        name: Tenant name
        slug: Tenant slug
        owner_user_id: Owner user ID
        subscription_tier: Subscription tier
        
    Returns:
        Tenant instance
    """
    # Set trial end date
    trial_ends_at = None
    if subscription_tier == SubscriptionTier.FREE:
        trial_ends_at = datetime.now(timezone.utc) + timedelta(days=14)
    
    return Tenant(
        name=name,
        slug=slug,
        owner_user_id=owner_user_id,
        subscription_tier=subscription_tier,
        trial_ends_at=trial_ends_at
    )


def create_user_tenant_association(
    user_id: str,
    tenant_id: str,
    role: TenantRole = TenantRole.MEMBER
) -> UserTenant:
    """
    Create association between user and tenant.
    
    Args:
        user_id: User identifier
        tenant_id: Tenant identifier
        role: User role in tenant
        
    Returns:
        UserTenant instance
    """
    return UserTenant(
        user_id=user_id,
        tenant_id=tenant_id,
        role=role,
        joined_at=datetime.now(timezone.utc)
    )


if __name__ == "__main__":
    # Example usage
    user = create_user(
        email="john.doe@example.com",
        first_name="John",
        last_name="Doe",
        role=UserRole.CONTENT_MANAGER
    )
    
    tenant = create_tenant(
        name="Example Company",
        slug="example-company",
        owner_user_id=user.id,
        subscription_tier=SubscriptionTier.PROFESSIONAL
    )
    
    user_tenant = create_user_tenant_association(
        user_id=user.id,
        tenant_id=tenant.id,
        role=TenantRole.OWNER
    )
    
    print(f"Created user: {user.display_name} ({user.email})")
    print(f"Created tenant: {tenant.name} ({tenant.slug})")
    print(f"User role in tenant: {user_tenant.role.value}")
    print(f"Can manage content: {user_tenant.has_permission('content.manage')}")
    print(f"Can manage users: {user_tenant.has_permission('users.manage')}")