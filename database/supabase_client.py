"""
Supabase client for the SEO Content Knowledge Graph System.

This module provides authentication, user management, and multi-tenant
data operations using Supabase as the backend service.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone

import structlog
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions
from postgrest.exceptions import APIError
from gotrue.errors import AuthError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from config.settings import get_settings

logger = structlog.get_logger(__name__)


class SupabaseConnectionError(Exception):
    """Raised when Supabase connection fails."""
    pass


class SupabaseAuthError(Exception):
    """Raised when Supabase authentication fails."""
    pass


class SupabaseDataError(Exception):
    """Raised when Supabase data operations fail."""
    pass


class SupabaseClient:
    """
    Supabase client for authentication and multi-tenant data management.
    
    Provides methods for user authentication, tenant management, and
    data operations with Row Level Security (RLS) for multi-tenancy.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
        service_role_key: Optional[str] = None,
        jwt_secret: Optional[str] = None,
    ):
        """
        Initialize Supabase client.

        Args:
            url: Supabase project URL
            key: Supabase anon key
            service_role_key: Supabase service role key
            jwt_secret: JWT secret for token validation
        """
        settings = get_settings()
        
        self.url = url or settings.supabase_url
        self.key = key or settings.supabase_key
        self.service_role_key = service_role_key or settings.supabase_service_role_key
        self.jwt_secret = jwt_secret or settings.supabase_jwt_secret
        
        self._client: Optional[Client] = None
        self._service_client: Optional[Client] = None
        self._is_connected = False

    def connect(self) -> None:
        """
        Establish connection to Supabase.
        
        Raises:
            SupabaseConnectionError: If connection fails
        """
        try:
            # Regular client with anon key
            self._client = create_client(
                supabase_url=self.url,
                supabase_key=self.key,
                options=ClientOptions(
                    auto_refresh_token=True,
                    persist_session=True,
                )
            )
            
            # Service role client for admin operations
            if self.service_role_key:
                self._service_client = create_client(
                    supabase_url=self.url,
                    supabase_key=self.service_role_key,
                    options=ClientOptions(
                        auto_refresh_token=False,
                        persist_session=False,
                    )
                )
            
            self._is_connected = True
            
            logger.info(
                "Supabase connection established",
                url=self.url,
                has_service_client=self._service_client is not None,
            )
            
        except Exception as e:
            logger.error(
                "Failed to connect to Supabase",
                url=self.url,
                error=str(e),
            )
            raise SupabaseConnectionError(f"Failed to connect to Supabase: {e}") from e

    def _ensure_connected(self) -> None:
        """Ensure client is connected."""
        if not self._client:
            raise SupabaseConnectionError("Client not initialized. Call connect() first.")

    def _get_service_client(self) -> Client:
        """Get service role client for admin operations."""
        if not self._service_client:
            raise SupabaseConnectionError("Service role client not available")
        return self._service_client

    # =============================================================================
    # Authentication
    # =============================================================================

    async def sign_up(
        self,
        email: str,
        password: str,
        user_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Sign up a new user.
        
        Args:
            email: User email
            password: User password
            user_metadata: Additional user metadata
            
        Returns:
            User data and session information
            
        Raises:
            SupabaseAuthError: If sign up fails
        """
        self._ensure_connected()
        
        try:
            response = self._client.auth.sign_up({
                "email": email,
                "password": password,
                "options": {
                    "data": user_metadata or {}
                }
            })
            
            if response.user is None:
                raise SupabaseAuthError("Sign up failed: No user returned")
            
            logger.info(
                "User signed up successfully",
                user_id=response.user.id,
                email=email,
            )
            
            return {
                "user": response.user,
                "session": response.session,
            }
            
        except AuthError as e:
            logger.error("Supabase sign up failed", email=email, error=str(e))
            raise SupabaseAuthError(f"Sign up failed: {e}") from e

    async def sign_in(self, email: str, password: str) -> Dict[str, Any]:
        """
        Sign in user with email and password.
        
        Args:
            email: User email
            password: User password
            
        Returns:
            User data and session information
            
        Raises:
            SupabaseAuthError: If sign in fails
        """
        self._ensure_connected()
        
        try:
            response = self._client.auth.sign_in_with_password({
                "email": email,
                "password": password,
            })
            
            if response.user is None or response.session is None:
                raise SupabaseAuthError("Sign in failed: Invalid credentials")
            
            logger.info(
                "User signed in successfully",
                user_id=response.user.id,
                email=email,
            )
            
            return {
                "user": response.user,
                "session": response.session,
            }
            
        except AuthError as e:
            logger.error("Supabase sign in failed", email=email, error=str(e))
            raise SupabaseAuthError(f"Sign in failed: {e}") from e

    async def sign_out(self) -> bool:
        """
        Sign out current user.
        
        Returns:
            True if sign out was successful
        """
        self._ensure_connected()
        
        try:
            self._client.auth.sign_out()
            logger.info("User signed out successfully")
            return True
            
        except AuthError as e:
            logger.error("Supabase sign out failed", error=str(e))
            return False

    async def refresh_session(self) -> Optional[Dict[str, Any]]:
        """
        Refresh current user session.
        
        Returns:
            Refreshed session information or None if failed
        """
        self._ensure_connected()
        
        try:
            response = self._client.auth.refresh_session()
            
            if response.session is None:
                return None
            
            logger.debug("Session refreshed successfully")
            
            return {
                "user": response.user,
                "session": response.session,
            }
            
        except AuthError as e:
            logger.error("Session refresh failed", error=str(e))
            return None

    async def get_current_user(self) -> Optional[Dict[str, Any]]:
        """
        Get current authenticated user.
        
        Returns:
            Current user data or None if not authenticated
        """
        self._ensure_connected()
        
        try:
            user = self._client.auth.get_user()
            return user if user else None
            
        except AuthError as e:
            logger.error("Failed to get current user", error=str(e))
            return None

    async def update_user(
        self,
        user_attributes: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Update current user attributes.
        
        Args:
            user_attributes: User attributes to update
            
        Returns:
            Updated user data or None if failed
        """
        self._ensure_connected()
        
        try:
            response = self._client.auth.update_user(user_attributes)
            
            logger.info(
                "User updated successfully",
                user_id=response.user.id if response.user else None,
            )
            
            return {
                "user": response.user,
            } if response.user else None
            
        except AuthError as e:
            logger.error("User update failed", error=str(e))
            return None

    # =============================================================================
    # Tenant Management
    # =============================================================================

    async def create_tenant(
        self,
        tenant_data: Dict[str, Any],
        owner_user_id: str,
    ) -> Optional[str]:
        """
        Create a new tenant.
        
        Args:
            tenant_data: Tenant information
            owner_user_id: Owner user ID
            
        Returns:
            Tenant ID if created successfully
            
        Raises:
            SupabaseDataError: If tenant creation fails
        """
        service_client = self._get_service_client()
        
        try:
            # Prepare tenant data
            tenant_record = {
                **tenant_data,
                "owner_user_id": owner_user_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            
            # Insert tenant
            response = service_client.table("tenants").insert(tenant_record).execute()
            
            if not response.data:
                raise SupabaseDataError("Failed to create tenant: No data returned")
            
            tenant_id = response.data[0]["id"]
            
            # Create user-tenant association
            await self.add_user_to_tenant(
                user_id=owner_user_id,
                tenant_id=tenant_id,
                role="owner",
            )
            
            logger.info(
                "Tenant created successfully",
                tenant_id=tenant_id,
                owner_user_id=owner_user_id,
            )
            
            return tenant_id
            
        except APIError as e:
            logger.error("Failed to create tenant", error=str(e))
            raise SupabaseDataError(f"Failed to create tenant: {e}") from e

    async def get_tenant(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """
        Get tenant information.
        
        Args:
            tenant_id: Tenant ID
            
        Returns:
            Tenant data or None if not found
        """
        self._ensure_connected()
        
        try:
            response = self._client.table("tenants").select("*").eq("id", tenant_id).execute()
            
            if not response.data:
                return None
            
            return response.data[0]
            
        except APIError as e:
            logger.error("Failed to get tenant", tenant_id=tenant_id, error=str(e))
            return None

    async def get_user_tenants(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get tenants for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of tenant data
        """
        self._ensure_connected()
        
        try:
            # Get user-tenant associations
            response = (
                self._client.table("user_tenants")
                .select("tenant_id, role, tenants(*)")
                .eq("user_id", user_id)
                .execute()
            )
            
            tenants = []
            for record in response.data:
                tenant_data = record["tenants"]
                tenant_data["user_role"] = record["role"]
                tenants.append(tenant_data)
            
            logger.debug(
                "Retrieved user tenants",
                user_id=user_id,
                tenant_count=len(tenants),
            )
            
            return tenants
            
        except APIError as e:
            logger.error("Failed to get user tenants", user_id=user_id, error=str(e))
            return []

    async def add_user_to_tenant(
        self,
        user_id: str,
        tenant_id: str,
        role: str = "member",
    ) -> bool:
        """
        Add user to tenant.
        
        Args:
            user_id: User ID
            tenant_id: Tenant ID
            role: User role in tenant
            
        Returns:
            True if user was added successfully
        """
        service_client = self._get_service_client()
        
        try:
            user_tenant_record = {
                "user_id": user_id,
                "tenant_id": tenant_id,
                "role": role,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            
            response = service_client.table("user_tenants").insert(user_tenant_record).execute()
            
            success = bool(response.data)
            
            logger.info(
                "User added to tenant",
                user_id=user_id,
                tenant_id=tenant_id,
                role=role,
                success=success,
            )
            
            return success
            
        except APIError as e:
            logger.error(
                "Failed to add user to tenant",
                user_id=user_id,
                tenant_id=tenant_id,
                error=str(e),
            )
            return False

    async def remove_user_from_tenant(
        self,
        user_id: str,
        tenant_id: str,
    ) -> bool:
        """
        Remove user from tenant.
        
        Args:
            user_id: User ID
            tenant_id: Tenant ID
            
        Returns:
            True if user was removed successfully
        """
        service_client = self._get_service_client()
        
        try:
            response = (
                service_client.table("user_tenants")
                .delete()
                .eq("user_id", user_id)
                .eq("tenant_id", tenant_id)
                .execute()
            )
            
            success = bool(response.data)
            
            logger.info(
                "User removed from tenant",
                user_id=user_id,
                tenant_id=tenant_id,
                success=success,
            )
            
            return success
            
        except APIError as e:
            logger.error(
                "Failed to remove user from tenant",
                user_id=user_id,
                tenant_id=tenant_id,
                error=str(e),
            )
            return False

    # =============================================================================
    # Content Management
    # =============================================================================

    async def create_content_record(
        self,
        content_data: Dict[str, Any],
        tenant_id: str,
        user_id: str,
    ) -> Optional[str]:
        """
        Create content record in Supabase.
        
        Args:
            content_data: Content data
            tenant_id: Tenant ID
            user_id: User ID
            
        Returns:
            Content ID if created successfully
        """
        self._ensure_connected()
        
        try:
            # Set the user context for RLS
            self._client.auth.set_session(self._client.auth.get_session())
            
            content_record = {
                **content_data,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            
            response = self._client.table("content").insert(content_record).execute()
            
            if not response.data:
                raise SupabaseDataError("Failed to create content: No data returned")
            
            content_id = response.data[0]["id"]
            
            logger.info(
                "Content record created",
                content_id=content_id,
                tenant_id=tenant_id,
                user_id=user_id,
            )
            
            return content_id
            
        except APIError as e:
            logger.error("Failed to create content record", error=str(e))
            raise SupabaseDataError(f"Failed to create content: {e}") from e

    async def get_content_record(
        self,
        content_id: str,
        tenant_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get content record.
        
        Args:
            content_id: Content ID
            tenant_id: Tenant ID
            
        Returns:
            Content data or None if not found
        """
        self._ensure_connected()
        
        try:
            response = (
                self._client.table("content")
                .select("*")
                .eq("id", content_id)
                .eq("tenant_id", tenant_id)
                .execute()
            )
            
            if not response.data:
                return None
            
            return response.data[0]
            
        except APIError as e:
            logger.error(
                "Failed to get content record",
                content_id=content_id,
                tenant_id=tenant_id,
                error=str(e),
            )
            return None

    async def list_content_records(
        self,
        tenant_id: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List content records for tenant.
        
        Args:
            tenant_id: Tenant ID
            filters: Additional filters
            limit: Maximum number of records
            offset: Offset for pagination
            
        Returns:
            List of content records
        """
        self._ensure_connected()
        
        try:
            query = self._client.table("content").select("*").eq("tenant_id", tenant_id)
            
            # Apply filters
            if filters:
                for key, value in filters.items():
                    if key == "status":
                        query = query.eq("status", value)
                    elif key == "content_type":
                        query = query.eq("content_type", value)
                    elif key == "author_id":
                        query = query.eq("author_id", value)
            
            # Apply pagination
            query = query.range(offset, offset + limit - 1)
            
            response = query.execute()
            
            logger.debug(
                "Listed content records",
                tenant_id=tenant_id,
                count=len(response.data),
                filters=filters,
            )
            
            return response.data
            
        except APIError as e:
            logger.error(
                "Failed to list content records",
                tenant_id=tenant_id,
                error=str(e),
            )
            return []

    async def update_content_record(
        self,
        content_id: str,
        tenant_id: str,
        updates: Dict[str, Any],
    ) -> bool:
        """
        Update content record.
        
        Args:
            content_id: Content ID
            tenant_id: Tenant ID
            updates: Fields to update
            
        Returns:
            True if update was successful
        """
        self._ensure_connected()
        
        try:
            updates["updated_at"] = datetime.now(timezone.utc).isoformat()
            
            response = (
                self._client.table("content")
                .update(updates)
                .eq("id", content_id)
                .eq("tenant_id", tenant_id)
                .execute()
            )
            
            success = bool(response.data)
            
            logger.info(
                "Content record updated",
                content_id=content_id,
                tenant_id=tenant_id,
                success=success,
            )
            
            return success
            
        except APIError as e:
            logger.error(
                "Failed to update content record",
                content_id=content_id,
                tenant_id=tenant_id,
                error=str(e),
            )
            return False

    async def delete_content_record(
        self,
        content_id: str,
        tenant_id: str,
    ) -> bool:
        """
        Delete content record.
        
        Args:
            content_id: Content ID
            tenant_id: Tenant ID
            
        Returns:
            True if deletion was successful
        """
        self._ensure_connected()
        
        try:
            response = (
                self._client.table("content")
                .delete()
                .eq("id", content_id)
                .eq("tenant_id", tenant_id)
                .execute()
            )
            
            success = bool(response.data)
            
            logger.info(
                "Content record deleted",
                content_id=content_id,
                tenant_id=tenant_id,
                success=success,
            )
            
            return success
            
        except APIError as e:
            logger.error(
                "Failed to delete content record",
                content_id=content_id,
                tenant_id=tenant_id,
                error=str(e),
            )
            return False

    # =============================================================================
    # Analytics and Monitoring
    # =============================================================================

    async def log_agent_activity(
        self,
        tenant_id: str,
        user_id: str,
        agent_type: str,
        activity_data: Dict[str, Any],
    ) -> bool:
        """
        Log agent activity for monitoring.
        
        Args:
            tenant_id: Tenant ID
            user_id: User ID
            agent_type: Type of agent
            activity_data: Activity details
            
        Returns:
            True if logged successfully
        """
        self._ensure_connected()
        
        try:
            log_record = {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "agent_type": agent_type,
                "activity_data": activity_data,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            
            response = self._client.table("agent_activity_logs").insert(log_record).execute()
            
            success = bool(response.data)
            
            logger.debug(
                "Agent activity logged",
                tenant_id=tenant_id,
                agent_type=agent_type,
                success=success,
            )
            
            return success
            
        except APIError as e:
            logger.error(
                "Failed to log agent activity",
                tenant_id=tenant_id,
                agent_type=agent_type,
                error=str(e),
            )
            return False

    def __repr__(self) -> str:
        """String representation of Supabase client."""
        return f"SupabaseClient(url={self.url}, connected={self._is_connected})"


# =============================================================================
# Utility Functions
# =============================================================================

def get_supabase_client() -> SupabaseClient:
    """
    Get a configured Supabase client instance.
    
    Returns:
        Configured SupabaseClient instance
    """
    client = SupabaseClient()
    client.connect()
    return client


async def initialize_supabase_schema() -> None:
    """Initialize Supabase database schema with tables and RLS policies."""
    client = get_supabase_client()
    service_client = client._get_service_client()
    
    # SQL for creating tables and RLS policies
    schema_sql = """
    -- Enable Row Level Security
    ALTER TABLE IF EXISTS tenants ENABLE ROW LEVEL SECURITY;
    ALTER TABLE IF EXISTS user_tenants ENABLE ROW LEVEL SECURITY;
    ALTER TABLE IF EXISTS content ENABLE ROW LEVEL SECURITY;
    ALTER TABLE IF EXISTS agent_activity_logs ENABLE ROW LEVEL SECURITY;
    
    -- Tenants table policies
    CREATE POLICY "Users can view their own tenants" ON tenants
        FOR SELECT USING (
            owner_user_id = auth.uid() OR
            id IN (SELECT tenant_id FROM user_tenants WHERE user_id = auth.uid())
        );
    
    CREATE POLICY "Users can create tenants" ON tenants
        FOR INSERT WITH CHECK (owner_user_id = auth.uid());
    
    CREATE POLICY "Tenant owners can update their tenants" ON tenants
        FOR UPDATE USING (owner_user_id = auth.uid());
    
    -- User-tenants table policies
    CREATE POLICY "Users can view their tenant associations" ON user_tenants
        FOR SELECT USING (user_id = auth.uid());
    
    -- Content table policies
    CREATE POLICY "Users can view content in their tenants" ON content
        FOR SELECT USING (
            tenant_id IN (SELECT tenant_id FROM user_tenants WHERE user_id = auth.uid())
        );
    
    CREATE POLICY "Users can create content in their tenants" ON content
        FOR INSERT WITH CHECK (
            tenant_id IN (SELECT tenant_id FROM user_tenants WHERE user_id = auth.uid()) AND
            user_id = auth.uid()
        );
    
    CREATE POLICY "Users can update their own content" ON content
        FOR UPDATE USING (user_id = auth.uid());
    
    CREATE POLICY "Users can delete their own content" ON content
        FOR DELETE USING (user_id = auth.uid());
    
    -- Agent activity logs policies
    CREATE POLICY "Users can view logs for their tenants" ON agent_activity_logs
        FOR SELECT USING (
            tenant_id IN (SELECT tenant_id FROM user_tenants WHERE user_id = auth.uid())
        );
    
    CREATE POLICY "Users can create activity logs" ON agent_activity_logs
        FOR INSERT WITH CHECK (user_id = auth.uid());
    """
    
    try:
        # Execute schema SQL
        service_client.postgrest.rpc("exec_sql", {"sql": schema_sql}).execute()
        logger.info("Supabase schema initialization completed")
        
    except Exception as e:
        logger.error("Failed to initialize Supabase schema", error=str(e))
        raise


if __name__ == "__main__":
    # Example usage and testing
    def main():
        client = SupabaseClient()
        try:
            client.connect()
            print("Supabase client test completed successfully")
        except Exception as e:
            print(f"Supabase client test failed: {e}")

    main()