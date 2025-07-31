"""
Unit tests for workspace settings API routes.

Tests workspace settings configuration, member management settings, and usage tracking.
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

# Import the main application
from web.main import app

# Create test client
client = TestClient(app)


class TestWorkspaceSettingsRoutes:
    """Test workspace settings API endpoints."""
    
    @pytest.fixture
    def mock_auth_user(self):
        """Mock authenticated user."""
        return {
            "id": "test-user-123",
            "email": "test@example.com",
            "display_name": "Test User",
            "organization_id": "demo-org"
        }
    
    @pytest.fixture
    def mock_workspace_settings(self):
        """Mock workspace settings data."""
        return {
            "workspace_name": "Test Workspace",
            "description": "A test workspace for unit tests",
            "avatar_url": "https://example.com/avatar.png",
            "seat_limit": 10,
            "default_member_role": "member",
            "auto_approve_invites": False,
            "current_seats_used": 3,
            "storage_used_gb": 2.5
        }
    
    @pytest.fixture
    def auth_headers(self):
        """Mock authorization headers."""
        return {"Authorization": "Bearer test-token"}
    
    def test_get_workspace_settings_success(self, mock_auth_user, mock_workspace_settings, auth_headers):
        """Test getting workspace settings successfully."""
        
        workspace_id = "workspace-123"
        
        with patch('web.api.workspace_settings_routes.get_current_user', return_value=mock_auth_user):
            response = client.get(f"/api/workspaces/{workspace_id}/settings", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            
            # Check response structure
            assert data["success"] is True
            assert "settings" in data
            assert "usage_stats" in data
            
            # Check settings structure
            settings = data["settings"]
            assert "workspace_name" in settings
            assert "seat_limit" in settings
            assert "current_seats_used" in settings
            
            # Check usage stats structure
            usage_stats = data["usage_stats"]
            assert "seats" in usage_stats
            assert "storage" in usage_stats
            assert "api_calls" in usage_stats
    
    def test_get_workspace_settings_unauthorized(self):
        """Test getting workspace settings without authentication."""
        workspace_id = "workspace-123"
        
        response = client.get(f"/api/workspaces/{workspace_id}/settings")
        assert response.status_code == 401
    
    def test_update_workspace_settings_success(self, mock_auth_user, auth_headers):
        """Test updating workspace settings successfully."""
        
        workspace_id = "workspace-123"
        update_data = {
            "workspace_name": "Updated Workspace Name",
            "description": "Updated description",
            "seat_limit": 25
        }
        
        with patch('web.api.workspace_settings_routes.get_current_user', return_value=mock_auth_user):
            response = client.put(
                f"/api/workspaces/{workspace_id}/settings",
                headers=auth_headers,
                json=update_data
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Check response structure
            assert data["success"] is True
            assert "settings" in data
            assert "usage_stats" in data
            assert "last_updated" in data
            
            # Verify updated values
            settings = data["settings"]
            assert settings["workspace_name"] == update_data["workspace_name"]
            assert settings["description"] == update_data["description"]
            assert settings["seat_limit"] == update_data["seat_limit"]
    
    def test_update_workspace_settings_invalid_data(self, mock_auth_user, auth_headers):
        """Test updating workspace settings with invalid data."""
        
        workspace_id = "workspace-123"
        invalid_data = {
            "seat_limit": -5,  # Invalid negative seat limit
            "workspace_name": ""  # Empty name
        }
        
        with patch('web.api.workspace_settings_routes.get_current_user', return_value=mock_auth_user):
            response = client.put(
                f"/api/workspaces/{workspace_id}/settings",
                headers=auth_headers,
                json=invalid_data
            )
            
            # Should fail validation
            assert response.status_code == 422
    
    def test_get_workspace_profile_success(self, mock_auth_user, auth_headers):
        """Test getting workspace profile successfully."""
        
        workspace_id = "workspace-123"
        
        with patch('web.api.workspace_settings_routes.get_current_user', return_value=mock_auth_user):
            response = client.get(f"/api/workspaces/{workspace_id}/settings/profile", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            
            # Check response structure
            assert data["success"] is True
            assert "profile" in data
            
            # Check profile structure
            profile = data["profile"]
            assert "workspace_name" in profile
            assert "description" in profile
            assert "avatar_url" in profile
            assert "created_at" in profile
    
    def test_get_workspace_usage_success(self, mock_auth_user, auth_headers):
        """Test getting workspace usage successfully."""
        
        workspace_id = "workspace-123"
        
        with patch('web.api.workspace_settings_routes.get_current_user', return_value=mock_auth_user):
            response = client.get(f"/api/workspaces/{workspace_id}/settings/usage", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            
            # Check response structure
            assert data["success"] is True
            assert "usage" in data
            assert "limits" in data
            
            # Check usage structure
            usage = data["usage"]
            assert "seats" in usage
            assert "storage" in usage
            assert "api_calls" in usage
            
            # Check limits structure
            limits = data["limits"]
            assert "seat_limit" in limits
            assert "can_add_member" in limits
            assert "seats_available" in limits
    
    def test_get_member_settings_success(self, mock_auth_user, auth_headers):
        """Test getting member management settings."""
        
        workspace_id = "workspace-123"
        
        with patch('web.api.workspace_settings_routes.get_current_user', return_value=mock_auth_user):
            response = client.get(f"/api/workspaces/{workspace_id}/settings/members", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            
            # Check response structure
            assert data["success"] is True
            assert "member_settings" in data
            
            # Check member settings structure
            member_settings = data["member_settings"]
            assert "seat_limit" in member_settings
            assert "default_member_role" in member_settings
            assert "auto_approve_invites" in member_settings
            assert "current_seats_used" in member_settings
            assert "seats_available" in member_settings
            assert "can_add_member" in member_settings
    
    def test_update_member_settings_success(self, mock_auth_user, auth_headers):
        """Test updating member management settings."""
        
        workspace_id = "workspace-123"
        member_settings = {
            "seat_limit": 50,
            "default_member_role": "admin",
            "auto_approve_invites": True
        }
        
        with patch('web.api.workspace_settings_routes.get_current_user', return_value=mock_auth_user):
            response = client.put(
                f"/api/workspaces/{workspace_id}/settings/members",
                headers=auth_headers,
                json=member_settings
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Check response structure
            assert data["success"] is True
            assert data["message"] == "Member settings updated successfully"
            assert "member_settings" in data
            assert "last_updated" in data
            
            # Verify updated values
            updated_settings = data["member_settings"]
            assert updated_settings["seat_limit"] == 50
            assert updated_settings["default_member_role"] == "admin"
            assert updated_settings["auto_approve_invites"] is True
    
    def test_update_member_settings_invalid_role(self, mock_auth_user, auth_headers):
        """Test updating member settings with invalid role."""
        
        workspace_id = "workspace-123"
        invalid_settings = {
            "default_member_role": "invalid_role"  # Invalid role
        }
        
        with patch('web.api.workspace_settings_routes.get_current_user', return_value=mock_auth_user):
            response = client.put(
                f"/api/workspaces/{workspace_id}/settings/members",
                headers=auth_headers,
                json=invalid_settings
            )
            
            # Should fail validation
            assert response.status_code == 422


class TestWorkspaceSettingsModels:
    """Test workspace settings data models."""
    
    def test_workspace_settings_creation(self):
        """Test creating WorkspaceSettings instance."""
        from models.user_models import WorkspaceSettings, TenantRole
        
        settings = WorkspaceSettings(
            workspace_name="Test Workspace",
            description="A test workspace",
            seat_limit=10,
            default_member_role=TenantRole.MEMBER,
            current_seats_used=3
        )
        
        assert settings.workspace_name == "Test Workspace"
        assert settings.seat_limit == 10
        assert settings.current_seats_used == 3
        assert settings.default_member_role == TenantRole.MEMBER
    
    def test_workspace_settings_usage_calculations(self):
        """Test workspace settings usage calculation methods."""
        from models.user_models import WorkspaceSettings, TenantRole
        
        settings = WorkspaceSettings(
            workspace_name="Test Workspace",
            seat_limit=10,
            current_seats_used=6
        )
        
        # Test usage percentage
        assert settings.get_seat_usage_percentage() == 60.0
        
        # Test can add member
        assert settings.can_add_member() is True
        
        # Test available seats
        assert settings.get_available_seats() == 4
        
        # Test at capacity
        settings.current_seats_used = 10
        assert settings.can_add_member() is False
        assert settings.get_available_seats() == 0
        assert settings.get_seat_usage_percentage() == 100.0
    
    def test_workspace_settings_update_request_validation(self):
        """Test workspace settings update request validation."""
        from models.user_models import WorkspaceSettingsUpdateRequest, TenantRole
        
        # Valid request
        valid_request = WorkspaceSettingsUpdateRequest(
            workspace_name="Updated Workspace",
            seat_limit=25,
            default_member_role=TenantRole.ADMIN
        )
        
        assert valid_request.workspace_name == "Updated Workspace"
        assert valid_request.seat_limit == 25
        assert valid_request.default_member_role == TenantRole.ADMIN
        
        # Partial update request
        partial_request = WorkspaceSettingsUpdateRequest(
            workspace_name="Only Name Updated"
        )
        
        assert partial_request.workspace_name == "Only Name Updated"
        assert partial_request.seat_limit is None
        assert partial_request.default_member_role is None


class TestWorkspaceSettingsIntegration:
    """Integration tests for workspace settings functionality."""
    
    def test_complete_settings_workflow(self, mock_auth_user, auth_headers):
        """Test complete workspace settings management workflow."""
        
        workspace_id = "integration-test-workspace"
        
        with patch('web.api.workspace_settings_routes.get_current_user', return_value=mock_auth_user):
            # 1. Get initial settings
            response = client.get(f"/api/workspaces/{workspace_id}/settings", headers=auth_headers)
            assert response.status_code == 200
            initial_settings = response.json()["settings"]
            
            # 2. Update workspace profile
            profile_update = {
                "workspace_name": "Integration Test Workspace",
                "description": "Updated via integration test"
            }
            
            response = client.put(
                f"/api/workspaces/{workspace_id}/settings",
                headers=auth_headers,
                json=profile_update
            )
            assert response.status_code == 200
            
            # 3. Update member settings
            member_update = {
                "seat_limit": 100,
                "default_member_role": "admin",
                "auto_approve_invites": True
            }
            
            response = client.put(
                f"/api/workspaces/{workspace_id}/settings/members",
                headers=auth_headers,
                json=member_update
            )
            assert response.status_code == 200
            
            # 4. Verify final state
            response = client.get(f"/api/workspaces/{workspace_id}/settings", headers=auth_headers)
            assert response.status_code == 200
            
            final_data = response.json()
            final_settings = final_data["settings"]
            
            # Verify all updates were applied
            assert final_settings["workspace_name"] == "Integration Test Workspace"
            assert final_settings["description"] == "Updated via integration test"
            assert final_settings["seat_limit"] == 100
            assert final_settings["default_member_role"] == "admin"
            assert final_settings["auto_approve_invites"] is True
            
            # Verify usage calculations
            usage_stats = final_data["usage_stats"]
            assert "seats" in usage_stats
            assert usage_stats["seats"]["limit"] == 100


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])