"""
Unit tests for workspace API routes.

Tests workspace creation, management, member collaboration, and invite code system.
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


class TestWorkspaceRoutes:
    """Test workspace API endpoints."""
    
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
    def mock_workspace(self):
        """Mock workspace data."""
        return {
            "id": "workspace-123",
            "name": "Test Workspace",
            "description": "A test workspace",
            "slug": "test-workspace",
            "user_role": "owner",
            "is_current": True,
            "member_count": 1,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
    
    @pytest.fixture
    def auth_headers(self):
        """Mock authorization headers."""
        return {"Authorization": "Bearer test-token"}
    
    def test_get_user_workspaces_success(self, mock_auth_user, mock_workspace, auth_headers):
        """Test getting user's workspaces successfully."""
        
        with patch('web.api.workspace_routes.get_current_user', return_value=mock_auth_user):
            response = client.get("/api/workspaces", headers=auth_headers)
            
            assert response.status_code == 200
            workspaces = response.json()
            
            # Should return at least one workspace (fallback)
            assert len(workspaces) >= 1
            assert isinstance(workspaces, list)
            
            # Check workspace structure
            workspace = workspaces[0]
            assert "id" in workspace
            assert "name" in workspace
            assert "user_role" in workspace
            assert "is_current" in workspace
            assert "member_count" in workspace
    
    def test_get_user_workspaces_unauthorized(self):
        """Test getting workspaces without authentication."""
        response = client.get("/api/workspaces")
        assert response.status_code == 401
    
    def test_create_workspace_success(self, mock_auth_user, auth_headers):
        """Test creating a new workspace successfully."""
        
        workspace_data = {
            "name": "New Test Workspace",
            "description": "A brand new workspace"
        }
        
        with patch('web.api.workspace_routes.get_current_user', return_value=mock_auth_user):
            response = client.post(
                "/api/workspaces",
                headers=auth_headers,
                json=workspace_data
            )
            
            assert response.status_code == 200
            created_workspace = response.json()
            
            # Check created workspace structure
            assert created_workspace["name"] == workspace_data["name"]
            assert created_workspace["description"] == workspace_data["description"]
            assert created_workspace["user_role"] == "owner"
            assert "id" in created_workspace
            assert "slug" in created_workspace
    
    def test_create_workspace_invalid_data(self, mock_auth_user, auth_headers):
        """Test creating workspace with invalid data."""
        
        invalid_data = {
            "name": "",  # Empty name should fail
            "description": "Valid description"
        }
        
        with patch('web.api.workspace_routes.get_current_user', return_value=mock_auth_user):
            response = client.post(
                "/api/workspaces",
                headers=auth_headers,
                json=invalid_data
            )
            
            # Should fail validation
            assert response.status_code == 422
    
    def test_create_workspace_unauthorized(self):
        """Test creating workspace without authentication."""
        workspace_data = {"name": "Test Workspace"}
        
        response = client.post("/api/workspaces", json=workspace_data)
        assert response.status_code == 401
    
    def test_update_workspace_success(self, mock_auth_user, auth_headers):
        """Test updating workspace successfully."""
        
        workspace_id = "workspace-123"
        update_data = {
            "name": "Updated Workspace Name",
            "description": "Updated description"
        }
        
        with patch('web.api.workspace_routes.get_current_user', return_value=mock_auth_user):
            response = client.put(
                f"/api/workspaces/{workspace_id}",
                headers=auth_headers,
                json=update_data
            )
            
            assert response.status_code == 200
            updated_workspace = response.json()
            
            # Check updated fields
            assert updated_workspace["name"] == update_data["name"]
            assert updated_workspace["id"] == workspace_id
    
    def test_delete_workspace_success(self, mock_auth_user, auth_headers):
        """Test deleting workspace successfully."""
        
        workspace_id = "workspace-123"
        
        with patch('web.api.workspace_routes.get_current_user', return_value=mock_auth_user):
            response = client.delete(
                f"/api/workspaces/{workspace_id}",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            result = response.json()
            assert result["success"] is True
    
    def test_switch_workspace_success(self, mock_auth_user, auth_headers):
        """Test switching workspace successfully."""
        
        workspace_id = "workspace-456"
        
        with patch('web.api.workspace_routes.get_current_user', return_value=mock_auth_user):
            with patch('web.api.workspace_routes.create_access_token', return_value="new-token"):
                response = client.post(
                    f"/api/workspaces/{workspace_id}/switch",
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                result = response.json()
                
                assert result["success"] is True
                assert result["workspace_id"] == workspace_id
                assert "access_token" in result
    
    def test_create_invite_code_success(self, mock_auth_user, auth_headers):
        """Test creating invite code successfully."""
        
        workspace_id = "workspace-123"
        invite_data = {
            "expires_in_hours": 168,  # 7 days
            "max_uses": 10
        }
        
        with patch('web.api.workspace_routes.get_current_user', return_value=mock_auth_user):
            response = client.post(
                f"/api/workspaces/{workspace_id}/invite-codes",
                headers=auth_headers,
                json=invite_data
            )
            
            assert response.status_code == 200
            invite_code = response.json()
            
            # Check invite code structure
            assert "code" in invite_code
            assert "expires_at" in invite_code
            assert invite_code["max_uses"] == 10
            assert invite_code["current_uses"] == 0
            assert invite_code["is_active"] is True
    
    def test_get_invite_codes_success(self, mock_auth_user, auth_headers):
        """Test getting workspace invite codes."""
        
        workspace_id = "workspace-123"
        
        with patch('web.api.workspace_routes.get_current_user', return_value=mock_auth_user):
            response = client.get(
                f"/api/workspaces/{workspace_id}/invite-codes",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            invite_codes = response.json()
            assert isinstance(invite_codes, list)
    
    def test_join_workspace_success(self, mock_auth_user, auth_headers):
        """Test joining workspace with valid invite code."""
        
        # First create an invite code in the global store
        from web.api.workspace_routes import invite_codes
        test_code = "TEST123"
        invite_codes[test_code] = {
            "workspace_id": "workspace-456",
            "created_by": "other-user",
            "expires_at": datetime.now(timezone.utc).replace(year=2025),  # Future date
            "max_uses": 10,
            "current_uses": 0,
            "is_active": True,
            "created_at": datetime.now(timezone.utc)
        }
        
        join_data = {"invite_code": test_code}
        
        with patch('web.api.workspace_routes.get_current_user', return_value=mock_auth_user):
            response = client.post(
                "/api/workspaces/join",
                headers=auth_headers,
                json=join_data
            )
            
            assert response.status_code == 200
            joined_workspace = response.json()
            
            assert joined_workspace["id"] == "workspace-456"
            assert joined_workspace["user_role"] == "member"
            
            # Verify invite code usage was incremented
            assert invite_codes[test_code]["current_uses"] == 1
    
    def test_join_workspace_invalid_code(self, mock_auth_user, auth_headers):
        """Test joining workspace with invalid invite code."""
        
        join_data = {"invite_code": "INVALID123"}
        
        with patch('web.api.workspace_routes.get_current_user', return_value=mock_auth_user):
            response = client.post(
                "/api/workspaces/join",
                headers=auth_headers,
                json=join_data
            )
            
            assert response.status_code == 400
            error = response.json()
            assert "Invalid invite code" in error["detail"]
    
    def test_join_workspace_expired_code(self, mock_auth_user, auth_headers):
        """Test joining workspace with expired invite code."""
        
        # Create expired invite code
        from web.api.workspace_routes import invite_codes
        expired_code = "EXPIRED123"
        invite_codes[expired_code] = {
            "workspace_id": "workspace-456",
            "created_by": "other-user",
            "expires_at": datetime.now(timezone.utc).replace(year=2020),  # Past date
            "max_uses": 10,
            "current_uses": 0,
            "is_active": True,
            "created_at": datetime.now(timezone.utc)
        }
        
        join_data = {"invite_code": expired_code}
        
        with patch('web.api.workspace_routes.get_current_user', return_value=mock_auth_user):
            response = client.post(
                "/api/workspaces/join",
                headers=auth_headers,
                json=join_data
            )
            
            assert response.status_code == 400
            error = response.json()
            assert "expired" in error["detail"].lower()
    
    def test_revoke_invite_code_success(self, mock_auth_user, auth_headers):
        """Test revoking invite code successfully."""
        
        # Create invite code
        from web.api.workspace_routes import invite_codes
        workspace_id = "workspace-123"
        test_code = "REVOKE123"
        invite_codes[test_code] = {
            "workspace_id": workspace_id,
            "created_by": mock_auth_user["id"],
            "expires_at": datetime.now(timezone.utc).replace(year=2025),
            "max_uses": 10,
            "current_uses": 0,
            "is_active": True,
            "created_at": datetime.now(timezone.utc)
        }
        
        with patch('web.api.workspace_routes.get_current_user', return_value=mock_auth_user):
            response = client.delete(
                f"/api/workspaces/{workspace_id}/invite-codes/{test_code}",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            result = response.json()
            assert result["success"] is True
            
            # Verify code was deactivated
            assert invite_codes[test_code]["is_active"] is False
    
    def test_get_workspace_members_success(self, mock_auth_user, auth_headers):
        """Test getting workspace members."""
        
        workspace_id = "workspace-123"
        
        with patch('web.api.workspace_routes.get_current_user', return_value=mock_auth_user):
            response = client.get(
                f"/api/workspaces/{workspace_id}/members",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            members = response.json()
            
            assert isinstance(members, list)
            assert len(members) >= 1  # At least the current user
            
            # Check member structure
            member = members[0]
            assert "user_id" in member
            assert "email" in member
            assert "role" in member
            assert "joined_at" in member
    
    def test_update_member_role_success(self, mock_auth_user, auth_headers):
        """Test updating member role successfully."""
        
        workspace_id = "workspace-123"
        member_user_id = "member-456"
        role_data = {"role": "admin"}
        
        with patch('web.api.workspace_routes.get_current_user', return_value=mock_auth_user):
            response = client.put(
                f"/api/workspaces/{workspace_id}/members/{member_user_id}",
                headers=auth_headers,
                json=role_data
            )
            
            assert response.status_code == 200
            result = response.json()
            assert result["success"] is True
    
    def test_remove_member_success(self, mock_auth_user, auth_headers):
        """Test removing member successfully."""
        
        workspace_id = "workspace-123"
        member_user_id = "member-456"
        
        with patch('web.api.workspace_routes.get_current_user', return_value=mock_auth_user):
            response = client.delete(
                f"/api/workspaces/{workspace_id}/members/{member_user_id}",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            result = response.json()
            assert result["success"] is True
    
    def test_remove_member_self_error(self, mock_auth_user, auth_headers):
        """Test that user cannot remove themselves."""
        
        workspace_id = "workspace-123"
        # Try to remove self
        member_user_id = mock_auth_user["id"]
        
        with patch('web.api.workspace_routes.get_current_user', return_value=mock_auth_user):
            response = client.delete(
                f"/api/workspaces/{workspace_id}/members/{member_user_id}",
                headers=auth_headers
            )
            
            assert response.status_code == 400
            error = response.json()
            assert "Cannot remove yourself" in error["detail"]


class TestWorkspaceUtilities:
    """Test workspace utility functions."""
    
    def test_workspace_create_request_validation(self):
        """Test workspace creation request validation."""
        from web.api.workspace_routes import WorkspaceCreateRequest
        
        # Valid request
        valid_request = WorkspaceCreateRequest(
            name="Test Workspace",
            description="A test workspace"
        )
        assert valid_request.name == "Test Workspace"
        assert valid_request.description == "A test workspace"
        
        # Test with minimal data
        minimal_request = WorkspaceCreateRequest(name="Minimal")
        assert minimal_request.name == "Minimal"
        assert minimal_request.description is None
    
    def test_invite_code_request_validation(self):
        """Test invite code creation request validation."""
        from web.api.workspace_routes import InviteCodeCreateRequest
        
        # Valid request
        valid_request = InviteCodeCreateRequest(
            expires_in_hours=24,
            max_uses=5
        )
        assert valid_request.expires_in_hours == 24
        assert valid_request.max_uses == 5
        
        # Default values
        default_request = InviteCodeCreateRequest()
        assert default_request.expires_in_hours == 168  # 7 days
        assert default_request.max_uses == 10
    
    def test_join_workspace_request_validation(self):
        """Test join workspace request validation."""
        from web.api.workspace_routes import JoinWorkspaceRequest
        
        # Valid request
        valid_request = JoinWorkspaceRequest(invite_code="ABC123XYZ")
        assert valid_request.invite_code == "ABC123XYZ"
        
        # Test minimum length
        with pytest.raises(ValueError):
            JoinWorkspaceRequest(invite_code="SHORT")


class TestWorkspaceIntegration:
    """Integration tests for workspace functionality."""
    
    def test_complete_workspace_flow(self, mock_auth_user, auth_headers):
        """Test complete workspace creation and management flow."""
        
        with patch('web.api.workspace_routes.get_current_user', return_value=mock_auth_user):
            # 1. Create workspace
            create_response = client.post(
                "/api/workspaces",
                headers=auth_headers,
                json={"name": "Integration Test Workspace"}
            )
            assert create_response.status_code == 200
            workspace = create_response.json()
            workspace_id = workspace["id"]
            
            # 2. Get workspaces (should include new one)
            list_response = client.get("/api/workspaces", headers=auth_headers)
            assert list_response.status_code == 200
            workspaces = list_response.json()
            assert any(w["id"] == workspace_id for w in workspaces)
            
            # 3. Create invite code
            invite_response = client.post(
                f"/api/workspaces/{workspace_id}/invite-codes",
                headers=auth_headers,
                json={"expires_in_hours": 24, "max_uses": 5}
            )
            assert invite_response.status_code == 200
            invite_code = invite_response.json()["code"]
            
            # 4. Get invite codes
            codes_response = client.get(
                f"/api/workspaces/{workspace_id}/invite-codes",
                headers=auth_headers
            )
            assert codes_response.status_code == 200
            codes = codes_response.json()
            assert any(c["code"] == invite_code for c in codes)
            
            # 5. Update workspace
            update_response = client.put(
                f"/api/workspaces/{workspace_id}",
                headers=auth_headers,
                json={"name": "Updated Integration Test Workspace"}
            )
            assert update_response.status_code == 200
            
            # 6. Get members
            members_response = client.get(
                f"/api/workspaces/{workspace_id}/members",
                headers=auth_headers
            )
            assert members_response.status_code == 200
            members = members_response.json()
            assert len(members) >= 1
            
            # 7. Delete workspace
            delete_response = client.delete(
                f"/api/workspaces/{workspace_id}",
                headers=auth_headers
            )
            assert delete_response.status_code == 200


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])