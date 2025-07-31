"""
Unit tests for general settings API routes.

Tests the settings API endpoints for application settings, API keys,
system configuration, and notification preferences.
"""

import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import HTTPException

# Mock the get_current_user_safe function
mock_user = {
    "id": "test-user-id",
    "organization_id": "test-org-id",
    "email": "test@example.com"
}

@pytest.fixture
def mock_auth():
    """Mock authentication for tests."""
    with patch('web.api.settings_routes.get_current_user_safe') as mock_get_user:
        mock_get_user.return_value = mock_user
        yield mock_get_user


@pytest.fixture
def mock_tenant_mapper():
    """Mock TenantOrgMapper for tests."""
    with patch('web.api.settings_routes.TenantOrgMapper') as mock_mapper:
        # Set up default mock behavior
        mock_mapper.get_app_settings = AsyncMock(return_value={})
        mock_mapper.update_app_settings = AsyncMock(return_value=True)
        yield mock_mapper


class TestSettingsRoutes:
    """Test suite for settings API routes."""
    
    def test_get_all_settings_default(self, mock_auth, mock_tenant_mapper):
        """Test getting settings with default values."""
        from web.api.settings_routes import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        
        # Mock empty settings (should return defaults)
        mock_tenant_mapper.get_app_settings.return_value = {}
        
        response = client.get("/api/settings/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["settings"]["checkin_frequency"] == "medium"
        assert data["settings"]["require_approval"] is True
        assert data["api_keys"]["openai"] == ""
        assert data["system_config"]["max_content_length"] == 50000
        assert data["notifications"]["email"] is True
        
        # Verify TenantOrgMapper was called correctly
        mock_tenant_mapper.get_app_settings.assert_called_once_with("test-org-id")
    
    def test_get_all_settings_existing(self, mock_auth, mock_tenant_mapper):
        """Test getting existing settings."""
        from web.api.settings_routes import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        
        # Mock existing settings
        existing_settings = {
            "settings": {
                "checkin_frequency": "high",
                "require_approval": False,
                "notify_low_confidence": False
            },
            "api_keys": {
                "openai": "sk-test123",
                "langfuse_public": "pk-test456",
                "langfuse_secret": "sk-secret789"
            },
            "system_config": {
                "max_content_length": 75000,
                "batch_size": 20,
                "enable_caching": False
            },
            "notifications": {
                "email": False,
                "browser": True,
                "email_address": "test@example.com",
                "report_frequency": "daily"
            },
            "last_updated": "2023-01-01T00:00:00Z"
        }
        mock_tenant_mapper.get_app_settings.return_value = existing_settings
        
        response = client.get("/api/settings/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["settings"]["checkin_frequency"] == "high"
        assert data["settings"]["require_approval"] is False
        assert data["api_keys"]["openai"] == "sk-test123"
        assert data["system_config"]["max_content_length"] == 75000
        assert data["notifications"]["email"] is False
        assert data["last_updated"] == "2023-01-01T00:00:00Z"
    
    def test_save_all_settings(self, mock_auth, mock_tenant_mapper):
        """Test saving all settings."""
        from web.api.settings_routes import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        
        # Mock successful save
        mock_tenant_mapper.update_app_settings.return_value = True
        
        settings_request = {
            "settings": {
                "checkin_frequency": "high",
                "require_approval": True,
                "notify_low_confidence": False
            },
            "api_keys": {
                "openai": "sk-newkey123",
                "langfuse_public": "pk-newkey456", 
                "langfuse_secret": "sk-newsecret789"
            },
            "system_config": {
                "max_content_length": 60000,
                "batch_size": 15,
                "enable_caching": True
            },
            "notifications": {
                "email": True,
                "browser": False,
                "email_address": "new@example.com",
                "report_frequency": "weekly"
            }
        }
        
        response = client.post("/api/settings/", json=settings_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["settings"]["checkin_frequency"] == "high"
        assert data["api_keys"]["openai"] == "sk-newkey123"
        assert data["system_config"]["max_content_length"] == 60000
        assert data["notifications"]["email_address"] == "new@example.com"
        assert "last_updated" in data
        
        # Verify save was called
        mock_tenant_mapper.update_app_settings.assert_called_once()
    
    def test_save_settings_failure(self, mock_auth, mock_tenant_mapper):
        """Test handling save failures."""
        from web.api.settings_routes import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        
        # Mock save failure
        mock_tenant_mapper.update_app_settings.return_value = False
        
        settings_request = {
            "settings": {"checkin_frequency": "high", "require_approval": True, "notify_low_confidence": False},
            "api_keys": {"openai": "", "langfuse_public": "", "langfuse_secret": ""},
            "system_config": {"max_content_length": 50000, "batch_size": 10, "enable_caching": True},
            "notifications": {"email": True, "browser": False, "email_address": "", "report_frequency": "weekly"}
        }
        
        response = client.post("/api/settings/", json=settings_request)
        
        assert response.status_code == 500
        assert "Failed to save settings" in response.json()["detail"]
    
    def test_get_api_keys_masked(self, mock_auth, mock_tenant_mapper):
        """Test getting API keys with masking."""
        from web.api.settings_routes import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        
        # Mock settings with API keys
        mock_settings = {
            "api_keys": {
                "openai": "sk-1234567890abcdef1234567890abcdef",
                "langfuse_public": "pk-abcdef1234567890abcdef1234567890",
                "langfuse_secret": "sk-fedcba0987654321fedcba0987654321"
            }
        }
        mock_tenant_mapper.get_app_settings.return_value = mock_settings
        
        response = client.get("/api/settings/api-keys")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        # Keys should be masked
        assert data["api_keys"]["openai"] == "sk-1...cdef"
        assert data["api_keys"]["langfuse_public"] == "pk-a...7890"
        assert data["api_keys"]["langfuse_secret"] == "sk-f...4321"
        # Flags should indicate presence
        assert data["has_openai"] is True
        assert data["has_langfuse"] is True
    
    def test_update_api_keys(self, mock_auth, mock_tenant_mapper):
        """Test updating API keys."""
        from web.api.settings_routes import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        
        # Mock existing settings
        mock_tenant_mapper.get_app_settings.return_value = {"some": "settings"}
        mock_tenant_mapper.update_app_settings.return_value = True
        
        api_keys_request = {
            "openai": "sk-newkey123",
            "langfuse_public": "pk-newpublic456",
            "langfuse_secret": "sk-newsecret789"
        }
        
        response = client.post("/api/settings/api-keys", json=api_keys_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["message"] == "API keys updated successfully"
        assert "last_updated" in data
        
        # Verify update was called with correct structure
        mock_tenant_mapper.update_app_settings.assert_called_once()
        call_args = mock_tenant_mapper.update_app_settings.call_args
        assert call_args[0][0] == "test-org-id"  # organization_id
        updated_settings = call_args[0][1]
        assert updated_settings["api_keys"]["openai"] == "sk-newkey123"
    
    def test_test_configuration(self, mock_auth, mock_tenant_mapper):
        """Test configuration testing endpoint."""
        from web.api.settings_routes import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        
        # Mock settings with API keys
        mock_settings = {
            "api_keys": {
                "openai": "sk-validkey123456789012345678901234567890",
                "langfuse_public": "pk-validpublic123456789012345678901234",
                "langfuse_secret": "sk-validsecret123456789012345678901234"
            }
        }
        mock_tenant_mapper.get_app_settings.return_value = mock_settings
        
        response = client.post("/api/settings/test-configuration")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert len(data["tests"]) == 2  # OpenAI and Langfuse tests
        
        # Check test results
        openai_test = next(t for t in data["tests"] if t["service"] == "OpenAI")
        langfuse_test = next(t for t in data["tests"] if t["service"] == "Langfuse")
        
        assert openai_test["status"] == "valid_format"
        assert langfuse_test["status"] == "valid_format"
    
    def test_test_configuration_invalid_keys(self, mock_auth, mock_tenant_mapper):
        """Test configuration testing with invalid API keys."""
        from web.api.settings_routes import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        
        # Mock settings with invalid API keys
        mock_settings = {
            "api_keys": {
                "openai": "invalid-key",
                "langfuse_public": "wrong-prefix",
                "langfuse_secret": "also-wrong"
            }
        }
        mock_tenant_mapper.get_app_settings.return_value = mock_settings
        
        response = client.post("/api/settings/test-configuration")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is False  # Should fail due to invalid formats
        
        # Check test results
        openai_test = next(t for t in data["tests"] if t["service"] == "OpenAI")
        langfuse_test = next(t for t in data["tests"] if t["service"] == "Langfuse")
        
        assert openai_test["status"] == "invalid_format" 
        assert langfuse_test["status"] == "invalid_format"
    
    def test_reset_to_defaults(self, mock_auth, mock_tenant_mapper):
        """Test resetting settings to defaults."""
        from web.api.settings_routes import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        
        mock_tenant_mapper.update_app_settings.return_value = True
        
        response = client.post("/api/settings/reset-defaults")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["message"] == "Settings reset to defaults successfully"
        assert "last_updated" in data
        
        # Verify defaults were saved
        mock_tenant_mapper.update_app_settings.assert_called_once()
        call_args = mock_tenant_mapper.update_app_settings.call_args
        saved_settings = call_args[0][1]
        
        # Check default values
        assert saved_settings["settings"]["checkin_frequency"] == "medium"
        assert saved_settings["settings"]["require_approval"] is True
        assert saved_settings["api_keys"]["openai"] == ""
        assert saved_settings["system_config"]["max_content_length"] == 50000
        assert saved_settings["notifications"]["email"] is True
    
    @pytest.mark.skip(reason="Auth error handling needs FastAPI exception handling setup")
    def test_authentication_required(self, mock_tenant_mapper):
        """Test that endpoints require authentication."""
        from web.api.settings_routes import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        
        # Don't mock authentication - should fail
        with patch('web.api.settings_routes.get_current_user_safe') as mock_get_user:
            mock_get_user.side_effect = HTTPException(status_code=401, detail="Unauthorized")
            
            response = client.get("/api/settings/")
            assert response.status_code == 401
            
            response = client.post("/api/settings/", json={})
            assert response.status_code == 401


if __name__ == "__main__":
    pytest.main([__file__, "-v"])