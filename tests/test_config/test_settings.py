"""
Tests for the Settings configuration module.
"""

import pytest
from unittest.mock import patch
import os

from config.settings import Settings, get_settings


class TestSettings:
    """Test suite for Settings configuration."""

    def test_settings_creation(self):
        """Test creation of Settings with default values."""
        settings = Settings()
        
        assert settings.environment == "development"
        assert settings.debug is False
        assert settings.openai_model == "gpt-4o-mini"
        assert settings.log_level == "INFO"

    def test_settings_database_urls(self):
        """Test database URL configurations."""
        settings = Settings()
        
        # Test default database URLs
        assert settings.neo4j_uri == "bolt://localhost:7687"
        assert settings.redis_url == "redis://localhost:6379"
        assert settings.supabase_url == "postgresql://postgres:postgres@localhost:5432/seo_content"
        assert settings.qdrant_url == "http://localhost:6333"

    def test_settings_security_config(self):
        """Test security configuration."""
        settings = Settings()
        
        # Test security settings
        assert settings.neo4j_username == "neo4j"
        assert settings.neo4j_password == "password"
        assert settings.supabase_key == "test-key-for-development"

    def test_settings_api_config(self):
        """Test API configuration."""
        settings = Settings()
        
        # Test API settings
        assert settings.web_port == 8000
        assert settings.web_host == "0.0.0.0"
        assert settings.rate_limit_per_minute == 100

    @patch.dict(os.environ, {"ENVIRONMENT": "production"})
    def test_settings_production_environment(self):
        """Test settings in production environment."""
        settings = Settings()
        
        assert settings.environment == "production"
        assert settings.debug is False

    @patch.dict(os.environ, {"NEO4J_URI": "bolt://custom-neo4j:7687"})
    def test_settings_custom_neo4j_url(self):
        """Test custom Neo4j URI from environment."""
        settings = Settings()
        
        assert settings.neo4j_uri == "bolt://custom-neo4j:7687"

    @patch.dict(os.environ, {"REDIS_URL": "redis://custom-redis:6379"})
    def test_settings_custom_redis_url(self):
        """Test custom Redis URL from environment."""
        settings = Settings()
        
        assert settings.redis_url == "redis://custom-redis:6379"

    @patch.dict(os.environ, {"NEO4J_PASSWORD": "custom-password"})
    def test_settings_custom_secret_key(self):
        """Test custom Neo4j password from environment."""
        settings = Settings()
        
        assert settings.neo4j_password == "custom-password"

    @patch.dict(os.environ, {"RATE_LIMIT_PER_MINUTE": "60"})
    def test_settings_custom_token_expire(self):
        """Test custom rate limit from environment."""
        settings = Settings()
        
        assert settings.rate_limit_per_minute == 60

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"})
    def test_settings_openai_api_key(self):
        """Test OpenAI API key from environment."""
        settings = Settings()
        
        assert settings.openai_api_key == "test-openai-key"

    def test_settings_validation(self):
        """Test settings validation."""
        settings = Settings()
        
        # Test that required settings are present
        assert settings.neo4j_uri is not None
        assert settings.openai_model is not None
        assert settings.environment is not None

    def test_settings_cors_validation(self):
        """Test rate limiting configuration."""
        settings = Settings()
        
        # Test that rate limiting is enabled by default
        assert settings.rate_limit_enabled is True
        assert settings.rate_limit_per_minute == 100

    def test_settings_database_timeout_config(self):
        """Test caching configurations."""
        settings = Settings()
        
        # Test default caching values
        assert settings.cache_enabled is True
        assert settings.cache_ttl_seconds == 3600

    @patch.dict(os.environ, {"CACHE_TTL_SECONDS": "7200"})
    def test_settings_custom_database_timeout(self):
        """Test custom cache TTL from environment."""
        settings = Settings()
        
        assert settings.cache_ttl_seconds == 7200


class TestGetSettings:
    """Test suite for get_settings function."""

    def test_get_settings_returns_settings_instance(self):
        """Test that get_settings returns a Settings instance."""
        settings = get_settings()
        
        assert isinstance(settings, Settings)

    def test_get_settings_singleton_behavior(self):
        """Test that get_settings returns the same instance (singleton)."""
        settings1 = get_settings()
        settings2 = get_settings()
        
        assert settings1 is settings2

    def test_get_settings_caching(self):
        """Test that get_settings caches the result."""
        # Reset the global settings first
        from config.settings import reload_settings
        reload_settings()
        
        settings1 = get_settings()
        settings2 = get_settings()
        
        assert settings1 is settings2

    def test_get_settings_with_environment_changes(self):
        """Test get_settings behavior with environment changes."""
        # Reset the global settings first
        from config.settings import reload_settings
        reload_settings()
        
        settings1 = get_settings()
        original_env = settings1.environment
        
        # The cached result should remain the same even if we change environment
        # (this tests that caching is working correctly)
        with patch.dict(os.environ, {"ENVIRONMENT": "test"}):
            settings2 = get_settings()
            assert settings2.environment == original_env  # Should be cached value