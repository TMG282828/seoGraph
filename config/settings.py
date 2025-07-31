"""
Configuration settings for the SEO Content Knowledge Graph System.
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings."""
    
    # Core AI Services
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", env="OPENAI_MODEL")
    openai_embedding_model: str = Field(default="text-embedding-ada-002", env="OPENAI_EMBEDDING_MODEL")
    
    # Search & SEO Services
    searxng_url: str = Field(default="http://localhost:8080", env="SEARXNG_URL")
    searxng_api_key: Optional[str] = Field(default=None, env="SEARXNG_API_KEY")
    
    # Database Services
    neo4j_uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    neo4j_username: str = Field(default="neo4j", env="NEO4J_USERNAME")
    neo4j_password: str = Field(default="password", env="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", env="NEO4J_DATABASE")
    
    qdrant_url: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(default="seo_content", env="QDRANT_COLLECTION_NAME")
    
    supabase_url: str = Field(default="http://localhost:54321", env="SUPABASE_URL")
    supabase_key: str = Field(default="demo-anon-key", env="SUPABASE_ANON_KEY")  # Use ANON_KEY as the main key
    supabase_anon_key: str = Field(default="demo-anon-key", env="SUPABASE_ANON_KEY")
    supabase_service_key: str = Field(default="demo-service-key-for-development", env="SUPABASE_SERVICE_ROLE_KEY")
    supabase_service_role_key: str = Field(default="demo-service-key-for-development", env="SUPABASE_SERVICE_ROLE_KEY")
    supabase_jwt_secret: str = Field(default="test-jwt-secret", env="SUPABASE_JWT_SECRET")
    
    # Google Services
    google_client_id: Optional[str] = Field(default=None, env="GOOGLE_CLIENT_ID")
    google_client_secret: Optional[str] = Field(default=None, env="GOOGLE_CLIENT_SECRET")
    google_redirect_uri: str = Field(default="http://localhost:8000/auth/google/callback", env="GOOGLE_REDIRECT_URI")
    
    # Authentication
    secret_key: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    jwt_secret_key: str = Field(default="your-secret-key-change-in-production", env="JWT_SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(default=30, env="REFRESH_TOKEN_EXPIRE_DAYS")
    
    # Optional Services
    langfuse_public_key: Optional[str] = Field(default=None, env="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: Optional[str] = Field(default=None, env="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", env="LANGFUSE_HOST")
    
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    # Application Configuration
    web_port: int = Field(default=8000, env="WEB_PORT")
    web_host: str = Field(default="0.0.0.0", env="WEB_HOST")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Content Processing
    max_content_length: int = Field(default=50000, env="MAX_CONTENT_LENGTH")
    max_file_size_mb: int = Field(default=10, env="MAX_FILE_SIZE_MB")
    allowed_file_extensions: str = Field(default=".txt,.md,.pdf,.docx,.html", env="ALLOWED_FILE_EXTENSIONS")
    
    # SEO Configuration
    seo_min_word_count: int = Field(default=300, env="SEO_MIN_WORD_COUNT")
    seo_max_keyword_density: float = Field(default=0.03, env="SEO_MAX_KEYWORD_DENSITY")
    seo_min_readability_score: int = Field(default=60, env="SEO_MIN_READABILITY_SCORE")
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(default=100, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    
    # Caching
    cache_ttl_seconds: int = Field(default=3600, env="CACHE_TTL_SECONDS")
    cache_enabled: bool = Field(default=True, env="CACHE_ENABLED")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"  # Allow extra fields from .env


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings (singleton pattern)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment (useful for testing)."""
    global _settings
    _settings = None
    return get_settings()