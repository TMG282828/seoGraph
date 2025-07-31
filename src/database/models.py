"""
Database models for SEO Content Knowledge Graph System.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class TrackedKeyword(Base):
    """Model for keywords being tracked in SEO Monitor."""
    __tablename__ = "tracked_keywords"
    
    id = Column(Integer, primary_key=True, index=True)
    keyword = Column(String(255), nullable=False, index=True)
    organization_id = Column(String(36), nullable=False, index=True)  # Multi-tenant support
    user_id = Column(String(255), nullable=True, index=True)  # Future user system
    domain = Column(String(255), nullable=True)
    target_url = Column(String(500), nullable=True)
    
    # SEO Metrics
    current_position = Column(Integer, nullable=True)
    previous_position = Column(Integer, nullable=True)
    search_volume = Column(Integer, nullable=True)
    difficulty = Column(Float, nullable=True)
    cpc = Column(Float, nullable=True)
    
    # Tracking metadata
    data_source = Column(String(50), default="algorithmic")  # algorithmic, google_ads, gsc
    is_active = Column(Boolean, default=True)
    notes = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_checked = Column(DateTime(timezone=True), nullable=True)
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'keyword': self.keyword,
            'organization_id': self.organization_id,
            'user_id': self.user_id,
            'domain': self.domain,
            'target_url': self.target_url,
            'current_position': self.current_position,
            'previous_position': self.previous_position,
            'search_volume': self.search_volume,
            'difficulty': self.difficulty,
            'cpc': self.cpc,
            'data_source': self.data_source,
            'is_active': self.is_active,
            'notes': self.notes,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'last_checked': self.last_checked.isoformat() if self.last_checked else None
        }

class KeywordHistory(Base):
    """Historical data for tracked keywords."""
    __tablename__ = "keyword_history"
    
    id = Column(Integer, primary_key=True, index=True)
    tracked_keyword_id = Column(Integer, nullable=False, index=True)
    position = Column(Integer, nullable=True)
    search_volume = Column(Integer, nullable=True)
    difficulty = Column(Float, nullable=True)
    cpc = Column(Float, nullable=True)
    organic_traffic = Column(Integer, nullable=True)
    
    # Data source and quality
    data_source = Column(String(50), nullable=False)
    confidence_score = Column(Float, default=0.8)
    
    # Timestamp
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'tracked_keyword_id': self.tracked_keyword_id,
            'position': self.position,
            'search_volume': self.search_volume,
            'difficulty': self.difficulty,
            'cpc': self.cpc,
            'organic_traffic': self.organic_traffic,
            'data_source': self.data_source,
            'confidence_score': self.confidence_score,
            'recorded_at': self.recorded_at.isoformat() if self.recorded_at else None
        }

class ContentBrief(Base):
    """Model for content briefs used in Content Studio."""
    __tablename__ = "content_briefs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), nullable=True, index=True)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    word_count = Column(Integer, nullable=True)
    keywords = Column(JSON, nullable=True)  # Array of keywords
    
    # Metadata
    source_type = Column(String(50), default="manual")  # manual, file_upload, url_import, google_drive
    original_filename = Column(String(255), nullable=True)
    source_url = Column(String(500), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'title': self.title,
            'content': self.content,
            'summary': self.summary,
            'word_count': self.word_count,
            'keywords': self.keywords or [],
            'source_type': self.source_type,
            'original_filename': self.original_filename,
            'source_url': self.source_url,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class SavedContent(Base):
    """Model for saved generated content."""
    __tablename__ = "saved_content"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), nullable=True, index=True)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    content_type = Column(String(100), default="article")  # article, blog_post, social_media, etc.
    
    # Content metrics
    word_count = Column(Integer, nullable=True)
    seo_score = Column(Float, nullable=True)
    readability_score = Column(Float, nullable=True)
    keywords = Column(JSON, nullable=True)
    
    # Generation context
    brief_used = Column(String(500), nullable=True)  # Brief title reference
    brief_id = Column(Integer, nullable=True)  # Reference to content_briefs table
    created_via = Column(String(50), default="manual")  # manual, chat_generation, batch_generation
    auto_saved = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'title': self.title,
            'content': self.content,
            'content_type': self.content_type,
            'word_count': self.word_count,
            'seo_score': self.seo_score,
            'readability_score': self.readability_score,
            'keywords': self.keywords or [],
            'brief_used': self.brief_used,
            'brief_id': self.brief_id,
            'created_via': self.created_via,
            'auto_saved': self.auto_saved,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class ContentItem(Base):
    """Model for Knowledge Base content items (compatible with Supabase schema)."""
    __tablename__ = "content_items"
    
    id = Column(String(36), primary_key=True, index=True)  # UUID format
    organization_id = Column(String(36), nullable=False, index=True)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    content_type = Column(String(100), default="document")  # document, article, webpage, etc.
    
    # Source information
    source_type = Column(String(50), default="manual")  # file_upload, url_import, manual
    source_url = Column(String(500), nullable=True)
    original_filename = Column(String(255), nullable=True)
    
    # Content metrics and analysis
    word_count = Column(Integer, nullable=True)
    seo_score = Column(Float, nullable=True)
    readability_score = Column(Float, nullable=True)
    keywords = Column(JSON, nullable=True)  # Array of extracted keywords
    
    # Processing status
    processing_status = Column(String(50), default="pending")  # pending, processing, completed, failed
    vector_embedding_status = Column(String(50), default="pending")  # For vector database sync
    
    # Metadata
    file_size = Column(Integer, nullable=True)
    file_type = Column(String(50), nullable=True)
    extraction_metadata = Column(JSON, nullable=True)
    
    # User and organization context
    created_by = Column(String(36), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    processed_at = Column(DateTime(timezone=True), nullable=True)
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'organization_id': self.organization_id,
            'title': self.title,
            'content': self.content,
            'content_type': self.content_type,
            'source_type': self.source_type,
            'source_url': self.source_url,
            'original_filename': self.original_filename,
            'word_count': self.word_count,
            'seo_score': self.seo_score,
            'readability_score': self.readability_score,
            'keywords': self.keywords or [],
            'processing_status': self.processing_status,
            'vector_embedding_status': self.vector_embedding_status,
            'file_size': self.file_size,
            'file_type': self.file_type,
            'extraction_metadata': self.extraction_metadata,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None
        }