"""
Content-related Pydantic models for the SEO Content Knowledge Graph System.

This module defines data models for content management, including content items,
content metadata, and content processing results.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator, model_validator
import structlog

logger = structlog.get_logger(__name__)


class ContentType(str, Enum):
    """Content type enumeration."""
    
    ARTICLE = "article"
    BLOG_POST = "blog_post"
    LANDING_PAGE = "landing_page"
    PRODUCT_PAGE = "product_page"
    CATEGORY_PAGE = "category_page"
    FAQ = "faq"
    GUIDE = "guide"
    TUTORIAL = "tutorial"
    CASE_STUDY = "case_study"
    WHITE_PAPER = "white_paper"
    PRESS_RELEASE = "press_release"
    NEWS = "news"
    REVIEW = "review"
    COMPARISON = "comparison"
    LISTICLE = "listicle"
    HOW_TO = "how_to"
    ABOUT = "about"
    CONTACT = "contact"
    LEGAL = "legal"
    OTHER = "other"


class ContentStatus(str, Enum):
    """Content status enumeration."""
    
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    REVIEW_REQUESTED = "review_requested"
    APPROVED = "approved"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    DELETED = "deleted"
    SCHEDULED = "scheduled"


class ContentPriority(str, Enum):
    """Content priority enumeration."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class ContentLanguage(str, Enum):
    """Supported content languages."""
    
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    RUSSIAN = "ru"
    ARABIC = "ar"


class ContentMetadata(BaseModel):
    """Content metadata and SEO information."""
    
    title_tag: Optional[str] = Field(None, max_length=60, description="HTML title tag")
    meta_description: Optional[str] = Field(None, max_length=160, description="Meta description")
    meta_keywords: List[str] = Field(default_factory=list, description="Meta keywords")
    canonical_url: Optional[str] = Field(None, description="Canonical URL")
    open_graph: Dict[str, Any] = Field(default_factory=dict, description="Open Graph metadata")
    twitter_card: Dict[str, Any] = Field(default_factory=dict, description="Twitter Card metadata")
    schema_markup: Dict[str, Any] = Field(default_factory=dict, description="Schema.org markup")
    robots_instructions: List[str] = Field(default_factory=list, description="Robots meta directives")
    
    @validator('title_tag')
    def validate_title_tag(cls, v):
        """Validate title tag length."""
        if v and len(v) > 60:
            logger.warning("Title tag exceeds recommended 60 characters", length=len(v))
        return v
    
    @validator('meta_description')
    def validate_meta_description(cls, v):
        """Validate meta description length."""
        if v and len(v) > 160:
            logger.warning("Meta description exceeds recommended 160 characters", length=len(v))
        return v


class ContentMetrics(BaseModel):
    """Content performance and analytics metrics."""
    
    word_count: int = Field(0, ge=0, description="Total word count")
    character_count: int = Field(0, ge=0, description="Total character count")
    paragraph_count: int = Field(0, ge=0, description="Number of paragraphs")
    sentence_count: int = Field(0, ge=0, description="Number of sentences")
    heading_count: Dict[str, int] = Field(default_factory=dict, description="Heading counts by level")
    
    readability_score: Optional[float] = Field(None, ge=0, le=100, description="Readability score")
    reading_time_minutes: Optional[float] = Field(None, ge=0, description="Estimated reading time")
    
    keyword_density: Dict[str, float] = Field(default_factory=dict, description="Keyword density percentages")
    internal_links: int = Field(0, ge=0, description="Number of internal links")
    external_links: int = Field(0, ge=0, description="Number of external links")
    images: int = Field(0, ge=0, description="Number of images")
    
    seo_score: Optional[float] = Field(None, ge=0, le=100, description="Overall SEO score")
    quality_score: Optional[float] = Field(None, ge=0, le=100, description="Content quality score")
    
    @validator('keyword_density')
    def validate_keyword_density(cls, v):
        """Ensure keyword density values are percentages."""
        for keyword, density in v.items():
            if not 0 <= density <= 100:
                raise ValueError(f"Keyword density for '{keyword}' must be between 0 and 100")
        return v


class ContentGap(BaseModel):
    """Content gap identification result."""
    
    topic: str = Field(..., min_length=1, description="Topic with content gap")
    priority_score: float = Field(..., ge=0.0, le=1.0, description="Gap priority score")
    search_volume: Optional[int] = Field(None, ge=0, description="Search volume for topic")
    competition_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Competition level")
    difficulty_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Content creation difficulty")
    
    related_topics: List[str] = Field(default_factory=list, description="Related topics")
    related_keywords: List[str] = Field(default_factory=list, description="Related keywords")
    
    suggested_content_type: ContentType = Field(..., description="Recommended content type")
    suggested_length: Optional[int] = Field(None, ge=0, description="Suggested word count")
    
    reasoning: str = Field(..., min_length=1, description="Gap analysis reasoning")
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="Analysis confidence")
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ContentItem(BaseModel):
    """Main content item model."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique content ID")
    title: str = Field(..., min_length=1, max_length=200, description="Content title")
    content: str = Field(..., min_length=1, description="Main content body")
    summary: Optional[str] = Field(None, max_length=500, description="Content summary")
    excerpt: Optional[str] = Field(None, max_length=300, description="Content excerpt")
    
    content_type: ContentType = Field(..., description="Type of content")
    status: ContentStatus = Field(ContentStatus.DRAFT, description="Content status")
    priority: ContentPriority = Field(ContentPriority.MEDIUM, description="Content priority")
    language: ContentLanguage = Field(ContentLanguage.ENGLISH, description="Content language")
    
    # Metadata and SEO
    metadata: ContentMetadata = Field(default_factory=ContentMetadata, description="SEO metadata")
    metrics: ContentMetrics = Field(default_factory=ContentMetrics, description="Content metrics")
    
    # Topics and keywords
    topics: List[str] = Field(default_factory=list, description="Content topics")
    keywords: List[str] = Field(default_factory=list, description="Target keywords")
    entities: List[str] = Field(default_factory=list, description="Named entities")
    categories: List[str] = Field(default_factory=list, description="Content categories")
    tags: List[str] = Field(default_factory=list, description="Content tags")
    
    # Ownership and permissions
    tenant_id: str = Field(..., description="Tenant identifier")
    author_id: str = Field(..., description="Author user ID")
    editor_id: Optional[str] = Field(None, description="Editor user ID")
    reviewer_id: Optional[str] = Field(None, description="Reviewer user ID")
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    published_at: Optional[datetime] = Field(None, description="Publication timestamp")
    scheduled_at: Optional[datetime] = Field(None, description="Scheduled publication time")
    
    # URLs and linking
    slug: Optional[str] = Field(None, description="URL slug")
    url_path: Optional[str] = Field(None, description="Full URL path")
    parent_content_id: Optional[str] = Field(None, description="Parent content ID")
    
    # Additional data
    custom_fields: Dict[str, Any] = Field(default_factory=dict, description="Custom fields")
    source_url: Optional[str] = Field(None, description="Source URL if imported")
    version: int = Field(1, ge=1, description="Content version")
    
    @validator('title')
    def validate_title(cls, v):
        """Validate title format."""
        if not v.strip():
            raise ValueError("Title cannot be empty or whitespace only")
        return v.strip()
    
    @validator('content')
    def validate_content(cls, v):
        """Validate content format."""
        if not v.strip():
            raise ValueError("Content cannot be empty or whitespace only")
        return v.strip()
    
    @validator('slug')
    def validate_slug(cls, v):
        """Validate URL slug format."""
        if v:
            # Basic slug validation - letters, numbers, hyphens only
            import re
            if not re.match(r'^[a-z0-9-]+$', v):
                raise ValueError("Slug must contain only lowercase letters, numbers, and hyphens")
        return v
    
    @validator('keywords')
    def validate_keywords(cls, v):
        """Validate keywords list."""
        # Remove duplicates and empty strings
        return list(set(keyword.strip() for keyword in v if keyword.strip()))
    
    @validator('topics')
    def validate_topics(cls, v):
        """Validate topics list."""
        # Remove duplicates and empty strings
        return list(set(topic.strip() for topic in v if topic.strip()))
    
    @model_validator(mode='before')
    @classmethod
    def validate_timestamps(cls, values):
        """Validate timestamp relationships."""
        created_at = values.get('created_at')
        updated_at = values.get('updated_at')
        published_at = values.get('published_at')
        scheduled_at = values.get('scheduled_at')
        
        if updated_at and created_at and updated_at < created_at:
            raise ValueError("Updated timestamp cannot be before created timestamp")
        
        if published_at and created_at and published_at < created_at:
            raise ValueError("Published timestamp cannot be before created timestamp")
        
        if scheduled_at and created_at and scheduled_at < created_at:
            raise ValueError("Scheduled timestamp cannot be before created timestamp")
        
        return values
    
    def update_metrics(self) -> None:
        """Update content metrics based on current content."""
        # Basic metrics calculation
        words = self.content.split()
        self.metrics.word_count = len(words)
        self.metrics.character_count = len(self.content)
        
        # Paragraph count (simple line-based)
        paragraphs = [p for p in self.content.split('\n\n') if p.strip()]
        self.metrics.paragraph_count = len(paragraphs)
        
        # Sentence count (basic period counting)
        sentences = [s for s in self.content.split('.') if s.strip()]
        self.metrics.sentence_count = len(sentences)
        
        # Reading time (average 200 words per minute)
        if self.metrics.word_count > 0:
            self.metrics.reading_time_minutes = self.metrics.word_count / 200
        
        # Update timestamp
        self.updated_at = datetime.now(timezone.utc)
    
    def to_graph_node_data(self) -> Dict[str, Any]:
        """Convert to Neo4j node data format."""
        return {
            "id": self.id,
            "title": self.title,
            "content_type": self.content_type.value,
            "status": self.status.value,
            "priority": self.priority.value,
            "language": self.language.value,
            "tenant_id": self.tenant_id,
            "author_id": self.author_id,
            "word_count": self.metrics.word_count,
            "readability_score": self.metrics.readability_score,
            "seo_score": self.metrics.seo_score,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "slug": self.slug,
            "meta_description": self.metadata.meta_description,
            "canonical_url": self.metadata.canonical_url,
        }


class ContentRequest(BaseModel):
    """Request model for content operations."""
    
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)
    content_type: ContentType
    topics: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    language: ContentLanguage = Field(ContentLanguage.ENGLISH)
    priority: ContentPriority = Field(ContentPriority.MEDIUM)
    metadata: Optional[ContentMetadata] = None
    custom_fields: Dict[str, Any] = Field(default_factory=dict)


class ContentResponse(BaseModel):
    """Response model for content operations."""
    
    content: ContentItem
    success: bool = True
    message: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class ContentAnalysisRequest(BaseModel):
    """Request model for content analysis."""
    
    content_id: Optional[str] = None
    content: Optional[str] = None
    include_seo_research: bool = False
    include_competitor_analysis: bool = False
    include_topic_extraction: bool = True
    include_keyword_analysis: bool = True
    include_readability_analysis: bool = True
    language: ContentLanguage = Field(ContentLanguage.ENGLISH)
    industry: Optional[str] = None
    
    @model_validator(mode='before')
    @classmethod
    def validate_content_source(cls, values):
        """Ensure either content_id or content is provided."""
        content_id = values.get('content_id')
        content = values.get('content')
        
        if not content_id and not content:
            raise ValueError("Either content_id or content must be provided")
        
        return values


class ContentAnalysisResult(BaseModel):
    """Result model for content analysis."""
    
    content_id: Optional[str] = None
    topics: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    
    readability_score: Optional[float] = None
    seo_score: Optional[float] = None
    quality_score: Optional[float] = None
    
    keyword_density: Dict[str, float] = Field(default_factory=dict)
    topic_relevance: Dict[str, float] = Field(default_factory=dict)
    
    suggestions: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    processing_time: Optional[float] = None
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ContentGenerationRequest(BaseModel):
    """Request model for content generation."""
    
    topic: str = Field(..., min_length=1)
    content_type: ContentType
    target_length: Optional[int] = Field(None, ge=100, le=10000)
    keywords: List[str] = Field(default_factory=list)
    tone: Optional[str] = Field(None, description="Content tone (e.g., professional, casual, friendly)")
    audience: Optional[str] = Field(None, description="Target audience")
    language: ContentLanguage = Field(ContentLanguage.ENGLISH)
    include_outline: bool = True
    include_meta_description: bool = True
    include_title_suggestions: int = Field(3, ge=1, le=10)
    
    brand_voice_config: Optional[Dict[str, Any]] = None
    seo_requirements: Optional[Dict[str, Any]] = None
    competitor_references: List[str] = Field(default_factory=list)


class ContentGenerationResult(BaseModel):
    """Result model for content generation."""
    
    generated_content: str
    title_suggestions: List[str] = Field(default_factory=list)
    meta_description: Optional[str] = None
    outline: Optional[List[str]] = None
    
    word_count: int = 0
    estimated_reading_time: Optional[float] = None
    
    used_keywords: List[str] = Field(default_factory=list)
    suggested_improvements: List[str] = Field(default_factory=list)
    
    generation_time: Optional[float] = None
    quality_score: Optional[float] = None
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ContentBatchOperation(BaseModel):
    """Model for batch content operations."""
    
    operation_type: str = Field(..., description="Type of batch operation")
    content_ids: List[str] = Field(..., min_items=1)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    tenant_id: str = Field(..., description="Tenant identifier")
    requested_by: str = Field(..., description="User ID who requested the operation")
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ContentBatchResult(BaseModel):
    """Result model for batch content operations."""
    
    operation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    total_items: int
    processed_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    
    results: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    processing_time: Optional[float] = None
    
    status: str = Field("pending", description="Operation status")


# =============================================================================
# Utility Functions
# =============================================================================

def create_content_from_request(
    request: ContentRequest,
    tenant_id: str,
    author_id: str
) -> ContentItem:
    """
    Create ContentItem from request data.
    
    Args:
        request: Content creation request
        tenant_id: Tenant identifier
        author_id: Author user ID
        
    Returns:
        ContentItem instance
    """
    content = ContentItem(
        title=request.title,
        content=request.content,
        content_type=request.content_type,
        topics=request.topics,
        keywords=request.keywords,
        language=request.language,
        priority=request.priority,
        tenant_id=tenant_id,
        author_id=author_id,
        custom_fields=request.custom_fields,
    )
    
    if request.metadata:
        content.metadata = request.metadata
    
    # Update metrics
    content.update_metrics()
    
    return content


def validate_content_data(data: Dict[str, Any]) -> List[str]:
    """
    Validate content data dictionary.
    
    Args:
        data: Content data to validate
        
    Returns:
        List of validation errors
    """
    errors = []
    
    try:
        ContentItem(**data)
    except Exception as e:
        errors.append(str(e))
    
    return errors


if __name__ == "__main__":
    # Example usage
    content = ContentItem(
        title="Example Article",
        content="This is an example article about SEO and content marketing.",
        content_type=ContentType.ARTICLE,
        tenant_id="example-tenant",
        author_id="example-author",
        topics=["SEO", "Content Marketing"],
        keywords=["seo", "content", "marketing"],
    )
    
    content.update_metrics()
    print(f"Created content: {content.title}")
    print(f"Word count: {content.metrics.word_count}")
    print(f"Reading time: {content.metrics.reading_time_minutes:.1f} minutes")