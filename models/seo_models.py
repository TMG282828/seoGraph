"""
SEO-related Pydantic models for the SEO Content Knowledge Graph System.

This module defines data models for SEO analysis, keyword research,
competitor analysis, and search performance tracking.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator, model_validator
import structlog

logger = structlog.get_logger(__name__)


class SearchEngine(str, Enum):
    """Supported search engines."""
    
    GOOGLE = "google"
    BING = "bing"
    YAHOO = "yahoo"
    DUCKDUCKGO = "duckduckgo"
    YANDEX = "yandex"
    BAIDU = "baidu"


class DeviceType(str, Enum):
    """Device types for search results."""
    
    DESKTOP = "desktop"
    MOBILE = "mobile"
    TABLET = "tablet"


class SearchIntent(str, Enum):
    """Search intent classification."""
    
    INFORMATIONAL = "informational"
    NAVIGATIONAL = "navigational"
    TRANSACTIONAL = "transactional"
    COMMERCIAL = "commercial"
    LOCAL = "local"


class TrendDirection(str, Enum):
    """Trend direction for keywords and topics."""
    
    RISING = "rising"
    STABLE = "stable"
    DECLINING = "declining"
    SEASONAL = "seasonal"
    VOLATILE = "volatile"


class CompetitionLevel(str, Enum):
    """Competition level classification."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class RankingPosition(BaseModel):
    """Search engine ranking position data."""
    
    position: int = Field(..., ge=1, le=100, description="Ranking position")
    url: str = Field(..., description="Ranking URL")
    title: Optional[str] = Field(None, description="Page title")
    snippet: Optional[str] = Field(None, description="Search snippet")
    
    search_engine: SearchEngine = Field(SearchEngine.GOOGLE, description="Search engine")
    device_type: DeviceType = Field(DeviceType.DESKTOP, description="Device type")
    location: Optional[str] = Field(None, description="Search location")
    language: Optional[str] = Field(None, description="Search language")
    
    date_checked: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    previous_position: Optional[int] = Field(None, description="Previous position")
    position_change: Optional[int] = Field(None, description="Position change")
    
    featured_snippet: bool = Field(False, description="Is featured snippet")
    knowledge_panel: bool = Field(False, description="Has knowledge panel")
    local_pack: bool = Field(False, description="In local pack")
    
    @validator('position_change')
    def calculate_position_change(cls, v, values):
        """Calculate position change if previous position is available."""
        if values.get('previous_position') and values.get('position'):
            return values['previous_position'] - values['position']
        return v


class KeywordData(BaseModel):
    """Keyword research and analysis data."""
    
    keyword: str = Field(..., min_length=1, description="Keyword text")
    search_volume: int = Field(0, ge=0, description="Monthly search volume")
    competition: float = Field(0.0, ge=0.0, le=1.0, description="Competition score")
    cpc: Optional[float] = Field(None, ge=0.0, description="Cost per click")
    
    difficulty: Optional[float] = Field(None, ge=0.0, le=100.0, description="Keyword difficulty")
    opportunity: Optional[float] = Field(None, ge=0.0, le=100.0, description="Opportunity score")
    
    trend_direction: TrendDirection = Field(TrendDirection.STABLE, description="Trend direction")
    trend_data: List[int] = Field(default_factory=list, description="Monthly trend data")
    
    search_intent: Optional[SearchIntent] = Field(None, description="Search intent")
    competition_level: Optional[CompetitionLevel] = Field(None, description="Competition level")
    
    related_keywords: List[str] = Field(default_factory=list, description="Related keywords")
    long_tail_keywords: List[str] = Field(default_factory=list, description="Long-tail variations")
    question_keywords: List[str] = Field(default_factory=list, description="Question-based keywords")
    
    local_search_volume: Dict[str, int] = Field(default_factory=dict, description="Local search volumes")
    seasonal_trends: Dict[str, float] = Field(default_factory=dict, description="Seasonal trend data")
    
    tenant_id: str = Field(..., description="Tenant identifier")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    @validator('keyword')
    def validate_keyword(cls, v):
        """Validate keyword format."""
        return v.strip().lower()
    
    @validator('competition_level')
    def set_competition_level(cls, v, values):
        """Set competition level based on competition score."""
        if v is None and 'competition' in values:
            competition = values['competition']
            if competition < 0.3:
                return CompetitionLevel.LOW
            elif competition < 0.6:
                return CompetitionLevel.MEDIUM
            elif competition < 0.8:
                return CompetitionLevel.HIGH
            else:
                return CompetitionLevel.VERY_HIGH
        return v


class SEOMetrics(BaseModel):
    """SEO performance metrics for content."""
    
    content_id: str = Field(..., description="Content identifier")
    
    # On-page SEO metrics
    title_optimization: float = Field(0.0, ge=0.0, le=100.0, description="Title optimization score")
    meta_description_optimization: float = Field(0.0, ge=0.0, le=100.0, description="Meta description score")
    heading_optimization: float = Field(0.0, ge=0.0, le=100.0, description="Heading structure score")
    keyword_optimization: float = Field(0.0, ge=0.0, le=100.0, description="Keyword optimization score")
    content_optimization: float = Field(0.0, ge=0.0, le=100.0, description="Content optimization score")
    
    # Content metrics
    word_count: int = Field(0, ge=0, description="Total word count")
    readability_score: float = Field(0.0, ge=0.0, le=100.0, description="Readability score")
    
    # Keyword metrics
    primary_keyword_density: float = Field(0.0, ge=0.0, le=100.0, description="Primary keyword density")
    keyword_density: Dict[str, float] = Field(default_factory=dict, description="All keyword densities")
    keyword_positions: Dict[str, List[int]] = Field(default_factory=dict, description="Keyword positions in text")
    
    # Link metrics
    internal_links: int = Field(0, ge=0, description="Number of internal links")
    external_links: int = Field(0, ge=0, description="Number of external links")
    broken_links: int = Field(0, ge=0, description="Number of broken links")
    
    # Media metrics
    images: int = Field(0, ge=0, description="Number of images")
    alt_text_coverage: float = Field(0.0, ge=0.0, le=100.0, description="Alt text coverage percentage")
    video_count: int = Field(0, ge=0, description="Number of videos")
    
    # Technical SEO
    page_load_time: Optional[float] = Field(None, ge=0.0, description="Page load time in seconds")
    mobile_friendly: Optional[bool] = Field(None, description="Mobile-friendly status")
    schema_markup_present: bool = Field(False, description="Schema markup present")
    
    # Overall scores
    overall_seo_score: float = Field(0.0, ge=0.0, le=100.0, description="Overall SEO score")
    content_quality_score: float = Field(0.0, ge=0.0, le=100.0, description="Content quality score")
    
    # Timestamps
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tenant_id: str = Field(..., description="Tenant identifier")
    
    def calculate_overall_score(self) -> float:
        """Calculate overall SEO score from component scores."""
        scores = [
            self.title_optimization * 0.15,
            self.meta_description_optimization * 0.10,
            self.heading_optimization * 0.15,
            self.keyword_optimization * 0.25,
            self.content_optimization * 0.20,
            self.readability_score * 0.15,
        ]
        
        self.overall_seo_score = sum(scores)
        return self.overall_seo_score


class CompetitorData(BaseModel):
    """Competitor analysis data."""
    
    competitor_name: str = Field(..., description="Competitor name or domain")
    domain: str = Field(..., description="Competitor domain")
    
    # SEO metrics
    domain_authority: Optional[float] = Field(None, ge=0.0, le=100.0, description="Domain authority")
    page_authority: Optional[float] = Field(None, ge=0.0, le=100.0, description="Page authority")
    trust_score: Optional[float] = Field(None, ge=0.0, le=100.0, description="Trust score")
    
    # Traffic metrics
    organic_traffic: Optional[int] = Field(None, ge=0, description="Estimated organic traffic")
    paid_traffic: Optional[int] = Field(None, ge=0, description="Estimated paid traffic")
    total_backlinks: Optional[int] = Field(None, ge=0, description="Total backlinks")
    referring_domains: Optional[int] = Field(None, ge=0, description="Referring domains")
    
    # Keyword metrics
    ranking_keywords: Optional[int] = Field(None, ge=0, description="Number of ranking keywords")
    top_keywords: List[str] = Field(default_factory=list, description="Top ranking keywords")
    content_gaps: List[str] = Field(default_factory=list, description="Content gaps vs competitor")
    
    # Content metrics
    total_pages: Optional[int] = Field(None, ge=0, description="Total indexed pages")
    content_types: Dict[str, int] = Field(default_factory=dict, description="Content type distribution")
    update_frequency: Optional[str] = Field(None, description="Content update frequency")
    
    # Social metrics
    social_signals: Dict[str, int] = Field(default_factory=dict, description="Social media signals")
    
    tenant_id: str = Field(..., description="Tenant identifier")
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @validator('domain')
    def validate_domain(cls, v):
        """Validate domain format."""
        import re
        domain_pattern = r'^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)*[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$'
        if not re.match(domain_pattern, v):
            raise ValueError("Invalid domain format")
        return v.lower()


class SERPFeature(BaseModel):
    """Search Engine Results Page (SERP) feature data."""
    
    keyword: str = Field(..., description="Search keyword")
    search_engine: SearchEngine = Field(SearchEngine.GOOGLE, description="Search engine")
    
    # SERP features
    featured_snippet: Optional[Dict[str, Any]] = Field(None, description="Featured snippet data")
    knowledge_panel: Optional[Dict[str, Any]] = Field(None, description="Knowledge panel data")
    local_pack: Optional[List[Dict[str, Any]]] = Field(None, description="Local pack results")
    people_also_ask: List[str] = Field(default_factory=list, description="People also ask questions")
    related_searches: List[str] = Field(default_factory=list, description="Related searches")
    
    # Ads
    top_ads: int = Field(0, ge=0, description="Number of top ads")
    bottom_ads: int = Field(0, ge=0, description="Number of bottom ads")
    shopping_ads: int = Field(0, ge=0, description="Number of shopping ads")
    
    # Organic results
    total_results: Optional[int] = Field(None, ge=0, description="Total search results")
    organic_results: List[Dict[str, Any]] = Field(default_factory=list, description="Top organic results")
    
    # Analysis
    search_intent: Optional[SearchIntent] = Field(None, description="Detected search intent")
    difficulty_score: Optional[float] = Field(None, ge=0.0, le=100.0, description="Ranking difficulty")
    
    tenant_id: str = Field(..., description="Tenant identifier")
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SEOAuditResult(BaseModel):
    """Comprehensive SEO audit result."""
    
    audit_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Audit identifier")
    content_id: Optional[str] = Field(None, description="Content identifier")
    url: Optional[str] = Field(None, description="URL audited")
    
    # Audit scores
    overall_score: float = Field(0.0, ge=0.0, le=100.0, description="Overall SEO score")
    technical_score: float = Field(0.0, ge=0.0, le=100.0, description="Technical SEO score")
    content_score: float = Field(0.0, ge=0.0, le=100.0, description="Content SEO score")
    
    # Issues and recommendations
    critical_issues: List[Dict[str, Any]] = Field(default_factory=list, description="Critical issues")
    warnings: List[Dict[str, Any]] = Field(default_factory=list, description="Warnings")
    recommendations: List[Dict[str, Any]] = Field(default_factory=list, description="Recommendations")
    
    # Detailed metrics
    metrics: SEOMetrics
    
    # Benchmark data
    industry_average: Optional[float] = Field(None, description="Industry average score")
    competitor_scores: Dict[str, float] = Field(default_factory=dict, description="Competitor scores")
    
    tenant_id: str = Field(..., description="Tenant identifier")
    audited_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    audited_by: str = Field(..., description="User who performed audit")


class SEORecommendation(BaseModel):
    """SEO improvement recommendation."""
    
    recommendation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content_id: Optional[str] = Field(None, description="Target content ID")
    
    category: str = Field(..., description="Recommendation category")
    priority: str = Field(..., description="Priority level (high, medium, low)")
    
    title: str = Field(..., description="Recommendation title")
    description: str = Field(..., description="Detailed description")
    impact: str = Field(..., description="Expected impact")
    effort: str = Field(..., description="Implementation effort")
    
    specific_actions: List[str] = Field(default_factory=list, description="Specific actions to take")
    success_metrics: List[str] = Field(default_factory=list, description="How to measure success")
    
    # Status tracking
    status: str = Field("pending", description="Implementation status")
    assigned_to: Optional[str] = Field(None, description="Assigned user ID")
    due_date: Optional[datetime] = Field(None, description="Due date")
    completed_at: Optional[datetime] = Field(None, description="Completion date")
    
    tenant_id: str = Field(..., description="Tenant identifier")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TrendAnalysis(BaseModel):
    """Trend analysis for topics and keywords."""
    
    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic: str = Field(..., description="Topic or keyword analyzed")
    
    # Trend data
    trend_direction: TrendDirection = Field(..., description="Overall trend direction")
    trend_strength: float = Field(0.0, ge=0.0, le=1.0, description="Trend strength")
    
    # Historical data
    historical_volume: List[Dict[str, Union[str, int]]] = Field(
        default_factory=list, 
        description="Historical search volume data"
    )
    seasonal_patterns: Dict[str, float] = Field(
        default_factory=dict, 
        description="Seasonal pattern analysis"
    )
    
    # Predictions
    predicted_volume: Optional[int] = Field(None, ge=0, description="Predicted next month volume")
    confidence_interval: Optional[Dict[str, int]] = Field(None, description="Prediction confidence")
    
    # Related trends
    related_topics: List[Dict[str, Any]] = Field(default_factory=list, description="Related trending topics")
    emerging_keywords: List[str] = Field(default_factory=list, description="Emerging related keywords")
    
    # Analysis metadata
    data_source: str = Field(..., description="Data source for analysis")
    analysis_period: str = Field(..., description="Analysis time period")
    
    tenant_id: str = Field(..., description="Tenant identifier")
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class KeywordCluster(BaseModel):
    """Keyword clustering result."""
    
    cluster_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    cluster_name: str = Field(..., description="Cluster name/topic")
    
    primary_keyword: str = Field(..., description="Primary keyword")
    secondary_keywords: List[str] = Field(default_factory=list, description="Secondary keywords")
    long_tail_keywords: List[str] = Field(default_factory=list, description="Long-tail keywords")
    
    # Cluster metrics
    total_search_volume: int = Field(0, ge=0, description="Total cluster search volume")
    average_difficulty: float = Field(0.0, ge=0.0, le=100.0, description="Average keyword difficulty")
    average_cpc: Optional[float] = Field(None, ge=0.0, description="Average CPC")
    
    # Content opportunities
    content_gaps: List[str] = Field(default_factory=list, description="Content gap opportunities")
    recommended_content_types: List[str] = Field(default_factory=list, description="Recommended content types")
    
    # Competition analysis
    top_competitors: List[str] = Field(default_factory=list, description="Top competing domains")
    difficulty_score: float = Field(0.0, ge=0.0, le=100.0, description="Overall cluster difficulty")
    
    tenant_id: str = Field(..., description="Tenant identifier")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# Request/Response Models
# =============================================================================

class KeywordResearchRequest(BaseModel):
    """Request model for keyword research."""
    
    seed_keywords: List[str] = Field(..., min_items=1, description="Seed keywords")
    language: str = Field("en", description="Target language")
    location: Optional[str] = Field(None, description="Target location")
    include_questions: bool = Field(True, description="Include question keywords")
    include_long_tail: bool = Field(True, description="Include long-tail keywords")
    max_results: int = Field(100, ge=1, le=1000, description="Maximum results")
    min_search_volume: int = Field(0, ge=0, description="Minimum search volume")
    max_difficulty: Optional[float] = Field(None, ge=0.0, le=100.0, description="Maximum difficulty")


class CompetitorAnalysisRequest(BaseModel):
    """Request model for competitor analysis."""
    
    competitor_domains: List[str] = Field(..., min_items=1, description="Competitor domains")
    keywords: List[str] = Field(default_factory=list, description="Keywords to analyze")
    include_content_gaps: bool = Field(True, description="Include content gap analysis")
    include_backlink_analysis: bool = Field(False, description="Include backlink analysis")
    include_paid_analysis: bool = Field(False, description="Include paid search analysis")


class SEOAuditRequest(BaseModel):
    """Request model for SEO audit."""
    
    content_id: Optional[str] = Field(None, description="Content ID to audit")
    url: Optional[str] = Field(None, description="URL to audit")
    include_technical: bool = Field(True, description="Include technical SEO audit")
    include_content: bool = Field(True, description="Include content SEO audit")
    include_competitors: bool = Field(False, description="Include competitor comparison")
    
    @model_validator(mode='before')
    @classmethod
    def validate_audit_target(cls, values):
        """Ensure either content_id or url is provided."""
        content_id = values.get('content_id')
        url = values.get('url')
        
        if not content_id and not url:
            raise ValueError("Either content_id or url must be provided")
        
        return values


class SEOAnalysisRequest(BaseModel):
    """Request model for SEO analysis."""
    
    content: Optional[str] = Field(None, description="Content text to analyze")
    content_id: Optional[str] = Field(None, description="Content ID to analyze")
    url: Optional[str] = Field(None, description="URL to analyze")
    
    # Analysis options
    target_keywords: List[str] = Field(default_factory=list, description="Target keywords")
    include_keyword_research: bool = Field(True, description="Include keyword research")
    include_competitor_analysis: bool = Field(True, description="Include competitor analysis")
    include_content_recommendations: bool = Field(True, description="Include content recommendations")
    
    # Context
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")
    language: str = Field("en", description="Content language")
    location: Optional[str] = Field(None, description="Target location")
    
    @model_validator(mode='before')
    @classmethod
    def validate_analysis_target(cls, values):
        """Ensure at least one content source is provided."""
        content = values.get('content')
        content_id = values.get('content_id')
        url = values.get('url')
        
        if not any([content, content_id, url]):
            raise ValueError("At least one of content, content_id, or url must be provided")
        
        return values


class SEOAnalysisResult(BaseModel):
    """Result model for SEO analysis."""
    
    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Input data
    content_analyzed: str = Field(..., description="Content that was analyzed")
    target_keywords: List[str] = Field(default_factory=list, description="Target keywords")
    
    # SEO metrics
    seo_metrics: SEOMetrics
    
    # Keyword analysis
    keyword_analysis: List[KeywordData] = Field(default_factory=list, description="Keyword analysis results")
    keyword_opportunities: List[str] = Field(default_factory=list, description="Keyword opportunities")
    
    # Content recommendations
    recommendations: List[SEORecommendation] = Field(default_factory=list, description="SEO recommendations")
    
    # Competitor insights
    competitor_analysis: Optional[List[CompetitorData]] = Field(None, description="Competitor analysis")
    content_gaps: List[str] = Field(default_factory=list, description="Content gaps")
    
    # Performance predictions
    predicted_rankings: Dict[str, int] = Field(default_factory=dict, description="Predicted keyword rankings")
    traffic_potential: Optional[int] = Field(None, description="Estimated traffic potential")
    
    # Analysis metadata
    confidence_score: float = Field(0.0, ge=0.0, le=1.0, description="Analysis confidence")
    analysis_duration: Optional[float] = Field(None, description="Analysis duration in seconds")
    
    tenant_id: str = Field(..., description="Tenant identifier")
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    analyzed_by: str = Field(..., description="User or agent who performed analysis")


# =============================================================================
# Utility Functions
# =============================================================================

def calculate_keyword_difficulty(
    competition: float,
    domain_authority_avg: float,
    serp_features: int
) -> float:
    """
    Calculate keyword difficulty score.
    
    Args:
        competition: Competition level (0-1)
        domain_authority_avg: Average domain authority of top results
        serp_features: Number of SERP features present
        
    Returns:
        Difficulty score (0-100)
    """
    # Weighted calculation
    difficulty = (
        competition * 40 +
        (domain_authority_avg / 100) * 40 +
        min(serp_features / 5, 1) * 20
    )
    
    return min(difficulty, 100.0)


def classify_search_intent(keyword: str, serp_features: Dict[str, Any]) -> SearchIntent:
    """
    Classify search intent based on keyword and SERP features.
    
    Args:
        keyword: Search keyword
        serp_features: SERP features data
        
    Returns:
        Classified search intent
    """
    keyword_lower = keyword.lower()
    
    # Commercial indicators
    commercial_words = ['buy', 'purchase', 'price', 'cost', 'deal', 'discount', 'shop']
    if any(word in keyword_lower for word in commercial_words):
        return SearchIntent.COMMERCIAL
    
    # Informational indicators
    info_words = ['how', 'what', 'why', 'when', 'where', 'guide', 'tutorial', 'learn']
    if any(word in keyword_lower for word in info_words):
        return SearchIntent.INFORMATIONAL
    
    # Navigational indicators
    if serp_features.get('knowledge_panel') or 'brand' in keyword_lower:
        return SearchIntent.NAVIGATIONAL
    
    # Local indicators
    local_words = ['near', 'nearby', 'local', 'location']
    if any(word in keyword_lower for word in local_words) or serp_features.get('local_pack'):
        return SearchIntent.LOCAL
    
    # Default to informational
    return SearchIntent.INFORMATIONAL


def group_keywords_by_intent(keywords: List[KeywordData]) -> Dict[SearchIntent, List[KeywordData]]:
    """
    Group keywords by search intent.
    
    Args:
        keywords: List of keyword data
        
    Returns:
        Dictionary mapping intent to keywords
    """
    grouped = {intent: [] for intent in SearchIntent}
    
    for keyword in keywords:
        intent = keyword.search_intent or SearchIntent.INFORMATIONAL
        grouped[intent].append(keyword)
    
    return grouped


class ContentGapOpportunity(BaseModel):
    """Content gap analysis opportunity."""
    
    gap_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic: str = Field(..., description="Gap topic")
    priority_score: float = Field(..., ge=0.0, le=1.0, description="Priority score")
    
    # Search metrics
    search_volume: int = Field(0, ge=0, description="Monthly search volume")
    competition_level: CompetitionLevel = Field(CompetitionLevel.MEDIUM, description="Competition level")
    trend_direction: TrendDirection = Field(TrendDirection.STABLE, description="Trend direction")
    
    # Keywords and content
    related_keywords: List[str] = Field(default_factory=list, description="Related keywords")
    competitor_coverage: Dict[str, int] = Field(default_factory=dict, description="Competitor coverage")
    content_suggestions: List[str] = Field(default_factory=list, description="Content suggestions")
    
    # Opportunity metrics
    estimated_traffic: int = Field(0, ge=0, description="Estimated monthly traffic")
    difficulty_score: float = Field(0.0, ge=0.0, le=1.0, description="Difficulty score")
    opportunity_score: float = Field(0.0, ge=0.0, le=1.0, description="Overall opportunity score")
    
    # Analysis metadata
    confidence_score: float = Field(0.0, ge=0.0, le=1.0, description="Analysis confidence")
    data_sources: List[str] = Field(default_factory=list, description="Data sources used")
    
    tenant_id: str = Field(..., description="Tenant identifier")
    identified_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = Field(None, description="Opportunity expiration")
    
    def calculate_opportunity_score(self) -> float:
        """Calculate overall opportunity score."""
        # Base score from search volume (normalized)
        volume_score = min(self.search_volume / 10000, 1.0)
        
        # Competition penalty
        competition_penalty = 1.0 - (self.difficulty_score * 0.5)
        
        # Trend boost
        trend_multiplier = {
            TrendDirection.RISING: 1.3,
            TrendDirection.STABLE: 1.0,
            TrendDirection.DECLINING: 0.7,
            TrendDirection.SEASONAL: 1.1,
            TrendDirection.VOLATILE: 0.8
        }.get(self.trend_direction, 1.0)
        
        # Competitor gap bonus
        competitor_gap = max(0, 1.0 - (len(self.competitor_coverage) / 10))
        
        # Calculate final score
        self.opportunity_score = min(
            volume_score * competition_penalty * trend_multiplier * (1 + competitor_gap),
            1.0
        )
        
        return self.opportunity_score


class GapAnalysisResult(BaseModel):
    """Gap analysis result with opportunities and insights."""
    
    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = Field(..., description="Tenant identifier")
    
    # Analysis parameters
    industry: str = Field(..., description="Industry analyzed")
    analysis_type: str = Field(..., description="Type of gap analysis")
    
    # Results
    opportunities: List[ContentGapOpportunity] = Field(default_factory=list)
    total_opportunities: int = Field(0, ge=0, description="Total opportunities found")
    
    # Insights
    top_topics: List[str] = Field(default_factory=list, description="Top opportunity topics")
    trending_themes: List[str] = Field(default_factory=list, description="Trending themes")
    competitor_insights: Dict[str, Any] = Field(default_factory=dict, description="Competitor insights")
    
    # Analysis metadata
    confidence_score: float = Field(0.0, ge=0.0, le=1.0, description="Overall confidence")
    analysis_duration: float = Field(0.0, ge=0.0, description="Analysis duration in seconds")
    data_freshness: Dict[str, datetime] = Field(default_factory=dict, description="Data freshness timestamps")
    
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    analyzed_by: str = Field(..., description="User or system that performed analysis")
    
    def get_high_priority_opportunities(self, limit: int = 10) -> List[ContentGapOpportunity]:
        """Get top high-priority opportunities."""
        sorted_opportunities = sorted(
            self.opportunities,
            key=lambda x: x.priority_score,
            reverse=True
        )
        return sorted_opportunities[:limit]
    
    def get_opportunities_by_trend(self, trend: TrendDirection) -> List[ContentGapOpportunity]:
        """Get opportunities filtered by trend direction."""
        return [opp for opp in self.opportunities if opp.trend_direction == trend]


class CompetitorContentAnalysis(BaseModel):
    """Competitor content analysis result."""
    
    competitor_domain: str = Field(..., description="Competitor domain")
    competitor_name: Optional[str] = Field(None, description="Competitor name")
    
    # Content metrics
    total_content_pieces: int = Field(0, ge=0, description="Total content pieces")
    content_frequency: float = Field(0.0, ge=0.0, description="Content publishing frequency")
    average_word_count: int = Field(0, ge=0, description="Average word count")
    
    # Topic coverage
    topics_covered: List[str] = Field(default_factory=list, description="Topics covered")
    topic_distribution: Dict[str, int] = Field(default_factory=dict, description="Topic distribution")
    content_gaps: List[str] = Field(default_factory=list, description="Content gaps vs our content")
    
    # Performance metrics
    estimated_traffic: int = Field(0, ge=0, description="Estimated monthly traffic")
    social_engagement: Dict[str, int] = Field(default_factory=dict, description="Social media engagement")
    backlink_profile: Dict[str, Any] = Field(default_factory=dict, description="Backlink profile")
    
    # SEO metrics
    keyword_rankings: Dict[str, int] = Field(default_factory=dict, description="Keyword rankings")
    serp_features: List[str] = Field(default_factory=list, description="SERP features captured")
    
    tenant_id: str = Field(..., description="Tenant identifier")
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


if __name__ == "__main__":
    # Example usage
    keyword = KeywordData(
        keyword="seo content strategy",
        search_volume=1200,
        competition=0.7,
        tenant_id="example-tenant",
        related_keywords=["content marketing", "seo strategy", "content planning"]
    )
    
    print(f"Created keyword: {keyword.keyword}")
    print(f"Competition level: {keyword.competition_level}")
    print(f"Search volume: {keyword.search_volume}")
    
    # Example SEO metrics
    metrics = SEOMetrics(
        content_id="example-content",
        title_optimization=85.0,
        meta_description_optimization=90.0,
        keyword_optimization=75.0,
        content_optimization=80.0,
        readability_score=78.0,
        tenant_id="example-tenant"
    )
    
    overall_score = metrics.calculate_overall_score()
    print(f"Overall SEO score: {overall_score:.1f}")
    
    # Example gap analysis opportunity
    opportunity = ContentGapOpportunity(
        topic="AI content automation",
        priority_score=0.8,
        search_volume=3500,
        competition_level=CompetitionLevel.MEDIUM,
        trend_direction=TrendDirection.RISING,
        related_keywords=["AI content", "content automation", "automated content"],
        tenant_id="example-tenant"
    )
    
    opportunity.calculate_opportunity_score()
    print(f"Gap opportunity: {opportunity.topic}")
    print(f"Opportunity score: {opportunity.opportunity_score:.2f}")