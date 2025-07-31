"""
Data models for competitor analysis functionality.

Contains all Pydantic models, dataclasses, and type definitions used
across the competitor analysis system.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from pydantic import BaseModel, Field

from database.neo4j_client import Neo4jClient
from database.qdrant_client import QdrantClient
from ...services.searxng_service import SearXNGService
from ...services.competitor_monitoring import CompetitorMonitoringService
from ...services.embedding_service import EmbeddingService


class CompetitorAnalysisError(Exception):
    """Raised when competitor analysis operations fail."""
    pass


@dataclass
class CompetitorAnalysisDeps:
    """Dependencies for competitor analysis agent."""
    neo4j_client: Neo4jClient
    qdrant_client: QdrantClient
    searxng_service: SearXNGService
    embedding_service: EmbeddingService
    competitor_monitoring: CompetitorMonitoringService
    tenant_id: str
    industry: Optional[str] = None


class CompetitorInsight(BaseModel):
    """Structured competitor insight data."""
    
    competitor_domain: str = Field(..., description="Competitor domain")
    insight_type: str = Field(..., description="Type of insight")
    insight_category: str = Field(..., description="Category of insight")
    
    # Core data
    title: str = Field(..., description="Insight title")
    description: str = Field(..., description="Detailed description")
    impact_score: float = Field(..., ge=0.0, le=1.0, description="Business impact score")
    urgency: str = Field(..., description="Urgency level (high, medium, low)")
    
    # Supporting data
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Relevant metrics")
    
    # Strategic recommendations
    recommendations: List[str] = Field(default_factory=list, description="Action recommendations")
    opportunity_size: Optional[str] = Field(None, description="Opportunity size estimate")
    
    # Confidence and validation
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in insight")
    data_sources: List[str] = Field(default_factory=list, description="Data sources used")
    
    # Temporal data
    identified_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    relevance_expires: Optional[datetime] = Field(None, description="When insight becomes stale")


class CompetitorProfile(BaseModel):
    """Comprehensive competitor profile."""
    
    domain: str = Field(..., description="Competitor domain")
    company_name: str = Field(..., description="Company name")
    
    # Content strategy
    content_volume: int = Field(0, ge=0, description="Total content pieces")
    content_frequency: float = Field(0.0, ge=0.0, description="Publishing frequency")
    primary_content_types: List[str] = Field(default_factory=list, description="Main content types")
    
    # SEO metrics
    estimated_domain_authority: float = Field(0.0, ge=0.0, le=100.0, description="Domain authority estimate")
    top_ranking_keywords: List[str] = Field(default_factory=list, description="Top ranking keywords")
    content_gaps: List[str] = Field(default_factory=list, description="Identified content gaps")
    
    # Strategic positioning
    target_audience: str = Field("", description="Primary target audience")
    competitive_advantages: List[str] = Field(default_factory=list, description="Key advantages")
    content_quality_score: float = Field(0.0, ge=0.0, le=100.0, description="Content quality assessment")
    
    # Performance indicators
    engagement_indicators: Dict[str, float] = Field(default_factory=dict, description="Engagement metrics")
    seo_optimization_level: str = Field("unknown", description="SEO optimization level")
    
    # Analysis metadata
    last_analyzed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    analysis_completeness: float = Field(0.0, ge=0.0, le=1.0, description="Analysis completeness score")
    data_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in data")


class KeywordGapAnalysis(BaseModel):
    """Results of keyword gap analysis between competitors."""
    
    # Input data
    our_keywords: List[str] = Field(..., description="Our current keywords")
    competitor_keywords: List[str] = Field(..., description="Competitor's keywords")
    
    # Gap analysis results
    competitor_only_keywords: List[str] = Field(..., description="Keywords only competitor ranks for")
    our_only_keywords: List[str] = Field(..., description="Keywords only we rank for")
    shared_keywords: List[str] = Field(..., description="Keywords both rank for")
    
    # Opportunity analysis
    high_opportunity_keywords: List[str] = Field(default_factory=list, description="High opportunity keywords")
    content_gap_opportunities: List[Dict[str, Any]] = Field(default_factory=list, description="Content gap opportunities")
    
    # Metrics
    keyword_overlap_ratio: float = Field(0.0, ge=0.0, le=1.0, description="Keyword overlap ratio")
    competitive_intensity: float = Field(0.0, ge=0.0, le=1.0, description="Competitive intensity score")
    
    # Strategic insights
    strategic_recommendations: List[str] = Field(default_factory=list, description="Strategic recommendations")
    priority_actions: List[str] = Field(default_factory=list, description="Priority actions to take")
    
    # Analysis metadata
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    analysis_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in analysis")