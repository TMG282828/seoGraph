"""
Graph data models for the SEO Content Knowledge Graph System.

This module contains all Pydantic models used for graph data representation.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    """Graph node model with comprehensive properties."""
    id: str
    label: str
    type: str
    size: int = 40
    weight: int = 50  # For frontend visualization
    color: str = "#666666"
    description: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    # Extended properties for different node types
    score: Optional[float] = None
    relevance: Optional[float] = None
    confidence: Optional[float] = None


class GraphEdge(BaseModel):
    """Graph edge model with relationship properties."""
    id: str
    source: str
    target: str
    type: str
    weight: float = 1.0
    color: str = "#999999"
    label: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    
    # Extended properties for different edge types
    strength: Optional[float] = None
    confidence: Optional[float] = None
    relationship_type: Optional[str] = None


class GraphData(BaseModel):
    """Complete graph data structure."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Graph statistics
    node_count: int = 0
    edge_count: int = 0
    density: float = 0.0
    average_degree: float = 0.0
    connected_components: int = 0
    
    # Timestamps
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    cache_key: Optional[str] = None


class GraphQuery(BaseModel):
    """Graph query parameters with comprehensive filtering options."""
    # Basic limits
    node_limit: int = Field(100, ge=1, le=1000)
    edge_limit: int = Field(200, ge=1, le=2000)
    
    # Organization context
    organization_id: Optional[str] = None
    
    # Content filters
    keyword_filter: Optional[str] = None
    content_type_filter: Optional[str] = None
    topic_filter: Optional[str] = None
    
    # Quality filters
    min_connection_strength: float = 0.1
    min_node_score: Optional[float] = None
    min_relevance: Optional[float] = None
    
    # Time filters
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    
    # Grouping and clustering
    cluster_by: Optional[str] = None
    group_similar: bool = False
    
    # Layout preferences
    layout_type: str = "force-directed"
    include_orphans: bool = True


class ContentNodeData(BaseModel):
    """Specialized data for content nodes."""
    content_id: str
    title: str
    word_count: int = 0
    seo_score: float = 0.0
    readability_score: float = 0.0
    content_type: str = "document"
    url: Optional[str] = None
    author: Optional[str] = None
    published_date: Optional[str] = None
    
    # SEO metrics
    keywords: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    backlinks: int = 0
    social_shares: int = 0


class KeywordNodeData(BaseModel):
    """Specialized data for keyword nodes."""
    keyword: str
    search_volume: int = 0
    competition: float = 0.0
    difficulty: str = "unknown"
    trend_direction: str = "stable"
    related_keywords: List[str] = Field(default_factory=list)
    intent: Optional[str] = None
    cpc: Optional[float] = None


class TopicNodeData(BaseModel):
    """Specialized data for topic nodes."""
    topic_name: str
    description: Optional[str] = None
    content_count: int = 0
    avg_seo_score: float = 0.0
    related_topics: List[str] = Field(default_factory=list)
    keyword_density: Dict[str, float] = Field(default_factory=dict)


class CompetitorNodeData(BaseModel):
    """Specialized data for competitor nodes."""
    competitor_name: str
    domain: str
    authority_score: float = 0.0
    content_overlap: float = 0.0
    keyword_overlap: float = 0.0
    shared_topics: List[str] = Field(default_factory=list)
    competitive_strength: str = "unknown"


# Response wrapper for API consistency
class GraphResponse(BaseModel):
    """Standard response wrapper for graph API endpoints."""
    success: bool = True
    graph: GraphData
    statistics: Dict[str, Any] = Field(default_factory=dict)
    message: Optional[str] = None
    cache_info: Optional[Dict[str, Any]] = None