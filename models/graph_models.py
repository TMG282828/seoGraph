"""
Graph-related Pydantic models for the SEO Content Knowledge Graph System.

This module defines data models for graph nodes, relationships, and
graph operations in the Neo4j knowledge graph.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Literal

from pydantic import BaseModel, Field, validator, root_validator
import structlog

from database.graph_schema import NodeLabel, RelationshipType

logger = structlog.get_logger(__name__)


class GraphNode(BaseModel):
    """Base model for graph nodes."""
    
    id: str = Field(..., description="Unique node identifier")
    label: NodeLabel = Field(..., description="Node label/type")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Node properties")
    
    tenant_id: Optional[str] = Field(None, description="Tenant identifier for multi-tenancy")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    @validator('properties')
    def validate_properties(cls, v, values):
        """Validate properties based on node label."""
        # This would validate against the schema definitions
        # For now, just ensure tenant_id is included if available
        if 'tenant_id' in values and values['tenant_id']:
            v['tenant_id'] = values['tenant_id']
        return v
    
    def to_cypher_create(self) -> str:
        """Generate Cypher CREATE statement for this node."""
        props_str = ", ".join([f"{k}: ${k}" for k in self.properties.keys()])
        return f"CREATE (n:{self.label.value} {{{props_str}}})"
    
    def to_cypher_merge(self, unique_key: str = "id") -> str:
        """Generate Cypher MERGE statement for this node."""
        return f"MERGE (n:{self.label.value} {{{unique_key}: ${unique_key}}})"


class GraphRelationship(BaseModel):
    """Base model for graph relationships."""
    
    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    relationship_type: RelationshipType = Field(..., description="Relationship type")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Relationship properties")
    
    source_label: Optional[NodeLabel] = Field(None, description="Source node label")
    target_label: Optional[NodeLabel] = Field(None, description="Target node label")
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    weight: float = Field(1.0, ge=0.0, description="Relationship weight/strength")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")
    
    @validator('properties')
    def set_default_properties(cls, v, values):
        """Set default relationship properties."""
        v.setdefault('created_at', datetime.now(timezone.utc).isoformat())
        v.setdefault('weight', values.get('weight', 1.0))
        if values.get('confidence') is not None:
            v['confidence'] = values['confidence']
        return v
    
    def to_cypher_create(self) -> str:
        """Generate Cypher CREATE statement for this relationship."""
        props_str = ", ".join([f"{k}: ${k}" for k in self.properties.keys()])
        rel_props = f" {{{props_str}}}" if props_str else ""
        
        return (
            f"MATCH (source:{self.source_label.value if self.source_label else ''} {{id: $source_id}}), "
            f"(target:{self.target_label.value if self.target_label else ''} {{id: $target_id}}) "
            f"CREATE (source)-[r:{self.relationship_type.value}{rel_props}]->(target)"
        )


class ContentNode(GraphNode):
    """Specialized node model for content."""
    
    label: Literal[NodeLabel.CONTENT] = Field(NodeLabel.CONTENT)
    
    title: str = Field(..., description="Content title")
    content_type: str = Field(..., description="Type of content")
    status: str = Field(..., description="Content status")
    author_id: str = Field(..., description="Author identifier")
    
    word_count: Optional[int] = Field(None, ge=0, description="Word count")
    readability_score: Optional[float] = Field(None, ge=0, le=100, description="Readability score")
    seo_score: Optional[float] = Field(None, ge=0, le=100, description="SEO score")
    
    published_at: Optional[datetime] = Field(None, description="Publication timestamp")
    
    def __init__(self, **data):
        """Initialize content node with properties."""
        super().__init__(**data)
        
        # Set properties from fields
        self.properties.update({
            'title': self.title,
            'content_type': self.content_type,
            'status': self.status,
            'author_id': self.author_id,
        })
        
        # Optional properties
        if self.word_count is not None:
            self.properties['word_count'] = self.word_count
        if self.readability_score is not None:
            self.properties['readability_score'] = self.readability_score
        if self.seo_score is not None:
            self.properties['seo_score'] = self.seo_score
        if self.published_at is not None:
            self.properties['published_at'] = self.published_at.isoformat()


class TopicNode(GraphNode):
    """Specialized node model for topics."""
    
    label: Literal[NodeLabel.TOPIC] = Field(NodeLabel.TOPIC)
    
    name: str = Field(..., description="Topic name")
    description: Optional[str] = Field(None, description="Topic description")
    category: Optional[str] = Field(None, description="Topic category")
    
    search_volume: Optional[int] = Field(None, ge=0, description="Search volume")
    trend_direction: Optional[str] = Field(None, description="Trend direction")
    importance_score: Optional[float] = Field(None, ge=0, le=1, description="Importance score")
    
    def __init__(self, **data):
        """Initialize topic node with properties."""
        super().__init__(**data)
        
        # Set properties from fields
        self.properties.update({
            'name': self.name,
        })
        
        # Optional properties
        if self.description:
            self.properties['description'] = self.description
        if self.category:
            self.properties['category'] = self.category
        if self.search_volume is not None:
            self.properties['search_volume'] = self.search_volume
        if self.trend_direction:
            self.properties['trend_direction'] = self.trend_direction
        if self.importance_score is not None:
            self.properties['importance_score'] = self.importance_score


class KeywordNode(GraphNode):
    """Specialized node model for keywords."""
    
    label: Literal[NodeLabel.KEYWORD] = Field(NodeLabel.KEYWORD)
    
    text: str = Field(..., description="Keyword text")
    search_volume: Optional[int] = Field(None, ge=0, description="Monthly search volume")
    competition: Optional[float] = Field(None, ge=0, le=1, description="Competition score")
    cpc: Optional[float] = Field(None, ge=0, description="Cost per click")
    difficulty: Optional[float] = Field(None, ge=0, le=100, description="Keyword difficulty")
    
    def __init__(self, **data):
        """Initialize keyword node with properties."""
        super().__init__(**data)
        
        # Set properties from fields
        self.properties.update({
            'text': self.text,
        })
        
        # Optional properties
        if self.search_volume is not None:
            self.properties['search_volume'] = self.search_volume
        if self.competition is not None:
            self.properties['competition'] = self.competition
        if self.cpc is not None:
            self.properties['cpc'] = self.cpc
        if self.difficulty is not None:
            self.properties['difficulty'] = self.difficulty


class EntityNode(GraphNode):
    """Specialized node model for entities."""
    
    label: Literal[NodeLabel.ENTITY] = Field(NodeLabel.ENTITY)
    
    name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Entity type (person, organization, etc.)")
    description: Optional[str] = Field(None, description="Entity description")
    aliases: List[str] = Field(default_factory=list, description="Entity aliases")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Recognition confidence")
    
    def __init__(self, **data):
        """Initialize entity node with properties."""
        super().__init__(**data)
        
        # Set properties from fields
        self.properties.update({
            'name': self.name,
            'type': self.entity_type,
        })
        
        # Optional properties
        if self.description:
            self.properties['description'] = self.description
        if self.aliases:
            self.properties['aliases'] = self.aliases
        if self.confidence is not None:
            self.properties['confidence'] = self.confidence


class SimilarityRelationship(GraphRelationship):
    """Specialized relationship for content similarity."""
    
    relationship_type: Literal[RelationshipType.SIMILAR_TO] = Field(RelationshipType.SIMILAR_TO)
    
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    algorithm: Optional[str] = Field(None, description="Similarity algorithm used")
    
    def __init__(self, **data):
        """Initialize similarity relationship with properties."""
        super().__init__(**data)
        
        # Set properties from fields
        self.properties.update({
            'similarity_score': self.similarity_score,
        })
        
        if self.algorithm:
            self.properties['algorithm'] = self.algorithm


class RelatestoRelationship(GraphRelationship):
    """Specialized relationship for content-topic relationships."""
    
    relationship_type: Literal[RelationshipType.RELATES_TO] = Field(RelationshipType.RELATES_TO)
    
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Relevance score")
    mention_count: Optional[int] = Field(None, ge=0, description="Number of mentions")
    prominence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Topic prominence")
    
    def __init__(self, **data):
        """Initialize relates-to relationship with properties."""
        super().__init__(**data)
        
        # Optional properties
        if self.relevance_score is not None:
            self.properties['relevance_score'] = self.relevance_score
        if self.mention_count is not None:
            self.properties['mention_count'] = self.mention_count
        if self.prominence is not None:
            self.properties['prominence'] = self.prominence


class GraphQuery(BaseModel):
    """Model for graph database queries."""
    
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    cypher: str = Field(..., description="Cypher query string")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Query parameters")
    
    description: Optional[str] = Field(None, description="Query description")
    expected_results: Optional[int] = Field(None, ge=0, description="Expected result count")
    timeout: Optional[int] = Field(None, ge=1, description="Query timeout in seconds")
    
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")
    created_by: Optional[str] = Field(None, description="User who created the query")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @validator('cypher')
    def validate_cypher(cls, v):
        """Basic Cypher query validation."""
        v = v.strip()
        if not v:
            raise ValueError("Cypher query cannot be empty")
        
        # Basic safety checks
        dangerous_keywords = ['DELETE', 'DETACH DELETE', 'REMOVE', 'SET', 'CREATE']
        upper_query = v.upper()
        
        # Allow these for content management, but log them
        for keyword in dangerous_keywords:
            if keyword in upper_query:
                logger.warning("Potentially destructive Cypher query", keyword=keyword, query=v[:100])
        
        return v


class GraphQueryResult(BaseModel):
    """Model for graph query results."""
    
    query_id: str = Field(..., description="Query identifier")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Query results")
    
    execution_time: Optional[float] = Field(None, ge=0, description="Execution time in seconds")
    records_returned: int = Field(0, ge=0, description="Number of records returned")
    
    success: bool = Field(True, description="Query success status")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    
    executed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GraphAnalytics(BaseModel):
    """Model for graph analytics and statistics."""
    
    tenant_id: str = Field(..., description="Tenant identifier")
    
    # Node counts
    total_nodes: int = Field(0, ge=0, description="Total number of nodes")
    content_nodes: int = Field(0, ge=0, description="Number of content nodes")
    topic_nodes: int = Field(0, ge=0, description="Number of topic nodes")
    keyword_nodes: int = Field(0, ge=0, description="Number of keyword nodes")
    entity_nodes: int = Field(0, ge=0, description="Number of entity nodes")
    
    # Relationship counts
    total_relationships: int = Field(0, ge=0, description="Total number of relationships")
    similarity_relationships: int = Field(0, ge=0, description="Number of similarity relationships")
    topic_relationships: int = Field(0, ge=0, description="Number of topic relationships")
    keyword_relationships: int = Field(0, ge=0, description="Number of keyword relationships")
    
    # Graph metrics
    average_degree: Optional[float] = Field(None, ge=0, description="Average node degree")
    graph_density: Optional[float] = Field(None, ge=0, le=1, description="Graph density")
    largest_component_size: Optional[int] = Field(None, ge=0, description="Largest connected component size")
    
    # Content metrics
    average_content_connections: Optional[float] = Field(None, ge=0, description="Average content connections")
    most_connected_topics: List[Dict[str, Any]] = Field(default_factory=list, description="Most connected topics")
    most_used_keywords: List[Dict[str, Any]] = Field(default_factory=list, description="Most used keywords")
    
    # Quality metrics
    orphaned_content: int = Field(0, ge=0, description="Content with no relationships")
    duplicate_topics: int = Field(0, ge=0, description="Potential duplicate topics")
    missing_keywords: int = Field(0, ge=0, description="Content without keywords")
    
    calculated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GraphOperation(BaseModel):
    """Model for graph operations and bulk updates."""
    
    operation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    operation_type: str = Field(..., description="Type of operation")
    
    # Operation details
    nodes_to_create: List[GraphNode] = Field(default_factory=list, description="Nodes to create")
    nodes_to_update: List[GraphNode] = Field(default_factory=list, description="Nodes to update")
    nodes_to_delete: List[str] = Field(default_factory=list, description="Node IDs to delete")
    
    relationships_to_create: List[GraphRelationship] = Field(default_factory=list, description="Relationships to create")
    relationships_to_update: List[GraphRelationship] = Field(default_factory=list, description="Relationships to update")
    relationships_to_delete: List[Dict[str, str]] = Field(default_factory=list, description="Relationships to delete")
    
    # Execution details
    tenant_id: str = Field(..., description="Tenant identifier")
    created_by: str = Field(..., description="User who created the operation")
    dry_run: bool = Field(False, description="Whether this is a dry run")
    
    # Status tracking
    status: str = Field("pending", description="Operation status")
    progress: float = Field(0.0, ge=0.0, le=100.0, description="Operation progress percentage")
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = Field(None, description="Operation start time")
    completed_at: Optional[datetime] = Field(None, description="Operation completion time")
    
    # Results
    nodes_created: int = Field(0, ge=0, description="Number of nodes created")
    nodes_updated: int = Field(0, ge=0, description="Number of nodes updated")
    nodes_deleted: int = Field(0, ge=0, description="Number of nodes deleted")
    relationships_created: int = Field(0, ge=0, description="Number of relationships created")
    relationships_updated: int = Field(0, ge=0, description="Number of relationships updated")
    relationships_deleted: int = Field(0, ge=0, description="Number of relationships deleted")
    
    errors: List[str] = Field(default_factory=list, description="Operation errors")


class GraphVisualization(BaseModel):
    """Model for graph visualization data."""
    
    visualization_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = Field(..., description="Tenant identifier")
    
    # Visualization parameters
    center_node_id: Optional[str] = Field(None, description="Center node for visualization")
    node_types: List[NodeLabel] = Field(default_factory=list, description="Node types to include")
    relationship_types: List[RelationshipType] = Field(default_factory=list, description="Relationship types to include")
    max_depth: int = Field(2, ge=1, le=5, description="Maximum depth from center node")
    max_nodes: int = Field(100, ge=1, le=1000, description="Maximum number of nodes")
    
    # Layout and styling
    layout_algorithm: str = Field("force-directed", description="Layout algorithm")
    node_size_attribute: Optional[str] = Field(None, description="Attribute for node sizing")
    edge_width_attribute: Optional[str] = Field(None, description="Attribute for edge width")
    color_scheme: str = Field("default", description="Color scheme")
    
    # Visualization data
    nodes: List[Dict[str, Any]] = Field(default_factory=list, description="Node data for visualization")
    edges: List[Dict[str, Any]] = Field(default_factory=list, description="Edge data for visualization")
    
    # Metadata
    created_by: str = Field(..., description="User who created the visualization")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# Request/Response Models
# =============================================================================

class GraphSearchRequest(BaseModel):
    """Request model for graph search operations."""
    
    search_type: str = Field(..., description="Type of search (node, relationship, path)")
    query: str = Field(..., description="Search query")
    
    node_types: List[NodeLabel] = Field(default_factory=list, description="Node types to search")
    relationship_types: List[RelationshipType] = Field(default_factory=list, description="Relationship types to include")
    
    filters: Dict[str, Any] = Field(default_factory=dict, description="Additional filters")
    limit: int = Field(50, ge=1, le=1000, description="Maximum results")
    
    tenant_id: str = Field(..., description="Tenant identifier")


class GraphSearchResponse(BaseModel):
    """Response model for graph search operations."""
    
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Search results")
    total_count: int = Field(0, ge=0, description="Total matching results")
    execution_time: float = Field(0.0, ge=0, description="Search execution time")
    
    success: bool = Field(True, description="Search success status")
    message: Optional[str] = Field(None, description="Response message")


# =============================================================================
# Utility Functions
# =============================================================================

def create_content_node(
    content_id: str,
    title: str,
    content_type: str,
    status: str,
    author_id: str,
    tenant_id: str,
    **kwargs
) -> ContentNode:
    """
    Create a content node with standard properties.
    
    Args:
        content_id: Content identifier
        title: Content title
        content_type: Type of content
        status: Content status
        author_id: Author identifier
        tenant_id: Tenant identifier
        **kwargs: Additional properties
        
    Returns:
        ContentNode instance
    """
    return ContentNode(
        id=content_id,
        title=title,
        content_type=content_type,
        status=status,
        author_id=author_id,
        tenant_id=tenant_id,
        **kwargs
    )


def create_topic_node(
    topic_id: str,
    name: str,
    tenant_id: str,
    **kwargs
) -> TopicNode:
    """
    Create a topic node with standard properties.
    
    Args:
        topic_id: Topic identifier
        name: Topic name
        tenant_id: Tenant identifier
        **kwargs: Additional properties
        
    Returns:
        TopicNode instance
    """
    return TopicNode(
        id=topic_id,
        name=name,
        tenant_id=tenant_id,
        **kwargs
    )


def create_similarity_relationship(
    source_id: str,
    target_id: str,
    similarity_score: float,
    algorithm: Optional[str] = None
) -> SimilarityRelationship:
    """
    Create a similarity relationship between content nodes.
    
    Args:
        source_id: Source content ID
        target_id: Target content ID
        similarity_score: Similarity score (0-1)
        algorithm: Algorithm used for similarity calculation
        
    Returns:
        SimilarityRelationship instance
    """
    return SimilarityRelationship(
        source_id=source_id,
        target_id=target_id,
        similarity_score=similarity_score,
        algorithm=algorithm,
        source_label=NodeLabel.CONTENT,
        target_label=NodeLabel.CONTENT,
        weight=similarity_score,
        confidence=similarity_score
    )


def create_topic_relationship(
    content_id: str,
    topic_id: str,
    relevance_score: Optional[float] = None
) -> RelatestoRelationship:
    """
    Create a relationship between content and topic.
    
    Args:
        content_id: Content identifier
        topic_id: Topic identifier
        relevance_score: Relevance score (0-1)
        
    Returns:
        RelatestoRelationship instance
    """
    return RelatestoRelationship(
        source_id=content_id,
        target_id=topic_id,
        relevance_score=relevance_score,
        source_label=NodeLabel.CONTENT,
        target_label=NodeLabel.TOPIC,
        weight=relevance_score or 1.0
    )


if __name__ == "__main__":
    # Example usage
    content_node = create_content_node(
        content_id="content-1",
        title="SEO Best Practices",
        content_type="article",
        status="published",
        author_id="author-1",
        tenant_id="tenant-1",
        word_count=1500,
        seo_score=85.0
    )
    
    topic_node = create_topic_node(
        topic_id="topic-1",
        name="SEO",
        tenant_id="tenant-1",
        search_volume=10000
    )
    
    relationship = create_topic_relationship(
        content_id=content_node.id,
        topic_id=topic_node.id,
        relevance_score=0.9
    )
    
    print(f"Created content node: {content_node.title}")
    print(f"Created topic node: {topic_node.name}")
    print(f"Created relationship: {relationship.relationship_type.value}")
    print(f"Cypher query: {relationship.to_cypher_create()}")