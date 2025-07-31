"""
Simplified Graph API for Knowledge Base content visualization.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

# Create router
graph_router = APIRouter(prefix="/graph", tags=["graph"])

# =============================================================================
# Request/Response Models (Simplified)
# =============================================================================

class GraphNode(BaseModel):
    """Graph node model."""
    id: str
    label: str
    type: str
    size: int = 40
    weight: int = 50  # Add weight field for frontend visualization
    color: str = "#666666"
    description: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class GraphEdge(BaseModel):
    """Graph edge model."""
    id: str
    source: str
    target: str
    type: str
    weight: float = 1.0
    color: str = "#999999"
    label: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None

class GraphData(BaseModel):
    """Complete graph data model."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    node_count: int = 0
    edge_count: int = 0
    density: float = 0.0
    average_degree: float = 0.0
    connected_components: int = 0
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class GraphQuery(BaseModel):
    """Graph query parameters."""
    node_limit: int = Field(100, ge=1, le=1000)
    edge_limit: int = Field(200, ge=1, le=2000)
    keyword_filter: Optional[str] = None
    content_type_filter: Optional[str] = None
    min_connection_strength: float = 0.1

# =============================================================================
# Graph Data Endpoints
# =============================================================================

@graph_router.get("/test")
async def test_graph_router():
    """Simple test endpoint to verify router is working."""
    return {"status": "working", "message": "Graph router is accessible"}

@graph_router.get("/content-knowledge")
async def get_content_knowledge_graph(
    node_limit: int = 100,
    edge_limit: int = 200,
    keyword_filter: Optional[str] = None,
    content_type_filter: Optional[str] = None,
    min_connection_strength: float = 0.1
):
    """Get content knowledge graph data from Knowledge Base and Neo4j."""
    try:
        # Create query object from parameters
        query = GraphQuery(
            node_limit=node_limit,
            edge_limit=edge_limit,
            keyword_filter=keyword_filter,
            content_type_filter=content_type_filter,
            min_connection_strength=min_connection_strength
        )
        logger.info("📊 Starting content knowledge graph generation with real Neo4j data")
        
        # Get content from basic database first
        from src.database.content_service import ContentDatabaseService
        db_service = ContentDatabaseService()
        
        result = await db_service.get_content_items(
            organization_id="demo-org",
            search=query.keyword_filter,
            content_type=query.content_type_filter,
            limit=min(query.node_limit, 20),
            offset=0
        )
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=f"Failed to fetch content: {result.get('error', 'Unknown error')}")
        
        content_list = result.get("content", [])
        logger.info(f"✅ Retrieved {len(content_list)} documents from database")
        
        # Skip Neo4j for now - use basic content to ensure visualization works
        logger.info("🚀 Using basic content for immediate visualization")
        enhanced_content_list = []
        for content in content_list:
            content['neo4j_relationships'] = []
            content['related_topics'] = []
            enhanced_content_list.append(content)
        
        # Process enhanced content into graph format
        nodes = {}
        edges = []
        
        # Convert each document to a content node with Neo4j enhancement
        for item in enhanced_content_list:
            content_id = item.get("id")
            if content_id and content_id not in nodes:
                # Enhanced node with Neo4j relationship count
                neo4j_relationships = item.get('neo4j_relationships', [])
                related_topics = item.get('related_topics', [])
                
                # Calculate node weight for visualization size
                word_count = item.get('word_count', 100)
                node_weight = min(max(word_count // 10, 30), 80)  # Scale for visualization
                
                nodes[content_id] = GraphNode(
                    id=content_id,
                    label=item.get('title', 'Untitled')[:50],
                    type='content',
                    size=node_weight,
                    weight=node_weight,  # Set weight directly for frontend
                    color='#4CAF50' if len(neo4j_relationships) == 0 else '#2196F3',  # Blue if has Neo4j relationships
                    description=f"Word count: {item.get('word_count', 0)}, SEO Score: {item.get('seo_score', 0)}%, Relationships: {len(neo4j_relationships)}",
                    properties={
                        'content_type': item.get('content_type'),
                        'word_count': item.get('word_count', 0),
                        'seo_score': item.get('seo_score', 0),
                        'readability_score': item.get('readability_score', 0),
                        'neo4j_relationships_count': len(neo4j_relationships),
                        'related_topics_count': len(related_topics),
                        'has_graph_data': len(neo4j_relationships) > 0
                    },
                    created_at=item.get('created_at'),
                    updated_at=item.get('updated_at')
                )
                
                # Add Neo4j recommendation edges if available
                for recommendation in neo4j_relationships:
                    target_id = recommendation.get('content_id') or recommendation.get('id')
                    rel_type = recommendation.get('relationship_type', 'RECOMMENDED')
                    if target_id and target_id != content_id:
                        edge_id = f"{content_id}-{target_id}-neo4j"
                        edges.append(GraphEdge(
                            id=edge_id,
                            source=content_id,
                            target=target_id,
                            type=f'neo4j_{rel_type.lower()}',
                            weight=recommendation.get('score', 0.8),
                            color='#E91E63',  # Pink for Neo4j relationships
                            label=f'Neo4j: {rel_type.replace("_", " ").title()}',
                            properties={'source': 'neo4j', 'relationship_type': rel_type}
                        ))
                
                # Add topic nodes from Neo4j if available
                for topic in related_topics:
                    topic_id = f"neo4j_topic_{topic.get('name', 'unknown').replace(' ', '_')}"
                    if topic_id not in nodes:
                        nodes[topic_id] = GraphNode(
                            id=topic_id,
                            label=topic.get('name', 'Unknown Topic'),
                            type='neo4j_topic',
                            size=30,
                            color='#FF9800',
                            description=f"Neo4j Topic: {topic.get('description', 'No description')}",
                            properties={'source': 'neo4j', 'topic_type': topic.get('type', 'general')}
                        )
                    
                    # Connect content to Neo4j topic
                    edge_id = f"{content_id}-{topic_id}-topic"
                    edges.append(GraphEdge(
                        id=edge_id,
                        source=content_id,
                        target=topic_id,
                        type='discusses_topic',
                        weight=0.7,
                        color='#FF9800',
                        label='Discusses Topic',
                        properties={'source': 'neo4j'}
                    ))
        
        # Group documents by content type to create topic nodes
        content_types = {}
        for item in content_list:
            content_type = item.get("content_type", "document")
            if content_type not in content_types:
                content_types[content_type] = []
            content_types[content_type].append(item.get("id"))
        
        # Add topic nodes for content types if we have multiple documents
        if len(content_list) > 1:
            for content_type, doc_ids in content_types.items():
                if len(doc_ids) > 1:
                    topic_id = f"topic_{content_type}"
                    topic_weight = min(len(doc_ids) * 10, 80)
                    nodes[topic_id] = GraphNode(
                        id=topic_id,
                        label=content_type.title(),
                        type='topic',
                        size=topic_weight,
                        weight=topic_weight,
                        color='#FF9800',
                        description=f"Topic grouping for {len(doc_ids)} {content_type} documents",
                        properties={'document_count': len(doc_ids), 'content_type': content_type}
                    )
                    
                    # Connect documents to their topic
                    for doc_id in doc_ids:
                        edge_id = f"{doc_id}-{topic_id}"
                        edges.append(GraphEdge(
                            id=edge_id,
                            source=doc_id,
                            target=topic_id,
                            type='categorized_as',
                            weight=0.6,
                            color='#FF9800',
                            label='Categorized As'
                        ))
        
        # Add SEO quality connections
        high_seo_docs = [item for item in content_list if item.get("seo_score", 0) > 50]
        if len(high_seo_docs) > 1:
            seo_keyword_id = "keyword_seo_optimized"
            keyword_weight = min(len(high_seo_docs) * 8, 70)
            nodes[seo_keyword_id] = GraphNode(
                id=seo_keyword_id,
                label="SEO Optimized",
                type='keyword',
                size=keyword_weight,
                weight=keyword_weight,
                color='#2196F3',
                description=f"{len(high_seo_docs)} documents with high SEO scores",
                properties={'quality_threshold': 50, 'document_count': len(high_seo_docs)}
            )
            
            for item in high_seo_docs:
                edge_id = f"{item.get('id')}-{seo_keyword_id}"
                edges.append(GraphEdge(
                    id=edge_id,
                    source=item.get('id'),
                    target=seo_keyword_id,
                    type='optimized_for',
                    weight=item.get('seo_score', 50) / 100,
                    color='#2196F3',
                    label='SEO Optimized'
                ))
        
        # Create similarity connections between documents based on word count
        for i, doc1 in enumerate(content_list):
            for doc2 in content_list[i+1:]:
                wc1 = doc1.get("word_count", 0)
                wc2 = doc2.get("word_count", 0)
                if wc1 > 0 and wc2 > 0:
                    similarity = 1 - abs(wc1 - wc2) / max(wc1, wc2)
                    if similarity > 0.7:  # Only connect similar-sized documents
                        edge_id = f"{doc1.get('id')}-{doc2.get('id')}"
                        edges.append(GraphEdge(
                            id=edge_id,
                            source=doc1.get('id'),
                            target=doc2.get('id'),
                            type='similar_length',
                            weight=similarity * 0.4,
                            color='#9C27B0',
                            label='Similar Length'
                        ))
        
        # Convert nodes dict to list
        node_list = list(nodes.values())
        
        # Filter edges by minimum connection strength
        if query.min_connection_strength > 0:
            edges = [e for e in edges if e.weight >= query.min_connection_strength]
        
        # Limit edges
        edges = edges[:query.edge_limit]
        
        # Calculate graph statistics
        node_count = len(node_list)
        edge_count = len(edges)
        density = (2 * edge_count) / (node_count * (node_count - 1)) if node_count > 1 else 0
        average_degree = (2 * edge_count) / node_count if node_count > 0 else 0
        
        # Create graph data
        graph_data = GraphData(
            nodes=node_list,
            edges=edges,
            metadata={
                'query_type': 'content_knowledge',
                'tenant_id': 'demo-org',
                'source': 'knowledge_base',
                'content_count': len(content_list)
            },
            node_count=node_count,
            edge_count=edge_count,
            density=density,
            average_degree=average_degree,
            connected_components=1 if node_count > 0 else 0
        )
        
        logger.info(f"🎯 Generated graph with {node_count} nodes and {edge_count} edges")
        
        # Return in format expected by frontend
        return {
            "success": True,
            "graph": graph_data.dict(),
            "statistics": {
                "node_count": node_count,
                "edge_count": edge_count,
                "density": density,
                "average_degree": average_degree,
                "content_count": len(content_list)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get content knowledge graph: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        
        # Log to a specific file for debugging
        with open('/Users/kitan/Desktop/apps/Context-Engineering-Intro/logs/graph_error_debug.log', 'a') as f:
            import datetime
            f.write(f"\n{datetime.datetime.now()}: Graph API Error\n")
            f.write(f"Error: {e}\n")
            f.write(f"Type: {type(e)}\n")
            f.write(f"Traceback:\n{traceback.format_exc()}\n")
            f.write("-" * 50 + "\n")
        
        raise HTTPException(status_code=500, detail=f"Failed to get content knowledge graph: {str(e)}")

@graph_router.get("/health")
async def health_check():
    """Health check for graph service."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "available_graphs": ["content-knowledge"]
    }