"""
Main Graph Router for the SEO Content Knowledge Graph System.

This module consolidates all graph endpoints into a single router.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import ValidationError

from .models import GraphQuery, GraphResponse
from .content_knowledge import get_content_knowledge_graph_data
from .competitor_landscape import get_competitor_landscape_graph_data
from .semantic_clusters import get_semantic_clusters_graph_data
from .keyword_network import get_keyword_network_graph_data

# Import authentication dependencies
try:
    from src.auth.auth_middleware import get_current_user, get_current_organization
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False
    logger.warning("Authentication middleware not available for graph endpoints")

logger = logging.getLogger(__name__)

# Create main graph router
graph_router = APIRouter(prefix="/graph", tags=["graph"])


# =============================================================================
# Health and Test Endpoints
# =============================================================================

@graph_router.get("/test")
async def test_graph_router():
    """Simple test endpoint to verify router is working."""
    return {
        "status": "working", 
        "message": "Modular graph router is accessible",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoints": [
            "/graph/content-knowledge",
            "/graph/content_relationships",
            "/graph/keyword-network",
            "/graph/competitor-landscape", 
            "/graph/semantic-clusters",
            "/graph/stats",
            "/graph/health"
        ]
    }


@graph_router.get("/health")
async def health_check():
    """Health check for graph service."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "available_graphs": [
            "content-knowledge",
            "competitor-landscape", 
            "semantic-clusters"
        ],
        "service": "modular_graph_api",
        "version": "2.0.0"
    }


# =============================================================================
# Graph Data Endpoints
# =============================================================================

async def get_organization_id(organization_id: Optional[str] = Query(None, description="Organization ID")):
    """Get organization ID from query parameter, authentication, or use demo-org as fallback."""
    # Use query parameter if provided
    if organization_id:
        logger.info(f"🏢 Using organization_id from URL parameter: '{organization_id}'")
        return organization_id
    
    # Try authentication
    if AUTH_AVAILABLE:
        try:
            auth_org_id = await get_current_organization()
            logger.info(f"🔐 Using organization_id from authentication: '{auth_org_id}'")
            return auth_org_id
        except Exception as e:
            logger.warning(f"⚠️ Authentication failed, using demo-org fallback: {e}")
            return "demo-org"
    
    logger.info("📝 Using demo-org fallback (no auth available)")
    return "demo-org"

@graph_router.get("/content-knowledge")
async def get_content_knowledge_graph(
    node_limit: int = Query(100, ge=1, le=1000, description="Maximum number of nodes"),
    edge_limit: int = Query(200, ge=1, le=2000, description="Maximum number of edges"),
    keyword_filter: Optional[str] = Query(None, description="Filter by keyword"),
    content_type_filter: Optional[str] = Query(None, description="Filter by content type"),
    min_connection_strength: float = Query(0.1, ge=0.0, le=1.0, description="Minimum edge weight"),
    organization_id: str = Depends(get_organization_id)
):
    """
    Get content knowledge graph data from Knowledge Base.
    
    This endpoint creates a visualization of content relationships,
    topic connections, and SEO quality groupings.
    """
    try:
        # Create query object with organization context
        query = GraphQuery(
            node_limit=node_limit,
            edge_limit=edge_limit,
            keyword_filter=keyword_filter,
            content_type_filter=content_type_filter,
            min_connection_strength=min_connection_strength,
            organization_id=organization_id if AUTH_AVAILABLE else "demo-org"
        )
        
        # Get graph data
        response = await get_content_knowledge_graph_data(query)
        
        # Return in format expected by frontend
        return {
            "success": response.success,
            "graph": response.graph.dict(),
            "statistics": response.statistics,
            "message": response.message
        }
        
    except ValidationError as e:
        logger.error(f"Validation error in content knowledge graph: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid query parameters: {e}")
    except Exception as e:
        logger.error(f"Failed to get content knowledge graph: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to generate content knowledge graph: {str(e)}")


# Alias endpoint for backward compatibility with frontend
@graph_router.get("/content_relationships")
async def get_content_relationships_graph(
    node_limit: int = Query(100, ge=1, le=1000, description="Maximum number of nodes"),
    edge_limit: int = Query(200, ge=1, le=2000, description="Maximum number of edges"),
    keyword_filter: Optional[str] = Query(None, description="Filter by keyword"),
    content_type_filter: Optional[str] = Query(None, description="Filter by content type"),
    min_connection_strength: float = Query(0.1, ge=0.0, le=1.0, description="Minimum edge weight"),
    organization_id: str = Depends(get_organization_id)
):
    """
    Get content knowledge graph data from Knowledge Base (alias for content-knowledge).
    
    This endpoint creates a visualization of content relationships,
    topic connections, and SEO quality groupings.
    """
    # Redirect to the main content knowledge graph endpoint
    return await get_content_knowledge_graph(
        node_limit=node_limit,
        edge_limit=edge_limit,
        keyword_filter=keyword_filter,
        content_type_filter=content_type_filter,
        min_connection_strength=min_connection_strength,
        organization_id=organization_id
    )


@graph_router.get("/keyword-network")
async def get_keyword_network_graph(
    node_limit: int = Query(80, ge=1, le=500, description="Maximum number of nodes"),
    edge_limit: int = Query(120, ge=1, le=1000, description="Maximum number of edges"),
    keyword_filter: Optional[str] = Query(None, description="Filter by keyword"),
    content_type_filter: Optional[str] = Query(None, description="Filter by content type"),
    min_connection_strength: float = Query(0.2, ge=0.0, le=1.0, description="Minimum edge weight")
):
    """
    Get keyword network graph data from real content analysis.
    
    This endpoint creates a visualization of keyword relationships,
    semantic connections, and search performance data.
    """
    try:
        # Create query object
        query = GraphQuery(
            node_limit=node_limit,
            edge_limit=edge_limit,
            keyword_filter=keyword_filter,
            content_type_filter=content_type_filter,
            min_connection_strength=min_connection_strength
        )
        
        # Get graph data
        response = await get_keyword_network_graph_data(query)
        
        # Return in format expected by frontend
        return {
            "success": response.success,
            "graph": response.graph.dict(),
            "statistics": response.statistics,
            "message": response.message
        }
        
    except ValidationError as e:
        logger.error(f"Validation error in keyword network graph: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid query parameters: {e}")
    except Exception as e:
        logger.error(f"Failed to get keyword network graph: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to generate keyword network graph: {str(e)}")


@graph_router.get("/competitor-landscape")
async def get_competitor_landscape_graph(
    node_limit: int = Query(100, ge=1, le=500, description="Maximum number of nodes"),
    edge_limit: int = Query(150, ge=1, le=1000, description="Maximum number of edges"),
    keyword_filter: Optional[str] = Query(None, description="Filter by keyword"),
    content_type_filter: Optional[str] = Query(None, description="Filter by content type"),
    min_connection_strength: float = Query(0.2, ge=0.0, le=1.0, description="Minimum edge weight")
):
    """
    Get competitor landscape graph data.
    
    This endpoint creates a visualization of competitive relationships
    based on content topics, keyword overlap, and SEO performance.
    """
    try:
        # Create query object
        query = GraphQuery(
            node_limit=node_limit,
            edge_limit=edge_limit,
            keyword_filter=keyword_filter,
            content_type_filter=content_type_filter,
            min_connection_strength=min_connection_strength
        )
        
        # Get graph data
        response = await get_competitor_landscape_graph_data(query)
        
        # Return in format expected by frontend
        return {
            "success": response.success,
            "graph": response.graph.dict(),
            "statistics": response.statistics,
            "message": response.message
        }
        
    except ValidationError as e:
        logger.error(f"Validation error in competitor landscape graph: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid query parameters: {e}")
    except Exception as e:
        logger.error(f"Failed to get competitor landscape graph: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to generate competitor landscape graph: {str(e)}")


@graph_router.get("/semantic-clusters")
async def get_semantic_clusters_graph(
    node_limit: int = Query(80, ge=1, le=400, description="Maximum number of nodes"),
    edge_limit: int = Query(120, ge=1, le=800, description="Maximum number of edges"),
    keyword_filter: Optional[str] = Query(None, description="Filter by keyword"),
    content_type_filter: Optional[str] = Query(None, description="Filter by content type"),
    min_connection_strength: float = Query(0.3, ge=0.0, le=1.0, description="Minimum edge weight"),
    cluster_by: Optional[str] = Query(None, description="Clustering method")
):
    """
    Get semantic clusters graph data.
    
    This endpoint creates clusters of semantically related content
    using AI-driven topic modeling and similarity analysis.
    """
    try:
        # Create query object
        query = GraphQuery(
            node_limit=node_limit,
            edge_limit=edge_limit,
            keyword_filter=keyword_filter,
            content_type_filter=content_type_filter,
            min_connection_strength=min_connection_strength,
            cluster_by=cluster_by
        )
        
        # Get graph data
        response = await get_semantic_clusters_graph_data(query)
        
        # Return in format expected by frontend
        return {
            "success": response.success,
            "graph": response.graph.dict(),
            "statistics": response.statistics,
            "message": response.message
        }
        
    except ValidationError as e:
        logger.error(f"Validation error in semantic clusters graph: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid query parameters: {e}")
    except Exception as e:
        logger.error(f"Failed to get semantic clusters graph: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to generate semantic clusters graph: {str(e)}")


# =============================================================================
# Statistics and Analytics Endpoints
# =============================================================================

@graph_router.get("/stats")
async def get_graph_statistics():
    """Get comprehensive graph statistics across all graph types."""
    try:
        logger.info("📊 Getting comprehensive graph statistics")
        
        # Get basic content stats
        from src.database.content_service import ContentDatabaseService
        db_service = ContentDatabaseService()
        
        content_result = await db_service.get_content_items(
            organization_id="demo-org",
            limit=100
        )
        
        content_count = len(content_result.get("content", [])) if content_result.get("success") else 0
        
        # Calculate derived statistics
        stats = {
            "content_statistics": {
                "total_content": content_count,
                "avg_content_per_graph": content_count / 3 if content_count > 0 else 0,
                "content_coverage": min(content_count / 50 * 100, 100)  # Percentage towards 50 content goal
            },
            "graph_performance": {
                "content_knowledge_health": "excellent" if content_count > 0 else "needs_content",
                "competitor_analysis_ready": content_count >= 5,
                "clustering_feasible": content_count >= 3,
                "recommendation_quality": "high" if content_count > 10 else "medium" if content_count > 5 else "low"
            },
            "system_metrics": {
                "total_graphs_available": 3,
                "avg_response_time_ms": 250,  # Estimated based on complexity
                "cache_hit_ratio": 0.0,  # No caching implemented yet
                "error_rate": 0.0
            },
            "data_quality": {
                "content_completeness": min(content_count / 20 * 100, 100),
                "seo_analysis_coverage": 100 if content_count > 0 else 0,
                "relationship_density": min(content_count * 0.1, 1.0),
                "topic_diversity": min(content_count / 10, 5)
            },
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get graph statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get graph statistics: {str(e)}")


# =============================================================================
# Utility Endpoints
# =============================================================================

@graph_router.get("/node-types")
async def get_supported_node_types():
    """Get all supported node types and their properties."""
    return {
        "node_types": {
            "content": {
                "description": "Content articles and documents",
                "color": "#22c55e",
                "properties": ["word_count", "seo_score", "content_type", "readability_score"],
                "size_based_on": "word_count and seo_score"
            },
            "topic": {
                "description": "Content topics and categories",
                "color": "#8b5cf6", 
                "properties": ["content_count", "avg_seo_score", "related_topics"],
                "size_based_on": "content_count"
            },
            "keyword": {
                "description": "SEO keywords and search terms",
                "color": "#f59e0b",
                "properties": ["search_volume", "competition", "difficulty", "trend_direction"],
                "size_based_on": "search_volume"
            },
            "competitor": {
                "description": "Competitive positions and market segments",
                "color": "#ef4444",
                "properties": ["authority_score", "content_overlap", "competitive_strength"],
                "size_based_on": "authority_score"
            },
            "cluster": {
                "description": "Semantic content clusters",
                "color": "#6366f1",
                "properties": ["cluster_size", "avg_seo_score", "coherence_level"],
                "size_based_on": "cluster_size and performance"
            },
            "opportunity": {
                "description": "Content gaps and opportunities",
                "color": "#8b5cf6",
                "properties": ["gap_type", "priority", "opportunity_score"],
                "size_based_on": "opportunity_score"
            }
        },
        "edge_types": {
            "content_similarity": "Similarity between content pieces",
            "topic_connection": "Content belongs to topic",
            "competitive_coverage": "Competitive position covers topic",
            "cluster_membership": "Content belongs to cluster",
            "neo4j_relationship": "Relationship from Neo4j graph database",
            "seo_boost": "Content has high SEO optimization",
            "improvement_opportunity": "Gap represents improvement opportunity"
        }
    }


@graph_router.get("/config")
async def get_graph_configuration():
    """Get current graph configuration and limits."""
    return {
        "limits": {
            "max_nodes": 1000,
            "max_edges": 2000,
            "max_content_nodes": 50,
            "max_clusters": 10
        },
        "defaults": {
            "node_limit": 100,
            "edge_limit": 200,
            "min_connection_strength": 0.1,
            "cache_ttl_seconds": 600
        },
        "features": {
            "neo4j_integration": True,
            "real_time_updates": False,
            "caching_enabled": False,
            "rate_limiting": False
        },
        "performance": {
            "estimated_response_time_ms": {
                "content_knowledge": 200,
                "competitor_landscape": 300,
                "semantic_clusters": 400
            },
            "memory_usage_mb": {
                "small_graph": 5,
                "medium_graph": 15,
                "large_graph": 40
            }
        }
    }