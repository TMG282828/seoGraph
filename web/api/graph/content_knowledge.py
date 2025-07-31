"""
Content Knowledge Graph module for the SEO Content Knowledge Graph System.

This module handles the content knowledge graph endpoint with real data integration.
"""

import logging
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from .models import GraphNode, GraphEdge, GraphData, GraphQuery, GraphResponse
from .utils import (
    calculate_node_size, get_node_color, extract_node_properties, 
    generate_node_description, calculate_graph_metrics
)

logger = logging.getLogger(__name__)


async def get_content_knowledge_graph_data(query: GraphQuery) -> GraphResponse:
    """
    Generate content knowledge graph data from Knowledge Base and Neo4j.
    
    This function creates a graph visualization of content relationships,
    topic connections, and SEO quality groupings.
    """
    try:
        logger.info("📊 Starting content knowledge graph generation")
        
        # Get content from basic database
        content_list = await _get_content_from_database(query)
        logger.info(f"✅ Retrieved {len(content_list)} documents from database")
        
        # Try to enhance with Neo4j data (with timeout protection)
        enhanced_content_list = await _enhance_with_neo4j_data(content_list)
        
        # Process content into graph format
        nodes, edges = _create_graph_elements(enhanced_content_list, query)
        
        # Create graph statistics
        graph_metrics = calculate_graph_metrics(nodes, edges)
        
        # Create graph data object
        graph_data = GraphData(
            nodes=nodes,
            edges=edges,
            metadata={
                'query_type': 'content_knowledge',
                'tenant_id': query.organization_id or 'demo-org',
                'source': 'knowledge_base',
                'content_count': len(content_list)
            },
            node_count=len(nodes),
            edge_count=len(edges),
            density=graph_metrics['density'],
            average_degree=graph_metrics['average_degree'],
            connected_components=graph_metrics['connected_components']
        )
        
        # Create response
        statistics = {
            'node_count': len(nodes),
            'edge_count': len(edges),
            'content_count': len(content_list),
            **graph_metrics
        }
        
        logger.info(f"🎯 Generated graph with {len(nodes)} nodes and {len(edges)} edges")
        
        return GraphResponse(
            success=True,
            graph=graph_data,
            statistics=statistics
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to generate content knowledge graph: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        raise


async def _get_content_from_database(query: GraphQuery) -> List[Dict[str, Any]]:
    """Get content items from the database with query filters."""
    from src.database.content_service import ContentDatabaseService
    
    # Debug: Log the query parameters
    actual_org_id = query.organization_id or "demo-org"
    logger.info(f"🔍 _get_content_from_database called with query.organization_id='{query.organization_id}', using actual_org_id='{actual_org_id}'")
    logger.info(f"📋 Query filters: search='{query.keyword_filter}', content_type='{query.content_type_filter}', limit={min(query.node_limit, 50)}")
    print(f"DEBUG: _get_content_from_database called with org_id='{actual_org_id}'")
    
    db_service = ContentDatabaseService()
    
    result = await db_service.get_content_items(
        organization_id=actual_org_id,
        search=query.keyword_filter,
        content_type=query.content_type_filter,
        limit=min(query.node_limit, 50),
        offset=0
    )
    
    logger.info(f"📊 Database result: success={result.get('success')}, content_count={len(result.get('content', []))}, total={result.get('total', 0)}")
    
    if not result.get("success"):
        logger.error(f"❌ Database query failed: {result.get('error', 'Unknown error')}")
        raise Exception(f"Failed to fetch content: {result.get('error', 'Unknown error')}")
    
    content_items = result.get("content", [])
    logger.info(f"🎯 Final result: returning {len(content_items)} content items")
    return content_items


async def _enhance_with_neo4j_data(content_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enhance content data with Neo4j relationships (with timeout protection).
    
    This function tries to add Neo4j relationship data to content items,
    but falls back gracefully if Neo4j is unavailable or slow.
    """
    enhanced_content_list = []
    
    try:
        def get_neo4j_data():
            try:
                from src.database.neo4j_client import neo4j_client
                logger.info("🔗 Attempting to enhance with Neo4j relationship data")
                
                # Set organization context
                neo4j_client.set_organization_context("demo-org")
                
                enhanced_list = []
                
                # Get topic hierarchy once for all content (optimization)
                try:
                    topic_hierarchy = neo4j_client.get_topic_hierarchy()
                    logger.info(f"Retrieved {len(topic_hierarchy)} topics from hierarchy")
                except Exception as e:
                    logger.warning(f"Failed to get topic hierarchy: {e}")
                    topic_hierarchy = []
                
                for content in content_list:
                    content_id = content.get('id')
                    
                    try:
                        # Get Neo4j recommendations (limit to 3 for performance)
                        recommendations = neo4j_client.get_content_recommendations(content_id, limit=3)
                        
                        content['neo4j_relationships'] = recommendations
                        content['related_topics'] = [
                            topic for topic in topic_hierarchy 
                            if topic.get('name')
                        ][:3]
                        enhanced_list.append(content)
                        
                        logger.info(f"Enhanced content {content_id} with {len(recommendations)} recommendations")
                        
                    except Exception as neo4j_error:
                        logger.warning(f"Failed to get Neo4j data for {content_id}: {neo4j_error}")
                        content['neo4j_relationships'] = []
                        content['related_topics'] = []
                        enhanced_list.append(content)
                        
                return enhanced_list
                
            except Exception as e:
                logger.warning(f"Neo4j client initialization failed: {e}")
                return content_list
        
        # Run Neo4j enhancement with timeout
        with ThreadPoolExecutor() as executor:
            future = executor.submit(get_neo4j_data)
            try:
                enhanced_content_list = future.result(timeout=3.0)
                logger.info("✅ Neo4j enhancement completed successfully")
            except TimeoutError:
                logger.warning("⏰ Neo4j enhancement timed out, using basic content")
                enhanced_content_list = content_list
                # Add empty Neo4j data to prevent errors
                for content in enhanced_content_list:
                    content['neo4j_relationships'] = []
                    content['related_topics'] = []
                    
    except Exception as neo4j_error:
        logger.warning(f"Neo4j enhancement failed, using basic content: {neo4j_error}")
        enhanced_content_list = content_list
        # Add empty Neo4j data to prevent errors
        for content in enhanced_content_list:
            content['neo4j_relationships'] = []
            content['related_topics'] = []
    
    return enhanced_content_list


def _create_graph_elements(content_list: List[Dict[str, Any]], 
                          query: GraphQuery) -> tuple[List[GraphNode], List[GraphEdge]]:
    """Create graph nodes and edges from content data."""
    nodes = {}
    edges = []
    
    # Create content nodes
    for item in content_list:
        content_id = item.get("id")
        if content_id and content_id not in nodes:
            neo4j_relationships = item.get('neo4j_relationships', [])
            related_topics = item.get('related_topics', [])
            
            # Calculate node properties
            node_weight = calculate_node_size(item, 'content')
            node_color = get_node_color('content', item)
            
            nodes[content_id] = GraphNode(
                id=content_id,
                label=item.get('title', 'Untitled')[:50],
                type='content',
                size=node_weight,
                weight=node_weight,
                color=node_color,
                description=generate_node_description(item, 'content'),
                properties=extract_node_properties(item, 'content'),
                created_at=item.get('created_at'),
                updated_at=item.get('updated_at')
            )
            
            # Add Neo4j relationship edges
            for recommendation in neo4j_relationships:
                target_id = recommendation.get('content_id') or recommendation.get('id')
                if target_id and target_id != content_id:
                    edge_id = f"{content_id}-{target_id}-neo4j"
                    edges.append(GraphEdge(
                        id=edge_id,
                        source=content_id,
                        target=target_id,
                        type='neo4j_relationship',
                        weight=recommendation.get('score', 0.8),
                        color='#E91E63',
                        label='Neo4j Recommendation',
                        properties={'source': 'neo4j', 'recommendation_type': 'content'}
                    ))
            
            # Add topic nodes from Neo4j
            for topic in related_topics:
                topic_name = topic.get('name', 'Unknown Topic')
                topic_id = f"neo4j_topic_{topic_name.replace(' ', '_')}"
                
                if topic_id not in nodes:
                    topic_weight = calculate_node_size(topic, 'topic')
                    nodes[topic_id] = GraphNode(
                        id=topic_id,
                        label=topic_name,
                        type='topic',
                        size=topic_weight,
                        weight=topic_weight,
                        color=get_node_color('topic'),
                        description=generate_node_description(topic, 'topic'),
                        properties=extract_node_properties(topic, 'topic')
                    )
                
                # Connect content to topic
                edge_id = f"{content_id}-{topic_id}-topic"
                edges.append(GraphEdge(
                    id=edge_id,
                    source=content_id,
                    target=topic_id,
                    type='topic_connection',
                    weight=0.7,
                    color=get_node_color('topic'),
                    label='Discusses Topic',
                    properties={'source': 'neo4j'}
                ))
    
    # Group documents by content type
    content_types = {}
    for item in content_list:
        content_type = item.get("content_type", "document")
        if content_type not in content_types:
            content_types[content_type] = []
        content_types[content_type].append(item.get("id"))
    
    # Add topic nodes for content types (if multiple documents)
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
                    color=get_node_color('topic'),
                    description=f"Topic grouping for {len(doc_ids)} {content_type} documents",
                    properties={
                        'document_count': len(doc_ids), 
                        'content_type': content_type
                    }
                )
                
                # Connect documents to their topic
                for doc_id in doc_ids:
                    if doc_id in nodes:
                        edge_id = f"{doc_id}-{topic_id}"
                        edges.append(GraphEdge(
                            id=edge_id,
                            source=doc_id,
                            target=topic_id,
                            type='content_cluster',
                            weight=0.6,
                            color=get_node_color('topic'),
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
            color=get_node_color('keyword'),
            description=f"{len(high_seo_docs)} documents with high SEO scores",
            properties={'quality_threshold': 50, 'document_count': len(high_seo_docs)}
        )
        
        for item in high_seo_docs:
            edge_id = f"{item.get('id')}-{seo_keyword_id}"
            edges.append(GraphEdge(
                id=edge_id,
                source=item.get('id'),
                target=seo_keyword_id,
                type='seo_boost',
                weight=item.get('seo_score', 50) / 100,
                color=get_node_color('keyword'),
                label='SEO Optimized'
            ))
    
    # Create similarity connections between documents
    for i, doc1 in enumerate(content_list):
        for doc2 in content_list[i+1:]:
            similarity = _calculate_content_similarity(doc1, doc2)
            if similarity > 0.7:  # Only connect similar documents
                edge_id = f"{doc1.get('id')}-{doc2.get('id')}"
                edges.append(GraphEdge(
                    id=edge_id,
                    source=doc1.get('id'),
                    target=doc2.get('id'),
                    type='content_similarity',
                    weight=similarity * 0.4,
                    color='#9C27B0',
                    label='Similar Content'
                ))
    
    # Filter edges by minimum connection strength
    if query.min_connection_strength > 0:
        edges = [e for e in edges if e.weight >= query.min_connection_strength]
    
    # Limit edges
    edges = edges[:query.edge_limit]
    
    # Convert nodes dict to list
    node_list = list(nodes.values())
    
    return node_list, edges


def _calculate_content_similarity(doc1: Dict[str, Any], doc2: Dict[str, Any]) -> float:
    """Calculate similarity between two documents based on various factors."""
    similarity_score = 0.0
    
    # Word count similarity
    wc1 = doc1.get("word_count", 0)
    wc2 = doc2.get("word_count", 0)
    if wc1 > 0 and wc2 > 0:
        wc_similarity = 1 - abs(wc1 - wc2) / max(wc1, wc2)
        similarity_score += wc_similarity * 0.3
    
    # Content type similarity
    if doc1.get("content_type") == doc2.get("content_type"):
        similarity_score += 0.4
    
    # SEO score similarity
    seo1 = doc1.get("seo_score", 0)
    seo2 = doc2.get("seo_score", 0)
    if abs(seo1 - seo2) < 20:  # Similar SEO scores
        similarity_score += 0.3
    
    return min(similarity_score, 1.0)