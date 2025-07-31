"""
Keyword Network Graph module for the SEO Content Knowledge Graph System.

This module generates keyword relationship networks based on real content analysis.
"""

import logging
from typing import Dict, List, Any
from collections import defaultdict

from .models import GraphNode, GraphEdge, GraphData, GraphQuery, GraphResponse
from .utils import (
    calculate_node_size, get_node_color, extract_node_properties, 
    generate_node_description, calculate_graph_metrics
)

logger = logging.getLogger(__name__)


async def get_keyword_network_graph_data(query: GraphQuery) -> GraphResponse:
    """
    Generate keyword network graph data from real content analysis.
    
    This function creates a visualization of keyword relationships,
    semantic connections, and search performance data.
    """
    try:
        logger.info("🔑 Starting keyword network graph generation")
        
        # Get content from database
        content_list = await _get_content_from_database(query)
        logger.info(f"✅ Retrieved {len(content_list)} documents for keyword analysis")
        
        # Extract keywords from content
        keyword_data = await _extract_keywords_from_content(content_list)
        
        # Generate keyword relationship network
        nodes, edges = _create_keyword_network(keyword_data, query)
        
        # Create graph statistics
        graph_metrics = calculate_graph_metrics(nodes, edges)
        
        # Create graph data object
        graph_data = GraphData(
            nodes=nodes,
            edges=edges,
            metadata={
                'query_type': 'keyword_network',
                'tenant_id': 'demo-org',
                'source': 'content_analysis',
                'content_count': len(content_list),
                'keyword_count': len([n for n in nodes if n.type == 'keyword'])
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
            'keyword_count': len([n for n in nodes if n.type == 'keyword']),
            **graph_metrics
        }
        
        logger.info(f"🎯 Generated keyword network with {len(nodes)} nodes and {len(edges)} edges")
        
        return GraphResponse(
            success=True,
            graph=graph_data,
            statistics=statistics
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to generate keyword network graph: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        raise


async def _get_content_from_database(query: GraphQuery) -> List[Dict[str, Any]]:
    """Get content items from the database with query filters."""
    from src.database.content_service import ContentDatabaseService
    
    db_service = ContentDatabaseService()
    
    result = await db_service.get_content_items(
        organization_id="demo-org",
        search=query.keyword_filter,
        content_type=query.content_type_filter,
        limit=min(query.node_limit, 50),
        offset=0
    )
    
    if not result.get("success"):
        raise Exception(f"Failed to fetch content: {result.get('error', 'Unknown error')}")
    
    return result.get("content", [])


async def _extract_keywords_from_content(content_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract and analyze keywords from content.
    
    This uses a combination of:
    1. Title/content analysis for obvious keywords
    2. Word frequency analysis
    3. SEO-related terms
    """
    keyword_frequency = defaultdict(int)
    keyword_content_map = defaultdict(set)
    keyword_properties = {}
    
    # Common SEO/content-related terms to look for
    seo_keywords = {
        'seo', 'optimization', 'content', 'marketing', 'digital', 'strategy',
        'keyword', 'search', 'engine', 'ranking', 'traffic', 'analytics',
        'performance', 'analysis', 'research', 'guide', 'tips', 'best',
        'practices', 'tools', 'data', 'insights', 'metrics', 'roi'
    }
    
    for content in content_list:
        content_id = content.get('id')
        title = content.get('title', '').lower()
        
        # Extract keywords from title
        title_words = [word.strip('.,!?()[]{}":;') for word in title.split()]
        title_words = [word for word in title_words if len(word) > 2]
        
        # Look for SEO keywords in title
        for word in title_words:
            clean_word = word.lower()
            if clean_word in seo_keywords or len(clean_word) > 4:
                keyword_frequency[clean_word] += 2  # Title words get higher weight
                keyword_content_map[clean_word].add(content_id)
        
        # Analyze content type
        content_type = content.get('content_type', 'document')
        if content_type != 'document':
            keyword_frequency[content_type] += 1
            keyword_content_map[content_type].add(content_id)
        
        # Add SEO score as a keyword dimension
        seo_score = content.get('seo_score', 0)
        if seo_score > 30:
            keyword_frequency['high_seo_quality'] += 1
            keyword_content_map['high_seo_quality'].add(content_id)
        elif seo_score > 0:
            keyword_frequency['needs_seo_improvement'] += 1
            keyword_content_map['needs_seo_improvement'].add(content_id)
    
    # Build keyword properties
    for keyword, freq in keyword_frequency.items():
        content_ids = keyword_content_map[keyword]
        avg_seo_score = 0
        
        # Calculate average SEO score for this keyword
        if content_ids:
            total_seo = sum(
                next((c.get('seo_score', 0) for c in content_list if c.get('id') in content_ids), 0)
                for _ in content_ids
            )
            avg_seo_score = total_seo / len(content_ids) if content_ids else 0
        
        keyword_properties[keyword] = {
            'frequency': freq,
            'content_count': len(content_ids),
            'content_ids': list(content_ids),
            'avg_seo_score': avg_seo_score,
            'importance': freq * len(content_ids),
            'keyword_type': _classify_keyword(keyword)
        }
    
    return {
        'keywords': keyword_properties,
        'content_list': content_list
    }


def _classify_keyword(keyword: str) -> str:
    """Classify keyword type for visualization."""
    seo_terms = {'seo', 'optimization', 'ranking', 'search', 'engine'}
    content_terms = {'content', 'marketing', 'strategy', 'guide', 'tips'}
    technical_terms = {'analytics', 'data', 'metrics', 'performance', 'tools'}
    quality_terms = {'high_seo_quality', 'needs_seo_improvement'}
    
    keyword_lower = keyword.lower()
    
    if keyword_lower in quality_terms:
        return 'quality_indicator'
    elif keyword_lower in seo_terms:
        return 'seo_keyword'
    elif keyword_lower in content_terms:
        return 'content_keyword'
    elif keyword_lower in technical_terms:
        return 'technical_keyword'
    else:
        return 'general_keyword'


def _create_keyword_network(keyword_data: Dict[str, Any], query: GraphQuery) -> tuple[List[GraphNode], List[GraphEdge]]:
    """Create keyword network nodes and edges."""
    nodes = {}
    edges = []
    
    keywords = keyword_data['keywords']
    content_list = keyword_data['content_list']
    
    # Create keyword nodes
    for keyword, props in keywords.items():
        if props['frequency'] >= 1:  # Only include keywords that appear at least once
            # Calculate node weight based on frequency and content coverage
            node_weight = int(min(props['importance'] * 5 + 20, 80))
            keyword_type = props['keyword_type']
            
            nodes[keyword] = GraphNode(
                id=keyword,
                label=keyword.title().replace('_', ' '),
                type='keyword',
                size=node_weight,
                weight=node_weight,
                color=get_node_color('keyword', {'keyword_type': keyword_type}),
                description=generate_node_description(props, 'keyword'),
                properties={
                    'frequency': props['frequency'],
                    'content_count': props['content_count'],
                    'avg_seo_score': props['avg_seo_score'],
                    'keyword_type': keyword_type,
                    'importance': props['importance']
                }
            )
    
    # Create content nodes (smaller, supporting nodes)
    for content in content_list:
        content_id = content.get('id')
        if content_id:
            base_size = calculate_node_size(content, 'content')
            node_weight = int(base_size * 7 // 10)  # Make smaller than keywords using integer math
            
            nodes[content_id] = GraphNode(
                id=content_id,
                label=content.get('title', 'Untitled')[:30],
                type='content',
                size=node_weight,
                weight=node_weight,
                color=get_node_color('content', content),
                description=generate_node_description(content, 'content'),
                properties=extract_node_properties(content, 'content')
            )
    
    # Create keyword-to-content edges
    for keyword, props in keywords.items():
        if keyword in nodes:
            for content_id in props['content_ids']:
                if content_id in nodes:
                    edge_weight = min(props['frequency'] * 0.2 + 0.3, 0.9)
                    edge_id = f"{keyword}-{content_id}"
                    
                    edges.append(GraphEdge(
                        id=edge_id,
                        source=keyword,
                        target=content_id,
                        type='keyword_appears_in',
                        weight=edge_weight,
                        color=get_node_color('keyword'),
                        label='Contains Keyword',
                        properties={
                            'frequency': props['frequency'],
                            'keyword_type': props['keyword_type']
                        }
                    ))
    
    # Create keyword-to-keyword edges (semantic relationships)
    keyword_list = list(keywords.keys())
    for i, keyword1 in enumerate(keyword_list):
        for keyword2 in keyword_list[i+1:]:
            if keyword1 in nodes and keyword2 in nodes:
                similarity = _calculate_keyword_similarity(
                    keywords[keyword1], keywords[keyword2]
                )
                
                if similarity > 0.3:  # Only connect related keywords
                    edge_id = f"{keyword1}-{keyword2}-semantic"
                    edge_weight = similarity * 0.6
                    
                    edges.append(GraphEdge(
                        id=edge_id,
                        source=keyword1,
                        target=keyword2,
                        type='semantic_relationship',
                        weight=edge_weight,
                        color='#9C27B0',
                        label='Related Keywords',
                        properties={
                            'similarity_score': similarity,
                            'relationship_type': 'semantic'
                        }
                    ))
    
    # Create keyword clusters by type
    keyword_types = defaultdict(list)
    for keyword, props in keywords.items():
        if keyword in nodes:
            keyword_types[props['keyword_type']].append(keyword)
    
    # Add cluster center nodes for keyword types with multiple keywords
    for keyword_type, type_keywords in keyword_types.items():
        if len(type_keywords) > 1:
            cluster_id = f"cluster_{keyword_type}"
            cluster_weight = int(min(len(type_keywords) * 8 + 30, 70))
            
            nodes[cluster_id] = GraphNode(
                id=cluster_id,
                label=keyword_type.replace('_', ' ').title(),
                type='cluster',
                size=cluster_weight,
                weight=cluster_weight,
                color=get_node_color('cluster'),
                description=f"Cluster of {len(type_keywords)} {keyword_type.replace('_', ' ')} keywords",
                properties={
                    'cluster_type': keyword_type,
                    'keyword_count': len(type_keywords)
                }
            )
            
            # Connect keywords to their cluster
            for keyword in type_keywords:
                edge_id = f"{keyword}-{cluster_id}"
                edges.append(GraphEdge(
                    id=edge_id,
                    source=keyword,
                    target=cluster_id,
                    type='cluster_membership',
                    weight=0.5,
                    color=get_node_color('cluster'),
                    label='Member Of',
                    properties={'cluster_type': keyword_type}
                ))
    
    # Filter edges by minimum connection strength
    if query.min_connection_strength > 0:
        edges = [e for e in edges if e.weight >= query.min_connection_strength]
    
    # Limit edges
    edges = edges[:query.edge_limit]
    
    # Convert nodes dict to list and limit
    node_list = list(nodes.values())[:query.node_limit]
    
    return node_list, edges


def _calculate_keyword_similarity(keyword1_props: Dict[str, Any], keyword2_props: Dict[str, Any]) -> float:
    """Calculate similarity between two keywords based on their properties."""
    similarity = 0.0
    
    # Same keyword type gets bonus
    if keyword1_props['keyword_type'] == keyword2_props['keyword_type']:
        similarity += 0.4
    
    # Similar content overlap
    content1 = set(keyword1_props['content_ids'])
    content2 = set(keyword2_props['content_ids'])
    
    if content1 and content2:
        overlap = len(content1.intersection(content2))
        total = len(content1.union(content2))
        if total > 0:
            content_similarity = overlap / total
            similarity += content_similarity * 0.5
    
    # Similar SEO scores
    seo1 = keyword1_props['avg_seo_score']
    seo2 = keyword2_props['avg_seo_score']
    if abs(seo1 - seo2) < 15:  # Similar SEO performance
        similarity += 0.2
    
    # Similar frequency/importance
    freq1 = keyword1_props['frequency']
    freq2 = keyword2_props['frequency']
    if freq1 > 0 and freq2 > 0:
        freq_similarity = 1 - abs(freq1 - freq2) / max(freq1, freq2)
        similarity += freq_similarity * 0.3
    
    return min(similarity, 1.0)