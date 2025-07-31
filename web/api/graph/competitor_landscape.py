"""
Competitor Landscape Graph module for the SEO Content Knowledge Graph System.

This module handles competitor analysis visualization based on content topics,
keyword overlap, and competitive positioning.
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


async def get_competitor_landscape_graph_data(query: GraphQuery) -> GraphResponse:
    """
    Generate competitor landscape graph data based on content analysis.
    
    This function creates a visualization of competitive relationships
    based on content topics, keyword overlap, and SEO performance.
    """
    try:
        logger.info("🏆 Starting competitor landscape graph generation")
        
        # Get content data for competitive analysis
        content_list = await _get_content_for_analysis(query)
        logger.info(f"✅ Retrieved {len(content_list)} documents for competitor analysis")
        
        # Analyze competitive landscape from content
        competitive_data = await _analyze_competitive_landscape(content_list)
        
        # Create graph elements
        nodes, edges = _create_competitor_graph_elements(competitive_data, query)
        
        # Calculate graph metrics
        graph_metrics = calculate_graph_metrics(nodes, edges)
        
        # Create graph data
        graph_data = GraphData(
            nodes=nodes,
            edges=edges,
            metadata={
                'query_type': 'competitor_landscape',
                'tenant_id': 'demo-org',
                'source': 'content_analysis',
                'analysis_type': 'competitive_positioning'
            },
            node_count=len(nodes),
            edge_count=len(edges),
            density=graph_metrics['density'],
            average_degree=graph_metrics['average_degree'],
            connected_components=graph_metrics['connected_components']
        )
        
        # Create response
        statistics = {
            'competitor_count': competitive_data.get('competitor_count', 0),
            'topic_overlap_avg': competitive_data.get('topic_overlap_avg', 0),
            'competitive_gaps': competitive_data.get('competitive_gaps', []),
            **graph_metrics
        }
        
        logger.info(f"🎯 Generated competitor landscape with {len(nodes)} nodes and {len(edges)} edges")
        
        return GraphResponse(
            success=True,
            graph=graph_data,
            statistics=statistics
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to generate competitor landscape graph: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        raise


async def _get_content_for_analysis(query: GraphQuery) -> List[Dict[str, Any]]:
    """Get content items for competitive analysis."""
    from src.database.content_service import ContentDatabaseService
    
    db_service = ContentDatabaseService()
    
    result = await db_service.get_content_items(
        organization_id="demo-org",
        search=query.keyword_filter,
        content_type=query.content_type_filter,
        limit=min(query.node_limit, 30),
        offset=0
    )
    
    if not result.get("success"):
        raise Exception(f"Failed to fetch content: {result.get('error', 'Unknown error')}")
    
    return result.get("content", [])


async def _analyze_competitive_landscape(content_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze competitive landscape based on content topics and performance.
    
    This creates a competitive analysis based on:
    1. Topic coverage and gaps
    2. Content performance tiers
    3. SEO positioning
    4. Content type distribution
    """
    if not content_list:
        raise Exception("No content available for competitive analysis")
    
    # Analyze topics and coverage
    topic_analysis = _analyze_topic_coverage(content_list)
    
    # Analyze performance tiers
    performance_tiers = _analyze_performance_tiers(content_list)
    
    # Identify competitive gaps
    competitive_gaps = _identify_competitive_gaps(content_list, topic_analysis)
    
    # Create competitive positioning
    competitive_positions = _create_competitive_positions(content_list, performance_tiers)
    
    return {
        'content_count': len(content_list),
        'competitor_count': len(competitive_positions),
        'topic_analysis': topic_analysis,
        'performance_tiers': performance_tiers,
        'competitive_gaps': competitive_gaps,
        'competitive_positions': competitive_positions,
        'topic_overlap_avg': sum(tier.get('topic_coverage', 0) for tier in performance_tiers.values()) / len(performance_tiers) if performance_tiers else 0
    }


def _analyze_topic_coverage(content_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze topic coverage across content."""
    topics = defaultdict(list)
    content_types = defaultdict(int)
    
    for content in content_list:
        content_type = content.get('content_type', 'document')
        content_types[content_type] += 1
        
        # Extract topics from content (simplified)
        title = content.get('title', '').lower()
        
        # Basic topic extraction based on common SEO terms
        if any(term in title for term in ['seo', 'optimization', 'search']):
            topics['SEO & Optimization'].append(content['id'])
        elif any(term in title for term in ['content', 'marketing', 'strategy']):
            topics['Content Marketing'].append(content['id'])
        elif any(term in title for term in ['technical', 'development', 'code']):
            topics['Technical'].append(content['id'])
        elif any(term in title for term in ['analytics', 'data', 'metrics']):
            topics['Analytics'].append(content['id'])
        else:
            topics['General'].append(content['id'])
    
    return {
        'topics': dict(topics),
        'content_types': dict(content_types),
        'topic_diversity': len(topics),
        'content_type_diversity': len(content_types)
    }


def _analyze_performance_tiers(content_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze content performance and create competitive tiers."""
    if not content_list:
        return {}
    
    # Sort content by SEO score
    sorted_content = sorted(content_list, key=lambda x: x.get('seo_score', 0), reverse=True)
    
    # Create performance tiers
    total_content = len(sorted_content)
    tier_size = max(1, total_content // 3)
    
    tiers = {
        'high_performers': {
            'content': sorted_content[:tier_size],
            'avg_seo_score': sum(c.get('seo_score', 0) for c in sorted_content[:tier_size]) / tier_size if tier_size > 0 else 0,
            'topic_coverage': len(set(c.get('content_type', 'unknown') for c in sorted_content[:tier_size])),
            'competitive_strength': 'strong'
        },
        'medium_performers': {
            'content': sorted_content[tier_size:tier_size*2],
            'avg_seo_score': sum(c.get('seo_score', 0) for c in sorted_content[tier_size:tier_size*2]) / tier_size if tier_size > 0 else 0,
            'topic_coverage': len(set(c.get('content_type', 'unknown') for c in sorted_content[tier_size:tier_size*2])),
            'competitive_strength': 'moderate'
        },
        'improvement_needed': {
            'content': sorted_content[tier_size*2:],
            'avg_seo_score': sum(c.get('seo_score', 0) for c in sorted_content[tier_size*2:]) / max(1, len(sorted_content[tier_size*2:])),
            'topic_coverage': len(set(c.get('content_type', 'unknown') for c in sorted_content[tier_size*2:])),
            'competitive_strength': 'weak'
        }
    }
    
    return tiers


def _identify_competitive_gaps(content_list: List[Dict[str, Any]], 
                              topic_analysis: Dict[str, Any]) -> List[str]:
    """Identify competitive gaps and opportunities."""
    gaps = []
    
    topics = topic_analysis.get('topics', {})
    
    # Check for missing or underrepresented topics
    if len(topics.get('SEO & Optimization', [])) < 2:
        gaps.append('SEO Optimization content')
    
    if len(topics.get('Technical', [])) < 1:
        gaps.append('Technical content')
    
    if len(topics.get('Analytics', [])) < 1:
        gaps.append('Analytics and metrics content')
    
    # Check for low-performing content types
    content_types = topic_analysis.get('content_types', {})
    if 'guide' not in content_types:
        gaps.append('Comprehensive guides')
    
    if 'tutorial' not in content_types:
        gaps.append('Tutorial content')
    
    # Check average performance
    avg_seo_score = sum(c.get('seo_score', 0) for c in content_list) / len(content_list) if content_list else 0
    if avg_seo_score < 50:
        gaps.append('Overall SEO optimization')
    
    return gaps


def _create_competitive_positions(content_list: List[Dict[str, Any]], 
                                 performance_tiers: Dict[str, Any]) -> Dict[str, Any]:
    """Create competitive positioning based on performance analysis."""
    positions = {}
    
    for tier_name, tier_data in performance_tiers.items():
        tier_content = tier_data.get('content', [])
        if tier_content:
            position_id = f"position_{tier_name}"
            positions[position_id] = {
                'name': tier_name.replace('_', ' ').title(),
                'content_count': len(tier_content),
                'avg_seo_score': tier_data.get('avg_seo_score', 0),
                'competitive_strength': tier_data.get('competitive_strength', 'unknown'),
                'topic_coverage': tier_data.get('topic_coverage', 0),
                'representative_content': [c.get('title', 'Untitled') for c in tier_content[:3]]
            }
    
    return positions


# Demo data fallback removed - now uses real content analysis only


def _create_competitor_graph_elements(competitive_data: Dict[str, Any], 
                                     query: GraphQuery) -> tuple[List[GraphNode], List[GraphEdge]]:
    """Create graph nodes and edges from competitive analysis data."""
    nodes = {}
    edges = []
    
    competitive_positions = competitive_data.get('competitive_positions', {})
    topic_analysis = competitive_data.get('topic_analysis', {})
    competitive_gaps = competitive_data.get('competitive_gaps', [])
    
    # Create competitive position nodes
    for position_id, position_data in competitive_positions.items():
        strength = position_data.get('competitive_strength', 'unknown')
        
        # Determine node color based on competitive strength
        if strength == 'strong':
            color = '#16a34a'  # Green
        elif strength == 'moderate':
            color = '#f59e0b'  # Orange
        else:
            color = '#ef4444'  # Red
        
        node_size = min(max(position_data.get('content_count', 0) * 15 + 40, 40), 90)
        
        nodes[position_id] = GraphNode(
            id=position_id,
            label=position_data.get('name', 'Unknown Position'),
            type='competitor',
            size=node_size,
            weight=node_size,
            color=color,
            description=f"Avg SEO: {position_data.get('avg_seo_score', 0):.1f}%, {position_data.get('content_count', 0)} articles",
            properties={
                'competitive_strength': strength,
                'avg_seo_score': position_data.get('avg_seo_score', 0),
                'content_count': position_data.get('content_count', 0),
                'topic_coverage': position_data.get('topic_coverage', 0)
            }
        )
    
    # Create topic nodes
    topics = topic_analysis.get('topics', {})
    for topic_name, content_ids in topics.items():
        topic_id = f"topic_{topic_name.replace(' ', '_').replace('&', 'and')}"
        
        if topic_id not in nodes:
            topic_size = min(max(len(content_ids) * 12 + 35, 35), 70)
            
            nodes[topic_id] = GraphNode(
                id=topic_id,
                label=topic_name,
                type='topic',
                size=topic_size,
                weight=topic_size,
                color=get_node_color('topic'),
                description=f"Topic with {len(content_ids)} articles",
                properties={
                    'content_count': len(content_ids),
                    'topic_category': topic_name,
                    'coverage_level': 'high' if len(content_ids) > 2 else 'low'
                }
            )
    
    # Create competitive gap nodes
    for i, gap in enumerate(competitive_gaps[:5]):  # Limit to 5 gaps
        gap_id = f"gap_{i}_{gap.replace(' ', '_')}"
        
        nodes[gap_id] = GraphNode(
            id=gap_id,
            label=f"Gap: {gap}",
            type='opportunity',
            size=45,
            weight=45,
            color='#8b5cf6',  # Purple for opportunities
            description=f"Competitive opportunity: {gap}",
            properties={
                'gap_type': gap,
                'priority': 'high' if i < 2 else 'medium',
                'opportunity_score': 100 - (i * 15)
            }
        )
    
    # Create edges between positions and topics
    for position_id, position_data in competitive_positions.items():
        coverage = position_data.get('topic_coverage', 0)
        strength = position_data.get('competitive_strength', 'unknown')
        
        # Connect to topics based on coverage
        topic_nodes = [node_id for node_id in nodes if nodes[node_id].type == 'topic']
        connected_topics = min(coverage, len(topic_nodes))
        
        for i, topic_id in enumerate(topic_nodes[:connected_topics]):
            edge_id = f"{position_id}-{topic_id}"
            
            # Edge weight based on competitive strength
            weight = 0.8 if strength == 'strong' else 0.6 if strength == 'moderate' else 0.4
            
            edges.append(GraphEdge(
                id=edge_id,
                source=position_id,
                target=topic_id,
                type='competitive_coverage',
                weight=weight,
                color=nodes[position_id].color,
                label='Covers Topic',
                properties={'coverage_strength': strength}
            ))
    
    # Create edges between competitive positions (rivalry)
    position_ids = list(competitive_positions.keys())
    for i, pos1 in enumerate(position_ids):
        for pos2 in position_ids[i+1:]:
            pos1_strength = competitive_positions[pos1].get('competitive_strength', 'unknown')
            pos2_strength = competitive_positions[pos2].get('competitive_strength', 'unknown')
            
            # Stronger competition between similar-strength positions
            if pos1_strength == pos2_strength:
                weight = 0.7
                color = '#ef4444'  # Red for direct competition
                label = 'Direct Competition'
            else:
                weight = 0.4
                color = '#f59e0b'  # Orange for indirect competition
                label = 'Market Presence'
            
            edge_id = f"{pos1}-{pos2}-competition"
            edges.append(GraphEdge(
                id=edge_id,
                source=pos1,
                target=pos2,
                type='competitive_rivalry',
                weight=weight,
                color=color,
                label=label,
                properties={'rivalry_type': 'market_competition'}
            ))
    
    # Connect gaps to relevant topics (opportunities)
    gap_nodes = [node_id for node_id in nodes if nodes[node_id].type == 'opportunity']
    topic_nodes = [node_id for node_id in nodes if nodes[node_id].type == 'topic']
    
    for gap_id in gap_nodes:
        # Connect each gap to 1-2 relevant topics
        for topic_id in topic_nodes[:2]:
            edge_id = f"{gap_id}-{topic_id}-opportunity"
            edges.append(GraphEdge(
                id=edge_id,
                source=gap_id,
                target=topic_id,
                type='improvement_opportunity',
                weight=0.6,
                color='#8b5cf6',
                label='Improvement Opportunity',
                properties={'opportunity_type': 'content_gap'}
            ))
    
    # Filter edges by query parameters
    if query.min_connection_strength > 0:
        edges = [e for e in edges if e.weight >= query.min_connection_strength]
    
    # Limit edges
    edges = edges[:query.edge_limit]
    
    # Convert nodes dict to list
    node_list = list(nodes.values())
    
    return node_list, edges