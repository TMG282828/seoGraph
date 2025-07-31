"""
Semantic Clusters Graph module for the SEO Content Knowledge Graph System.

This module handles semantic clustering of content based on AI analysis,
topic modeling, and content similarity.
"""

import logging
import math
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict, Counter

from .models import GraphNode, GraphEdge, GraphData, GraphQuery, GraphResponse
from .utils import (
    calculate_node_size, get_node_color, extract_node_properties, 
    generate_node_description, calculate_graph_metrics
)

logger = logging.getLogger(__name__)


async def get_semantic_clusters_graph_data(query: GraphQuery) -> GraphResponse:
    """
    Generate semantic clusters graph data based on content analysis.
    
    This function creates clusters of semantically related content
    using AI-driven topic modeling and similarity analysis.
    """
    try:
        logger.info("🧠 Starting semantic clusters graph generation")
        
        # Get content data for clustering
        content_list = await _get_content_for_clustering(query)
        logger.info(f"✅ Retrieved {len(content_list)} documents for clustering")
        
        # Perform semantic clustering analysis
        clustering_data = await _perform_semantic_clustering(content_list)
        
        # Create graph elements
        nodes, edges = _create_cluster_graph_elements(clustering_data, query)
        
        # Calculate graph metrics
        graph_metrics = calculate_graph_metrics(nodes, edges)
        
        # Create graph data
        graph_data = GraphData(
            nodes=nodes,
            edges=edges,
            metadata={
                'query_type': 'semantic_clusters',
                'tenant_id': 'demo-org',
                'source': 'semantic_analysis',
                'clustering_algorithm': 'content_similarity'
            },
            node_count=len(nodes),
            edge_count=len(edges),
            density=graph_metrics['density'],
            average_degree=graph_metrics['average_degree'],
            connected_components=graph_metrics['connected_components']
        )
        
        # Create response
        statistics = {
            'cluster_count': clustering_data.get('cluster_count', 0),
            'avg_cluster_size': clustering_data.get('avg_cluster_size', 0),
            'semantic_coherence': clustering_data.get('semantic_coherence', 0),
            'topic_diversity': clustering_data.get('topic_diversity', 0),
            **graph_metrics
        }
        
        logger.info(f"🎯 Generated semantic clusters with {len(nodes)} nodes and {len(edges)} edges")
        
        return GraphResponse(
            success=True,
            graph=graph_data,
            statistics=statistics
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to generate semantic clusters graph: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        raise


async def _get_content_for_clustering(query: GraphQuery) -> List[Dict[str, Any]]:
    """Get content items for semantic clustering."""
    from src.database.content_service import ContentDatabaseService
    
    db_service = ContentDatabaseService()
    
    result = await db_service.get_content_items(
        organization_id="demo-org",
        search=query.keyword_filter,
        content_type=query.content_type_filter,
        limit=min(query.node_limit, 40),
        offset=0
    )
    
    if not result.get("success"):
        raise Exception(f"Failed to fetch content: {result.get('error', 'Unknown error')}")
    
    return result.get("content", [])


async def _perform_semantic_clustering(content_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Perform semantic clustering analysis on content.
    
    This creates clusters based on:
    1. Content similarity (word count, SEO score patterns)
    2. Topic extraction from titles and content
    3. Content type groupings
    4. Performance-based clustering
    """
    if not content_list:
        raise Exception("No content available for semantic clustering")
    
    # Extract semantic features
    semantic_features = _extract_semantic_features(content_list)
    
    # Perform clustering
    clusters = _cluster_content_by_similarity(content_list, semantic_features)
    
    # Analyze cluster quality
    cluster_analysis = _analyze_cluster_quality(clusters, semantic_features)
    
    # Identify cluster themes
    cluster_themes = _identify_cluster_themes(clusters)
    
    # Calculate semantic coherence
    semantic_coherence = _calculate_semantic_coherence(clusters, semantic_features)
    
    return {
        'content_count': len(content_list),
        'cluster_count': len(clusters),
        'avg_cluster_size': sum(len(cluster['content']) for cluster in clusters.values()) / len(clusters) if clusters else 0,
        'clusters': clusters,
        'cluster_themes': cluster_themes,
        'semantic_features': semantic_features,
        'cluster_analysis': cluster_analysis,
        'semantic_coherence': semantic_coherence,
        'topic_diversity': len(set(feature.get('primary_topic', 'unknown') for feature in semantic_features.values()))
    }


def _extract_semantic_features(content_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Extract semantic features from content for clustering."""
    features = {}
    
    for content in content_list:
        content_id = content.get('id')
        if not content_id:
            continue
        
        title = content.get('title', '').lower()
        word_count = content.get('word_count', 0)
        seo_score = content.get('seo_score', 0)
        content_type = content.get('content_type', 'document')
        
        # Extract primary topic from title (simplified NLP)
        primary_topic = _extract_primary_topic(title)
        
        # Extract key terms
        key_terms = _extract_key_terms(title)
        
        # Calculate content complexity
        complexity = _calculate_content_complexity(word_count, seo_score)
        
        # Performance tier
        performance_tier = _get_performance_tier(seo_score)
        
        features[content_id] = {
            'primary_topic': primary_topic,
            'key_terms': key_terms,
            'word_count': word_count,
            'seo_score': seo_score,
            'content_type': content_type,
            'complexity': complexity,
            'performance_tier': performance_tier,
            'title_length': len(title),
            'semantic_vector': _create_semantic_vector(title, word_count, seo_score)
        }
    
    return features


def _extract_primary_topic(title: str) -> str:
    """Extract primary topic from content title."""
    title_lower = title.lower()
    
    # Define topic keywords
    topic_mapping = {
        'seo': ['seo', 'search', 'optimization', 'ranking', 'serp'],
        'content_marketing': ['content', 'marketing', 'strategy', 'blog', 'article'],
        'technical': ['technical', 'development', 'code', 'implementation', 'api'],
        'analytics': ['analytics', 'data', 'metrics', 'measurement', 'tracking'],
        'social_media': ['social', 'media', 'facebook', 'twitter', 'linkedin'],
        'email_marketing': ['email', 'newsletter', 'campaign', 'automation'],
        'conversion': ['conversion', 'cro', 'optimization', 'rate', 'funnel'],
        'branding': ['brand', 'branding', 'identity', 'logo', 'voice'],
        'user_experience': ['ux', 'ui', 'experience', 'usability', 'design']
    }
    
    # Find best matching topic
    for topic, keywords in topic_mapping.items():
        if any(keyword in title_lower for keyword in keywords):
            return topic
    
    return 'general'


def _extract_key_terms(title: str) -> List[str]:
    """Extract key terms from title."""
    # Simple term extraction (in production, use proper NLP)
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
    
    words = title.lower().split()
    key_terms = [word.strip('.,!?;:') for word in words if word not in stop_words and len(word) > 2]
    
    return key_terms[:5]  # Top 5 key terms


def _calculate_content_complexity(word_count: int, seo_score: float) -> str:
    """Calculate content complexity level."""
    if word_count > 2000 and seo_score > 70:
        return 'high'
    elif word_count > 1000 and seo_score > 50:
        return 'medium'
    else:
        return 'low'


def _get_performance_tier(seo_score: float) -> str:
    """Get performance tier based on SEO score."""
    if seo_score >= 70:
        return 'high_performer'
    elif seo_score >= 40:
        return 'medium_performer'
    else:
        return 'needs_improvement'


def _create_semantic_vector(title: str, word_count: int, seo_score: float) -> List[float]:
    """Create a simple semantic vector for similarity calculations."""
    # Simplified semantic vector (in production, use embeddings)
    vector = []
    
    # Title-based features
    vector.append(len(title) / 100.0)  # Normalized title length
    vector.append(1.0 if 'seo' in title.lower() else 0.0)
    vector.append(1.0 if 'content' in title.lower() else 0.0)
    vector.append(1.0 if 'marketing' in title.lower() else 0.0)
    vector.append(1.0 if 'strategy' in title.lower() else 0.0)
    
    # Performance features
    vector.append(word_count / 5000.0)  # Normalized word count
    vector.append(seo_score / 100.0)    # Normalized SEO score
    
    return vector


def _cluster_content_by_similarity(content_list: List[Dict[str, Any]], 
                                  semantic_features: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Cluster content based on semantic similarity."""
    clusters = {}
    content_ids = list(semantic_features.keys())
    clustered = set()
    cluster_id = 0
    
    # Group by primary topic first
    topic_groups = defaultdict(list)
    for content_id, features in semantic_features.items():
        topic = features.get('primary_topic', 'general')
        topic_groups[topic].append(content_id)
    
    # Create clusters from topic groups
    for topic, content_ids_in_topic in topic_groups.items():
        if len(content_ids_in_topic) >= 2:  # Only create clusters with 2+ items
            cluster_key = f"cluster_{cluster_id}_{topic}"
            
            # Get content objects for this cluster
            cluster_content = [
                content for content in content_list 
                if content.get('id') in content_ids_in_topic
            ]
            
            clusters[cluster_key] = {
                'id': cluster_key,
                'name': topic.replace('_', ' ').title(),
                'content': cluster_content,
                'primary_topic': topic,
                'content_ids': content_ids_in_topic,
                'size': len(content_ids_in_topic),
                'avg_seo_score': sum(features.get('seo_score', 0) for content_id in content_ids_in_topic for features in [semantic_features.get(content_id, {})]) / len(content_ids_in_topic),
                'avg_word_count': sum(features.get('word_count', 0) for content_id in content_ids_in_topic for features in [semantic_features.get(content_id, {})]) / len(content_ids_in_topic),
                'content_types': list(set(features.get('content_type', 'unknown') for content_id in content_ids_in_topic for features in [semantic_features.get(content_id, {})]))
            }
            
            clustered.update(content_ids_in_topic)
            cluster_id += 1
    
    # Handle singleton content (create individual clusters for unclustered content)
    unclustered = [content_id for content_id in content_ids if content_id not in clustered]
    for content_id in unclustered:
        content_obj = next((c for c in content_list if c.get('id') == content_id), None)
        if content_obj:
            features = semantic_features.get(content_id, {})
            cluster_key = f"cluster_{cluster_id}_singleton"
            
            clusters[cluster_key] = {
                'id': cluster_key,
                'name': f"Individual: {features.get('primary_topic', 'General').title()}",
                'content': [content_obj],
                'primary_topic': features.get('primary_topic', 'general'),
                'content_ids': [content_id],
                'size': 1,
                'avg_seo_score': features.get('seo_score', 0),
                'avg_word_count': features.get('word_count', 0),
                'content_types': [features.get('content_type', 'unknown')]
            }
            cluster_id += 1
    
    return clusters


def _analyze_cluster_quality(clusters: Dict[str, Dict[str, Any]], 
                            semantic_features: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the quality of clustering results."""
    if not clusters:
        return {'avg_coherence': 0, 'cluster_distribution': {}}
    
    cluster_sizes = [cluster['size'] for cluster in clusters.values()]
    
    analysis = {
        'total_clusters': len(clusters),
        'avg_cluster_size': sum(cluster_sizes) / len(cluster_sizes),
        'largest_cluster': max(cluster_sizes),
        'smallest_cluster': min(cluster_sizes),
        'cluster_distribution': Counter(cluster_sizes),
        'topic_distribution': Counter(cluster['primary_topic'] for cluster in clusters.values())
    }
    
    return analysis


def _identify_cluster_themes(clusters: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Identify themes for each cluster."""
    themes = {}
    
    for cluster_id, cluster_data in clusters.items():
        primary_topic = cluster_data.get('primary_topic', 'general')
        size = cluster_data.get('size', 0)
        avg_seo = cluster_data.get('avg_seo_score', 0)
        
        # Generate theme description
        if size >= 3:
            theme = f"Major {primary_topic.replace('_', ' ').title()} Hub"
        elif avg_seo > 70:
            theme = f"High-Performance {primary_topic.replace('_', ' ').title()}"
        elif avg_seo < 40:
            theme = f"Improvement Opportunity in {primary_topic.replace('_', ' ').title()}"
        else:
            theme = f"{primary_topic.replace('_', ' ').title()} Content"
        
        themes[cluster_id] = theme
    
    return themes


def _calculate_semantic_coherence(clusters: Dict[str, Dict[str, Any]], 
                                 semantic_features: Dict[str, Dict[str, Any]]) -> float:
    """Calculate overall semantic coherence of clustering."""
    if not clusters or len(clusters) <= 1:
        return 0.0
    
    total_coherence = 0.0
    cluster_count = 0
    
    for cluster_data in clusters.values():
        content_ids = cluster_data.get('content_ids', [])
        if len(content_ids) >= 2:
            # Calculate intra-cluster similarity
            similarities = []
            for i, id1 in enumerate(content_ids):
                for id2 in content_ids[i+1:]:
                    similarity = _calculate_vector_similarity(
                        semantic_features.get(id1, {}).get('semantic_vector', []),
                        semantic_features.get(id2, {}).get('semantic_vector', [])
                    )
                    similarities.append(similarity)
            
            if similarities:
                cluster_coherence = sum(similarities) / len(similarities)
                total_coherence += cluster_coherence
                cluster_count += 1
    
    return total_coherence / cluster_count if cluster_count > 0 else 0.0


def _calculate_vector_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate similarity between two semantic vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    
    # Cosine similarity
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


# Demo data fallback removed - now uses real content analysis only


def _create_cluster_graph_elements(clustering_data: Dict[str, Any], 
                                  query: GraphQuery) -> tuple[List[GraphNode], List[GraphEdge]]:
    """Create graph nodes and edges from clustering data."""
    nodes = {}
    edges = []
    
    clusters = clustering_data.get('clusters', {})
    cluster_themes = clustering_data.get('cluster_themes', {})
    
    # Create cluster center nodes
    for cluster_id, cluster_data in clusters.items():
        cluster_size = cluster_data.get('size', 0)
        avg_seo = cluster_data.get('avg_seo_score', 0)
        primary_topic = cluster_data.get('primary_topic', 'general')
        
        # Determine cluster color based on performance and size
        if cluster_size >= 3 and avg_seo > 70:
            color = '#16a34a'  # Green for strong clusters
        elif cluster_size >= 2 and avg_seo > 50:
            color = '#f59e0b'  # Orange for moderate clusters
        elif cluster_size == 1:
            color = '#8b5cf6'  # Purple for singleton clusters
        else:
            color = '#6b7280'  # Gray for weak clusters
        
        # Calculate node size based on cluster size and performance
        node_size = min(max(cluster_size * 20 + avg_seo * 0.3, 30), 100)
        
        cluster_name = cluster_data.get('name', f'Cluster {cluster_id}')
        theme = cluster_themes.get(cluster_id, 'Content Cluster')
        
        nodes[cluster_id] = GraphNode(
            id=cluster_id,
            label=cluster_name,
            type='cluster',
            size=int(node_size),
            weight=int(node_size),
            color=color,
            description=f"{theme} - {cluster_size} articles, Avg SEO: {avg_seo:.1f}%",
            properties={
                'cluster_size': cluster_size,
                'avg_seo_score': avg_seo,
                'avg_word_count': cluster_data.get('avg_word_count', 0),
                'primary_topic': primary_topic,
                'theme': theme,
                'content_types': cluster_data.get('content_types', []),
                'coherence_level': 'high' if avg_seo > 70 else 'medium' if avg_seo > 40 else 'low'
            }
        )
        
        # Create individual content nodes within cluster
        for content in cluster_data.get('content', []):
            content_id = content.get('id')
            if content_id and content_id not in nodes:
                content_size = min(max(content.get('word_count', 0) // 100 + 20, 15), 40)
                
                nodes[content_id] = GraphNode(
                    id=content_id,
                    label=content.get('title', 'Untitled')[:30],
                    type='content',
                    size=content_size,
                    weight=content_size,
                    color='#22c55e',
                    description=f"{content.get('word_count', 0)} words, SEO: {content.get('seo_score', 0):.1f}%",
                    properties={
                        'word_count': content.get('word_count', 0),
                        'seo_score': content.get('seo_score', 0),
                        'content_type': content.get('content_type', 'document'),
                        'cluster_member': cluster_id
                    }
                )
                
                # Connect content to cluster center
                edge_id = f"{content_id}-{cluster_id}"
                edges.append(GraphEdge(
                    id=edge_id,
                    source=content_id,
                    target=cluster_id,
                    type='cluster_membership',
                    weight=0.8,
                    color=color,
                    label='Member Of',
                    properties={'membership_type': 'cluster_content'}
                ))
    
    # Create topic-based super-clusters
    topic_clusters = defaultdict(list)
    for cluster_id, cluster_data in clusters.items():
        topic = cluster_data.get('primary_topic', 'general')
        topic_clusters[topic].append(cluster_id)
    
    # Create topic nodes and connections
    for topic, cluster_ids in topic_clusters.items():
        if len(cluster_ids) > 1:  # Only create topic nodes for multiple clusters
            topic_id = f"topic_{topic}"
            topic_size = min(len(cluster_ids) * 25 + 40, 80)
            
            nodes[topic_id] = GraphNode(
                id=topic_id,
                label=topic.replace('_', ' ').title(),
                type='topic',
                size=topic_size,
                weight=topic_size,
                color=get_node_color('topic'),
                description=f"Topic connecting {len(cluster_ids)} clusters",
                properties={
                    'cluster_count': len(cluster_ids),
                    'topic_category': topic,
                    'super_cluster': True
                }
            )
            
            # Connect clusters to topic
            for cluster_id in cluster_ids:
                edge_id = f"{cluster_id}-{topic_id}"
                edges.append(GraphEdge(
                    id=edge_id,
                    source=cluster_id,
                    target=topic_id,
                    type='topic_grouping',
                    weight=0.6,
                    color=get_node_color('topic'),
                    label='Topic Group',
                    properties={'grouping_type': 'semantic_topic'}
                ))
    
    # Create inter-cluster connections for similar clusters
    cluster_ids = list(clusters.keys())
    for i, cluster1 in enumerate(cluster_ids):
        for cluster2 in cluster_ids[i+1:]:
            cluster1_data = clusters[cluster1]
            cluster2_data = clusters[cluster2]
            
            # Calculate cluster similarity
            similarity = _calculate_cluster_similarity(cluster1_data, cluster2_data)
            
            if similarity > 0.5:  # Connect similar clusters
                edge_id = f"{cluster1}-{cluster2}-similarity"
                edges.append(GraphEdge(
                    id=edge_id,
                    source=cluster1,
                    target=cluster2,
                    type='cluster_similarity',
                    weight=similarity * 0.5,
                    color='#9ca3af',
                    label='Similar Clusters',
                    properties={'similarity_score': similarity}
                ))
    
    # Filter edges by query parameters
    if query.min_connection_strength > 0:
        edges = [e for e in edges if e.weight >= query.min_connection_strength]
    
    # Limit edges
    edges = edges[:query.edge_limit]
    
    # Convert nodes dict to list
    node_list = list(nodes.values())
    
    return node_list, edges


def _calculate_cluster_similarity(cluster1: Dict[str, Any], cluster2: Dict[str, Any]) -> float:
    """Calculate similarity between two clusters."""
    similarity = 0.0
    
    # Topic similarity
    if cluster1.get('primary_topic') == cluster2.get('primary_topic'):
        similarity += 0.4
    
    # Size similarity
    size1 = cluster1.get('size', 0)
    size2 = cluster2.get('size', 0)
    if size1 > 0 and size2 > 0:
        size_similarity = 1 - abs(size1 - size2) / max(size1, size2)
        similarity += size_similarity * 0.3
    
    # Performance similarity
    seo1 = cluster1.get('avg_seo_score', 0)
    seo2 = cluster2.get('avg_seo_score', 0)
    if abs(seo1 - seo2) < 20:  # Similar SEO performance
        similarity += 0.3
    
    return min(similarity, 1.0)