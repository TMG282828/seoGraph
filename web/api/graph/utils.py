"""
Graph utilities for the SEO Content Knowledge Graph System.

This module contains utility functions for graph calculations, styling, and processing.
"""

import math
from typing import Any, Dict, List, Set, Tuple
from .models import GraphNode, GraphEdge


def determine_node_type(node_data: Dict[str, Any]) -> str:
    """Determine node type based on data structure."""
    if 'content_id' in node_data or 'title' in node_data:
        return 'content'
    elif 'keyword' in node_data or 'search_volume' in node_data:
        return 'keyword'
    elif 'topic_name' in node_data or 'content_count' in node_data:
        return 'topic'
    elif 'competitor_name' in node_data or 'domain' in node_data:
        return 'competitor'
    else:
        return 'unknown'


def calculate_node_size(node_data: Dict[str, Any], node_type: str) -> int:
    """Calculate node size based on type and data."""
    base_size = 40
    
    if node_type == 'content':
        # Size based on word count and SEO score
        word_count = node_data.get('word_count', 0)
        seo_score = node_data.get('seo_score', 0)
        size = base_size + (word_count // 100) + (int(seo_score) // 10)
        return min(max(int(size), 20), 80)
    
    elif node_type == 'keyword':
        # Size based on search volume
        search_volume = node_data.get('search_volume', 0)
        size = base_size + math.log10(max(search_volume, 1)) * 5
        return min(max(int(size), 25), 70)
    
    elif node_type == 'topic':
        # Size based on content count
        content_count = node_data.get('content_count', 0)
        size = base_size + content_count * 8
        return min(max(int(size), 30), 90)
    
    elif node_type == 'competitor':
        # Size based on authority score
        authority = node_data.get('authority_score', 0)
        size = base_size + authority * 0.5
        return min(max(int(size), 35), 75)
    
    return int(base_size)


def get_node_color(node_type: str, properties: Dict[str, Any] = None) -> str:
    """Get node color based on type and properties."""
    base_colors = {
        'content': '#22c55e',     # Green
        'keyword': '#f59e0b',     # Orange
        'topic': '#8b5cf6',       # Purple
        'competitor': '#ef4444',  # Red
        'unknown': '#6b7280'      # Gray
    }
    
    base_color = base_colors.get(node_type, base_colors['unknown'])
    
    # Modify color based on properties
    if properties:
        if node_type == 'content':
            seo_score = properties.get('seo_score', 0)
            if seo_score > 80:
                return '#16a34a'  # Darker green for high SEO
            elif seo_score < 30:
                return '#84cc16'  # Lighter green for low SEO
        
        elif node_type == 'keyword':
            competition = properties.get('competition', 0)
            if competition > 0.8:
                return '#dc2626'  # Red for high competition
            elif competition < 0.3:
                return '#65a30d'  # Green for low competition
        
        elif node_type == 'topic':
            avg_score = properties.get('avg_seo_score', 0)
            if avg_score > 70:
                return '#7c3aed'  # Darker purple for high-performing topics
    
    return base_color


def get_edge_color(edge_type: str) -> str:
    """Get edge color based on relationship type."""
    colors = {
        'content_similarity': '#9ca3af',
        'keyword_relation': '#fbbf24',
        'topic_connection': '#a78bfa',
        'competitor_overlap': '#f87171',
        'seo_boost': '#34d399',
        'content_cluster': '#60a5fa',
        'neo4j_relationship': '#e91e63',
        'default': '#999999'
    }
    return colors.get(edge_type, colors['default'])


def extract_node_properties(node_data: Dict[str, Any], node_type: str) -> Dict[str, Any]:
    """Extract relevant properties for a node based on its type."""
    base_props = {
        'node_type': node_type,
        'data_source': node_data.get('source', 'database')
    }
    
    if node_type == 'content':
        base_props.update({
            'word_count': node_data.get('word_count', 0),
            'seo_score': node_data.get('seo_score', 0),
            'readability_score': node_data.get('readability_score', 0),
            'content_type': node_data.get('content_type', 'document'),
            'author': node_data.get('author'),
            'published_date': node_data.get('published_date')
        })
    
    elif node_type == 'keyword':
        base_props.update({
            'search_volume': node_data.get('search_volume', 0),
            'competition': node_data.get('competition', 0),
            'difficulty': node_data.get('difficulty', 'unknown'),
            'trend_direction': node_data.get('trend_direction', 'stable'),
            'intent': node_data.get('intent')
        })
    
    elif node_type == 'topic':
        base_props.update({
            'content_count': node_data.get('content_count', 0),
            'avg_seo_score': node_data.get('avg_seo_score', 0),
            'related_topics_count': len(node_data.get('related_topics', []))
        })
    
    elif node_type == 'competitor':
        base_props.update({
            'domain': node_data.get('domain'),
            'authority_score': node_data.get('authority_score', 0),
            'content_overlap': node_data.get('content_overlap', 0),
            'keyword_overlap': node_data.get('keyword_overlap', 0),
            'competitive_strength': node_data.get('competitive_strength', 'unknown')
        })
    
    return base_props


def generate_node_description(node_data: Dict[str, Any], node_type: str) -> str:
    """Generate a descriptive text for a node."""
    if node_type == 'content':
        title = node_data.get('title', 'Untitled')
        word_count = node_data.get('word_count', 0)
        seo_score = node_data.get('seo_score', 0)
        return f"{title} - {word_count} words, SEO: {seo_score:.1f}%"
    
    elif node_type == 'keyword':
        keyword = node_data.get('keyword', 'Unknown')
        volume = node_data.get('search_volume', 0)
        competition = node_data.get('competition', 0)
        return f"{keyword} - Volume: {volume:,}, Competition: {competition:.1f}"
    
    elif node_type == 'topic':
        topic = node_data.get('topic_name', 'Unknown Topic')
        count = node_data.get('content_count', 0)
        avg_score = node_data.get('avg_seo_score', 0)
        return f"{topic} - {count} articles, Avg SEO: {avg_score:.1f}%"
    
    elif node_type == 'competitor':
        name = node_data.get('competitor_name', 'Unknown')
        authority = node_data.get('authority_score', 0)
        overlap = node_data.get('content_overlap', 0)
        return f"{name} - Authority: {authority:.1f}, Overlap: {overlap:.1f}%"
    
    return f"{node_type.title()} node"


def calculate_graph_metrics(nodes: List[GraphNode], edges: List[GraphEdge]) -> Dict[str, float]:
    """Calculate comprehensive graph metrics."""
    node_count = len(nodes)
    edge_count = len(edges)
    
    if node_count == 0:
        return {
            'density': 0.0,
            'average_degree': 0.0,
            'clustering_coefficient': 0.0,
            'connected_components': 0
        }
    
    # Calculate density
    max_edges = node_count * (node_count - 1) / 2
    density = edge_count / max_edges if max_edges > 0 else 0.0
    
    # Calculate average degree
    average_degree = (2 * edge_count) / node_count if node_count > 0 else 0.0
    
    # Calculate connected components
    connected_components = calculate_connected_components(nodes, edges)
    
    # Calculate clustering coefficient (simplified)
    clustering_coefficient = calculate_clustering_coefficient(nodes, edges)
    
    return {
        'density': density,
        'average_degree': average_degree,
        'clustering_coefficient': clustering_coefficient,
        'connected_components': connected_components
    }


def calculate_connected_components(nodes: List[GraphNode], edges: List[GraphEdge]) -> int:
    """Calculate the number of connected components in the graph."""
    if not nodes:
        return 0
    
    # Build adjacency list
    adjacency = {node.id: set() for node in nodes}
    for edge in edges:
        if edge.source in adjacency and edge.target in adjacency:
            adjacency[edge.source].add(edge.target)
            adjacency[edge.target].add(edge.source)
    
    visited = set()
    components = 0
    
    def dfs(node_id: str):
        if node_id in visited:
            return
        visited.add(node_id)
        for neighbor in adjacency[node_id]:
            dfs(neighbor)
    
    for node in nodes:
        if node.id not in visited:
            dfs(node.id)
            components += 1
    
    return components


def calculate_clustering_coefficient(nodes: List[GraphNode], edges: List[GraphEdge]) -> float:
    """Calculate the average clustering coefficient of the graph."""
    if len(nodes) < 3:
        return 0.0
    
    # Build adjacency list
    adjacency = {node.id: set() for node in nodes}
    for edge in edges:
        if edge.source in adjacency and edge.target in adjacency:
            adjacency[edge.source].add(edge.target)
            adjacency[edge.target].add(edge.source)
    
    total_coefficient = 0.0
    node_count = 0
    
    for node in nodes:
        neighbors = adjacency[node.id]
        if len(neighbors) < 2:
            continue
        
        # Count triangles
        triangles = 0
        possible_triangles = len(neighbors) * (len(neighbors) - 1) / 2
        
        neighbor_list = list(neighbors)
        for i in range(len(neighbor_list)):
            for j in range(i + 1, len(neighbor_list)):
                if neighbor_list[j] in adjacency[neighbor_list[i]]:
                    triangles += 1
        
        if possible_triangles > 0:
            total_coefficient += triangles / possible_triangles
            node_count += 1
    
    return total_coefficient / node_count if node_count > 0 else 0.0


def filter_nodes_by_centrality(nodes: List[GraphNode], edges: List[GraphEdge], 
                              limit: int) -> List[GraphNode]:
    """Filter nodes by centrality measures, keeping the most important ones."""
    if len(nodes) <= limit:
        return nodes
    
    # Calculate degree centrality
    degree_count = {node.id: 0 for node in nodes}
    for edge in edges:
        if edge.source in degree_count:
            degree_count[edge.source] += 1
        if edge.target in degree_count:
            degree_count[edge.target] += 1
    
    # Sort nodes by degree centrality and other factors
    def node_importance(node: GraphNode) -> float:
        degree = degree_count.get(node.id, 0)
        size_score = node.size / 100.0
        weight_score = node.weight / 100.0
        return degree * 2 + size_score + weight_score
    
    sorted_nodes = sorted(nodes, key=node_importance, reverse=True)
    return sorted_nodes[:limit]


def create_node_clusters(nodes: List[GraphNode], edges: List[GraphEdge], 
                        max_clusters: int = 5) -> Dict[str, List[GraphNode]]:
    """Create node clusters based on connectivity and properties."""
    if not nodes:
        return {}
    
    # Simple clustering based on node type and connectivity
    clusters = {}
    
    # Group by type first
    type_groups = {}
    for node in nodes:
        if node.type not in type_groups:
            type_groups[node.type] = []
        type_groups[node.type].append(node)
    
    # Create clusters from type groups
    cluster_id = 0
    for node_type, type_nodes in type_groups.items():
        if len(type_nodes) <= 10:  # Small groups stay together
            clusters[f"cluster_{cluster_id}_{node_type}"] = type_nodes
            cluster_id += 1
        else:  # Large groups get subdivided
            chunk_size = len(type_nodes) // max_clusters + 1
            for i in range(0, len(type_nodes), chunk_size):
                chunk = type_nodes[i:i + chunk_size]
                clusters[f"cluster_{cluster_id}_{node_type}"] = chunk
                cluster_id += 1
    
    return clusters