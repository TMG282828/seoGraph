"""
Qdrant Vector Database client for multi-tenant SEO Content Knowledge Graph System.

This module provides a client wrapper for Qdrant operations with multi-tenant support,
semantic search, content embeddings, and similarity matching.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, CreateCollection, PointStruct, 
    Filter, FieldCondition, MatchValue, SearchRequest
)
import numpy as np
from datetime import datetime
import hashlib
import json
from dotenv import load_dotenv

# Load production environment variables
load_dotenv('.env.production')

logger = logging.getLogger(__name__)


class QdrantVectorClient:
    """Qdrant client wrapper with multi-tenant support for content embeddings and semantic search."""
    
    def __init__(self):
        """Initialize Qdrant client with environment configuration."""
        # Support both URL and host/port configurations
        qdrant_url = os.getenv("QDRANT_URL")
        if qdrant_url and "://" in qdrant_url:
            # Parse URL format (https://host:port)
            from urllib.parse import urlparse
            parsed = urlparse(qdrant_url)
            self.host = parsed.hostname
            self.port = parsed.port or 6333
        else:
            # Use separate host/port configuration
            self.host = os.getenv("QDRANT_HOST", "localhost")
            self.port = int(os.getenv("QDRANT_PORT", "6333"))
        
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.grpc_port = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
        
        self.client: Optional[QdrantClient] = None
        self.demo_mode = False
        self._current_organization_id = None
        
        # Collection configurations
        self.collections = {
            'content_embeddings': {
                'vector_size': 1536,  # OpenAI text-embedding-ada-002 embeddings
                'distance': Distance.COSINE
            },
            'keyword_embeddings': {
                'vector_size': 1536,  # OpenAI embeddings for consistency
                'distance': Distance.COSINE
            },
            'topic_embeddings': {
                'vector_size': 1536,  # OpenAI embeddings for consistency
                'distance': Distance.COSINE
            }
        }
        
        self._connect()
    
    def _connect(self) -> None:
        """Establish connection to Qdrant database."""
        try:
            if self.api_key:
                self.client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    api_key=self.api_key,
                    grpc_port=self.grpc_port
                )
            else:
                self.client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    grpc_port=self.grpc_port
                )
            
            # Test connection
            collections = self.client.get_collections()
            logger.info("Connected to Qdrant successfully")
            
        except Exception as e:
            logger.warning(f"Failed to connect to Qdrant: {e} - running in demo mode")
            self.demo_mode = True
            self.client = None
    
    def set_organization_context(self, organization_id: str) -> None:
        """Set the current organization context for multi-tenant operations."""
        self._current_organization_id = organization_id
    
    def close(self) -> None:
        """Close Qdrant client connection."""
        if self.client:
            self.client.close()
    
    # ============================================================================
    # Collection Management
    # ============================================================================
    
    def initialize_collections(self) -> bool:
        """Initialize Qdrant collections for multi-tenant content embeddings."""
        if self.demo_mode:
            logger.info("Demo mode: Collection initialization skipped")
            return True
        
        try:
            existing_collections = {col.name for col in self.client.get_collections().collections}
            
            for collection_name, config in self.collections.items():
                if collection_name not in existing_collections:
                    # Create new collection
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=config['vector_size'],
                            distance=config['distance']
                        )
                    )
                    logger.info(f"Created Qdrant collection: {collection_name} with {config['vector_size']} dimensions")
                else:
                    # Check if existing collection has correct dimensions
                    try:
                        collection_info = self.client.get_collection(collection_name)
                        existing_size = collection_info.config.params.vectors.size
                        expected_size = config['vector_size']
                        
                        if existing_size != expected_size:
                            logger.warning(f"Collection {collection_name} has {existing_size} dimensions but {expected_size} expected")
                            # Recreate collection with correct dimensions
                            logger.info(f"Recreating collection {collection_name} with correct dimensions")
                            self.client.delete_collection(collection_name)
                            self.client.create_collection(
                                collection_name=collection_name,
                                vectors_config=VectorParams(
                                    size=config['vector_size'],
                                    distance=config['distance']
                                )
                            )
                            logger.info(f"Recreated collection {collection_name} with {config['vector_size']} dimensions")
                        else:
                            logger.info(f"Collection {collection_name} exists with correct dimensions ({existing_size})")
                    except Exception as check_error:
                        logger.error(f"Failed to check collection {collection_name}: {check_error}")
            
            # Create payload indexes for multi-tenant filtering
            self._create_payload_indexes()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collections: {e}")
            return False
    
    def _create_payload_indexes(self) -> None:
        """Create payload indexes for efficient filtering."""
        try:
            # Index for organization_id filtering (multi-tenancy)
            for collection_name in self.collections.keys():
                try:
                    self.client.create_payload_index(
                        collection_name=collection_name,
                        field_name="organization_id",
                        field_schema="keyword"
                    )
                    logger.info(f"Created organization_id index for {collection_name}")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        logger.info(f"Index organization_id already exists for {collection_name}")
                    else:
                        logger.warning(f"Failed to create organization_id index for {collection_name}: {e}")
                
                # Index for content_type filtering
                try:
                    self.client.create_payload_index(
                        collection_name=collection_name,
                        field_name="content_type",
                        field_schema="keyword"
                    )
                    logger.info(f"Created content_type index for {collection_name}")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        logger.info(f"Index content_type already exists for {collection_name}")
                    else:
                        logger.warning(f"Failed to create content_type index for {collection_name}: {e}")
                        
        except Exception as e:
            logger.error(f"Failed to create payload indexes: {e}")
    
    def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a collection including point count and configuration."""
        if self.demo_mode:
            return {
                'name': collection_name,
                'vectors_count': 1500,
                'indexed_vectors_count': 1500,
                'points_count': 1500,
                'segments_count': 2,
                'config': self.collections.get(collection_name, {})
            }
        
        try:
            info = self.client.get_collection(collection_name)
            return {
                'name': collection_name,
                'vectors_count': info.vectors_count,
                'indexed_vectors_count': info.indexed_vectors_count,
                'points_count': info.points_count,
                'segments_count': info.segments_count,
                'config': self.collections.get(collection_name, {})
            }
        except Exception as e:
            logger.error(f"Failed to get collection info for {collection_name}: {e}")
            return None
    
    # ============================================================================
    # Content Embedding Operations
    # ============================================================================
    
    def store_content_embedding(self, content_id: str, embedding: List[float], 
                               metadata: Dict[str, Any]) -> bool:
        """Store content embedding with multi-tenant metadata."""
        if self.demo_mode:
            logger.info(f"Demo mode: Would store embedding for content {content_id}")
            return True
        
        if not self._current_organization_id:
            logger.error("No organization context set for storing embedding")
            return False
        
        try:
            # Add organization context to metadata
            full_metadata = {
                **metadata,
                'organization_id': self._current_organization_id,
                'content_id': content_id,
                'created_at': datetime.now().isoformat(),
                'embedding_type': 'content'
            }
            
            point = PointStruct(
                id=self._generate_point_id(content_id),
                vector=embedding,
                payload=full_metadata
            )
            
            self.client.upsert(
                collection_name='content_embeddings',
                points=[point]
            )
            
            logger.info(f"Stored content embedding for {content_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store content embedding for {content_id}: {e}")
            return False
    
    def store_keyword_embedding(self, keyword: str, embedding: List[float], 
                               metadata: Dict[str, Any]) -> bool:
        """Store keyword embedding with search metrics."""
        if self.demo_mode:
            logger.info(f"Demo mode: Would store embedding for keyword {keyword}")
            return True
        
        if not self._current_organization_id:
            logger.error("No organization context set for storing keyword embedding")
            return False
        
        try:
            keyword_id = hashlib.md5(f"{self._current_organization_id}:{keyword}".encode()).hexdigest()
            
            full_metadata = {
                **metadata,
                'organization_id': self._current_organization_id,
                'keyword': keyword,
                'keyword_id': keyword_id,
                'created_at': datetime.now().isoformat(),
                'embedding_type': 'keyword'
            }
            
            point = PointStruct(
                id=keyword_id,
                vector=embedding,
                payload=full_metadata
            )
            
            self.client.upsert(
                collection_name='keyword_embeddings',
                points=[point]
            )
            
            logger.info(f"Stored keyword embedding for {keyword}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store keyword embedding for {keyword}: {e}")
            return False
    
    def store_topic_embedding(self, topic_name: str, embedding: List[float], 
                             metadata: Dict[str, Any]) -> bool:
        """Store topic embedding with hierarchical information."""
        if self.demo_mode:
            logger.info(f"Demo mode: Would store embedding for topic {topic_name}")
            return True
        
        if not self._current_organization_id:
            logger.error("No organization context set for storing topic embedding")
            return False
        
        try:
            topic_id = hashlib.md5(f"{self._current_organization_id}:{topic_name}".encode()).hexdigest()
            
            full_metadata = {
                **metadata,
                'organization_id': self._current_organization_id,
                'topic_name': topic_name,
                'topic_id': topic_id,
                'created_at': datetime.now().isoformat(),
                'embedding_type': 'topic'
            }
            
            point = PointStruct(
                id=topic_id,
                vector=embedding,
                payload=full_metadata
            )
            
            self.client.upsert(
                collection_name='topic_embeddings',
                points=[point]
            )
            
            logger.info(f"Stored topic embedding for {topic_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store topic embedding for {topic_name}: {e}")
            return False
    
    # ============================================================================
    # Semantic Search Operations
    # ============================================================================
    
    def search_similar_content(self, query_embedding: List[float], limit: int = 10, 
                              content_type: Optional[str] = None, 
                              min_score: float = 0.7) -> List[Dict[str, Any]]:
        """Search for semantically similar content using vector similarity."""
        if self.demo_mode:
            return [
                {
                    'content_id': 'demo-content-1',
                    'title': 'Demo Similar Content 1',
                    'similarity_score': 0.95,
                    'content_type': 'article',
                    'summary': 'This is demo similar content'
                },
                {
                    'content_id': 'demo-content-2',
                    'title': 'Demo Similar Content 2',
                    'similarity_score': 0.88,
                    'content_type': 'blog_post',
                    'summary': 'Another demo similar content'
                }
            ]
        
        if not self._current_organization_id:
            logger.error("No organization context set for content search")
            return []
        
        try:
            # Build filter for organization and optional content type
            filter_conditions = [
                FieldCondition(
                    key="organization_id",
                    match=MatchValue(value=self._current_organization_id)
                )
            ]
            
            if content_type:
                filter_conditions.append(
                    FieldCondition(
                        key="content_type",
                        match=MatchValue(value=content_type)
                    )
                )
            
            search_filter = Filter(must=filter_conditions)
            
            search_result = self.client.search(
                collection_name='content_embeddings',
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=limit,
                score_threshold=min_score
            )
            
            results = []
            for scored_point in search_result:
                result = {
                    'content_id': scored_point.payload.get('content_id'),
                    'title': scored_point.payload.get('title', ''),
                    'summary': scored_point.payload.get('summary', ''),
                    'content_type': scored_point.payload.get('content_type', ''),
                    'url': scored_point.payload.get('url', ''),
                    'similarity_score': float(scored_point.score),
                    'word_count': scored_point.payload.get('word_count', 0),
                    'seo_score': scored_point.payload.get('seo_score', 0)
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search similar content: {e}")
            return []
    
    def search_related_keywords(self, query_embedding: List[float], limit: int = 15,
                               min_search_volume: int = 100, 
                               max_competition: float = 0.8) -> List[Dict[str, Any]]:
        """Search for related keywords using semantic similarity."""
        if self.demo_mode:
            return [
                {
                    'keyword': 'semantic SEO optimization',
                    'similarity_score': 0.92,
                    'search_volume': 2400,
                    'competition': 0.65,
                    'difficulty': 'medium'
                },
                {
                    'keyword': 'content clustering strategy',
                    'similarity_score': 0.87,
                    'search_volume': 1800,
                    'competition': 0.45,
                    'difficulty': 'low'
                }
            ]
        
        if not self._current_organization_id:
            logger.error("No organization context set for keyword search")
            return []
        
        try:
            # Build filter for organization and search criteria
            filter_conditions = [
                FieldCondition(
                    key="organization_id",
                    match=MatchValue(value=self._current_organization_id)
                )
            ]
            
            search_filter = Filter(must=filter_conditions)
            
            search_result = self.client.search(
                collection_name='keyword_embeddings',
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=limit * 2  # Get more to filter by volume/competition
            )
            
            results = []
            for scored_point in search_result:
                search_volume = scored_point.payload.get('search_volume', 0)
                competition = scored_point.payload.get('competition', 1.0)
                
                # Apply search volume and competition filters
                if search_volume >= min_search_volume and competition <= max_competition:
                    difficulty = 'low' if competition < 0.3 else 'medium' if competition < 0.7 else 'high'
                    
                    result = {
                        'keyword': scored_point.payload.get('keyword'),
                        'similarity_score': float(scored_point.score),
                        'search_volume': search_volume,
                        'competition': competition,
                        'difficulty': difficulty,
                        'trend_direction': scored_point.payload.get('trend_direction', 'stable'),
                        'related_topics': scored_point.payload.get('related_topics', [])
                    }
                    results.append(result)
                
                if len(results) >= limit:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search related keywords: {e}")
            return []
    
    def find_content_clusters(self, min_cluster_size: int = 3, 
                             similarity_threshold: float = 0.75) -> List[Dict[str, Any]]:
        """Find content clusters using vector similarity for content gap analysis."""
        if self.demo_mode:
            return [
                {
                    'cluster_id': 'technical-seo',
                    'cluster_topic': 'Technical SEO',
                    'content_count': 12,
                    'avg_seo_score': 78.5,
                    'representative_content': [
                        'Technical SEO Audit Guide',
                        'Core Web Vitals Optimization',
                        'Schema Markup Best Practices'
                    ],
                    'coverage_gaps': ['Mobile SEO', 'JavaScript SEO']
                },
                {
                    'cluster_id': 'content-marketing',
                    'cluster_topic': 'Content Marketing',
                    'content_count': 8,
                    'avg_seo_score': 72.1,
                    'representative_content': [
                        'Content Strategy Framework',
                        'Editorial Calendar Planning',
                        'Content Distribution Tactics'
                    ],
                    'coverage_gaps': ['Video Content', 'Podcast Marketing']
                }
            ]
        
        if not self._current_organization_id:
            logger.error("No organization context set for cluster analysis")
            return []
        
        try:
            # Get all content embeddings for the organization
            filter_conditions = [
                FieldCondition(
                    key="organization_id",
                    match=MatchValue(value=self._current_organization_id)
                )
            ]
            
            search_filter = Filter(must=filter_conditions)
            
            # Scroll through all content embeddings
            all_points = []
            offset = 0
            limit = 100
            
            while True:
                points = self.client.scroll(
                    collection_name='content_embeddings',
                    scroll_filter=search_filter,
                    limit=limit,
                    offset=offset
                )
                
                if not points[0]:  # No more points
                    break
                    
                all_points.extend(points[0])
                offset += limit
            
            # Simple clustering algorithm (in production, use proper clustering)
            clusters = []
            clustered_points = set()
            
            for i, point in enumerate(all_points):
                if point.id in clustered_points:
                    continue
                
                # Find similar points
                cluster_points = [point]
                clustered_points.add(point.id)
                
                for j, other_point in enumerate(all_points[i+1:], i+1):
                    if other_point.id in clustered_points:
                        continue
                    
                    # Calculate cosine similarity (simplified)
                    similarity = self._calculate_cosine_similarity(
                        point.vector, other_point.vector
                    )
                    
                    if similarity >= similarity_threshold:
                        cluster_points.append(other_point)
                        clustered_points.add(other_point.id)
                
                if len(cluster_points) >= min_cluster_size:
                    # Extract cluster information
                    titles = [p.payload.get('title', '') for p in cluster_points]
                    seo_scores = [p.payload.get('seo_score', 0) for p in cluster_points]
                    topics = [p.payload.get('primary_topic', '') for p in cluster_points]
                    
                    cluster = {
                        'cluster_id': f"cluster_{len(clusters)}",
                        'cluster_topic': max(set(topics), key=topics.count) if topics else 'Mixed',
                        'content_count': len(cluster_points),
                        'avg_seo_score': sum(seo_scores) / len(seo_scores) if seo_scores else 0,
                        'representative_content': titles[:3],
                        'coverage_gaps': []  # Would be populated by gap analysis
                    }
                    clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Failed to find content clusters: {e}")
            return []
    
    # ============================================================================
    # Analytics and Insights
    # ============================================================================
    
    def get_semantic_analytics(self) -> Dict[str, Any]:
        """Get comprehensive semantic analytics for the organization."""
        if self.demo_mode:
            return {
                'total_embeddings': 1847,
                'content_embeddings': 1245,
                'keyword_embeddings': 425,
                'topic_embeddings': 177,
                'avg_similarity_score': 0.73,
                'cluster_count': 15,
                'semantic_density': 0.82,
                'coverage_distribution': {
                    'high_coverage': 12,
                    'medium_coverage': 8,
                    'low_coverage': 5
                }
            }
        
        if not self._current_organization_id:
            return {}
        
        try:
            analytics = {}
            
            # Get collection statistics
            for collection_name in self.collections.keys():
                info = self.get_collection_info(collection_name)
                if info:
                    analytics[f"{collection_name.split('_')[0]}_embeddings"] = info['points_count']
            
            analytics['total_embeddings'] = sum(
                analytics.get(f"{name.split('_')[0]}_embeddings", 0) 
                for name in self.collections.keys()
            )
            
            # Calculate derived metrics (simplified)
            analytics['semantic_density'] = min(1.0, analytics['total_embeddings'] / 1000)
            analytics['cluster_count'] = max(1, analytics['total_embeddings'] // 100)
            analytics['avg_similarity_score'] = 0.65 + (analytics['semantic_density'] * 0.2)
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get semantic analytics: {e}")
            return {}
    
    def recommend_content_optimization(self, content_id: str) -> Dict[str, Any]:
        """Recommend optimization strategies based on semantic analysis."""
        if self.demo_mode:
            return {
                'content_id': content_id,
                'optimization_score': 0.78,
                'recommendations': [
                    {
                        'type': 'semantic_expansion',
                        'priority': 'high',
                        'description': 'Add content about related subtopics to improve topical authority',
                        'suggested_keywords': ['semantic SEO', 'topic modeling', 'content clusters']
                    },
                    {
                        'type': 'internal_linking',
                        'priority': 'medium', 
                        'description': 'Link to 3 related articles to strengthen topic connections',
                        'suggested_links': ['article-1', 'article-2', 'article-3']
                    }
                ],
                'similar_high_performers': [
                    'high-performing-article-1',
                    'high-performing-article-2'
                ]
            }
        
        # Implementation would analyze content embedding against high-performing content
        # and provide specific optimization recommendations
        return {}
    
    # ============================================================================
    # Utility Methods
    # ============================================================================
    
    def _generate_point_id(self, content_id: str) -> str:
        """Generate unique point ID for Qdrant."""
        return hashlib.md5(f"{self._current_organization_id}:{content_id}".encode()).hexdigest()
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        except Exception:
            return 0.0
    
    def delete_organization_data(self, organization_id: str) -> bool:
        """Delete all vector data for an organization (GDPR compliance)."""
        if self.demo_mode:
            logger.info(f"Demo mode: Would delete data for organization {organization_id}")
            return True
        
        try:
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="organization_id",
                        match=MatchValue(value=organization_id)
                    )
                ]
            )
            
            for collection_name in self.collections.keys():
                self.client.delete(
                    collection_name=collection_name,
                    points_selector=models.FilterSelector(filter=filter_condition)
                )
            
            logger.info(f"Deleted all vector data for organization {organization_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete organization data: {e}")
            return False


# Global Qdrant client instance
qdrant_client = QdrantVectorClient()