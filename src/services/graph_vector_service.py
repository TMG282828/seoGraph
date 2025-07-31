"""
Graph-Vector Integration Service for SEO Content Knowledge Graph System.

This service orchestrates the interaction between Neo4j (graph relationships) 
and Qdrant (vector embeddings) to provide unified content knowledge operations.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json

from ..database.neo4j_client import neo4j_client
from ..database.qdrant_client import qdrant_client
from ..database.supabase_client import supabase_client
from .embedding_service import embedding_service

logger = logging.getLogger(__name__)


class GraphVectorService:
    """Unified service for graph and vector operations with multi-tenant support."""
    
    def __init__(self):
        """Initialize the service with database clients."""
        self.neo4j = neo4j_client
        self.qdrant = qdrant_client
        self.supabase = supabase_client
        self.embedding_service = embedding_service
        self._current_organization_id = None
    
    def set_organization_context(self, organization_id: str) -> None:
        """Set organization context for all operations."""
        self._current_organization_id = organization_id
        self.neo4j.set_organization_context(organization_id)
        self.qdrant.set_organization_context(organization_id)
    
    # ============================================================================
    # Initialization and Schema Management
    # ============================================================================
    
    async def initialize_for_organization(self, organization_id: str, 
                                        org_config: Dict[str, Any]) -> bool:
        """Initialize graph and vector infrastructure for a new organization."""
        try:
            self.set_organization_context(organization_id)
            
            # Initialize Neo4j schema if needed
            schema_success = self.neo4j.initialize_schema()
            if not schema_success:
                logger.error(f"Failed to initialize Neo4j schema for org {organization_id}")
                return False
            
            # Initialize Qdrant collections if needed
            collections_success = self.qdrant.initialize_collections()
            if not collections_success:
                logger.error(f"Failed to initialize Qdrant collections for org {organization_id}")
                return False
            
            # Create organization node in graph
            org_node_success = self.neo4j.create_organization_node(
                organization_id, 
                org_config.get('name', ''),
                org_config
            )
            
            if not org_node_success:
                logger.error(f"Failed to create organization node for {organization_id}")
                return False
            
            logger.info(f"Successfully initialized graph-vector infrastructure for org {organization_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize infrastructure for org {organization_id}: {e}")
            return False
    
    # ============================================================================
    # Content Operations
    # ============================================================================
    
    async def process_content_comprehensive(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive content processing that creates graph relationships 
        and vector embeddings simultaneously.
        """
        try:
            content_id = content_data.get('id')
            if not content_id:
                logger.error("Content ID is required for processing")
                return {'success': False, 'error': 'Content ID required'}
            
            if not self._current_organization_id:
                logger.error("Organization context not set")
                return {'success': False, 'error': 'Organization context required'}
            
            # Step 1: Create content node in Neo4j
            logger.info(f"Creating Neo4j node for content {content_id}")
            neo4j_content_id = self.neo4j.create_content_node(content_data)
            
            # Step 2: Generate and store content embedding
            content_text = self._extract_content_text(content_data)
            if content_text:
                logger.info(f"Generating embedding for content {content_id}")
                embedding = await self.embedding_service.generate_content_embedding(content_text)
                
                if embedding:
                    embedding_metadata = {
                        'title': content_data.get('title', ''),
                        'summary': content_data.get('summary', ''),
                        'content_type': content_data.get('content_type', ''),
                        'url': content_data.get('url', ''),
                        'word_count': content_data.get('word_count', 0),
                        'seo_score': content_data.get('seo_score', 0),
                        'readability_score': content_data.get('readability_score', 0),
                        'primary_topic': content_data.get('primary_topic', ''),
                        'publish_date': content_data.get('publish_date', ''),
                        'source': content_data.get('source', 'manual')
                    }
                    
                    embedding_success = self.qdrant.store_content_embedding(
                        content_id, embedding, embedding_metadata
                    )
                    
                    if not embedding_success:
                        logger.warning(f"Failed to store embedding for content {content_id}")
            
            # Step 3: Process topics and create relationships
            topics = content_data.get('topics', [])
            if topics:
                logger.info(f"Linking content {content_id} to {len(topics)} topics")
                self.neo4j.link_content_to_topics(content_id, topics)
                
                # Store topic embeddings
                for topic in topics:
                    await self._process_topic_embedding(topic, content_id)
            
            # Step 4: Process keywords and create relationships
            keywords = content_data.get('keywords', [])
            if keywords:
                logger.info(f"Linking content {content_id} to {len(keywords)} keywords")
                self.neo4j.link_content_to_keywords(content_id, keywords)
                
                # Store keyword embeddings
                for keyword_data in keywords:
                    await self._process_keyword_embedding(keyword_data, content_id)
            
            # Step 5: Create content relationships with similar content
            await self._create_content_relationships(content_id, embedding if 'embedding' in locals() else None)
            
            # Step 6: Update Supabase with processing status
            await self._update_content_processing_status(content_id, 'completed')
            
            logger.info(f"Successfully processed content {content_id} comprehensively")
            return {
                'success': True,
                'content_id': content_id,
                'neo4j_id': neo4j_content_id,
                'embedding_stored': 'embedding' in locals() and embedding is not None,
                'topics_processed': len(topics),
                'keywords_processed': len(keywords)
            }
            
        except Exception as e:
            logger.error(f"Failed to process content comprehensively: {e}")
            await self._update_content_processing_status(content_id, 'failed', str(e))
            return {'success': False, 'error': str(e)}
    
    async def search_content_unified(self, query: str, limit: int = 20, 
                                   filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Unified content search combining text search (Neo4j) and semantic search (Qdrant).
        """
        try:
            results = []
            
            # Generate query embedding for semantic search
            query_embedding = await self.embedding_service.generate_query_embedding(query)
            
            # Semantic search using Qdrant
            if query_embedding:
                semantic_results = self.qdrant.search_similar_content(
                    query_embedding, 
                    limit=limit,
                    content_type=filters.get('content_type') if filters else None,
                    min_score=0.6
                )
                
                # Enhance results with graph data
                for result in semantic_results:
                    content_id = result['content_id']
                    
                    # Get graph relationships
                    recommendations = self.neo4j.get_content_recommendations(content_id, limit=3)
                    result['related_content'] = recommendations
                    
                    # Get topic hierarchy
                    # This would require additional Neo4j queries
                    result['topic_path'] = []
                    
                    results.append(result)
            
            # Text-based search using Neo4j
            text_results = self.neo4j.search_content(query, limit=limit//2)
            
            # Merge and deduplicate results
            content_ids_seen = {r['content_id'] for r in results}
            for text_result in text_results:
                if text_result['id'] not in content_ids_seen:
                    # Convert Neo4j format to unified format
                    unified_result = {
                        'content_id': text_result['id'],
                        'title': text_result['title'],
                        'summary': text_result['summary'],
                        'content_type': text_result['content_type'],
                        'url': text_result['url'],
                        'seo_score': text_result['seo_score'],
                        'similarity_score': text_result['relevance_score'],
                        'search_type': 'text_match',
                        'related_content': []
                    }
                    results.append(unified_result)
            
            # Sort by combined relevance score
            results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Failed to perform unified content search: {e}")
            return []
    
    # ============================================================================
    # Analytics and Insights
    # ============================================================================
    
    async def get_comprehensive_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics combining graph and vector insights."""
        try:
            analytics = {}
            
            # Get graph analytics
            graph_stats = self.neo4j.get_organization_stats()
            analytics['graph'] = graph_stats
            
            # Get vector analytics
            vector_stats = self.qdrant.get_semantic_analytics()
            analytics['vector'] = vector_stats
            
            # Get content analytics from Supabase
            content_stats = await self.supabase.get_content_analytics()
            analytics['content'] = content_stats
            
            # Calculate combined metrics
            analytics['combined'] = {
                'knowledge_density': self._calculate_knowledge_density(graph_stats, vector_stats),
                'semantic_coverage': self._calculate_semantic_coverage(graph_stats, vector_stats),
                'content_connectivity': self._calculate_content_connectivity(graph_stats),
                'optimization_score': self._calculate_optimization_score(analytics)
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive analytics: {e}")
            return {}
    
    async def discover_content_opportunities(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Discover content opportunities using both graph and vector analysis."""
        try:
            opportunities = []
            
            # Get content gaps from graph analysis
            graph_gaps = self.neo4j.find_content_gaps(limit=limit//2)
            
            # Get semantic opportunities from vector analysis
            # This would involve analyzing vector clusters and identifying gaps
            
            # Combine and rank opportunities
            for gap in graph_gaps:
                opportunity = {
                    'type': 'content_gap',
                    'topic': gap['topic'],
                    'priority_score': gap['priority_score'],
                    'keyword_opportunities': gap['keyword_opportunities'],
                    'search_volume_potential': gap['search_volume_potential'],
                    'competition_level': gap['competition_level'],
                    'source': 'graph_analysis',
                    'recommended_actions': [
                        f"Create comprehensive content about {gap['topic']}",
                        f"Target {gap['keyword_opportunities']} related keywords",
                        f"Focus on {gap['competition_level']} competition keywords"
                    ]
                }
                opportunities.append(opportunity)
            
            # Sort by priority score
            opportunities.sort(key=lambda x: x['priority_score'], reverse=True)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Failed to discover content opportunities: {e}")
            return []
    
    async def analyze_content_performance(self, content_id: str) -> Dict[str, Any]:
        """Comprehensive content performance analysis using graph and vector data."""
        try:
            analysis = {
                'content_id': content_id,
                'graph_analysis': {},
                'semantic_analysis': {},
                'recommendations': []
            }
            
            # Graph-based analysis
            recommendations = self.neo4j.get_content_recommendations(content_id, limit=10)
            analysis['graph_analysis'] = {
                'related_content_count': len(recommendations),
                'avg_related_seo_score': sum(r['seo_score'] for r in recommendations) / len(recommendations) if recommendations else 0,
                'topic_connectivity': len(set(r.get('topic', '') for r in recommendations))
            }
            
            # Vector-based analysis
            semantic_recommendations = self.qdrant.recommend_content_optimization(content_id)
            analysis['semantic_analysis'] = semantic_recommendations
            
            # Generate actionable recommendations
            if analysis['graph_analysis']['related_content_count'] < 3:
                analysis['recommendations'].append({
                    'type': 'internal_linking',
                    'priority': 'high',
                    'description': 'Increase internal linking to improve topic authority'
                })
            
            if analysis['graph_analysis']['avg_related_seo_score'] > 80:
                analysis['recommendations'].append({
                    'type': 'cluster_optimization',
                    'priority': 'medium', 
                    'description': 'Content is in a high-performing cluster - optimize for featured snippets'
                })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze content performance for {content_id}: {e}")
            return {}
    
    # ============================================================================
    # Private Helper Methods
    # ============================================================================
    
    def _extract_content_text(self, content_data: Dict[str, Any]) -> str:
        """Extract text for embedding generation."""
        title = content_data.get('title', '')
        summary = content_data.get('summary', '')
        content = content_data.get('content', '')
        
        # Combine title, summary, and content for embedding
        return f"{title} {summary} {content}".strip()
    
    async def _process_topic_embedding(self, topic: str, content_id: str) -> None:
        """Process and store topic embedding."""
        try:
            topic_embedding = await self.embedding_service.generate_topic_embedding(topic)
            if topic_embedding:
                metadata = {
                    'topic_name': topic,
                    'related_content': [content_id],
                    'content_count': 1
                }
                self.qdrant.store_topic_embedding(topic, topic_embedding, metadata)
        except Exception as e:
            logger.error(f"Failed to process topic embedding for {topic}: {e}")
    
    async def _process_keyword_embedding(self, keyword_data: Dict[str, Any], content_id: str) -> None:
        """Process and store keyword embedding."""
        try:
            keyword = keyword_data.get('keyword', '')
            if keyword:
                keyword_embedding = await self.embedding_service.generate_keyword_embedding(keyword)
                if keyword_embedding:
                    metadata = {
                        'keyword': keyword,
                        'search_volume': keyword_data.get('search_volume', 0),
                        'competition': keyword_data.get('competition', 0),
                        'related_content': [content_id]
                    }
                    self.qdrant.store_keyword_embedding(keyword, keyword_embedding, metadata)
        except Exception as e:
            logger.error(f"Failed to process keyword embedding: {e}")
    
    async def _create_content_relationships(self, content_id: str, embedding: Optional[List[float]]) -> None:
        """Create relationships with similar content."""
        try:
            if not embedding:
                return
            
            # Find similar content using embeddings
            similar_content = self.qdrant.search_similar_content(
                embedding, limit=5, min_score=0.7
            )
            
            # Create relationships in Neo4j
            related_ids = [item['content_id'] for item in similar_content if item['content_id'] != content_id]
            if related_ids:
                self.neo4j.create_content_relationships(content_id, related_ids, 'SEMANTICALLY_SIMILAR')
                
        except Exception as e:
            logger.error(f"Failed to create content relationships for {content_id}: {e}")
    
    async def _update_content_processing_status(self, content_id: str, status: str, 
                                              error_message: Optional[str] = None) -> None:
        """Update content processing status in Supabase."""
        try:
            update_data = {
                'processing_status': status,
                'processed_at': datetime.now().isoformat()
            }
            if error_message:
                update_data['processing_error'] = error_message
            
            # This would update the content_items table in Supabase
            # Implementation depends on Supabase client structure
            
        except Exception as e:
            logger.error(f"Failed to update processing status for {content_id}: {e}")
    
    def _calculate_knowledge_density(self, graph_stats: Dict, vector_stats: Dict) -> float:
        """Calculate knowledge density score combining graph and vector metrics."""
        try:
            graph_density = graph_stats.get('total_relationships', 0) / max(graph_stats.get('total_nodes', 1), 1)
            vector_density = vector_stats.get('semantic_density', 0)
            return min(1.0, (graph_density * 0.6 + vector_density * 0.4))
        except Exception:
            return 0.0
    
    def _calculate_semantic_coverage(self, graph_stats: Dict, vector_stats: Dict) -> float:
        """Calculate semantic coverage score."""
        try:
            graph_coverage = min(1.0, graph_stats.get('coverage_score', 0) / 100)
            vector_embeddings = vector_stats.get('total_embeddings', 0)
            vector_coverage = min(1.0, vector_embeddings / 1000)
            return (graph_coverage * 0.5 + vector_coverage * 0.5)
        except Exception:
            return 0.0
    
    def _calculate_content_connectivity(self, graph_stats: Dict) -> float:
        """Calculate content connectivity score."""
        try:
            return min(1.0, graph_stats.get('avg_connections_per_content', 0) / 10)
        except Exception:
            return 0.0
    
    def _calculate_optimization_score(self, analytics: Dict) -> float:
        """Calculate overall optimization score."""
        try:
            knowledge_density = analytics['combined'].get('knowledge_density', 0)
            semantic_coverage = analytics['combined'].get('semantic_coverage', 0)
            content_connectivity = analytics['combined'].get('content_connectivity', 0)
            
            return (knowledge_density * 0.4 + semantic_coverage * 0.3 + content_connectivity * 0.3) * 100
        except Exception:
            return 0.0


# Global graph-vector service instance
graph_vector_service = GraphVectorService()