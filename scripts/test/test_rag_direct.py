#!/usr/bin/env python3
"""
Direct RAG Integration Test for Neo4j and Qdrant.

This script tests the RAG components directly without initializing the full agent system.
"""

import os
import sys
import asyncio
import logging
import json
from typing import Dict, List, Any
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.database.neo4j_client import Neo4jClient
from src.database.qdrant_client import QdrantVectorClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DirectRAGTester:
    """Direct tester for RAG components without agent dependencies."""
    
    def __init__(self):
        self.organization_id = "test-org-001"
        self.neo4j_client = None
        self.qdrant_client = None
        self.test_results = {
            'connection_status': {},
            'basic_operations': {},
            'data_operations': {},
            'performance_metrics': {},
            'errors': []
        }
    
    def setup_clients(self) -> bool:
        """Initialize and test database clients."""
        logger.info("🔧 Setting up RAG clients...")
        
        try:
            # Initialize Neo4j client
            self.neo4j_client = Neo4jClient()
            self.neo4j_client.set_organization_context(self.organization_id)
            
            # Initialize Qdrant client  
            self.qdrant_client = QdrantVectorClient()
            self.qdrant_client.set_organization_context(self.organization_id)
            
            # Check connection status
            neo4j_connected = not self.neo4j_client.demo_mode
            qdrant_connected = not self.qdrant_client.demo_mode
            
            self.test_results['connection_status'] = {
                'neo4j': neo4j_connected,
                'qdrant': qdrant_connected,
                'neo4j_mode': 'connected' if neo4j_connected else 'demo',
                'qdrant_mode': 'connected' if qdrant_connected else 'demo'
            }
            
            logger.info(f"Neo4j: {'✅ Connected' if neo4j_connected else '⚠️  Demo mode'}")
            logger.info(f"Qdrant: {'✅ Connected' if qdrant_connected else '⚠️  Demo mode'}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup clients: {e}")
            self.test_results['errors'].append(f"Client setup failed: {e}")
            return False
    
    def test_neo4j_basic_operations(self) -> Dict[str, Any]:
        """Test basic Neo4j operations."""
        logger.info("🧪 Testing Neo4j basic operations...")
        
        results = {
            'schema_init': False,
            'organization_create': False,
            'content_create': False,
            'search_content': False,
            'get_stats': False
        }
        
        try:
            # Test schema initialization
            schema_result = self.neo4j_client.initialize_schema()
            results['schema_init'] = schema_result
            logger.info(f"   Schema init: {'✅' if schema_result else '❌'}")
            
            # Test organization creation
            org_result = self.neo4j_client.create_organization_node(
                self.organization_id,
                "Test Organization", 
                {"test_mode": True}
            )
            results['organization_create'] = org_result
            logger.info(f"   Organization: {'✅' if org_result else '❌'}")
            
            # Test content creation
            sample_content = {
                'id': 'rag-test-001',
                'title': 'RAG Integration Test Content',
                'content_type': 'article',
                'summary': 'Test content for RAG integration validation',
                'word_count': 1500,
                'seo_score': 75.0,
                'readability_score': 80.0,
                'author': 'RAG Tester'
            }
            
            content_id = self.neo4j_client.create_content_node(sample_content)
            results['content_create'] = bool(content_id)
            logger.info(f"   Content create: {'✅' if content_id else '❌'}")
            
            # Test content search
            search_results = self.neo4j_client.search_content("RAG", limit=5)
            results['search_content'] = bool(search_results)
            logger.info(f"   Content search: {'✅' if results['search_content'] else '❌'}")
            
            # Test organization stats
            org_stats = self.neo4j_client.get_organization_stats()
            results['get_stats'] = bool(org_stats)
            logger.info(f"   Organization stats: {'✅' if results['get_stats'] else '❌'}")
            
            if org_stats:
                logger.info(f"      Nodes: {org_stats.get('total_nodes', 0)}")
                logger.info(f"      Relationships: {org_stats.get('total_relationships', 0)}")
            
        except Exception as e:
            logger.error(f"❌ Neo4j basic operations error: {e}")
            self.test_results['errors'].append(f"Neo4j basic operations: {e}")
        
        return results
    
    def test_qdrant_basic_operations(self) -> Dict[str, Any]:
        """Test basic Qdrant operations."""
        logger.info("🧪 Testing Qdrant basic operations...")
        
        results = {
            'collections_init': False,
            'content_embedding': False,
            'keyword_embedding': False,
            'similarity_search': False,
            'get_analytics': False
        }
        
        try:
            # Test collection initialization
            collections_result = self.qdrant_client.initialize_collections()
            results['collections_init'] = collections_result
            logger.info(f"   Collections init: {'✅' if collections_result else '❌'}")
            
            # Generate mock embeddings (1536-dimensional for OpenAI compatibility)
            mock_embedding = [0.1 + (i * 0.001) for i in range(1536)]
            
            # Test content embedding storage
            content_metadata = {
                'title': 'RAG Test Content',
                'content_type': 'article',
                'summary': 'Test content for embedding',
                'word_count': 1500,
                'seo_score': 75.0,
                'primary_topic': 'RAG Testing'
            }
            
            content_embed_result = self.qdrant_client.store_content_embedding(
                'rag-test-001', mock_embedding, content_metadata
            )
            results['content_embedding'] = content_embed_result
            logger.info(f"   Content embedding: {'✅' if content_embed_result else '❌'}")
            
            # Test keyword embedding storage
            keyword_embed_result = self.qdrant_client.store_keyword_embedding(
                'rag integration testing',
                mock_embedding,
                {
                    'search_volume': 1200,
                    'competition': 0.45,
                    'trend_direction': 'stable'
                }
            )
            results['keyword_embedding'] = keyword_embed_result
            logger.info(f"   Keyword embedding: {'✅' if keyword_embed_result else '❌'}")
            
            # Test similarity search
            similar_content = self.qdrant_client.search_similar_content(
                mock_embedding, limit=5, min_score=0.5
            )
            results['similarity_search'] = bool(similar_content)
            logger.info(f"   Similarity search: {'✅' if results['similarity_search'] else '❌'}")
            
            if similar_content:
                logger.info(f"      Found {len(similar_content)} similar items")
            
            # Test analytics
            analytics = self.qdrant_client.get_semantic_analytics()
            results['get_analytics'] = bool(analytics)
            logger.info(f"   Get analytics: {'✅' if results['get_analytics'] else '❌'}")
            
            if analytics:
                logger.info(f"      Total embeddings: {analytics.get('total_embeddings', 0)}")
                logger.info(f"      Semantic density: {analytics.get('semantic_density', 0):.2f}")
            
        except Exception as e:
            logger.error(f"❌ Qdrant basic operations error: {e}")
            self.test_results['errors'].append(f"Qdrant basic operations: {e}")
        
        return results
    
    def test_data_integration(self) -> Dict[str, Any]:
        """Test data integration between Neo4j and Qdrant."""
        logger.info("🧪 Testing data integration...")
        
        results = {
            'create_test_data': False,
            'cross_reference': False,
            'content_enrichment': False
        }
        
        try:
            # Create test content in Neo4j
            test_content = {
                'id': 'integration-test-001',
                'title': 'Advanced Content Strategy for SEO Success',
                'content_type': 'guide', 
                'summary': 'Comprehensive guide covering content strategy, keyword research, and performance optimization',
                'word_count': 3200,
                'seo_score': 88.5,
                'readability_score': 82.3,
                'author': 'Content Expert'
            }
            
            content_id = self.neo4j_client.create_content_node(test_content)
            
            # Link to topics and keywords
            if content_id:
                topics = ['Content Strategy', 'SEO Optimization', 'Keyword Research']
                self.neo4j_client.link_content_to_topics(content_id, topics)
                
                keywords = [
                    {'keyword': 'content strategy', 'weight': 0.9, 'density': 2.3, 'relevance': 0.95, 'search_volume': 8900, 'competition': 0.72},
                    {'keyword': 'seo content planning', 'weight': 0.8, 'density': 1.8, 'relevance': 0.88, 'search_volume': 4500, 'competition': 0.56},
                    {'keyword': 'keyword research tools', 'weight': 0.85, 'density': 1.6, 'relevance': 0.91, 'search_volume': 6200, 'competition': 0.64}
                ]
                self.neo4j_client.link_content_to_keywords(content_id, keywords)
                
                results['create_test_data'] = True
                logger.info("   ✅ Test data created in Neo4j")
            
            # Create corresponding embeddings in Qdrant
            if results['create_test_data']:
                # Create content embedding
                content_embedding = [0.2 + (i * 0.0005) for i in range(1536)]  # Mock embedding
                content_metadata = {
                    'title': test_content['title'],
                    'content_type': test_content['content_type'],
                    'summary': test_content['summary'],
                    'word_count': test_content['word_count'],
                    'seo_score': test_content['seo_score'],
                    'primary_topic': 'Content Strategy',
                    'keywords': ['content strategy', 'seo content planning', 'keyword research tools']
                }
                
                embed_success = self.qdrant_client.store_content_embedding(
                    content_id, content_embedding, content_metadata
                )
                
                # Create keyword embeddings
                for keyword_data in keywords:
                    keyword_embedding = [0.15 + (i * 0.0003) for i in range(1536)]
                    keyword_success = self.qdrant_client.store_keyword_embedding(
                        keyword_data['keyword'],
                        keyword_embedding,
                        {
                            'search_volume': keyword_data['search_volume'],
                            'competition': keyword_data['competition'],
                            'related_content': [content_id]
                        }
                    )
                
                results['cross_reference'] = embed_success
                logger.info(f"   Cross-reference: {'✅' if embed_success else '❌'}")
            
            # Test content enrichment workflow
            if results['cross_reference']:
                # Search for related content in Neo4j
                neo4j_related = self.neo4j_client.search_content("content strategy", limit=3)
                
                # Search for similar content in Qdrant
                query_embedding = [0.18 + (i * 0.0004) for i in range(1536)]
                qdrant_similar = self.qdrant_client.search_similar_content(query_embedding, limit=3)
                
                enrichment_success = bool(neo4j_related or qdrant_similar)
                results['content_enrichment'] = enrichment_success
                logger.info(f"   Content enrichment: {'✅' if enrichment_success else '❌'}")
                
                if neo4j_related:
                    logger.info(f"      Neo4j found {len(neo4j_related)} related items")
                if qdrant_similar:
                    logger.info(f"      Qdrant found {len(qdrant_similar)} similar items")
            
        except Exception as e:
            logger.error(f"❌ Data integration error: {e}")
            self.test_results['errors'].append(f"Data integration: {e}")
        
        return results
    
    def measure_performance(self) -> Dict[str, Any]:
        """Measure performance of RAG operations."""
        logger.info("📊 Measuring performance...")
        
        metrics = {
            'neo4j_operations': {},
            'qdrant_operations': {},
            'data_volumes': {}
        }
        
        try:
            # Measure Neo4j operations
            start_time = datetime.now()
            search_results = self.neo4j_client.search_content("test", limit=10)
            neo4j_search_time = (datetime.now() - start_time).total_seconds()
            
            start_time = datetime.now()
            org_stats = self.neo4j_client.get_organization_stats()
            neo4j_stats_time = (datetime.now() - start_time).total_seconds()
            
            metrics['neo4j_operations'] = {
                'search_time': neo4j_search_time,
                'stats_time': neo4j_stats_time,
                'results_count': len(search_results) if search_results else 0
            }
            
            # Measure Qdrant operations
            mock_embedding = [0.1] * 1536
            
            start_time = datetime.now()
            similar_content = self.qdrant_client.search_similar_content(mock_embedding, limit=10)
            qdrant_search_time = (datetime.now() - start_time).total_seconds()
            
            start_time = datetime.now()
            analytics = self.qdrant_client.get_semantic_analytics()
            qdrant_analytics_time = (datetime.now() - start_time).total_seconds()
            
            metrics['qdrant_operations'] = {
                'search_time': qdrant_search_time,
                'analytics_time': qdrant_analytics_time,
                'results_count': len(similar_content) if similar_content else 0
            }
            
            # Data volumes
            if org_stats:
                metrics['data_volumes']['neo4j_nodes'] = org_stats.get('total_nodes', 0)
                metrics['data_volumes']['neo4j_relationships'] = org_stats.get('total_relationships', 0)
            
            if analytics:
                metrics['data_volumes']['qdrant_embeddings'] = analytics.get('total_embeddings', 0)
                metrics['data_volumes']['cluster_count'] = analytics.get('cluster_count', 0)
            
            logger.info(f"   Neo4j search: {neo4j_search_time:.3f}s")
            logger.info(f"   Qdrant search: {qdrant_search_time:.3f}s")
            logger.info(f"   Data volumes: Neo4j nodes={metrics['data_volumes'].get('neo4j_nodes', 0)}, "
                       f"Qdrant embeddings={metrics['data_volumes'].get('qdrant_embeddings', 0)}")
            
        except Exception as e:
            logger.error(f"❌ Performance measurement error: {e}")
            self.test_results['errors'].append(f"Performance measurement: {e}")
        
        return metrics
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all RAG tests."""
        logger.info("🚀 Starting Direct RAG Integration Tests...")
        logger.info("="*60)
        
        # Setup
        if not self.setup_clients():
            return self.test_results
        
        # Run tests
        logger.info("")
        self.test_results['basic_operations'] = {
            'neo4j': self.test_neo4j_basic_operations(),
            'qdrant': self.test_qdrant_basic_operations()
        }
        
        logger.info("")
        self.test_results['data_operations'] = self.test_data_integration()
        
        logger.info("")
        self.test_results['performance_metrics'] = self.measure_performance()
        
        # Generate summary
        self.generate_summary()
        
        return self.test_results
    
    def generate_summary(self):
        """Generate test summary."""
        logger.info("")
        logger.info("="*60)
        logger.info("📋 TEST SUMMARY")
        logger.info("="*60)
        
        # Connection status
        conn_status = self.test_results['connection_status']
        logger.info(f"🔗 Connections:")
        logger.info(f"   Neo4j: {'✅ Connected' if conn_status.get('neo4j') else '⚠️  Demo mode'}")
        logger.info(f"   Qdrant: {'✅ Connected' if conn_status.get('qdrant') else '⚠️  Demo mode'}")
        
        # Test results
        neo4j_tests = self.test_results['basic_operations'].get('neo4j', {})
        neo4j_passed = sum(1 for v in neo4j_tests.values() if v)
        neo4j_total = len(neo4j_tests)
        
        qdrant_tests = self.test_results['basic_operations'].get('qdrant', {})
        qdrant_passed = sum(1 for v in qdrant_tests.values() if v)
        qdrant_total = len(qdrant_tests)
        
        data_tests = self.test_results['data_operations']
        data_passed = sum(1 for v in data_tests.values() if v)
        data_total = len(data_tests)
        
        logger.info(f"🧪 Test Results:")
        logger.info(f"   Neo4j Basic: {neo4j_passed}/{neo4j_total} passed")
        logger.info(f"   Qdrant Basic: {qdrant_passed}/{qdrant_total} passed")
        logger.info(f"   Data Integration: {data_passed}/{data_total} passed")
        
        # Performance
        perf = self.test_results['performance_metrics']
        if perf:
            neo4j_perf = perf.get('neo4j_operations', {})
            qdrant_perf = perf.get('qdrant_operations', {})
            logger.info(f"⚡ Performance:")
            logger.info(f"   Neo4j search: {neo4j_perf.get('search_time', 0):.3f}s")
            logger.info(f"   Qdrant search: {qdrant_perf.get('search_time', 0):.3f}s")
        
        # Overall assessment
        total_passed = neo4j_passed + qdrant_passed + data_passed
        total_tests = neo4j_total + qdrant_total + data_total
        success_rate = (total_passed / total_tests) * 100 if total_tests > 0 else 0
        
        logger.info(f"")
        logger.info(f"🎯 Overall Success Rate: {success_rate:.1f}% ({total_passed}/{total_tests})")
        
        if self.test_results['errors']:
            logger.warning(f"⚠️  Errors encountered:")
            for error in self.test_results['errors'][:3]:  # Show first 3 errors
                logger.warning(f"   - {error}")
        
        if success_rate >= 80:
            logger.info("✅ RAG integration is working well!")
        elif success_rate >= 60:
            logger.warning("⚠️  RAG integration is partially functional")
        else:
            logger.error("❌ RAG integration needs attention")
        
        logger.info("="*60)


def main():
    """Run the direct RAG tests."""
    tester = DirectRAGTester()
    results = tester.run_all_tests()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"rag_direct_test_results_{timestamp}.json"
    
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"📄 Results saved to: {results_file}")
    except Exception as e:
        logger.warning(f"Failed to save results: {e}")
    
    return results


if __name__ == "__main__":
    main()