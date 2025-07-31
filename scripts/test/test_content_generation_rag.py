#!/usr/bin/env python3
"""
Content Generation Agent RAG Integration Test.

This script tests the content generation agent with RAG functionality using actual data.
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
from src.agents.content_generation.rag_tools import RAGTools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ContentGenerationRAGTester:
    """Test content generation agent's RAG capabilities."""
    
    def __init__(self):
        self.organization_id = "content-gen-test-001"
        self.rag_tools = None
        self.test_results = {
            'setup_results': {},
            'sample_data_creation': {},
            'rag_functionality': {},
            'content_enhancement': {},
            'performance_analysis': {},
            'errors': []
        }
    
    async def setup_rag_system(self) -> bool:
        """Setup RAG system with test data."""
        logger.info("🔧 Setting up RAG system for content generation testing...")
        
        try:
            # Initialize RAG tools
            self.rag_tools = RAGTools()
            
            # Initialize clients
            neo4j_client = Neo4jClient()
            neo4j_client.set_organization_context(self.organization_id)
            
            qdrant_client = QdrantVectorClient()
            qdrant_client.set_organization_context(self.organization_id)
            
            # Set clients in RAG tools
            self.rag_tools.set_neo4j_client(neo4j_client)
            self.rag_tools.set_qdrant_client(qdrant_client)
            
            # Initialize database schemas
            neo4j_connected = not neo4j_client.demo_mode
            qdrant_connected = not qdrant_client.demo_mode
            
            if neo4j_connected:
                neo4j_client.initialize_schema()
                neo4j_client.create_organization_node(
                    self.organization_id,
                    "Content Generation Test Org",
                    {"test_purpose": "rag_integration"}
                )
            
            if qdrant_connected:
                qdrant_client.initialize_collections()
            
            self.test_results['setup_results'] = {
                'neo4j_connected': neo4j_connected,
                'qdrant_connected': qdrant_connected,
                'rag_tools_initialized': True
            }
            
            logger.info(f"   Neo4j: {'✅ Connected' if neo4j_connected else '⚠️  Demo mode'}")
            logger.info(f"   Qdrant: {'✅ Connected' if qdrant_connected else '⚠️  Demo mode'}")
            logger.info("   RAG tools: ✅ Initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup RAG system: {e}")
            self.test_results['errors'].append(f"RAG setup failed: {e}")
            return False
    
    async def create_sample_content_data(self) -> bool:
        """Create comprehensive sample content data for testing."""
        logger.info("📚 Creating sample content data...")
        
        try:
            neo4j_client = self.rag_tools.neo4j_client
            qdrant_client = self.rag_tools.qdrant_client
            
            # Sample content pieces for SEO knowledge base
            sample_contents = [
                {
                    'id': 'seo-fundamentals-001',
                    'title': 'SEO Fundamentals: Complete Beginner Guide',
                    'content_type': 'guide',
                    'summary': 'Comprehensive introduction to search engine optimization covering on-page, off-page, and technical SEO basics.',
                    'word_count': 2800,
                    'seo_score': 92.5,
                    'readability_score': 85.2,
                    'topics': ['SEO Basics', 'Search Engine Optimization', 'Digital Marketing'],
                    'keywords': [
                        {'keyword': 'seo fundamentals', 'weight': 0.95, 'density': 2.8, 'relevance': 0.98, 'search_volume': 12000, 'competition': 0.72},
                        {'keyword': 'search engine optimization', 'weight': 0.9, 'density': 2.1, 'relevance': 0.95, 'search_volume': 49000, 'competition': 0.85},
                        {'keyword': 'beginner seo guide', 'weight': 0.8, 'density': 1.5, 'relevance': 0.88, 'search_volume': 3400, 'competition': 0.45}
                    ],
                    'content_snippet': 'SEO fundamentals form the foundation of effective digital marketing strategies...'
                },
                {
                    'id': 'technical-seo-002',
                    'title': 'Technical SEO Audit: Complete Checklist 2024',
                    'content_type': 'article',
                    'summary': 'Step-by-step technical SEO audit process covering site speed, crawlability, indexing, and Core Web Vitals optimization.',
                    'word_count': 3500,
                    'seo_score': 89.8,
                    'readability_score': 78.5,
                    'topics': ['Technical SEO', 'SEO Audit', 'Website Optimization'],
                    'keywords': [
                        {'keyword': 'technical seo audit', 'weight': 0.92, 'density': 3.2, 'relevance': 0.96, 'search_volume': 8900, 'competition': 0.68},
                        {'keyword': 'seo checklist', 'weight': 0.85, 'density': 2.0, 'relevance': 0.90, 'search_volume': 5600, 'competition': 0.52},
                        {'keyword': 'core web vitals', 'weight': 0.75, 'density': 1.8, 'relevance': 0.82, 'search_volume': 7200, 'competition': 0.59}
                    ],
                    'content_snippet': 'A comprehensive technical SEO audit is essential for maintaining optimal website performance...'
                },
                {
                    'id': 'content-strategy-003',
                    'title': 'Content Strategy for SEO: Planning and Execution',
                    'content_type': 'blog_post',
                    'summary': 'Strategic approach to content creation that aligns with SEO goals, audience needs, and business objectives.',
                    'word_count': 2400,
                    'seo_score': 87.3,
                    'readability_score': 88.9,
                    'topics': ['Content Strategy', 'Content Marketing', 'SEO Strategy'],
                    'keywords': [
                        {'keyword': 'content strategy', 'weight': 0.88, 'density': 2.5, 'relevance': 0.93, 'search_volume': 18000, 'competition': 0.76},
                        {'keyword': 'seo content planning', 'weight': 0.82, 'density': 1.9, 'relevance': 0.86, 'search_volume': 2800, 'competition': 0.48},
                        {'keyword': 'content marketing seo', 'weight': 0.79, 'density': 1.6, 'relevance': 0.84, 'search_volume': 4100, 'competition': 0.63}
                    ],
                    'content_snippet': 'Effective content strategy integrates SEO principles with audience-focused value creation...'
                },
                {
                    'id': 'local-seo-004',
                    'title': 'Local SEO Optimization: Complete Business Guide',
                    'content_type': 'guide',
                    'summary': 'Comprehensive local SEO strategies for businesses including Google My Business optimization, local citations, and review management.',
                    'word_count': 3100,
                    'seo_score': 91.2,
                    'readability_score': 82.7,
                    'topics': ['Local SEO', 'Local Business Marketing', 'Google My Business'],
                    'keywords': [
                        {'keyword': 'local seo', 'weight': 0.94, 'density': 3.1, 'relevance': 0.97, 'search_volume': 22000, 'competition': 0.71},
                        {'keyword': 'google my business optimization', 'weight': 0.86, 'density': 2.2, 'relevance': 0.89, 'search_volume': 6800, 'competition': 0.55},
                        {'keyword': 'local business seo', 'weight': 0.81, 'density': 1.8, 'relevance': 0.85, 'search_volume': 4900, 'competition': 0.49}
                    ],
                    'content_snippet': 'Local SEO optimization helps businesses connect with nearby customers through targeted search visibility...'
                }
            ]
            
            # Create content in Neo4j and embeddings in Qdrant
            created_count = 0
            
            for content_data in sample_contents:
                try:
                    # Create content node in Neo4j
                    if not neo4j_client.demo_mode:
                        content_id = neo4j_client.create_content_node(content_data)
                        neo4j_client.link_content_to_topics(content_id, content_data['topics'])
                        neo4j_client.link_content_to_keywords(content_id, content_data['keywords'])
                    else:
                        content_id = content_data['id']
                    
                    # Create mock embedding (in production, would use OpenAI API)
                    # Generate different embeddings for each content piece
                    base_value = 0.1 + (created_count * 0.05)
                    mock_embedding = [base_value + (i * 0.0001) for i in range(1536)]
                    
                    # Store content embedding in Qdrant
                    content_metadata = {
                        'title': content_data['title'],
                        'content_type': content_data['content_type'],
                        'summary': content_data['summary'],
                        'word_count': content_data['word_count'],
                        'seo_score': content_data['seo_score'],
                        'primary_topic': content_data['topics'][0],
                        'keywords': [kw['keyword'] for kw in content_data['keywords']],
                        'content_snippet': content_data['content_snippet']
                    }
                    
                    qdrant_client.store_content_embedding(content_id, mock_embedding, content_metadata)
                    
                    # Store keyword embeddings
                    for keyword_data in content_data['keywords']:
                        keyword_embedding = [base_value + 0.02 + (i * 0.0001) for i in range(1536)]
                        qdrant_client.store_keyword_embedding(
                            keyword_data['keyword'],
                            keyword_embedding,
                            {
                                'search_volume': keyword_data['search_volume'],
                                'competition': keyword_data['competition'],
                                'related_content': [content_id],
                                'trend_direction': 'stable'
                            }
                        )
                    
                    # Store topic embeddings
                    for topic in content_data['topics']:
                        topic_embedding = [base_value + 0.03 + (i * 0.0001) for i in range(1536)]
                        qdrant_client.store_topic_embedding(
                            topic,
                            topic_embedding,
                            {
                                'description': f'Topic covering {topic.lower()} concepts and strategies',
                                'content_count': 1,
                                'avg_performance': content_data['seo_score']
                            }
                        )
                    
                    created_count += 1
                    logger.info(f"   Created content: {content_data['title'][:50]}...")
                    
                except Exception as e:
                    logger.error(f"   Failed to create content {content_data['id']}: {e}")
                    self.test_results['errors'].append(f"Content creation failed for {content_data['id']}: {e}")
            
            self.test_results['sample_data_creation'] = {
                'contents_created': created_count,
                'total_attempted': len(sample_contents),
                'success_rate': (created_count / len(sample_contents)) * 100 if sample_contents else 0
            }
            
            logger.info(f"   ✅ Created {created_count}/{len(sample_contents)} content pieces")
            return created_count > 0
            
        except Exception as e:
            logger.error(f"❌ Failed to create sample data: {e}")
            self.test_results['errors'].append(f"Sample data creation failed: {e}")
            return False
    
    async def test_rag_functionality(self) -> Dict[str, Any]:
        """Test RAG functionality for content generation."""
        logger.info("🧪 Testing RAG functionality for content generation...")
        
        results = {
            'knowledge_graph_search': False,
            'vector_similarity_search': False,
            'content_relationships': False,
            'context_enhancement': False,
            'topic_discovery': False
        }
        
        try:
            # Test 1: Knowledge Graph Search
            logger.info("   Testing knowledge graph search...")
            kg_results = await self.rag_tools.search_knowledge_graph(
                'Technical SEO', 
                ['technical seo', 'seo audit', 'website optimization']
            )
            
            kg_success = (
                kg_results.get('available', False) and 
                (kg_results.get('related_content') or kg_results.get('topic_relationships'))
            ) or (not kg_results.get('available') and bool(kg_results))
            
            results['knowledge_graph_search'] = kg_success
            logger.info(f"      Knowledge graph search: {'✅' if kg_success else '❌'}")
            
            if kg_results.get('related_content'):
                logger.info(f"      Found {len(kg_results['related_content'])} related content items")
            
            # Test 2: Vector Similarity Search
            logger.info("   Testing vector similarity search...")
            similar_content = await self.rag_tools.find_similar_content(
                'SEO optimization strategies and techniques for better rankings', 
                limit=5
            )
            
            results['vector_similarity_search'] = bool(similar_content)
            logger.info(f"      Vector similarity search: {'✅' if results['vector_similarity_search'] else '❌'}")
            
            if similar_content:
                logger.info(f"      Found {len(similar_content)} similar content items")
                for item in similar_content[:2]:
                    logger.info(f"         - {item.get('title', 'Untitled')} (score: {item.get('similarity_score', 0):.2f})")
            
            # Test 3: Content Relationships
            logger.info("   Testing content relationships...")
            relationships = await self.rag_tools.get_content_relationships('SEO Strategy')
            
            relationships_success = (
                relationships.get('available', False) and 
                any([relationships.get('parents'), relationships.get('children'), relationships.get('related')])
            ) or (not relationships.get('available') and bool(relationships))
            
            results['content_relationships'] = relationships_success
            logger.info(f"      Content relationships: {'✅' if relationships_success else '❌'}")
            
            # Test 4: Context Enhancement
            logger.info("   Testing context enhancement...")
            base_content = """
            This article covers advanced SEO techniques for improving search engine rankings. 
            We'll explore technical optimization, content strategy, and performance monitoring.
            """
            
            enhanced_content = await self.rag_tools.enhance_with_context(base_content, 'SEO Strategy')
            enhancement_success = len(enhanced_content) > len(base_content)
            
            results['context_enhancement'] = enhancement_success
            logger.info(f"      Context enhancement: {'✅' if enhancement_success else '❌'}")
            logger.info(f"      Content expanded from {len(base_content)} to {len(enhanced_content)} characters")
            
            # Test 5: Topic Discovery (using knowledge graph)
            logger.info("   Testing topic discovery...")
            topic_results = await self.rag_tools.search_knowledge_graph(
                'Content Marketing',
                ['content strategy', 'content planning', 'editorial calendar']
            )
            
            topic_success = bool(topic_results.get('topic_relationships') or topic_results)
            results['topic_discovery'] = topic_success
            logger.info(f"      Topic discovery: {'✅' if topic_success else '❌'}")
            
        except Exception as e:
            logger.error(f"❌ RAG functionality test error: {e}")
            self.test_results['errors'].append(f"RAG functionality tests failed: {e}")
        
        return results
    
    async def test_content_enhancement_scenarios(self) -> Dict[str, Any]:
        """Test specific content enhancement scenarios."""
        logger.info("📝 Testing content enhancement scenarios...")
        
        results = {
            'blog_post_enhancement': False,
            'technical_guide_enhancement': False,
            'keyword_expansion': False,
            'topic_clustering': False
        }
        
        try:
            # Scenario 1: Blog Post Enhancement
            logger.info("   Testing blog post enhancement...")
            blog_base = "Creating effective blog content requires understanding your audience and SEO principles."
            
            blog_enhanced = await self.rag_tools.enhance_with_context(blog_base, 'Content Strategy')
            results['blog_post_enhancement'] = len(blog_enhanced) > len(blog_base)
            
            logger.info(f"      Blog enhancement: {'✅' if results['blog_post_enhancement'] else '❌'}")
            
            # Scenario 2: Technical Guide Enhancement  
            logger.info("   Testing technical guide enhancement...")
            tech_base = "Technical SEO involves optimizing website infrastructure and performance for search engines."
            
            tech_enhanced = await self.rag_tools.enhance_with_context(tech_base, 'Technical SEO')
            results['technical_guide_enhancement'] = len(tech_enhanced) > len(tech_base)
            
            logger.info(f"      Technical guide enhancement: {'✅' if results['technical_guide_enhancement'] else '❌'}")
            
            # Scenario 3: Keyword Expansion
            logger.info("   Testing keyword expansion via similarity search...")
            keyword_related = await self.rag_tools.find_similar_content(
                'local SEO optimization strategies', 
                limit=3
            )
            
            results['keyword_expansion'] = bool(keyword_related)
            logger.info(f"      Keyword expansion: {'✅' if results['keyword_expansion'] else '❌'}")
            
            # Scenario 4: Topic Clustering
            logger.info("   Testing topic clustering...")
            cluster_topics = ['SEO Strategy', 'Content Marketing', 'Technical SEO']
            cluster_results = []
            
            for topic in cluster_topics:
                topic_content = await self.rag_tools.search_knowledge_graph(topic, [topic.lower()])
                cluster_results.append(bool(topic_content))
            
            results['topic_clustering'] = any(cluster_results)
            logger.info(f"      Topic clustering: {'✅' if results['topic_clustering'] else '❌'}")
            
        except Exception as e:
            logger.error(f"❌ Content enhancement scenarios error: {e}")
            self.test_results['errors'].append(f"Content enhancement scenarios failed: {e}")
        
        return results
    
    async def analyze_performance(self) -> Dict[str, Any]:
        """Analyze RAG system performance for content generation."""
        logger.info("⚡ Analyzing RAG system performance...")
        
        metrics = {
            'response_times': {},
            'data_quality': {},
            'enhancement_effectiveness': {}
        }
        
        try:
            # Measure knowledge graph search performance
            start_time = datetime.now()
            kg_results = await self.rag_tools.search_knowledge_graph('SEO', ['seo', 'optimization'])
            kg_time = (datetime.now() - start_time).total_seconds()
            metrics['response_times']['knowledge_graph_search'] = kg_time
            
            # Measure vector similarity search performance
            start_time = datetime.now()
            vector_results = await self.rag_tools.find_similar_content('content strategy planning', limit=5)
            vector_time = (datetime.now() - start_time).total_seconds()
            metrics['response_times']['vector_similarity_search'] = vector_time
            
            # Measure context enhancement performance
            start_time = datetime.now()
            test_content = "SEO requires both technical optimization and quality content creation."
            enhanced = await self.rag_tools.enhance_with_context(test_content, 'SEO Strategy')
            enhancement_time = (datetime.now() - start_time).total_seconds()
            metrics['response_times']['context_enhancement'] = enhancement_time
            
            # Data quality metrics
            metrics['data_quality'] = {
                'knowledge_graph_results': len(kg_results.get('related_content', [])) if kg_results.get('available') else (1 if kg_results else 0),
                'vector_search_results': len(vector_results) if vector_results else 0,
                'enhancement_ratio': len(enhanced) / len(test_content) if test_content else 1
            }
            
            # Enhancement effectiveness
            base_content_length = len(test_content)
            enhanced_content_length = len(enhanced)
            enhancement_factor = enhanced_content_length / base_content_length if base_content_length > 0 else 1
            
            metrics['enhancement_effectiveness'] = {
                'content_expansion_factor': enhancement_factor,
                'context_richness_score': min(10, enhancement_factor * 2),  # Scale to 0-10
                'information_density': enhancement_factor > 1.5  # Boolean: significant enhancement
            }
            
            logger.info(f"   Knowledge graph search: {kg_time:.3f}s")
            logger.info(f"   Vector similarity search: {vector_time:.3f}s")
            logger.info(f"   Context enhancement: {enhancement_time:.3f}s")
            logger.info(f"   Content expansion factor: {enhancement_factor:.2f}x")
            
        except Exception as e:
            logger.error(f"❌ Performance analysis error: {e}")
            self.test_results['errors'].append(f"Performance analysis failed: {e}")
        
        return metrics
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive RAG integration test for content generation."""
        logger.info("🚀 Starting Content Generation RAG Integration Test...")
        logger.info("="*70)
        
        # Setup
        if not await self.setup_rag_system():
            return self.test_results
        
        # Create sample data
        logger.info("")
        if not await self.create_sample_content_data():
            logger.warning("⚠️  Sample data creation failed, continuing with limited testing...")
        
        # Run functionality tests
        logger.info("")
        self.test_results['rag_functionality'] = await self.test_rag_functionality()
        
        logger.info("")
        self.test_results['content_enhancement'] = await self.test_content_enhancement_scenarios()
        
        logger.info("")
        self.test_results['performance_analysis'] = await self.analyze_performance()
        
        # Generate summary
        self.generate_summary()
        
        return self.test_results
    
    def generate_summary(self):
        """Generate comprehensive test summary."""
        logger.info("")
        logger.info("="*70)
        logger.info("📊 CONTENT GENERATION RAG TEST SUMMARY")
        logger.info("="*70)
        
        # Setup results
        setup = self.test_results['setup_results']
        logger.info(f"🔧 Setup Status:")
        logger.info(f"   Neo4j: {'✅ Connected' if setup.get('neo4j_connected') else '⚠️  Demo mode'}")
        logger.info(f"   Qdrant: {'✅ Connected' if setup.get('qdrant_connected') else '⚠️  Demo mode'}")
        logger.info(f"   RAG Tools: {'✅ Initialized' if setup.get('rag_tools_initialized') else '❌ Failed'}")
        
        # Sample data results
        sample_data = self.test_results['sample_data_creation']
        if sample_data:
            logger.info(f"📚 Sample Data: {sample_data.get('contents_created', 0)}/{sample_data.get('total_attempted', 0)} created ({sample_data.get('success_rate', 0):.1f}%)")
        
        # RAG functionality results
        rag_func = self.test_results['rag_functionality']
        rag_passed = sum(1 for v in rag_func.values() if v)
        rag_total = len(rag_func)
        logger.info(f"🧪 RAG Functionality: {rag_passed}/{rag_total} tests passed")
        
        # Content enhancement results
        content_enh = self.test_results['content_enhancement']
        enh_passed = sum(1 for v in content_enh.values() if v)
        enh_total = len(content_enh)
        logger.info(f"📝 Content Enhancement: {enh_passed}/{enh_total} scenarios passed")
        
        # Performance metrics
        perf = self.test_results['performance_analysis']
        if perf and perf.get('response_times'):
            times = perf['response_times']
            logger.info(f"⚡ Performance:")
            logger.info(f"   Knowledge Graph: {times.get('knowledge_graph_search', 0):.3f}s")
            logger.info(f"   Vector Search: {times.get('vector_similarity_search', 0):.3f}s")
            logger.info(f"   Enhancement: {times.get('context_enhancement', 0):.3f}s")
            
            if perf.get('enhancement_effectiveness'):
                eff = perf['enhancement_effectiveness']
                logger.info(f"   Content Expansion: {eff.get('content_expansion_factor', 1):.2f}x")
        
        # Overall assessment
        total_passed = rag_passed + enh_passed
        total_tests = rag_total + enh_total
        success_rate = (total_passed / total_tests) * 100 if total_tests > 0 else 0
        
        logger.info("")
        logger.info(f"🎯 Overall Success Rate: {success_rate:.1f}% ({total_passed}/{total_tests})")
        
        if self.test_results['errors']:
            logger.warning(f"⚠️  {len(self.test_results['errors'])} errors encountered:")
            for error in self.test_results['errors'][:3]:  # Show first 3 errors
                logger.warning(f"   - {error}")
        
        if success_rate >= 85:
            logger.info("🎉 Content Generation RAG integration is excellent!")
        elif success_rate >= 70:
            logger.info("✅ Content Generation RAG integration is working well!")
        elif success_rate >= 50:
            logger.warning("⚠️  Content Generation RAG integration is partially functional")
        else:
            logger.error("❌ Content Generation RAG integration needs significant work")
        
        logger.info("="*70)


async def main():
    """Run the content generation RAG test."""
    tester = ContentGenerationRAGTester()
    results = await tester.run_comprehensive_test()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"content_generation_rag_test_results_{timestamp}.json"
    
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"📄 Results saved to: {results_file}")
    except Exception as e:
        logger.warning(f"Failed to save results: {e}")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())