"""
Unit tests for Content Generation RAG Tools module.

Tests RAG-enhanced tools for knowledge graph search, vector similarity search,
content relationships, and context enhancement functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

from src.agents.content_generation.rag_tools import RAGTools


class TestRAGTools:
    """Test cases for RAGTools class."""
    
    @pytest.fixture
    def rag_tools(self):
        """Create RAGTools instance for testing."""
        return RAGTools()
    
    @pytest.fixture
    def mock_neo4j_client(self):
        """Create mock Neo4j client."""
        mock_client = Mock()
        mock_client.demo_mode = False
        mock_client.search_content = Mock(return_value=[
            {
                "title": "Technical SEO Guide",
                "summary": "Comprehensive technical SEO optimization guide",
                "content_type": "article",
                "seo_score": 85.5
            },
            {
                "title": "SEO Audit Checklist",
                "summary": "Complete SEO audit process and checklist",
                "content_type": "guide",
                "seo_score": 78.2
            }
        ])
        mock_client.get_organization_stats = Mock(return_value={
            "total_nodes": 150,
            "total_relationships": 300
        })
        return mock_client
    
    @pytest.fixture
    def mock_qdrant_client(self):
        """Create mock Qdrant client."""
        mock_client = Mock()
        mock_client.demo_mode = False
        mock_client.search_similar_content = Mock(return_value=[
            {
                "title": "Advanced SEO Techniques",
                "summary": "Advanced SEO strategies and techniques",
                "similarity_score": 0.85,
                "content_type": "article",
                "keywords": ["advanced seo", "seo techniques"]
            },
            {
                "title": "Technical SEO Fundamentals",
                "summary": "Basic technical SEO principles",
                "similarity_score": 0.78,
                "content_type": "guide",
                "keywords": ["technical seo", "seo basics"]
            }
        ])
        return mock_client
    
    def test_rag_tools_initialization(self, rag_tools):
        """Test RAGTools initialization."""
        assert rag_tools.neo4j_client is None
        assert rag_tools.qdrant_client is None
    
    def test_set_neo4j_client(self, rag_tools, mock_neo4j_client):
        """Test Neo4j client setting."""
        rag_tools.set_neo4j_client(mock_neo4j_client)
        assert rag_tools.neo4j_client == mock_neo4j_client
    
    def test_set_qdrant_client(self, rag_tools, mock_qdrant_client):
        """Test Qdrant client setting."""
        rag_tools.set_qdrant_client(mock_qdrant_client)
        assert rag_tools.qdrant_client == mock_qdrant_client
    
    @pytest.mark.asyncio
    async def test_search_knowledge_graph_no_client(self, rag_tools):
        """Test knowledge graph search without client."""
        topic = "Technical SEO"
        keywords = ["technical seo", "seo audit"]
        
        result = await rag_tools.search_knowledge_graph(topic, keywords)
        
        assert isinstance(result, dict)
        assert result["available"] is False
        assert "related_topics" in result
        assert "connections" in result
    
    @pytest.mark.asyncio
    async def test_search_knowledge_graph_with_client(self, rag_tools, mock_neo4j_client):
        """Test knowledge graph search with client."""
        rag_tools.set_neo4j_client(mock_neo4j_client)
        
        topic = "Technical SEO"
        keywords = ["technical seo", "seo audit"]
        
        result = await rag_tools.search_knowledge_graph(topic, keywords)
        
        assert isinstance(result, dict)
        assert result["available"] is True
        assert "related_content" in result
        assert "topic_relationships" in result
        assert result["source"] == "neo4j_knowledge_graph"
        
        # Check that search was called with combined query
        expected_query = f"{topic} {' '.join(keywords)}"
        mock_neo4j_client.search_content.assert_called_once_with(expected_query, limit=10)
    
    @pytest.mark.asyncio
    async def test_search_knowledge_graph_demo_mode(self, rag_tools):
        """Test knowledge graph search in demo mode."""
        mock_client = Mock()
        mock_client.demo_mode = True
        mock_client.search_content = Mock(return_value=[])
        
        rag_tools.set_neo4j_client(mock_client)
        
        result = await rag_tools.search_knowledge_graph("Test Topic", ["keyword"])
        
        assert isinstance(result, dict)
        assert "related_content" in result
    
    @pytest.mark.asyncio
    async def test_find_similar_content_no_client(self, rag_tools):
        """Test vector similarity search without client."""
        query = "Technical SEO optimization techniques"
        
        result = await rag_tools.find_similar_content(query, limit=5)
        
        assert isinstance(result, list)
        assert len(result) == 0
    
    @pytest.mark.asyncio
    async def test_find_similar_content_with_client(self, rag_tools, mock_qdrant_client):
        """Test vector similarity search with client."""
        rag_tools.set_qdrant_client(mock_qdrant_client)
        
        query = "Technical SEO optimization techniques"
        limit = 5
        
        result = await rag_tools.find_similar_content(query, limit=limit)
        
        assert isinstance(result, list)
        assert len(result) >= 0
        
        # Verify client was called with mock embedding
        mock_qdrant_client.search_similar_content.assert_called_once()
        
        # Check call arguments
        call_args = mock_qdrant_client.search_similar_content.call_args
        assert call_args[1]["limit"] == limit
        assert call_args[1]["min_score"] == 0.5
        
        # Verify embedding was generated (mock embedding)
        embedding_arg = call_args[1]["query_embedding"]
        assert isinstance(embedding_arg, list)
        assert len(embedding_arg) == 1536  # OpenAI embedding dimension
    
    @pytest.mark.asyncio
    async def test_find_similar_content_result_processing(self, rag_tools, mock_qdrant_client):
        """Test similar content result processing."""
        rag_tools.set_qdrant_client(mock_qdrant_client)
        
        # Test with dict-format results
        mock_qdrant_client.search_similar_content.return_value = [
            {
                "title": "Test Content",
                "summary": "Test summary",
                "score": 0.85,
                "content_type": "article",
                "keywords": ["test"]
            }
        ]
        
        result = await rag_tools.find_similar_content("test query")
        
        assert len(result) == 1
        assert result[0]["title"] == "Test Content"
        assert result[0]["similarity_score"] == 0.85
        assert result[0]["content"] == "Test summary"  # Should use summary as content
    
    @pytest.mark.asyncio
    async def test_get_content_relationships_no_client(self, rag_tools):
        """Test content relationships without client."""
        topic = "SEO Strategy"
        
        result = await rag_tools.get_content_relationships(topic)
        
        assert isinstance(result, dict)
        assert result["available"] is False
        assert "relationships" in result
    
    @pytest.mark.asyncio
    async def test_get_content_relationships_with_client(self, rag_tools, mock_neo4j_client):
        """Test content relationships with client."""
        rag_tools.set_neo4j_client(mock_neo4j_client)
        
        topic = "SEO Strategy"
        
        result = await rag_tools.get_content_relationships(topic)
        
        assert isinstance(result, dict)
        assert result["available"] is True
        assert "parents" in result
        assert "children" in result
        assert "related" in result
        assert "mentioning_content" in result
        
        # Should have called search_content
        mock_neo4j_client.search_content.assert_called_with(topic, limit=5)
    
    @pytest.mark.asyncio
    async def test_enhance_with_context_no_clients(self, rag_tools):
        """Test context enhancement without clients."""
        content = "Basic SEO content."
        topic = "Technical SEO"
        
        result = await rag_tools.enhance_with_context(content, topic)
        
        # Should return original content if no enhancement possible
        assert result == content
    
    @pytest.mark.asyncio
    async def test_enhance_with_context_with_clients(self, rag_tools, mock_neo4j_client, mock_qdrant_client):
        """Test context enhancement with clients."""
        rag_tools.set_neo4j_client(mock_neo4j_client)
        rag_tools.set_qdrant_client(mock_qdrant_client)
        
        content = "Basic SEO content."
        topic = "Technical SEO"
        
        result = await rag_tools.enhance_with_context(content, topic)
        
        assert isinstance(result, str)
        assert len(result) > len(content)  # Should be enhanced
        
        # Should contain enhancement sections
        assert "Related Context from Knowledge Base:" in result or len(result) > len(content)
    
    @pytest.mark.asyncio
    async def test_enhance_with_context_content_structure(self, rag_tools, mock_neo4j_client, mock_qdrant_client):
        """Test enhanced content structure."""
        rag_tools.set_neo4j_client(mock_neo4j_client)
        rag_tools.set_qdrant_client(mock_qdrant_client)
        
        # Setup mock responses
        mock_neo4j_client.search_content.return_value = [
            {"title": "Related Article", "summary": "Related content summary"}
        ]
        
        mock_qdrant_client.search_similar_content.return_value = [
            {"title": "Similar Content", "summary": "Similar content summary", "score": 0.8}
        ]
        
        content = "Original content about technical SEO."
        topic = "Technical SEO"
        
        result = await rag_tools.enhance_with_context(content, topic)
        
        # Should contain original content
        assert "Original content about technical SEO." in result
        
        # Should contain knowledge base context
        if "Related Context from Knowledge Base:" in result:
            assert "Related Article" in result
        
        # Should contain similar content insights
        if "Similar Content Insights:" in result:
            assert "Similar Content" in result
    
    @pytest.mark.asyncio
    async def test_error_handling_in_knowledge_graph_search(self, rag_tools, mock_neo4j_client):
        """Test error handling in knowledge graph search."""
        # Setup client to raise exception
        mock_neo4j_client.search_content.side_effect = Exception("Database error")
        rag_tools.set_neo4j_client(mock_neo4j_client)
        
        result = await rag_tools.search_knowledge_graph("Test Topic", ["keyword"])
        
        assert result["available"] is False
        assert "error" in result
        assert "Database error" in result["error"]
    
    @pytest.mark.asyncio
    async def test_error_handling_in_vector_search(self, rag_tools, mock_qdrant_client):
        """Test error handling in vector search."""
        # Setup client to raise exception
        mock_qdrant_client.search_similar_content.side_effect = Exception("Vector search error")
        rag_tools.set_qdrant_client(mock_qdrant_client)
        
        result = await rag_tools.find_similar_content("test query")
        
        assert isinstance(result, list)
        assert len(result) == 0
    
    @pytest.mark.asyncio
    async def test_mock_embedding_generation(self, rag_tools, mock_qdrant_client):
        """Test mock embedding generation."""
        rag_tools.set_qdrant_client(mock_qdrant_client)
        
        await rag_tools.find_similar_content("test query with different length")
        
        # Verify embedding was generated with correct dimension
        call_args = mock_qdrant_client.search_similar_content.call_args
        embedding = call_args[1]["query_embedding"]
        
        assert len(embedding) == 1536
        assert all(isinstance(val, float) for val in embedding)
        assert embedding[0] == 0.1  # First value should be base value
    
    @pytest.mark.asyncio
    async def test_concurrent_rag_operations(self, rag_tools, mock_neo4j_client, mock_qdrant_client):
        """Test concurrent RAG operations."""
        rag_tools.set_neo4j_client(mock_neo4j_client)
        rag_tools.set_qdrant_client(mock_qdrant_client)
        
        tasks = [
            rag_tools.search_knowledge_graph("Topic 1", ["keyword1"]),
            rag_tools.find_similar_content("query 1"),
            rag_tools.get_content_relationships("Topic 2"),
            rag_tools.enhance_with_context("content", "Topic 3")
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        assert len(results) == 4
        assert all(not isinstance(result, Exception) for result in results)
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, rag_tools, mock_neo4j_client, mock_qdrant_client):
        """Test performance benchmarks for RAG operations."""
        rag_tools.set_neo4j_client(mock_neo4j_client)
        rag_tools.set_qdrant_client(mock_qdrant_client)
        
        import time
        
        # Test knowledge graph search performance
        start_time = time.time()
        await rag_tools.search_knowledge_graph("Test Topic", ["keyword"])
        kg_time = time.time() - start_time
        
        # Should complete quickly with mocked clients
        assert kg_time < 1.0
        
        # Test vector search performance
        start_time = time.time()
        await rag_tools.find_similar_content("test query")
        vector_time = time.time() - start_time
        
        assert vector_time < 1.0
    
    def test_client_mode_detection(self, rag_tools):
        """Test client mode detection (connected vs demo)."""
        # Test with demo mode client
        demo_client = Mock()
        demo_client.demo_mode = True
        
        connected_client = Mock()
        connected_client.demo_mode = False
        
        rag_tools.set_neo4j_client(demo_client)
        assert rag_tools.neo4j_client.demo_mode is True
        
        rag_tools.set_neo4j_client(connected_client)
        assert rag_tools.neo4j_client.demo_mode is False
    
    @pytest.mark.asyncio
    async def test_knowledge_graph_topic_relationships(self, rag_tools, mock_neo4j_client):
        """Test topic relationship handling in knowledge graph search."""
        rag_tools.set_neo4j_client(mock_neo4j_client)
        
        # Mock organization stats to simulate having data
        mock_neo4j_client.get_organization_stats.return_value = {"total_nodes": 100}
        
        result = await rag_tools.search_knowledge_graph("Test Topic", ["keyword"])
        
        assert "topic_relationships" in result
        assert isinstance(result["topic_relationships"], list)
        
        # Should have added a mock relationship
        if result["topic_relationships"]:
            relationship = result["topic_relationships"][0]
            assert "topic" in relationship
            assert "relationship" in relationship
            assert "score" in relationship


class TestRAGToolsIntegration:
    """Integration tests for RAG tools with real-like scenarios."""
    
    @pytest.fixture
    def configured_rag_tools(self, mock_neo4j_client, mock_qdrant_client):
        """RAG tools with both clients configured."""
        tools = RAGTools()
        tools.set_neo4j_client(mock_neo4j_client)
        tools.set_qdrant_client(mock_qdrant_client)
        return tools
    
    @pytest.mark.asyncio
    async def test_full_rag_workflow(self, configured_rag_tools):
        """Test complete RAG workflow for content enhancement."""
        topic = "Technical SEO Optimization"
        base_content = "This guide covers technical SEO best practices."
        
        # Step 1: Search knowledge graph
        kg_results = await configured_rag_tools.search_knowledge_graph(
            topic, ["technical seo", "optimization"]
        )
        
        # Step 2: Find similar content
        similar_content = await configured_rag_tools.find_similar_content(
            base_content, limit=3
        )
        
        # Step 3: Get relationships
        relationships = await configured_rag_tools.get_content_relationships(topic)
        
        # Step 4: Enhance with context
        enhanced_content = await configured_rag_tools.enhance_with_context(
            base_content, topic
        )
        
        # Verify all steps completed successfully
        assert kg_results["available"] is True
        assert isinstance(similar_content, list)
        assert relationships["available"] is True
        assert len(enhanced_content) > len(base_content)
    
    @pytest.mark.asyncio
    async def test_content_enrichment_quality(self, configured_rag_tools):
        """Test quality of content enrichment."""
        original_content = "SEO is important for websites."
        topic = "SEO Strategy"
        
        enhanced = await configured_rag_tools.enhance_with_context(original_content, topic)
        
        # Enhanced content should be significantly longer
        assert len(enhanced) > len(original_content) * 2
        
        # Should contain original content
        assert original_content in enhanced


if __name__ == "__main__":
    pytest.main([__file__])