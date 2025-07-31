"""
Unit tests for Content Generation Agent main module.

Tests agent initialization, task execution, and integration with RAG tools.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any, List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

from src.agents.content_generation.agent import ContentGenerationAgent, ContentGenerationRequest
from src.agents.base_agent import AgentContext, AgentResult


class TestContentGenerationAgent:
    """Test cases for ContentGenerationAgent class."""
    
    @pytest.fixture
    def agent(self):
        """Create a ContentGenerationAgent instance for testing."""
        return ContentGenerationAgent()
    
    @pytest.fixture
    def mock_context(self):
        """Create a mock AgentContext for testing."""
        return AgentContext(
            organization_id="test-org-123",
            user_id="test-user-456",
            brand_voice_config={
                "tone": "professional",
                "style": "informative"
            },
            seo_preferences={
                "target_keyword_density": 1.5,
                "content_length_preference": "medium"
            }
        )
    
    @pytest.fixture
    def sample_request(self):
        """Create a sample ContentGenerationRequest."""
        return {
            "content_type": "blog_post",
            "topic": "Technical SEO Best Practices",
            "target_keywords": ["technical seo", "seo audit", "website optimization"],
            "content_length": "medium",
            "writing_style": "informational",
            "target_audience": "seo professionals",
            "outline_only": False,
            "include_meta_tags": True,
            "use_knowledge_graph": True,
            "use_vector_search": True
        }
    
    def test_agent_initialization(self, agent):
        """Test that agent initializes correctly."""
        assert agent.name == "content_generation"
        assert "content generation" in agent.description.lower()
        assert agent.tools is not None
        assert agent.prompts is not None
        assert hasattr(agent, '_agent')  # Pydantic AI agent
    
    def test_content_generation_request_validation(self, sample_request):
        """Test ContentGenerationRequest model validation."""
        request = ContentGenerationRequest(**sample_request)
        
        assert request.content_type == "blog_post"
        assert request.topic == "Technical SEO Best Practices"
        assert len(request.target_keywords) == 3
        assert request.content_length == "medium"
        assert request.use_knowledge_graph is True
    
    def test_content_generation_request_defaults(self):
        """Test ContentGenerationRequest with default values."""
        minimal_request = {"topic": "Test Topic"}
        request = ContentGenerationRequest(**minimal_request)
        
        assert request.content_type == "blog_post"
        assert request.content_length == "medium"
        assert request.writing_style == "informational"
        assert request.outline_only is False
        assert request.use_knowledge_graph is True
    
    @pytest.mark.asyncio
    async def test_get_brand_voice_config(self, agent, mock_context):
        """Test brand voice configuration retrieval."""
        agent._current_context = mock_context
        
        brand_voice = await agent._get_brand_voice_config()
        
        assert brand_voice["tone"] == "professional"
        assert brand_voice["style"] == "informative"
    
    @pytest.mark.asyncio
    async def test_get_seo_preferences(self, agent, mock_context):
        """Test SEO preferences retrieval."""
        agent._current_context = mock_context
        
        seo_prefs = await agent._get_seo_preferences()
        
        assert seo_prefs["target_keyword_density"] == 1.5
        assert seo_prefs["content_length_preference"] == "medium"
    
    @pytest.mark.asyncio
    @patch('src.agents.content_generation.agent.ContentGenerationAgent._generate_outline_only')
    async def test_execute_task_outline_only(self, mock_outline, agent, mock_context, sample_request):
        """Test task execution for outline-only generation."""
        sample_request["outline_only"] = True
        
        mock_outline.return_value = {
            "generation_type": "outline",
            "topic": "Technical SEO Best Practices",
            "title_variations": ["Title 1", "Title 2"],
            "content_outline": {"sections": []},
            "confidence_score": 0.9
        }
        
        result = await agent._execute_task(sample_request, mock_context)
        
        mock_outline.assert_called_once()
        assert result["generation_type"] == "outline"
        assert result["confidence_score"] == 0.9
    
    @pytest.mark.asyncio
    @patch('src.agents.content_generation.agent.ContentGenerationAgent._generate_full_content')
    async def test_execute_task_full_content(self, mock_full, agent, mock_context, sample_request):
        """Test task execution for full content generation."""
        mock_full.return_value = {
            "generation_type": "full_content",
            "topic": "Technical SEO Best Practices",
            "content": "Generated content...",
            "word_count": 800,
            "confidence_score": 0.85
        }
        
        result = await agent._execute_task(sample_request, mock_context)
        
        mock_full.assert_called_once()
        assert result["generation_type"] == "full_content"
        assert result["word_count"] == 800
    
    @pytest.mark.asyncio
    @patch('src.agents.content_generation.tools.ContentGenerationTools.generate_title_variations')
    @patch('src.agents.content_generation.tools.ContentGenerationTools.create_content_outline')
    async def test_generate_outline_only(self, mock_outline, mock_titles, agent, sample_request):
        """Test outline generation functionality."""
        mock_titles.return_value = [
            "Technical SEO Best Practices: Complete Guide",
            "Ultimate Technical SEO Optimization Guide"
        ]
        
        mock_outline.return_value = {
            "title": "Technical SEO Best Practices",
            "sections": [
                {"title": "Introduction", "estimated_words": 200},
                {"title": "Technical Audit", "estimated_words": 400}
            ]
        }
        
        request = ContentGenerationRequest(**sample_request)
        brand_voice = {"tone": "professional"}
        seo_requirements = {"target_keyword_density": 1.5}
        
        result = await agent._generate_outline_only(request, brand_voice, seo_requirements)
        
        assert result["generation_type"] == "outline"
        assert len(result["title_variations"]) == 2
        assert "sections" in result["content_outline"]
        assert result["confidence_score"] == 0.9
    
    @pytest.mark.asyncio
    @patch('src.agents.content_generation.agent.ContentGenerationAgent._generate_outline_only')
    @patch('src.agents.content_generation.tools.ContentGenerationTools.create_introduction')
    @patch('src.agents.content_generation.tools.ContentGenerationTools.generate_section_content')
    @patch('src.agents.content_generation.tools.ContentGenerationTools.create_conclusion')
    async def test_generate_full_content(self, mock_conclusion, mock_section, mock_intro, 
                                       mock_outline, agent, sample_request, mock_context):
        """Test full content generation functionality."""
        # Setup mocks
        mock_outline.return_value = {
            "title_variations": ["Test Title"],
            "content_outline": {
                "sections": [
                    {"title": "Section 1", "keywords": ["keyword1"], "estimated_words": 200}
                ]
            }
        }
        
        mock_intro.return_value = "This is the introduction..."
        mock_section.return_value = "Section content here..."
        mock_conclusion.return_value = "In conclusion..."
        
        # Mock the agent's tools methods
        agent.tools.combine_content_sections = Mock(return_value="Full combined content")
        agent.tools.generate_meta_tags = AsyncMock(return_value={"title": "Meta Title"})
        agent.tools.suggest_internal_links = AsyncMock(return_value=[])
        agent.tools.optimize_for_featured_snippets = AsyncMock(return_value="Optimized content")
        agent.tools.analyze_generated_content = AsyncMock(return_value={"quality_score": 85})
        agent.tools.calculate_readability_score = Mock(return_value=75.0)
        agent.tools.calculate_content_seo_score = AsyncMock(return_value=80.0)
        agent.tools.check_brand_voice_compliance = AsyncMock(return_value={"compliance_score": 90})
        agent.tools.generate_improvement_suggestions = AsyncMock(return_value=["Suggestion 1"])
        
        request = ContentGenerationRequest(**sample_request)
        brand_voice = {"tone": "professional"}
        seo_requirements = {"target_keyword_density": 1.5}
        
        result = await agent._generate_full_content(request, brand_voice, seo_requirements, mock_context)
        
        assert result["generation_type"] == "full_content"
        assert result["content"] == "Optimized content"
        assert "meta_tags" in result
        assert result["confidence_score"] == 0.85
    
    @pytest.mark.asyncio
    async def test_execute_with_error_handling(self, agent, mock_context):
        """Test error handling in task execution."""
        # Test with invalid request data
        invalid_request = {"invalid_field": "value"}
        
        with patch.object(agent, '_execute_task', side_effect=ValueError("Invalid request")):
            result = await agent.execute(invalid_request, mock_context)
            
            assert result.success is False
            assert "Invalid request" in result.error_message
            assert result.agent_name == "content_generation"
    
    @pytest.mark.asyncio
    @patch('src.agents.base_agent.neo4j_client')
    @patch('src.agents.base_agent.qdrant_client')
    async def test_organization_context_setting(self, mock_qdrant, mock_neo4j, agent, mock_context, sample_request):
        """Test that organization context is properly set for services."""
        with patch.object(agent, '_execute_task', return_value={"test": "result"}):
            await agent.execute(sample_request, mock_context)
            
            mock_neo4j.set_organization_context.assert_called_with("test-org-123")
            mock_qdrant.set_organization_context.assert_called_with("test-org-123")
    
    def test_system_prompt_retrieval(self, agent):
        """Test system prompt retrieval."""
        with patch.object(agent.prompts, 'get_system_prompt', return_value="Test system prompt"):
            prompt = agent._get_system_prompt()
            assert prompt == "Test system prompt"
    
    @pytest.mark.asyncio
    async def test_rag_client_initialization(self, agent):
        """Test RAG client initialization."""
        # Test when clients are available
        assert hasattr(agent, 'neo4j_client')
        assert hasattr(agent, 'qdrant_client')
        
        # Verify tools have access to clients
        if agent.tools.rag_tools:
            assert hasattr(agent.tools.rag_tools, 'neo4j_client')
            assert hasattr(agent.tools.rag_tools, 'qdrant_client')
    
    @pytest.mark.asyncio
    async def test_concurrent_execution(self, agent, mock_context, sample_request):
        """Test concurrent execution of multiple content generation tasks."""
        with patch.object(agent, '_execute_task', return_value={"result": "success"}):
            # Create multiple tasks
            tasks = [
                agent.execute(sample_request, mock_context) 
                for _ in range(3)
            ]
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 3
            assert all(result.success for result in results)


class TestContentGenerationRequest:
    """Test cases for ContentGenerationRequest model."""
    
    def test_valid_content_types(self):
        """Test validation of content types."""
        valid_types = ["blog_post", "article", "guide", "landing_page", "product_description"]
        
        for content_type in valid_types:
            request = ContentGenerationRequest(
                topic="Test Topic",
                content_type=content_type
            )
            assert request.content_type == content_type
    
    def test_keyword_list_handling(self):
        """Test keyword list processing."""
        request = ContentGenerationRequest(
            topic="Test Topic",
            target_keywords=["keyword1", "keyword2", "keyword3"]
        )
        
        assert len(request.target_keywords) == 3
        assert "keyword1" in request.target_keywords
    
    def test_rag_configuration(self):
        """Test RAG-specific configuration options."""
        request = ContentGenerationRequest(
            topic="Test Topic",
            use_knowledge_graph=False,
            use_vector_search=True,
            similarity_threshold=0.8,
            max_related_content=3
        )
        
        assert request.use_knowledge_graph is False
        assert request.use_vector_search is True
        assert request.similarity_threshold == 0.8
        assert request.max_related_content == 3
    
    def test_human_in_loop_config(self):
        """Test human-in-the-loop configuration."""
        hil_config = {
            "review_required": True,
            "confidence_threshold": 0.8
        }
        
        request = ContentGenerationRequest(
            topic="Test Topic",
            human_in_loop=hil_config
        )
        
        assert request.human_in_loop["review_required"] is True
        assert request.human_in_loop["confidence_threshold"] == 0.8


if __name__ == "__main__":
    pytest.main([__file__])