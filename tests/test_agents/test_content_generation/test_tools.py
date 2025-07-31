"""
Unit tests for Content Generation Tools module.

Tests content generation tools including outline creation, title generation,
content writing, meta tag generation, and SEO optimization functions.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

from src.agents.content_generation.tools import ContentGenerationTools
from src.agents.content_generation.agent import ContentGenerationRequest


class TestContentGenerationTools:
    """Test cases for ContentGenerationTools class."""
    
    @pytest.fixture
    def tools(self):
        """Create ContentGenerationTools instance for testing."""
        return ContentGenerationTools()
    
    @pytest.fixture
    def sample_keywords(self):
        """Sample keywords for testing."""
        return ["technical seo", "seo audit", "website optimization"]
    
    @pytest.fixture
    def sample_content_request(self):
        """Sample content generation request."""
        return ContentGenerationRequest(
            topic="Technical SEO Best Practices",
            target_keywords=["technical seo", "seo audit"],
            content_type="blog_post",
            content_length="medium"
        )
    
    @pytest.mark.asyncio
    async def test_create_content_outline(self, tools, sample_keywords):
        """Test content outline creation."""
        topic = "Technical SEO Best Practices"
        content_type = "blog_post"
        
        result = await tools.create_content_outline(topic, sample_keywords, content_type)
        
        assert isinstance(result, dict)
        assert "title" in result
        assert "sections" in result
        assert "estimated_total_words" in result
        assert result["title"] == topic
        assert len(result["sections"]) >= 3  # Should have intro, body sections, conclusion
        
        # Check section structure
        for section in result["sections"]:
            assert "title" in section
            assert "keywords" in section
            assert "estimated_words" in section
            assert isinstance(section["estimated_words"], int)
    
    @pytest.mark.asyncio
    async def test_generate_title_variations(self, tools, sample_keywords):
        """Test title variation generation."""
        topic = "Technical SEO Best Practices"
        target_audience = "seo professionals"
        
        result = await tools.generate_title_variations(topic, sample_keywords, target_audience)
        
        assert isinstance(result, list)
        assert len(result) >= 3  # Should generate multiple variations
        
        for title in result:
            assert isinstance(title, str)
            assert len(title.strip()) > 0
            assert any(keyword.lower() in title.lower() for keyword in sample_keywords)
    
    @pytest.mark.asyncio
    async def test_create_introduction(self, tools, sample_keywords):
        """Test introduction creation."""
        title = "Technical SEO Best Practices: Complete Guide"
        hook_type = "question"
        
        result = await tools.create_introduction(title, sample_keywords, hook_type)
        
        assert isinstance(result, str)
        assert len(result) > 100  # Should be substantial
        assert any(keyword.lower() in result.lower() for keyword in sample_keywords)
        
        # For question hook, should contain a question mark
        if hook_type == "question":
            assert "?" in result
    
    @pytest.mark.asyncio
    async def test_generate_section_content(self, tools, sample_keywords):
        """Test section content generation."""
        section_title = "Technical SEO Audit Process"
        word_count = 300
        
        result = await tools.generate_section_content(section_title, sample_keywords, word_count)
        
        assert isinstance(result, str)
        assert len(result) > 200  # Should meet minimum length
        
        # Check keyword integration
        content_lower = result.lower()
        assert any(keyword.lower() in content_lower for keyword in sample_keywords)
    
    @pytest.mark.asyncio
    async def test_create_conclusion(self, tools):
        """Test conclusion creation."""
        main_points = [
            "Technical SEO Audit Process",
            "Site Speed Optimization", 
            "Core Web Vitals"
        ]
        cta_type = "action"
        
        result = await tools.create_conclusion(main_points, cta_type)
        
        assert isinstance(result, str)
        assert len(result) > 100
        
        # Should reference main points
        content_lower = result.lower()
        assert any(point.lower() in content_lower for point in main_points)
    
    @pytest.mark.asyncio
    async def test_suggest_internal_links(self, tools):
        """Test internal link suggestions."""
        content = """
        Technical SEO is essential for website optimization. 
        A comprehensive SEO audit helps identify issues.
        Site speed and Core Web Vitals are critical ranking factors.
        """
        topic = "Technical SEO"
        
        result = await tools.suggest_internal_links(content, topic)
        
        assert isinstance(result, list)
        
        for link in result:
            assert isinstance(link, dict)
            assert "anchor_text" in link
            assert "suggested_url" in link
            assert "context" in link
    
    @pytest.mark.asyncio
    async def test_generate_meta_tags(self, tools, sample_keywords):
        """Test meta tag generation."""
        title = "Technical SEO Best Practices: Complete Guide"
        content = "A comprehensive guide covering technical SEO optimization..."
        
        result = await tools.generate_meta_tags(title, content, sample_keywords)
        
        assert isinstance(result, dict)
        assert "meta_title" in result
        assert "meta_description" in result
        assert "keywords" in result
        
        # Check meta title length (should be under 60 characters)
        assert len(result["meta_title"]) <= 60
        
        # Check meta description length (should be under 160 characters)
        assert len(result["meta_description"]) <= 160
        
        # Keywords should be comma-separated string
        assert isinstance(result["keywords"], str)
        assert any(keyword in result["keywords"] for keyword in sample_keywords)
    
    @pytest.mark.asyncio
    async def test_optimize_for_featured_snippets(self, tools):
        """Test featured snippet optimization."""
        content = """
        What is technical SEO? Technical SEO involves optimizing 
        website infrastructure for search engines. Key elements include
        site speed, crawlability, and structured data.
        """
        target_query = "what is technical seo"
        
        result = await tools.optimize_for_featured_snippets(content, target_query)
        
        assert isinstance(result, str)
        assert len(result) >= len(content)  # Should be enhanced
        
        # Should contain structured elements for snippets
        result_lower = result.lower()
        assert "what is" in result_lower or "technical seo" in result_lower
    
    def test_combine_content_sections(self, tools):
        """Test content section combination."""
        title = "Technical SEO Guide"
        introduction = "This guide covers technical SEO..."
        sections_content = {
            "Section 1": "Content for section 1...",
            "Section 2": "Content for section 2..."
        }
        conclusion = "In conclusion, technical SEO is important..."
        
        result = tools.combine_content_sections(
            title, introduction, sections_content, conclusion
        )
        
        assert isinstance(result, str)
        assert title in result
        assert introduction in result
        assert conclusion in result
        
        for section_title, section_content in sections_content.items():
            assert section_title in result
            assert section_content in result
    
    def test_estimate_word_count(self, tools):
        """Test word count estimation."""
        outline = {
            "sections": [
                {"estimated_words": 200},
                {"estimated_words": 300},
                {"estimated_words": 150}
            ]
        }
        content_length = "medium"
        
        result = tools.estimate_word_count(outline, content_length)
        
        assert isinstance(result, int)
        assert result > 0
        
        # Should be based on sections plus intro/conclusion estimates
        expected_base = sum(section["estimated_words"] for section in outline["sections"])
        assert result >= expected_base
    
    @pytest.mark.asyncio
    async def test_analyze_outline_seo(self, tools, sample_keywords):
        """Test outline SEO analysis."""
        outline = {
            "title": "Technical SEO Best Practices",
            "sections": [
                {"title": "SEO Audit Process", "keywords": ["seo audit"]},
                {"title": "Technical Optimization", "keywords": ["technical seo"]}
            ]
        }
        
        result = await tools.analyze_outline_seo(outline, sample_keywords)
        
        assert isinstance(result, dict)
        assert "keyword_coverage" in result
        assert "structure_score" in result
        assert "recommendations" in result
        
        assert isinstance(result["keyword_coverage"], float)
        assert 0 <= result["keyword_coverage"] <= 1
        assert isinstance(result["structure_score"], (int, float))
        assert isinstance(result["recommendations"], list)
    
    def test_calculate_readability_score(self, tools):
        """Test readability score calculation."""
        content = """
        Technical SEO is an important aspect of search engine optimization.
        It involves optimizing your website's technical elements.
        This includes site speed, mobile-friendliness, and crawlability.
        A good technical SEO foundation helps search engines understand your content.
        """
        
        result = tools.calculate_readability_score(content)
        
        assert isinstance(result, (int, float))
        assert 0 <= result <= 100
    
    @pytest.mark.asyncio
    async def test_calculate_content_seo_score(self, tools, sample_keywords):
        """Test content SEO score calculation."""
        content = """
        Technical SEO best practices are essential for website optimization.
        A comprehensive seo audit helps identify technical issues.
        Focus on site speed, crawlability, and website optimization strategies.
        """
        
        result = await tools.calculate_content_seo_score(content, sample_keywords)
        
        assert isinstance(result, (int, float))
        assert 0 <= result <= 100
    
    @pytest.mark.asyncio
    async def test_check_brand_voice_compliance(self, tools):
        """Test brand voice compliance checking."""
        content = "This professional guide provides comprehensive technical SEO insights."
        brand_voice = {
            "tone": "professional",
            "style": "informative",
            "target_reading_level": "intermediate"
        }
        
        result = await tools.check_brand_voice_compliance(content, brand_voice)
        
        assert isinstance(result, dict)
        assert "compliance_score" in result
        assert "tone_match" in result
        assert "style_match" in result
        assert "recommendations" in result
        
        assert isinstance(result["compliance_score"], (int, float))
        assert 0 <= result["compliance_score"] <= 100
    
    @pytest.mark.asyncio
    async def test_generate_improvement_suggestions(self, tools, sample_content_request):
        """Test improvement suggestion generation."""
        content = "Basic technical SEO content with minimal optimization."
        
        result = await tools.generate_improvement_suggestions(content, sample_content_request)
        
        assert isinstance(result, list)
        
        for suggestion in result:
            assert isinstance(suggestion, str)
            assert len(suggestion.strip()) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_generated_content(self, tools, sample_content_request):
        """Test generated content analysis."""
        content = """
        Technical SEO Best Practices: A Comprehensive Guide
        
        Technical SEO is crucial for website optimization and search rankings.
        This guide covers seo audit processes and website optimization strategies.
        """
        brand_voice = {"tone": "professional", "style": "informative"}
        
        result = await tools.analyze_generated_content(content, sample_content_request, brand_voice)
        
        assert isinstance(result, dict)
        assert "quality_score" in result
        assert "keyword_optimization" in result
        assert "structure_analysis" in result
        assert "readability_metrics" in result
        
        assert isinstance(result["quality_score"], (int, float))
        assert 0 <= result["quality_score"] <= 100
    
    @pytest.mark.asyncio
    async def test_error_handling(self, tools):
        """Test error handling in tools methods."""
        # Test with invalid/empty inputs
        with patch('src.agents.content_generation.tools.logger') as mock_logger:
            result = await tools.create_content_outline("", [], "invalid_type")
            
            # Should handle gracefully and return default structure
            assert isinstance(result, dict)
            mock_logger.warning.assert_called()
    
    def test_rag_tools_integration(self, tools):
        """Test RAG tools integration."""
        # Test that RAG tools are properly initialized when available
        if hasattr(tools, 'rag_tools') and tools.rag_tools:
            assert hasattr(tools.rag_tools, 'search_knowledge_graph')
            assert hasattr(tools.rag_tools, 'find_similar_content')
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, tools, sample_keywords):
        """Test performance benchmarks for key operations."""
        import time
        
        # Test outline creation performance
        start_time = time.time()
        await tools.create_content_outline("Test Topic", sample_keywords, "blog_post")
        outline_time = time.time() - start_time
        
        # Should complete within reasonable time (< 2 seconds)
        assert outline_time < 2.0
        
        # Test title generation performance
        start_time = time.time()
        await tools.generate_title_variations("Test Topic", sample_keywords, "general")
        title_time = time.time() - start_time
        
        assert title_time < 1.0
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, tools, sample_keywords):
        """Test concurrent tool operations."""
        tasks = [
            tools.create_content_outline("Topic 1", sample_keywords, "blog_post"),
            tools.generate_title_variations("Topic 2", sample_keywords, "general"),
            tools.create_introduction("Title", sample_keywords, "question")
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All operations should complete successfully
        assert len(results) == 3
        assert all(not isinstance(result, Exception) for result in results)


class TestContentOptimizationHelpers:
    """Test cases for content optimization helper functions."""
    
    @pytest.fixture
    def tools(self):
        return ContentGenerationTools()
    
    def test_extract_keywords_from_content(self, tools):
        """Test keyword extraction from content."""
        content = """
        Technical SEO optimization involves website performance analysis.
        Core Web Vitals and site speed are important ranking factors.
        SEO audit tools help identify optimization opportunities.
        """
        
        # This would test a helper method if it exists
        # keywords = tools._extract_keywords_from_content(content)
        # assert isinstance(keywords, list)
        # assert "technical seo" in [kw.lower() for kw in keywords]
    
    def test_content_structure_validation(self, tools):
        """Test content structure validation."""
        content = """
        # Main Title
        
        ## Introduction
        This is the introduction section.
        
        ## Section 1
        Content for section 1.
        
        ## Conclusion
        Final thoughts.
        """
        
        # This would test structure validation if implemented
        # structure = tools._analyze_content_structure(content)
        # assert structure["has_headings"] == True
        # assert structure["heading_hierarchy"] == True


if __name__ == "__main__":
    pytest.main([__file__])