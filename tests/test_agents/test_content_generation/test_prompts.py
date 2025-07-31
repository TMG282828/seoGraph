"""
Unit tests for Content Generation Prompts module.

Tests prompt templates, system prompt generation, and dynamic prompt creation
for different content types and scenarios.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

from src.agents.content_generation.prompts import ContentGenerationPrompts
from src.agents.content_generation.agent import ContentGenerationRequest


class TestContentGenerationPrompts:
    """Test cases for ContentGenerationPrompts class."""
    
    @pytest.fixture
    def prompts(self):
        """Create ContentGenerationPrompts instance for testing."""
        return ContentGenerationPrompts()
    
    @pytest.fixture
    def sample_request(self):
        """Sample content generation request."""
        return ContentGenerationRequest(
            topic="Technical SEO Best Practices",
            target_keywords=["technical seo", "seo audit", "website optimization"],
            content_type="blog_post",
            content_length="medium",
            writing_style="informational",
            target_audience="seo professionals"
        )
    
    @pytest.fixture
    def sample_brand_voice(self):
        """Sample brand voice configuration."""
        return {
            "tone": "professional",
            "style": "informative",
            "personality": "expert",
            "target_reading_level": "intermediate",
            "voice_characteristics": [
                "authoritative",
                "helpful",
                "data-driven"
            ]
        }
    
    @pytest.fixture
    def sample_seo_requirements(self):
        """Sample SEO requirements."""
        return {
            "target_keyword_density": 1.5,
            "content_length_preference": "medium",
            "internal_linking_style": "contextual",
            "meta_optimization": True,
            "featured_snippet_optimization": True
        }
    
    def test_prompts_initialization(self, prompts):
        """Test ContentGenerationPrompts initialization."""
        assert prompts is not None
        assert hasattr(prompts, 'get_system_prompt')
        assert hasattr(prompts, 'get_outline_prompt')
        assert hasattr(prompts, 'get_full_content_prompt')
    
    def test_get_system_prompt(self, prompts):
        """Test system prompt generation."""
        system_prompt = prompts.get_system_prompt()
        
        assert isinstance(system_prompt, str)
        assert len(system_prompt) > 100  # Should be comprehensive
        
        # Should contain key elements
        system_prompt_lower = system_prompt.lower()
        assert "content generation" in system_prompt_lower
        assert "seo" in system_prompt_lower
        assert "brand voice" in system_prompt_lower or "brand" in system_prompt_lower
        assert "keyword" in system_prompt_lower
    
    def test_get_outline_prompt(self, prompts, sample_request, sample_brand_voice, sample_seo_requirements):
        """Test outline prompt generation."""
        outline_prompt = prompts.get_outline_prompt(
            sample_request, sample_brand_voice, sample_seo_requirements
        )
        
        assert isinstance(outline_prompt, str)
        assert len(outline_prompt) > 200  # Should be detailed
        
        # Should include request details
        assert sample_request.topic in outline_prompt
        assert sample_request.content_type in outline_prompt
        
        # Should include target keywords
        for keyword in sample_request.target_keywords:
            assert keyword in outline_prompt.lower()
        
        # Should include brand voice elements
        assert sample_brand_voice["tone"] in outline_prompt
        assert sample_brand_voice["style"] in outline_prompt
    
    def test_get_full_content_prompt(self, prompts, sample_request, sample_brand_voice, sample_seo_requirements):
        """Test full content prompt generation."""
        title = "Technical SEO Best Practices: Complete Guide"
        content_outline = {
            "title": title,
            "sections": [
                {"title": "Introduction", "keywords": ["technical seo"], "estimated_words": 200},
                {"title": "Technical Audit Process", "keywords": ["seo audit"], "estimated_words": 400},
                {"title": "Conclusion", "keywords": ["website optimization"], "estimated_words": 150}
            ]
        }
        
        full_prompt = prompts.get_full_content_prompt(
            sample_request, title, content_outline, sample_brand_voice, sample_seo_requirements
        )
        
        assert isinstance(full_prompt, str)
        assert len(full_prompt) > 500  # Should be comprehensive
        
        # Should include all components
        assert title in full_prompt
        assert sample_request.topic in full_prompt
        
        # Should include outline sections
        for section in content_outline["sections"]:
            assert section["title"] in full_prompt
        
        # Should include SEO requirements
        assert str(sample_seo_requirements["target_keyword_density"]) in full_prompt
    
    def test_outline_prompt_content_type_variations(self, prompts, sample_brand_voice, sample_seo_requirements):
        """Test outline prompt for different content types."""
        content_types = ["blog_post", "article", "guide", "landing_page", "product_description"]
        
        for content_type in content_types:
            request = ContentGenerationRequest(
                topic="Test Topic",
                content_type=content_type,
                target_keywords=["test keyword"]
            )
            
            prompt = prompts.get_outline_prompt(request, sample_brand_voice, sample_seo_requirements)
            
            assert content_type in prompt
            assert isinstance(prompt, str)
            assert len(prompt) > 100
    
    def test_outline_prompt_writing_style_variations(self, prompts, sample_brand_voice, sample_seo_requirements):
        """Test outline prompt for different writing styles."""
        writing_styles = ["informational", "persuasive", "narrative", "technical"]
        
        for style in writing_styles:
            request = ContentGenerationRequest(
                topic="Test Topic",
                writing_style=style,
                target_keywords=["test keyword"]
            )
            
            prompt = prompts.get_outline_prompt(request, sample_brand_voice, sample_seo_requirements)
            
            assert style in prompt
            assert isinstance(prompt, str)
            assert len(prompt) > 100
    
    def test_full_content_prompt_length_variations(self, prompts, sample_brand_voice, sample_seo_requirements):
        """Test full content prompt for different content lengths."""
        lengths = ["short", "medium", "long"]
        title = "Test Title"
        outline = {"title": title, "sections": [{"title": "Section 1", "keywords": ["test"]}]}
        
        for length in lengths:
            request = ContentGenerationRequest(
                topic="Test Topic",
                content_length=length,
                target_keywords=["test keyword"]
            )
            
            prompt = prompts.get_full_content_prompt(
                request, title, outline, sample_brand_voice, sample_seo_requirements
            )
            
            assert length in prompt
            assert isinstance(prompt, str)
            assert len(prompt) > 200
    
    def test_prompt_brand_voice_integration(self, prompts, sample_request, sample_seo_requirements):
        """Test brand voice integration in prompts."""
        brand_voices = [
            {"tone": "casual", "style": "conversational"},
            {"tone": "formal", "style": "academic"},
            {"tone": "friendly", "style": "approachable"}
        ]
        
        for brand_voice in brand_voices:
            outline_prompt = prompts.get_outline_prompt(
                sample_request, brand_voice, sample_seo_requirements
            )
            
            assert brand_voice["tone"] in outline_prompt
            assert brand_voice["style"] in outline_prompt
    
    def test_prompt_seo_requirements_integration(self, prompts, sample_request, sample_brand_voice):
        """Test SEO requirements integration in prompts."""
        seo_configs = [
            {"target_keyword_density": 1.0, "internal_linking_style": "contextual"},
            {"target_keyword_density": 2.0, "internal_linking_style": "explicit"},
            {"target_keyword_density": 1.5, "meta_optimization": True}
        ]
        
        for seo_config in seo_configs:
            outline_prompt = prompts.get_outline_prompt(
                sample_request, sample_brand_voice, seo_config
            )
            
            assert str(seo_config["target_keyword_density"]) in outline_prompt
            
            if "internal_linking_style" in seo_config:
                assert seo_config["internal_linking_style"] in outline_prompt
    
    def test_prompt_keyword_integration(self, prompts, sample_brand_voice, sample_seo_requirements):
        """Test keyword integration in prompts."""
        keywords_sets = [
            ["single keyword"],
            ["keyword one", "keyword two"],
            ["primary keyword", "secondary keyword", "long tail keyword phrase"]
        ]
        
        for keywords in keywords_sets:
            request = ContentGenerationRequest(
                topic="Test Topic",
                target_keywords=keywords
            )
            
            prompt = prompts.get_outline_prompt(request, sample_brand_voice, sample_seo_requirements)
            
            # All keywords should appear in prompt
            for keyword in keywords:
                assert keyword in prompt.lower()
    
    def test_prompt_audience_targeting(self, prompts, sample_brand_voice, sample_seo_requirements):
        """Test audience targeting in prompts."""
        audiences = ["beginners", "professionals", "executives", "technical experts", "general audience"]
        
        for audience in audiences:
            request = ContentGenerationRequest(
                topic="Test Topic",
                target_audience=audience,
                target_keywords=["test keyword"]
            )
            
            prompt = prompts.get_outline_prompt(request, sample_brand_voice, sample_seo_requirements)
            
            assert audience in prompt
    
    def test_prompt_template_consistency(self, prompts, sample_request, sample_brand_voice, sample_seo_requirements):
        """Test that prompts maintain consistent structure."""
        # Generate multiple prompts
        outline_prompt1 = prompts.get_outline_prompt(sample_request, sample_brand_voice, sample_seo_requirements)
        outline_prompt2 = prompts.get_outline_prompt(sample_request, sample_brand_voice, sample_seo_requirements)
        
        # Should be identical for same inputs
        assert outline_prompt1 == outline_prompt2
        
        # Test full content prompt consistency
        title = "Test Title"
        outline = {"title": title, "sections": []}
        
        full_prompt1 = prompts.get_full_content_prompt(
            sample_request, title, outline, sample_brand_voice, sample_seo_requirements
        )
        full_prompt2 = prompts.get_full_content_prompt(
            sample_request, title, outline, sample_brand_voice, sample_seo_requirements
        )
        
        assert full_prompt1 == full_prompt2
    
    def test_prompt_error_handling(self, prompts):
        """Test error handling in prompt generation."""
        # Test with minimal request
        minimal_request = ContentGenerationRequest(topic="Test")
        empty_brand_voice = {}
        empty_seo_requirements = {}
        
        # Should not raise exceptions
        outline_prompt = prompts.get_outline_prompt(minimal_request, empty_brand_voice, empty_seo_requirements)
        assert isinstance(outline_prompt, str)
        assert len(outline_prompt) > 0
        
        # Test full content prompt with minimal data
        title = "Test"
        minimal_outline = {"title": title, "sections": []}
        
        full_prompt = prompts.get_full_content_prompt(
            minimal_request, title, minimal_outline, empty_brand_voice, empty_seo_requirements
        )
        assert isinstance(full_prompt, str)
        assert len(full_prompt) > 0
    
    def test_prompt_special_characters_handling(self, prompts, sample_brand_voice, sample_seo_requirements):
        """Test handling of special characters in prompts."""
        request_with_special_chars = ContentGenerationRequest(
            topic="Test Topic with \"Quotes\" & Special Characters!",
            target_keywords=["keyword with spaces", "keyword-with-dashes", "keyword's apostrophe"],
            target_audience="C-level executives & decision makers"
        )
        
        prompt = prompts.get_outline_prompt(request_with_special_chars, sample_brand_voice, sample_seo_requirements)
        
        # Should handle special characters without breaking
        assert isinstance(prompt, str)
        assert len(prompt) > 100
        assert request_with_special_chars.topic in prompt
    
    def test_prompt_length_optimization(self, prompts, sample_request, sample_brand_voice, sample_seo_requirements):
        """Test that prompts are optimized for length."""
        outline_prompt = prompts.get_outline_prompt(sample_request, sample_brand_voice, sample_seo_requirements)
        
        # Should be comprehensive but not excessive
        assert 200 <= len(outline_prompt) <= 2000
        
        # Test full content prompt
        title = "Test Title"
        outline = {"title": title, "sections": [{"title": "Section", "keywords": ["test"]}]}
        full_prompt = prompts.get_full_content_prompt(
            sample_request, title, outline, sample_brand_voice, sample_seo_requirements
        )
        
        assert 500 <= len(full_prompt) <= 3000
    
    def test_system_prompt_components(self, prompts):
        """Test system prompt contains necessary components."""
        system_prompt = prompts.get_system_prompt()
        
        # Should contain role definition
        assert "content generation" in system_prompt.lower() or "content creator" in system_prompt.lower()
        
        # Should contain capabilities
        assert "seo" in system_prompt.lower()
        assert "keyword" in system_prompt.lower()
        
        # Should contain constraints or guidelines
        assert "quality" in system_prompt.lower() or "professional" in system_prompt.lower()


class TestPromptTemplateHelpers:
    """Test cases for prompt template helper functions."""
    
    @pytest.fixture
    def prompts(self):
        return ContentGenerationPrompts()
    
    def test_prompt_sanitization(self, prompts):
        """Test prompt input sanitization if implemented."""
        # This would test sanitization helpers if they exist
        # For example: prompts._sanitize_input("test string with \"quotes\"")
        pass
    
    def test_prompt_formatting_helpers(self, prompts):
        """Test prompt formatting helper functions if implemented."""
        # This would test formatting helpers if they exist
        # For example: prompts._format_keywords_list(["keyword1", "keyword2"])
        pass
    
    def test_conditional_prompt_sections(self, prompts, sample_request, sample_brand_voice, sample_seo_requirements):
        """Test conditional sections in prompts."""
        # Test with RAG enabled
        request_with_rag = ContentGenerationRequest(
            topic="Test Topic",
            use_knowledge_graph=True,
            use_vector_search=True,
            target_keywords=["test"]
        )
        
        prompt_with_rag = prompts.get_outline_prompt(request_with_rag, sample_brand_voice, sample_seo_requirements)
        
        # Test without RAG
        request_without_rag = ContentGenerationRequest(
            topic="Test Topic",
            use_knowledge_graph=False,
            use_vector_search=False,
            target_keywords=["test"]
        )
        
        prompt_without_rag = prompts.get_outline_prompt(request_without_rag, sample_brand_voice, sample_seo_requirements)
        
        # Prompts should be different based on RAG settings
        assert prompt_with_rag != prompt_without_rag


if __name__ == "__main__":
    pytest.main([__file__])