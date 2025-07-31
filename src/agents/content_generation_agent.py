"""
Content Generation Agent for SEO Content Knowledge Graph System.

This file imports the modular content generation agent from the content_generation package.
The agent has been refactored into separate modules for better maintainability.
"""

# Import the modular agent implementation
from .content_generation import ContentGenerationAgent, ContentGenerationRequest

# Re-export for backwards compatibility
__all__ = ['ContentGenerationAgent', 'ContentGenerationRequest']