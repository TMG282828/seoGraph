"""
Agents package for the SEO Content Knowledge Graph System.

DEPRECATED: This package provides backward compatibility for legacy imports.
New code should import directly from src.agents.

This package contains various AI agents for content analysis, SEO research,
content generation, graph management, and quality assurance.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from 'agents' is deprecated. Use 'src.agents' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from consolidated structure
try:
    # Import modularized agents
    from src.agents.content_generation import (
        ContentGenerationAgent, 
        create_content_generation_agent, 
        ContentGenerationRequest as ContentBrief
    )
    from src.agents.competitor_analysis import (
        create_competitor_analysis_agent
    )
    
    # Import non-modularized agents
    from src.agents.content_analysis_agent import ContentAnalysisAgent
    from src.agents.seo_research_agent import SEOResearchAgent  
    from src.agents.graph_management_agent import GraphManagementAgent
    from src.agents.quality_assurance_agent import QualityAssuranceAgent
    
    # Import trend analysis from legacy location if not yet moved
    try:
        from .trend_analysis import TrendAnalysisAgent, TrendData, TrendInsight
    except ImportError:
        # If trend analysis not available, create placeholder
        TrendAnalysisAgent = None
        TrendData = None
        TrendInsight = None
    
    # Create compatibility functions for the non-modularized agents
    def create_content_analysis_agent(*args, **kwargs):
        """Backward compatibility function for content analysis agent creation."""
        return ContentAnalysisAgent()
    
    def create_seo_research_agent(*args, **kwargs):
        """Backward compatibility function for SEO research agent creation."""
        return SEOResearchAgent()
    
    def create_graph_management_agent(*args, **kwargs):
        """Backward compatibility function for graph management agent creation."""
        return GraphManagementAgent()
    
    def create_quality_assurance_agent(*args, **kwargs):
        """Backward compatibility function for quality assurance agent creation."""
        return QualityAssuranceAgent()
    
    def create_trend_analysis_agent(*args, **kwargs):
        """Backward compatibility function for trend analysis agent creation."""
        if TrendAnalysisAgent:
            return TrendAnalysisAgent()
        else:
            raise ImportError("TrendAnalysisAgent not available - need to migrate to new structure")
    
    def create_content_brief(*args, **kwargs):
        """Backward compatibility function for content brief creation."""
        return ContentBrief(*args, **kwargs)

except ImportError as e:
    # Fallback to legacy implementations if new structure not available
    warnings.warn(f"Failed to import from new structure, falling back to legacy: {e}")
    from .content_analysis import ContentAnalysisAgent, create_content_analysis_agent
    from .seo_research import SEOResearchAgent, create_seo_research_agent
    from .content_generation import ContentGenerationAgent, create_content_generation_agent, ContentBrief, create_content_brief
    from .graph_management import GraphManagementAgent, create_graph_management_agent
    from .quality_assurance import QualityAssuranceAgent, create_quality_assurance_agent

__all__ = [
    "ContentAnalysisAgent",
    "create_content_analysis_agent",
    "SEOResearchAgent", 
    "create_seo_research_agent",
    "ContentGenerationAgent",
    "create_content_generation_agent",
    "ContentBrief",
    "create_content_brief",
    "GraphManagementAgent",
    "create_graph_management_agent",
    "QualityAssuranceAgent",
    "create_quality_assurance_agent",
    "TrendAnalysisAgent",
    "TrendData", 
    "TrendInsight",
    "create_trend_analysis_agent",
    "create_competitor_analysis_agent",
]