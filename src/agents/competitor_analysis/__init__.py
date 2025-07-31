"""
Competitor Analysis Agent Package.

Provides comprehensive competitor analysis capabilities including content strategy
analysis, keyword gap identification, and competitive intelligence generation.

This package is organized into focused modules:
- models: Data models and type definitions
- content_analyzer: Content strategy analysis functionality
- keyword_analyzer: Keyword gap analysis and opportunities
- analysis_workflows: Main analysis workflows and helper functions
- agent: Pydantic AI agent interface

Example usage:
    from src.agents.competitor_analysis import competitor_analysis_agent, analyze_competitor_simple
    
    # Simple analysis
    profile = await analyze_competitor_simple("example.com")
    
    # Full agent usage
    result = await competitor_analysis_agent.run(
        "Analyze competitor profile for example.com",
        deps=deps
    )
"""

from .agent import (
    competitor_analysis_agent,
    analyze_competitor_profile,
    identify_competitor_opportunities,
    monitor_competitor_changes,
    analyze_competitor_simple
)
from .models import (
    CompetitorAnalysisError,
    CompetitorAnalysisDeps, 
    CompetitorInsight,
    CompetitorProfile,
    KeywordGapAnalysis
)
from .content_analyzer import ContentAnalyzer
from .keyword_analyzer import KeywordAnalyzer
from .analysis_workflows import (
    analyze_competitor_profile as workflow_analyze_profile,
    identify_competitor_opportunities as workflow_identify_opportunities,
    monitor_competitor_changes as workflow_monitor_changes
)

# Import dependencies for compatibility function
from typing import Dict, Any, Optional
try:
    from ...database.neo4j_client import Neo4jClient
    from ...database.qdrant_client import QdrantClient
    from ...database.supabase_client import SupabaseClient
    from ...services.embedding_service import EmbeddingService
except ImportError:
    # Fallback imports from legacy structure
    try:
        from database.neo4j_client import Neo4jClient
        from database.qdrant_client import QdrantClient
        from database.supabase_client import SupabaseClient
        from services.embedding_service import EmbeddingService
    except ImportError:
        # Define dummy classes if imports fail
        Neo4jClient = None
        QdrantClient = None
        SupabaseClient = None
        EmbeddingService = None


async def create_competitor_analysis_agent(
    tenant_id: str,
    neo4j_client: Optional[Neo4jClient] = None,
    qdrant_client: Optional[QdrantClient] = None,
    supabase_client: Optional[SupabaseClient] = None,
    embedding_service: Optional[EmbeddingService] = None
):
    """
    Create a configured Competitor Analysis Agent.
    
    Backward compatibility function for legacy imports.
    
    Args:
        tenant_id: Tenant identifier
        neo4j_client: Neo4j client instance
        qdrant_client: Qdrant client instance
        supabase_client: Supabase client instance
        embedding_service: Embedding service instance
        
    Returns:
        Configured CompetitorAnalysisDeps instance for use with the agent
    """
    # Create dependencies object that can be used with the modular agent
    deps = CompetitorAnalysisDeps(
        tenant_id=tenant_id,
        neo4j_client=neo4j_client,
        qdrant_client=qdrant_client,
        supabase_client=supabase_client,
        embedding_service=embedding_service
    )
    
    return deps


__all__ = [
    # Main agent and functions
    'competitor_analysis_agent',
    'analyze_competitor_profile', 
    'identify_competitor_opportunities',
    'monitor_competitor_changes',
    'analyze_competitor_simple',
    
    # Models
    'CompetitorAnalysisError',
    'CompetitorAnalysisDeps',
    'CompetitorInsight', 
    'CompetitorProfile',
    'KeywordGapAnalysis',
    
    # Analyzers
    'ContentAnalyzer',
    'KeywordAnalyzer',
    
    # Workflow functions
    'workflow_analyze_profile',
    'workflow_identify_opportunities', 
    'workflow_monitor_changes',
    
    # Compatibility functions
    'create_competitor_analysis_agent'
]