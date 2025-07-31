"""
Competitor Analysis Agent for the SEO Content Knowledge Graph System.

This module provides a Pydantic AI agent specialized in competitor intelligence with
content analysis, keyword gap identification, and strategic insights.

REFACTORED: This file now imports from the modularized competitor_analysis package
for better maintainability and organization.
"""

# Import all functionality from the new modular structure
from src.agents.competitor_analysis import (
    # Main agent and functions
    competitor_analysis_agent,
    analyze_competitor_profile,
    identify_competitor_opportunities,
    monitor_competitor_changes,
    analyze_competitor_simple,
    
    # Models
    CompetitorAnalysisError,
    CompetitorAnalysisDeps,
    CompetitorInsight,
    CompetitorProfile,
    KeywordGapAnalysis,
    
    # Analyzers
    ContentAnalyzer,
    KeywordAnalyzer,
)

# Backward compatibility functions for CLI
def create_competitor_analysis_agent():
    """Create competitor analysis agent for CLI usage."""
    return competitor_analysis_agent


# Main execution for backward compatibility
if __name__ == "__main__":
    # Example usage and testing
    async def main():
        # Test competitor analysis
        competitor_domain = "example.com"
        
        profile = await analyze_competitor_simple(
            competitor_domain=competitor_domain,
            our_keywords=["content marketing", "seo", "digital strategy"],
            tenant_id="test-tenant"
        )
        
        if profile:
            print(f"Competitor analysis completed for: {profile.domain}")
            print(f"Content volume: {profile.content_volume}")
            print(f"Primary content types: {profile.primary_content_types}")
        else:
            print("Competitor analysis failed")

    import asyncio
    asyncio.run(main())