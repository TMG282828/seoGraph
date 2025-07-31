"""
Main Competitor Analysis Agent module.

Provides the primary Pydantic AI agent interface for competitor analysis
functionality with comprehensive content and keyword analysis capabilities.
"""

import structlog
from pydantic_ai import Agent, RunContext

from config.settings import get_settings
from .models import CompetitorAnalysisDeps, CompetitorProfile
from .analysis_workflows import (
    analyze_competitor_profile,
    identify_competitor_opportunities,
    monitor_competitor_changes,
    analyze_competitor_simple
)

logger = structlog.get_logger(__name__)


# Create the competitor analysis agent
competitor_analysis_agent = Agent(
    'openai:gpt-4o',
    deps_type=CompetitorAnalysisDeps,
    system_prompt="""
You are a specialized Competitor Analysis Agent for the SEO Content Knowledge Graph System.

Your primary responsibilities:
1. Analyze competitor content strategies and identify strengths/weaknesses
2. Perform keyword gap analysis and identify content opportunities
3. Generate actionable competitive intelligence insights
4. Provide strategic recommendations for competitive advantage
5. Monitor competitor activity and detect strategic shifts

Key capabilities:
- Comprehensive competitor content analysis
- Keyword gap identification and opportunity assessment
- Content strategy pattern recognition
- Competitive intelligence gathering
- Strategic recommendation generation

When analyzing competitors, consider:
- Content volume, frequency, and quality patterns
- Topic coverage and content type distribution
- SEO optimization and technical implementation
- Keyword targeting and ranking strategies
- Content engagement and user experience
- Strategic positioning and messaging

Always provide specific, actionable insights with confidence scores and supporting evidence.
Focus on strategic implications and business opportunities rather than just data.

Response format should be structured and include:
- Executive summary of key findings
- Detailed analysis results
- Strategic recommendations with priority levels
- Confidence scores for insights
- Supporting evidence and data sources
"""
)


@competitor_analysis_agent.tool
async def analyze_competitor_profile_tool(
    ctx: RunContext[CompetitorAnalysisDeps],
    competitor_domain: str,
    include_content_analysis: bool = True,
    include_keyword_analysis: bool = True,
    analysis_depth: str = "comprehensive"
) -> str:
    """
    Analyze comprehensive competitor profile.
    
    Args:
        competitor_domain: Competitor domain to analyze
        include_content_analysis: Whether to include content strategy analysis
        include_keyword_analysis: Whether to include keyword analysis
        analysis_depth: Depth of analysis (quick, standard, comprehensive)
        
    Returns:
        Formatted analysis results
    """
    try:
        profile = await analyze_competitor_profile(
            ctx.deps,
            competitor_domain,
            include_content_analysis,
            include_keyword_analysis,
            analysis_depth
        )
        
        # Format results for agent response
        result = f"""
## Competitor Analysis: {competitor_domain}

### Executive Summary
- **Analysis Depth**: {profile.get('analysis_depth')}
- **Analysis Completeness**: {profile.get('analysis_completeness', 0):.1%}
- **Content Volume**: {profile.get('content_volume', 0)} pieces analyzed
- **SEO Maturity**: {profile.get('seo_optimization_level', 'unknown')}

### Content Strategy Analysis
"""
        
        content_strategy = profile.get('content_strategy')
        if content_strategy:
            result += f"""
- **Primary Content Types**: {', '.join(profile.get('primary_content_types', []))}
- **Content Quality Score**: {profile.get('content_quality_score', 0):.2f}
- **Publishing Frequency**: {content_strategy.get('publishing_patterns', {}).get('estimated_monthly_frequency', 0):.1f} posts/month
- **Content Optimization**: {content_strategy.get('content_quality', {}).get('content_optimization_level', 'unknown')}
"""
        
        keyword_analysis = profile.get('keyword_analysis')
        if keyword_analysis:
            result += f"""
### Keyword Analysis
- **Total Keywords**: {keyword_analysis.get('total_competitor_keywords', 0)}
- **Keyword Overlap**: {keyword_analysis.get('keyword_overlap_ratio', 0):.1%}
- **Competitive Intensity**: {keyword_analysis.get('competitive_intensity', 0):.2f}
- **High Opportunity Keywords**: {len(keyword_analysis.get('high_opportunity_keywords', []))}
"""
        
        insights = profile.get('competitive_insights', [])
        if insights:
            result += "\n### Key Insights\n"
            for insight in insights[:5]:  # Top 5 insights
                if hasattr(insight, 'title'):
                    result += f"- **{insight.title}**: {insight.description}\n"
                elif isinstance(insight, dict):
                    result += f"- {insight.get('title', 'Insight')}: {insight.get('description', '')}\n"
        
        recommendations = profile.get('strategic_recommendations', [])
        if recommendations:
            result += "\n### Strategic Recommendations\n"
            for i, rec in enumerate(recommendations[:5], 1):
                result += f"{i}. {rec}\n"
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to analyze competitor profile: {e}")
        return f"Analysis failed: {str(e)}"


@competitor_analysis_agent.tool
async def identify_opportunities_tool(
    ctx: RunContext[CompetitorAnalysisDeps],
    competitor_domains: str,  # Comma-separated domains
    focus_area: str = "content_gaps",
    priority_level: str = "high"
) -> str:
    """
    Identify opportunities from competitor analysis.
    
    Args:
        competitor_domains: Comma-separated list of competitor domains
        focus_area: Focus area (content_gaps, keyword_gaps, strategy_gaps, all)
        priority_level: Priority level (high, medium, low)
        
    Returns:
        Formatted opportunity analysis
    """
    try:
        domains = [d.strip() for d in competitor_domains.split(',')]
        
        opportunities = await identify_competitor_opportunities(
            ctx.deps,
            domains,
            focus_area,
            priority_level
        )
        
        result = f"""
## Competitive Opportunities Analysis

### Summary
- **Competitors Analyzed**: {len(domains)}
- **Focus Area**: {opportunities.get('focus_area')}
- **Priority Level**: {opportunities.get('priority_level')}
- **Total Opportunities**: {opportunities.get('total_opportunities', 0)}

### Top Opportunities
"""
        
        for i, opp in enumerate(opportunities.get('opportunities', [])[:10], 1):
            priority_score = opp.get('priority_score', 0)
            result += f"""
{i}. **{opp.get('opportunity', 'Opportunity')}**
   - Priority Score: {priority_score:.2f}
   - Type: {opp.get('type', 'unknown')}
   - Domain: {opp.get('competitor_domain', 'unknown')}
   - Evidence: {opp.get('evidence', 'No evidence provided')}
"""
        
        insights = opportunities.get('summary_insights', [])
        if insights:
            result += "\n### Strategic Insights\n"
            for insight in insights:
                result += f"- {insight}\n"
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to identify opportunities: {e}")
        return f"Opportunity analysis failed: {str(e)}"


@competitor_analysis_agent.tool
async def monitor_changes_tool(
    ctx: RunContext[CompetitorAnalysisDeps],
    competitor_domain: str,
    monitoring_period: str = "30d",
    change_threshold: float = 0.1
) -> str:
    """
    Monitor competitor changes over time.
    
    Args:
        competitor_domain: Competitor domain to monitor
        monitoring_period: Period to monitor (7d, 30d, 90d)
        change_threshold: Threshold for significant changes (0.0-1.0)
        
    Returns:
        Formatted change analysis
    """
    try:
        changes = await monitor_competitor_changes(
            ctx.deps,
            competitor_domain,
            monitoring_period,
            change_threshold
        )
        
        if 'error' in changes:
            return f"Monitoring not available: {changes['error']}\nRecommendation: {changes.get('recommendation', '')}"
        
        result = f"""
## Competitor Change Monitoring: {competitor_domain}

### Monitoring Summary
- **Period**: {changes.get('monitoring_period')}
- **Change Threshold**: {changes.get('change_threshold')}
- **Changes Detected**: {len(changes.get('changes_detected', []))}
- **Alerts Generated**: {len(changes.get('alerts', []))}

### Detected Changes
"""
        
        for change in changes.get('changes_detected', []):
            result += f"""
- **{change.get('type', 'Change')}**: {change.get('description', 'No description')}
  - Significance: {change.get('significance', 'unknown')}
  - Change Amount: {change.get('change', 0):.1%}
"""
        
        alerts = changes.get('alerts', [])
        if alerts:
            result += "\n### Alerts\n"
            for alert in alerts:
                result += f"- **{alert.get('severity', 'INFO')}**: {alert.get('message', 'No message')}\n"
        
        implications = changes.get('strategic_implications', [])
        if implications:
            result += "\n### Strategic Implications\n"
            for implication in implications:
                result += f"- {implication}\n"
        
        result += f"\n### Recommendation\n{changes.get('recommendation', 'No specific recommendation')}"
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to monitor competitor changes: {e}")
        return f"Change monitoring failed: {str(e)}"


# Export main functions for backward compatibility
__all__ = [
    'competitor_analysis_agent',
    'analyze_competitor_profile',
    'identify_competitor_opportunities', 
    'monitor_competitor_changes',
    'analyze_competitor_simple'
]