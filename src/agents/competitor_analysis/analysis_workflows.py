"""
Analysis workflows for competitor analysis.

Contains main workflow functions and helper utilities for comprehensive
competitor analysis operations.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog

from database.neo4j_client import Neo4jClient
from .models import (
    CompetitorAnalysisDeps, 
    CompetitorAnalysisError, 
    CompetitorInsight,
    CompetitorProfile
)
from .content_analyzer import ContentAnalyzer
from .keyword_analyzer import KeywordAnalyzer

logger = structlog.get_logger(__name__)


async def analyze_competitor_profile(
    deps: CompetitorAnalysisDeps,
    competitor_domain: str,
    include_content_analysis: bool = True,
    include_keyword_analysis: bool = True,
    analysis_depth: str = "comprehensive"
) -> Dict[str, Any]:
    """
    Analyze comprehensive competitor profile.
    
    Args:
        deps: Analysis dependencies
        competitor_domain: Competitor domain to analyze
        include_content_analysis: Whether to include content strategy analysis
        include_keyword_analysis: Whether to include keyword analysis
        analysis_depth: Depth of analysis (quick, standard, comprehensive)
        
    Returns:
        Comprehensive competitor profile with strategic insights
    """
    try:
        logger.info(f"Analyzing competitor profile: {competitor_domain}")
        
        # Initialize analyzers
        content_analyzer = ContentAnalyzer(
            deps.searxng_service,
            deps.embedding_service
        )
        
        keyword_analyzer = KeywordAnalyzer(deps.searxng_service)
        
        # Basic profile information
        profile = {
            'domain': competitor_domain,
            'company_name': competitor_domain.replace('.com', '').replace('.', ' ').title(),
            'analysis_depth': analysis_depth,
            'analysis_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Content strategy analysis
        if include_content_analysis:
            try:
                sample_size = 100 if analysis_depth == "comprehensive" else 50 if analysis_depth == "standard" else 25
                content_strategy = await content_analyzer.analyze_competitor_content_strategy(
                    competitor_domain, sample_size
                )
                
                profile['content_strategy'] = content_strategy
                
                # Extract key metrics
                if content_strategy:
                    profile['content_volume'] = content_strategy.get('content_sample_size', 0)
                    profile['primary_content_types'] = content_strategy.get('content_types', {}).get('primary_types', [])
                    profile['content_quality_score'] = content_strategy.get('content_quality', {}).get('average_quality_score', 0)
                    profile['seo_optimization_level'] = content_strategy.get('seo_patterns', {}).get('seo_maturity', 'unknown')
                
            except Exception as e:
                logger.error(f"Content analysis failed for {competitor_domain}: {e}")
                profile['content_analysis_error'] = str(e)
        
        # Keyword analysis
        if include_keyword_analysis:
            try:
                # Get our keywords for comparison
                our_keywords = await get_our_keywords(deps.tenant_id, deps.industry)
                
                if our_keywords:
                    keyword_gaps = await keyword_analyzer.analyze_keyword_gaps(
                        our_keywords, competitor_domain, deps.industry or "general"
                    )
                    
                    profile['keyword_analysis'] = {
                        'total_competitor_keywords': len(keyword_gaps.competitor_keywords),
                        'keyword_overlap_ratio': keyword_gaps.keyword_overlap_ratio,
                        'competitive_intensity': keyword_gaps.competitive_intensity,
                        'high_opportunity_keywords': keyword_gaps.high_opportunity_keywords,
                        'content_gap_opportunities': keyword_gaps.content_gap_opportunities,
                        'strategic_recommendations': keyword_gaps.strategic_recommendations
                    }
                else:
                    profile['keyword_analysis'] = {'error': 'No baseline keywords available for comparison'}
                    
            except Exception as e:
                logger.error(f"Keyword analysis failed for {competitor_domain}: {e}")
                profile['keyword_analysis_error'] = str(e)
        
        # Generate competitive insights
        competitive_insights = await generate_competitive_insights(profile)
        profile['competitive_insights'] = competitive_insights
        
        # Generate strategic recommendations
        strategic_recommendations = await generate_strategic_recommendations(profile)
        profile['strategic_recommendations'] = strategic_recommendations
        
        # Calculate analysis completeness
        completeness_score = calculate_analysis_completeness(profile)
        profile['analysis_completeness'] = completeness_score
        
        return profile
        
    except Exception as e:
        logger.error(f"Failed to analyze competitor profile: {e}")
        raise CompetitorAnalysisError(f"Competitor profile analysis failed: {e}")


async def identify_competitor_opportunities(
    deps: CompetitorAnalysisDeps,
    competitor_domains: List[str],
    focus_area: str = "content_gaps",
    priority_level: str = "high"
) -> Dict[str, Any]:
    """
    Identify opportunities from competitor analysis.
    
    Args:
        deps: Analysis dependencies
        competitor_domains: List of competitor domains to analyze
        focus_area: Focus area for opportunities (content_gaps, keyword_gaps, strategy_gaps)
        priority_level: Priority level for opportunities (high, medium, low)
        
    Returns:
        Prioritized list of competitive opportunities
    """
    try:
        logger.info(f"Identifying opportunities from {len(competitor_domains)} competitors")
        
        opportunities = []
        
        for domain in competitor_domains:
            try:
                # Get basic competitor analysis
                competitor_profile = await analyze_competitor_profile(
                    deps, domain, include_content_analysis=True, include_keyword_analysis=True
                )
                
                # Extract opportunities based on focus area
                if focus_area == "content_gaps":
                    content_opps = await extract_content_opportunities(competitor_profile)
                    opportunities.extend(content_opps)
                elif focus_area == "keyword_gaps":
                    keyword_opps = await extract_keyword_opportunities(competitor_profile)
                    opportunities.extend(keyword_opps)
                elif focus_area == "strategy_gaps":
                    strategy_opps = await extract_strategy_opportunities(competitor_profile)
                    opportunities.extend(strategy_opps)
                else:
                    # Extract all types of opportunities
                    content_opps = await extract_content_opportunities(competitor_profile)
                    keyword_opps = await extract_keyword_opportunities(competitor_profile)
                    strategy_opps = await extract_strategy_opportunities(competitor_profile)
                    opportunities.extend(content_opps + keyword_opps + strategy_opps)
                
                # Rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"Failed to analyze competitor {domain}: {e}")
                continue
        
        # Filter by priority level
        if priority_level == "high":
            opportunities = [opp for opp in opportunities if opp.get('priority_score', 0) > 0.7]
        elif priority_level == "medium":
            opportunities = [opp for opp in opportunities if 0.4 <= opp.get('priority_score', 0) <= 0.7]
        elif priority_level == "low":
            opportunities = [opp for opp in opportunities if opp.get('priority_score', 0) < 0.4]
        
        # Sort by priority score
        opportunities.sort(key=lambda x: x.get('priority_score', 0), reverse=True)
        
        # Generate summary insights
        summary_insights = await generate_opportunity_insights(opportunities, focus_area)
        
        return {
            'total_opportunities': len(opportunities),
            'focus_area': focus_area,
            'priority_level': priority_level,
            'opportunities': opportunities[:20],  # Top 20 opportunities
            'summary_insights': summary_insights,
            'competitor_domains_analyzed': competitor_domains,
            'analysis_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to identify competitor opportunities: {e}")
        raise CompetitorAnalysisError(f"Opportunity identification failed: {e}")


async def monitor_competitor_changes(
    deps: CompetitorAnalysisDeps,
    competitor_domain: str,
    monitoring_period: str = "30d",
    change_threshold: float = 0.1
) -> Dict[str, Any]:
    """
    Monitor competitor changes over time.
    
    Args:
        deps: Analysis dependencies
        competitor_domain: Competitor domain to monitor
        monitoring_period: Period to monitor (7d, 30d, 90d)
        change_threshold: Threshold for significant changes (0.0-1.0)
        
    Returns:
        Competitor change analysis with alerts
    """
    try:
        logger.info(f"Monitoring competitor changes: {competitor_domain}")
        
        # Get current competitor profile
        current_profile = await analyze_competitor_profile(
            deps, competitor_domain, include_content_analysis=True, include_keyword_analysis=True
        )
        
        # Get historical data (placeholder - would load from database)
        historical_profile = await get_historical_competitor_profile(
            competitor_domain, monitoring_period, deps.neo4j_client
        )
        
        if not historical_profile:
            return {
                'error': f'No historical data available for {competitor_domain}',
                'recommendation': 'Continue monitoring to establish baseline'
            }
        
        # Analyze changes
        changes = await analyze_competitor_changes(
            current_profile, historical_profile, change_threshold
        )
        
        # Generate alerts
        alerts = await generate_change_alerts(changes, change_threshold)
        
        # Generate strategic implications
        strategic_implications = await analyze_strategic_implications(changes)
        
        return {
            'competitor_domain': competitor_domain,
            'monitoring_period': monitoring_period,
            'change_threshold': change_threshold,
            'changes_detected': changes,
            'alerts': alerts,
            'strategic_implications': strategic_implications,
            'recommendation': await generate_monitoring_recommendations(changes, alerts),
            'analysis_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to monitor competitor changes: {e}")
        raise CompetitorAnalysisError(f"Competitor monitoring failed: {e}")


# =============================================================================
# Helper Functions
# =============================================================================

async def get_our_keywords(tenant_id: str, industry: Optional[str]) -> List[str]:
    """Get our keywords for comparison."""
    # Placeholder - would query from database
    return [
        "content marketing", "seo optimization", "digital strategy",
        "content creation", "marketing automation", "social media",
        "analytics", "performance tracking", "conversion optimization"
    ]


async def generate_competitive_insights(profile: Dict[str, Any]) -> List[CompetitorInsight]:
    """Generate competitive insights from profile analysis."""
    insights = []
    
    try:
        # Content strategy insights
        content_strategy = profile.get('content_strategy', {})
        if content_strategy:
            quality_score = content_strategy.get('content_quality', {}).get('average_quality_score', 0)
            
            if quality_score > 0.7:
                insights.append(CompetitorInsight(
                    competitor_domain=profile['domain'],
                    insight_type='content_quality',
                    insight_category='strength',
                    title='High Content Quality',
                    description=f'Competitor maintains high content quality (score: {quality_score:.2f})',
                    impact_score=0.8,
                    urgency='medium',
                    recommendations=['Improve content quality standards', 'Invest in content review processes'],
                    confidence=0.7,
                    data_sources=['content_analysis']
                ))
        
        # Keyword analysis insights
        keyword_analysis = profile.get('keyword_analysis', {})
        if keyword_analysis:
            opportunities = keyword_analysis.get('high_opportunity_keywords', [])
            
            if opportunities:
                insights.append(CompetitorInsight(
                    competitor_domain=profile['domain'],
                    insight_type='keyword_opportunity',
                    insight_category='opportunity',
                    title='High-Value Keyword Opportunities',
                    description=f'Identified {len(opportunities)} high-opportunity keywords',
                    impact_score=0.9,
                    urgency='high',
                    recommendations=['Target identified opportunity keywords', 'Create content around these keywords'],
                    confidence=0.8,
                    data_sources=['keyword_analysis']
                ))
        
        return insights
        
    except Exception as e:
        logger.error(f"Failed to generate competitive insights: {e}")
        return []


async def generate_strategic_recommendations(profile: Dict[str, Any]) -> List[str]:
    """Generate strategic recommendations from profile analysis."""
    recommendations = []
    
    try:
        # Content-based recommendations
        content_strategy = profile.get('content_strategy', {})
        if content_strategy:
            frequency = content_strategy.get('publishing_patterns', {}).get('estimated_monthly_frequency', 0)
            if frequency > 15:
                recommendations.append("Consider increasing content publishing frequency to match competitor")
            
            quality_level = content_strategy.get('content_quality', {}).get('content_optimization_level', 'unknown')
            if quality_level == 'high':
                recommendations.append("Invest in content quality improvements to compete effectively")
        
        # Keyword-based recommendations
        keyword_analysis = profile.get('keyword_analysis', {})
        if keyword_analysis:
            overlap_ratio = keyword_analysis.get('keyword_overlap_ratio', 0)
            if overlap_ratio < 0.3:
                recommendations.append("Low keyword overlap - identify and target competitor's successful keywords")
            
            opportunities = keyword_analysis.get('high_opportunity_keywords', [])
            if opportunities:
                recommendations.append(f"Prioritize {len(opportunities)} identified high-opportunity keywords")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Continue monitoring competitor for strategic opportunities")
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Failed to generate strategic recommendations: {e}")
        return []


def calculate_analysis_completeness(profile: Dict[str, Any]) -> float:
    """Calculate completeness score for competitor analysis."""
    try:
        completeness_factors = {
            'basic_info': 1.0 if profile.get('domain') else 0.0,
            'content_analysis': 1.0 if profile.get('content_strategy') else 0.0,
            'keyword_analysis': 1.0 if profile.get('keyword_analysis') else 0.0,
            'insights': 1.0 if profile.get('competitive_insights') else 0.0,
            'recommendations': 1.0 if profile.get('strategic_recommendations') else 0.0
        }
        
        return sum(completeness_factors.values()) / len(completeness_factors)
        
    except Exception as e:
        logger.error(f"Failed to calculate analysis completeness: {e}")
        return 0.0


async def extract_content_opportunities(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract content opportunities from profile analysis."""
    opportunities = []
    
    try:
        content_strategy = profile.get('content_strategy', {})
        if content_strategy:
            # Content type opportunities
            content_types = content_strategy.get('content_types', {})
            if content_types:
                primary_types = content_types.get('primary_types', [])
                for content_type in primary_types:
                    opportunities.append({
                        'type': 'content_format',
                        'opportunity': f"Create more {content_type.replace('_', ' ')} content",
                        'priority_score': 0.7,
                        'competitor_domain': profile.get('domain'),
                        'evidence': f"Competitor focuses on {content_type}"
                    })
        
        return opportunities
        
    except Exception as e:
        logger.error(f"Failed to extract content opportunities: {e}")
        return []


async def extract_keyword_opportunities(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract keyword opportunities from profile analysis."""
    opportunities = []
    
    try:
        keyword_analysis = profile.get('keyword_analysis', {})
        if keyword_analysis:
            high_opp_keywords = keyword_analysis.get('high_opportunity_keywords', [])
            
            for keyword_data in high_opp_keywords:
                if isinstance(keyword_data, dict):
                    opportunities.append({
                        'type': 'keyword_opportunity',
                        'opportunity': f"Target keyword: {keyword_data.get('keyword')}",
                        'priority_score': keyword_data.get('opportunity_score', 0.5),
                        'competitor_domain': profile.get('domain'),
                        'evidence': f"Competitor ranks #{keyword_data.get('competitor_position')} for this keyword"
                    })
        
        return opportunities
        
    except Exception as e:
        logger.error(f"Failed to extract keyword opportunities: {e}")
        return []


async def extract_strategy_opportunities(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract strategic opportunities from profile analysis."""
    opportunities = []
    
    try:
        # SEO strategy opportunities
        content_strategy = profile.get('content_strategy', {})
        if content_strategy:
            seo_maturity = content_strategy.get('seo_patterns', {}).get('seo_maturity', 'unknown')
            if seo_maturity == 'basic':
                opportunities.append({
                    'type': 'seo_strategy',
                    'opportunity': 'Exploit competitor\'s basic SEO implementation',
                    'priority_score': 0.8,
                    'competitor_domain': profile.get('domain'),
                    'evidence': 'Competitor has basic SEO optimization'
                })
        
        return opportunities
        
    except Exception as e:
        logger.error(f"Failed to extract strategy opportunities: {e}")
        return []


async def generate_opportunity_insights(opportunities: List[Dict[str, Any]], focus_area: str) -> List[str]:
    """Generate insights from opportunity analysis."""
    insights = []
    
    try:
        if not opportunities:
            return ["No significant opportunities identified in current analysis"]
        
        # General insights
        insights.append(f"Identified {len(opportunities)} opportunities in {focus_area}")
        
        # Priority insights
        high_priority = [opp for opp in opportunities if opp.get('priority_score', 0) > 0.7]
        if high_priority:
            insights.append(f"{len(high_priority)} high-priority opportunities require immediate attention")
        
        # Type-specific insights
        if focus_area == "content_gaps":
            insights.append("Focus on content creation to address identified gaps")
        elif focus_area == "keyword_gaps":
            insights.append("Prioritize SEO optimization for opportunity keywords")
        elif focus_area == "strategy_gaps":
            insights.append("Consider strategic pivots to exploit competitor weaknesses")
        
        return insights
        
    except Exception as e:
        logger.error(f"Failed to generate opportunity insights: {e}")
        return []


async def get_historical_competitor_profile(domain: str, period: str, neo4j_client: Neo4jClient) -> Optional[Dict[str, Any]]:
    """Get historical competitor profile from database."""
    # Placeholder - would query historical data from Neo4j
    return None


async def analyze_competitor_changes(current: Dict[str, Any], historical: Dict[str, Any], threshold: float) -> List[Dict[str, Any]]:
    """Analyze changes between current and historical profiles."""
    changes = []
    
    try:
        # Compare content volume
        current_volume = current.get('content_volume', 0)
        historical_volume = historical.get('content_volume', 0)
        
        if historical_volume > 0:
            volume_change = (current_volume - historical_volume) / historical_volume
            if abs(volume_change) > threshold:
                changes.append({
                    'type': 'content_volume',
                    'change': volume_change,
                    'description': f"Content volume changed by {volume_change:.1%}",
                    'significance': 'high' if abs(volume_change) > 0.3 else 'medium'
                })
        
        return changes
        
    except Exception as e:
        logger.error(f"Failed to analyze competitor changes: {e}")
        return []


async def generate_change_alerts(changes: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    """Generate alerts from competitor changes."""
    alerts = []
    
    try:
        for change in changes:
            if change.get('significance') == 'high':
                alerts.append({
                    'type': 'competitor_change',
                    'severity': 'high',
                    'message': f"Significant change detected: {change.get('description')}",
                    'recommended_action': 'Monitor closely and adjust strategy if needed'
                })
        
        return alerts
        
    except Exception as e:
        logger.error(f"Failed to generate change alerts: {e}")
        return []


async def analyze_strategic_implications(changes: List[Dict[str, Any]]) -> List[str]:
    """Analyze strategic implications of competitor changes."""
    implications = []
    
    try:
        for change in changes:
            change_type = change.get('type')
            
            if change_type == 'content_volume':
                if change.get('change', 0) > 0.2:
                    implications.append("Competitor is scaling content production - consider increasing our output")
                elif change.get('change', 0) < -0.2:
                    implications.append("Competitor reducing content - opportunity to capture market share")
        
        return implications
        
    except Exception as e:
        logger.error(f"Failed to analyze strategic implications: {e}")
        return []


async def generate_monitoring_recommendations(changes: List[Dict[str, Any]], alerts: List[Dict[str, Any]]) -> str:
    """Generate monitoring recommendations."""
    try:
        if not changes and not alerts:
            return "Continue regular monitoring - no significant changes detected"
        
        if alerts:
            return "Immediate strategic review recommended due to significant competitor changes"
        
        if changes:
            return "Monitor competitor activity more closely - changes detected"
        
        return "Maintain current monitoring schedule"
        
    except Exception as e:
        logger.error(f"Failed to generate monitoring recommendations: {e}")
        return "Error generating recommendations"


# =============================================================================
# Utility Functions
# =============================================================================

async def analyze_competitor_simple(competitor_domain: str, 
                                  our_keywords: List[str] = None,
                                  tenant_id: str = "default") -> Optional[CompetitorProfile]:
    """
    Simple function to analyze a competitor.
    
    Args:
        competitor_domain: Competitor domain to analyze
        our_keywords: Our keywords for comparison
        tenant_id: Tenant identifier
        
    Returns:
        CompetitorProfile if successful, None if failed
    """
    try:
        # This would need proper dependency initialization in production
        logger.info(f"Simple competitor analysis for: {competitor_domain}")
        
        # Placeholder for simple analysis
        profile = CompetitorProfile(
            domain=competitor_domain,
            company_name=competitor_domain.replace('.com', '').replace('.', ' ').title(),
            content_volume=0,
            content_frequency=0.0,
            primary_content_types=[],
            estimated_domain_authority=0.0,
            top_ranking_keywords=[],
            content_gaps=[],
            target_audience="",
            competitive_advantages=[],
            content_quality_score=0.0,
            engagement_indicators={},
            seo_optimization_level="unknown",
            analysis_completeness=0.0,
            data_confidence=0.0
        )
        
        return profile
        
    except Exception as e:
        logger.error(f"Failed to analyze competitor: {e}")
        return None