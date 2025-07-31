"""
SERP Analysis and Content Generation Integration Service.

This service connects SERP (Search Engine Results Page) analysis with content generation
workflows to create SEO-optimized content based on ranking data and competitor insights.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from .unified_seo_data_service import UnifiedSEODataService, UnifiedKeywordData
from .serpbear_client import serpbear_client
from ..agents.content_generation.agent import ContentGenerationAgent, ContentGenerationRequest

logger = logging.getLogger(__name__)


class SERPContentInsight(BaseModel):
    """SERP-based insights for content generation."""
    keyword: str
    current_position: Optional[int] = None
    target_position: int = Field(default=10, description="Target ranking position")
    opportunity_score: float = Field(description="Content opportunity score (0-100)")
    content_gaps: List[str] = Field(default_factory=list)
    competitor_insights: Dict[str, Any] = Field(default_factory=dict)
    recommended_content_type: str = Field(default="blog_post")
    estimated_traffic_potential: int = Field(default=0)
    difficulty_score: float = Field(default=0.0)


class SERPContentStrategy(BaseModel):
    """Content strategy based on SERP analysis."""
    primary_keywords: List[str] = Field(default_factory=list)
    secondary_keywords: List[str] = Field(default_factory=list)
    content_priorities: List[SERPContentInsight] = Field(default_factory=list)
    competitor_gaps: Dict[str, List[str]] = Field(default_factory=dict)
    recommended_content_calendar: List[Dict[str, Any]] = Field(default_factory=list)
    total_opportunity_score: float = 0.0


class SERPContentIntegrationService:
    """
    Service for integrating SERP analysis with content generation workflows.
    
    This service analyzes keyword rankings, identifies content opportunities,
    and generates SEO-optimized content recommendations.
    """
    
    def __init__(self):
        """Initialize the SERP content integration service."""
        self.unified_seo_service = UnifiedSEODataService()
        self.content_agent = None
        logger.info("SERP Content Integration Service initialized")
    
    async def initialize(self):
        """Initialize the service and its dependencies."""
        try:
            # Initialize content generation agent if available
            from ..agents.content_generation.agent import ContentGenerationAgent
            self.content_agent = ContentGenerationAgent()
            logger.info("Content generation agent initialized")
        except ImportError:
            logger.warning("Content generation agent not available")
    
    async def analyze_content_opportunities(
        self, 
        domain: str, 
        keywords: Optional[List[str]] = None,
        days_back: int = 30
    ) -> SERPContentStrategy:
        """
        Analyze SERP data to identify content creation opportunities.
        
        Args:
            domain: Domain to analyze
            keywords: Specific keywords to analyze (if None, uses all tracked keywords)
            days_back: Number of days to look back for trend analysis
            
        Returns:
            Content strategy with prioritized opportunities
        """
        logger.info(f"Analyzing content opportunities for domain: {domain}")
        
        try:
            # Get unified keyword data
            if keywords:
                keyword_data = []
                for keyword in keywords:
                    data = await self.unified_seo_service.get_unified_keyword_data(domain, keyword)
                    if data:
                        keyword_data.append(data)
            else:
                keyword_data = await self.unified_seo_service.get_all_domain_keywords(domain)
            
            if not keyword_data:
                logger.warning(f"No keyword data found for domain: {domain}")
                return SERPContentStrategy()
            
            # Analyze each keyword for content opportunities
            content_insights = []
            for data in keyword_data:
                insight = await self._analyze_keyword_opportunity(data, days_back)
                if insight:
                    content_insights.append(insight)
            
            # Sort by opportunity score
            content_insights.sort(key=lambda x: x.opportunity_score, reverse=True)
            
            # Generate strategy
            strategy = await self._generate_content_strategy(content_insights)
            
            logger.info(f"Generated content strategy with {len(content_insights)} opportunities")
            return strategy
            
        except Exception as e:
            logger.error(f"Failed to analyze content opportunities: {e}")
            return SERPContentStrategy()
    
    async def _analyze_keyword_opportunity(
        self, 
        keyword_data: UnifiedKeywordData, 
        days_back: int = 30
    ) -> Optional[SERPContentInsight]:
        """Analyze individual keyword for content opportunity."""
        try:
            # Calculate opportunity score based on multiple factors
            opportunity_score = 0.0
            content_gaps = []
            
            # Factor 1: Current position (worse position = higher opportunity)
            if keyword_data.position:
                if keyword_data.position > 20:
                    opportunity_score += 30  # High opportunity for unranked content
                elif keyword_data.position > 10:
                    opportunity_score += 20  # Medium opportunity for page 2
                elif keyword_data.position > 5:
                    opportunity_score += 15  # Some opportunity for bottom of page 1
                else:
                    opportunity_score += 5   # Low opportunity for top positions
            else:
                opportunity_score += 35  # Very high opportunity for untracked keywords
            
            # Factor 2: Search volume (higher volume = higher opportunity)
            if keyword_data.search_volume:
                if keyword_data.search_volume > 10000:
                    opportunity_score += 25
                elif keyword_data.search_volume > 1000:
                    opportunity_score += 20
                elif keyword_data.search_volume > 100:
                    opportunity_score += 15
                else:
                    opportunity_score += 10
            
            # Factor 3: Competition level (lower competition = higher opportunity)
            if keyword_data.competition_score:
                if keyword_data.competition_score < 0.3:
                    opportunity_score += 20  # Low competition
                elif keyword_data.competition_score < 0.7:
                    opportunity_score += 10  # Medium competition
                else:
                    opportunity_score += 5   # High competition
            
            # Factor 4: Position trend (declining = opportunity)
            if keyword_data.position and keyword_data.previous_position:
                position_change = keyword_data.previous_position - keyword_data.position
                if position_change < -5:  # Dropped significantly
                    opportunity_score += 15
                elif position_change < 0:  # Dropped slightly
                    opportunity_score += 10
            
            # Identify content gaps based on competitor analysis
            if keyword_data.position and keyword_data.position > 10:
                content_gaps.append("Missing comprehensive content for target keyword")
                content_gaps.append("Potential for better on-page optimization")
            
            if not keyword_data.ranking_url:
                content_gaps.append("No dedicated content page for this keyword")
            
            # Determine recommended content type
            recommended_type = "blog_post"
            if keyword_data.search_volume and keyword_data.search_volume > 5000:
                recommended_type = "comprehensive_guide"
            elif "how to" in keyword_data.keyword.lower():
                recommended_type = "tutorial"
            elif "best" in keyword_data.keyword.lower():
                recommended_type = "comparison_guide"
            
            # Calculate estimated traffic potential
            traffic_potential = 0
            if keyword_data.search_volume and keyword_data.position:
                # Simplified CTR model based on position
                ctr_by_position = {
                    1: 0.284, 2: 0.147, 3: 0.103, 4: 0.073, 5: 0.053,
                    6: 0.040, 7: 0.031, 8: 0.024, 9: 0.019, 10: 0.016
                }
                target_ctr = ctr_by_position.get(min(10, max(1, 10)), 0.005)  # Target top 10
                current_ctr = ctr_by_position.get(keyword_data.position, 0.001)
                traffic_potential = int(keyword_data.search_volume * (target_ctr - current_ctr))
            
            return SERPContentInsight(
                keyword=keyword_data.keyword,
                current_position=keyword_data.position,
                opportunity_score=min(100, opportunity_score),
                content_gaps=content_gaps,
                competitor_insights={
                    "competition_level": keyword_data.competition,
                    "competition_score": keyword_data.competition_score,
                    "avg_cpc": keyword_data.cpc
                },
                recommended_content_type=recommended_type,
                estimated_traffic_potential=max(0, traffic_potential),
                difficulty_score=keyword_data.competition_score or 0.5
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze keyword opportunity for {keyword_data.keyword}: {e}")
            return None
    
    async def _generate_content_strategy(
        self, 
        content_insights: List[SERPContentInsight]
    ) -> SERPContentStrategy:
        """Generate comprehensive content strategy from insights."""
        try:
            # Categorize keywords
            primary_keywords = []
            secondary_keywords = []
            
            for insight in content_insights:
                if insight.opportunity_score > 70:
                    primary_keywords.append(insight.keyword)
                elif insight.opportunity_score > 40:
                    secondary_keywords.append(insight.keyword)
            
            # Generate content calendar recommendations
            content_calendar = []
            for i, insight in enumerate(content_insights[:10]):  # Top 10 opportunities
                calendar_entry = {
                    "priority": i + 1,
                    "keyword": insight.keyword,
                    "content_type": insight.recommended_content_type,
                    "opportunity_score": insight.opportunity_score,
                    "estimated_traffic": insight.estimated_traffic_potential,
                    "timeline": "immediate" if insight.opportunity_score > 80 else "within_month",
                    "content_gaps": insight.content_gaps[:3]  # Top 3 gaps
                }
                content_calendar.append(calendar_entry)
            
            # Calculate total opportunity score
            total_score = sum(insight.opportunity_score for insight in content_insights)
            
            return SERPContentStrategy(
                primary_keywords=primary_keywords,
                secondary_keywords=secondary_keywords,
                content_priorities=content_insights,
                recommended_content_calendar=content_calendar,
                total_opportunity_score=total_score
            )
            
        except Exception as e:
            logger.error(f"Failed to generate content strategy: {e}")
            return SERPContentStrategy()
    
    async def generate_serp_optimized_content(
        self,
        keyword: str,
        domain: str,
        content_type: str = "blog_post",
        include_competitors: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Generate content optimized for SERP performance.
        
        Args:
            keyword: Target keyword
            domain: Domain to optimize for
            content_type: Type of content to generate
            include_competitors: Whether to include competitor analysis
            
        Returns:
            Generated content with SEO optimization
        """
        if not self.content_agent:
            logger.error("Content generation agent not available")
            return None
        
        try:
            logger.info(f"Generating SERP-optimized content for keyword: {keyword}")
            
            # Get SERP data for the keyword
            keyword_data = await self.unified_seo_service.get_unified_keyword_data(domain, keyword)
            if not keyword_data:
                logger.warning(f"No SERP data found for keyword: {keyword}")
                return None
            
            # Analyze content opportunity
            insight = await self._analyze_keyword_opportunity(keyword_data)
            if not insight:
                logger.warning(f"Could not analyze opportunity for keyword: {keyword}")
                return None
            
            # Prepare content generation request with SERP insights
            request = ContentGenerationRequest(
                content_type=content_type,
                topic=keyword,
                target_keywords=[keyword] + (keyword_data.related_keywords or [])[:4],
                content_length="medium" if insight.opportunity_score > 60 else "short",
                writing_style="informational",
                target_audience="general",
                seo_requirements={
                    "target_position": insight.target_position,
                    "current_position": insight.current_position,
                    "opportunity_score": insight.opportunity_score,
                    "search_volume": keyword_data.search_volume,
                    "competition_score": keyword_data.competition_score,
                    "content_gaps": insight.content_gaps
                },
                competitor_analysis_data={
                    "competition_level": keyword_data.competition,
                    "avg_cpc": keyword_data.cpc,
                    "top_ranking_urls": []  # Could be enhanced with actual competitor URLs
                } if include_competitors else None
            )
            
            # Generate content using the agent
            result = await self.content_agent.generate_content(request)
            
            if result and result.success:
                # Enhance result with SERP-specific metadata
                enhanced_result = result.data.copy()
                enhanced_result["serp_optimization"] = {
                    "target_keyword": keyword,
                    "current_position": insight.current_position,
                    "target_position": insight.target_position,
                    "opportunity_score": insight.opportunity_score,
                    "estimated_traffic_potential": insight.estimated_traffic_potential,
                    "content_gaps_addressed": insight.content_gaps,
                    "optimization_recommendations": [
                        f"Target keyword density: 1-2% for '{keyword}'",
                        "Include target keyword in H1, H2, and meta description",
                        "Optimize for featured snippet potential",
                        "Include related keywords naturally throughout content"
                    ]
                }
                
                logger.info(f"Generated SERP-optimized content for {keyword} with opportunity score {insight.opportunity_score}")
                return enhanced_result
            else:
                logger.error(f"Content generation failed for keyword: {keyword}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to generate SERP-optimized content: {e}")
            return None
    
    async def get_content_recommendations(
        self, 
        domain: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get prioritized content recommendations based on SERP analysis.
        
        Args:
            domain: Domain to analyze
            limit: Maximum number of recommendations
            
        Returns:
            List of content recommendations with priorities
        """
        try:
            # Analyze opportunities
            strategy = await self.analyze_content_opportunities(domain)
            
            recommendations = []
            for insight in strategy.content_priorities[:limit]:
                recommendation = {
                    "keyword": insight.keyword,
                    "priority": "high" if insight.opportunity_score > 70 else "medium" if insight.opportunity_score > 40 else "low",
                    "opportunity_score": insight.opportunity_score,
                    "current_position": insight.current_position,
                    "estimated_traffic": insight.estimated_traffic_potential,
                    "recommended_content_type": insight.recommended_content_type,
                    "content_gaps": insight.content_gaps,
                    "timeline": "immediate" if insight.opportunity_score > 80 else "this_month" if insight.opportunity_score > 60 else "next_month",
                    "difficulty": "easy" if insight.difficulty_score < 0.3 else "medium" if insight.difficulty_score < 0.7 else "hard"
                }
                recommendations.append(recommendation)
            
            logger.info(f"Generated {len(recommendations)} content recommendations for {domain}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to get content recommendations: {e}")
            return []


# Global service instance
serp_content_integration_service = SERPContentIntegrationService()