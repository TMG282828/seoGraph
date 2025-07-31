"""
Keyword analysis functionality for competitor analysis.

Analyzes competitor keywords, identifies gaps, and provides strategic
keyword opportunities and recommendations.
"""

import asyncio
import re
from typing import Any, Dict, List

import structlog
from cachetools import TTLCache

from services.searxng_service import SearXNGService
from .models import KeywordGapAnalysis, CompetitorAnalysisError

logger = structlog.get_logger(__name__)


class KeywordAnalyzer:
    """
    Analyzes competitor keywords and identifies gaps.
    
    Provides comprehensive keyword analysis including:
    - Competitor keyword extraction and analysis
    - Gap identification between our keywords and competitor keywords
    - High-opportunity keyword identification
    - Content gap analysis and recommendations
    - Strategic keyword recommendations
    """
    
    def __init__(self, searxng_service: SearXNGService):
        """
        Initialize KeywordAnalyzer.
        
        Args:
            searxng_service: Service for web search operations
        """
        self.searxng_service = searxng_service
        self.cache = TTLCache(maxsize=500, ttl=3600)  # 1 hour cache
    
    async def analyze_keyword_gaps(self, 
                                 our_keywords: List[str],
                                 competitor_domain: str,
                                 industry: str) -> KeywordGapAnalysis:
        """
        Analyze keyword gaps between us and competitor.
        
        Args:
            our_keywords: List of our current target keywords
            competitor_domain: Competitor domain to analyze
            industry: Industry context for analysis
            
        Returns:
            Comprehensive keyword gap analysis results
        """
        try:
            cache_key = f"keyword_gaps_{competitor_domain}_{hash(tuple(our_keywords))}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Get competitor keywords
            competitor_keywords = await self._extract_competitor_keywords(competitor_domain, industry)
            
            # Perform gap analysis
            competitor_only = list(set(competitor_keywords) - set(our_keywords))
            our_only = list(set(our_keywords) - set(competitor_keywords))
            shared = list(set(our_keywords) & set(competitor_keywords))
            
            # Analyze opportunities
            high_opportunity = await self._identify_high_opportunity_keywords(
                competitor_only, competitor_domain
            )
            
            content_gaps = await self._identify_content_gap_opportunities(
                competitor_only, our_keywords, industry
            )
            
            # Calculate metrics
            total_keywords = len(set(our_keywords + competitor_keywords))
            overlap_ratio = len(shared) / total_keywords if total_keywords > 0 else 0
            
            competitive_intensity = min(len(shared) / len(our_keywords), 1.0) if our_keywords else 0
            
            # Generate recommendations
            strategic_recommendations = await self._generate_keyword_recommendations(
                competitor_only, our_only, shared, high_opportunity
            )
            
            priority_actions = await self._generate_priority_actions(
                high_opportunity, content_gaps
            )
            
            gap_analysis = KeywordGapAnalysis(
                our_keywords=our_keywords,
                competitor_keywords=competitor_keywords,
                competitor_only_keywords=competitor_only,
                our_only_keywords=our_only,
                shared_keywords=shared,
                high_opportunity_keywords=high_opportunity,
                content_gap_opportunities=content_gaps,
                keyword_overlap_ratio=overlap_ratio,
                competitive_intensity=competitive_intensity,
                strategic_recommendations=strategic_recommendations,
                priority_actions=priority_actions
            )
            
            # Cache results
            self.cache[cache_key] = gap_analysis
            
            return gap_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze keyword gaps: {e}")
            raise CompetitorAnalysisError(f"Keyword gap analysis failed: {e}")
    
    async def _extract_competitor_keywords(self, domain: str, industry: str) -> List[str]:
        """Extract keywords from competitor content."""
        try:
            # Search for competitor content
            queries = [
                f"site:{domain} {industry}",
                f"site:{domain} blog",
                f"site:{domain} articles"
            ]
            
            all_keywords = set()
            
            for query in queries:
                try:
                    results = await self.searxng_service.search(
                        query=query,
                        engines=['google'],
                        safesearch=1
                    )
                    
                    # Extract keywords from titles and snippets
                    for result in results:
                        title = result.get('title', '')
                        snippet = result.get('snippet', '')
                        
                        # Simple keyword extraction
                        text = f"{title} {snippet}".lower()
                        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
                        
                        # Filter and add relevant keywords
                        for word in words:
                            if len(word) > 3 and word not in {
                                'the', 'and', 'are', 'was', 'will', 'been', 'have', 'has', 'had',
                                'this', 'that', 'these', 'those', 'with', 'for', 'from', 'not'
                            }:
                                all_keywords.add(word)
                    
                    # Rate limiting
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.warning(f"Failed to extract keywords from query '{query}': {e}")
                    continue
            
            return list(all_keywords)[:100]  # Limit to top 100 keywords
            
        except Exception as e:
            logger.error(f"Failed to extract competitor keywords: {e}")
            return []
    
    async def _identify_high_opportunity_keywords(self, 
                                                competitor_keywords: List[str],
                                                competitor_domain: str) -> List[Dict[str, Any]]:
        """Identify high-opportunity keywords from competitor analysis."""
        try:
            opportunities = []
            
            for keyword in competitor_keywords[:30]:  # Analyze top 30 keywords
                try:
                    # Search for the keyword to assess competition
                    results = await self.searxng_service.search(
                        query=keyword,
                        engines=['google'],
                        safesearch=1
                    )
                    
                    # Calculate opportunity score
                    competitor_position = None
                    total_results = len(results)
                    
                    for i, result in enumerate(results):
                        if competitor_domain in result.get('url', ''):
                            competitor_position = i + 1
                            break
                    
                    if competitor_position and competitor_position <= 10:
                        # Calculate opportunity metrics
                        opportunity_score = (11 - competitor_position) / 10.0
                        difficulty_score = min(total_results / 100, 1.0)
                        
                        # Estimate search volume (simplified)
                        estimated_volume = max(1000 - (total_results * 10), 100)
                        
                        opportunities.append({
                            'keyword': keyword,
                            'competitor_position': competitor_position,
                            'opportunity_score': opportunity_score,
                            'difficulty_score': difficulty_score,
                            'estimated_volume': estimated_volume,
                            'competition_level': 'high' if total_results > 50 else 'medium' if total_results > 20 else 'low'
                        })
                    
                    # Rate limiting
                    await asyncio.sleep(0.3)
                    
                except Exception as e:
                    logger.warning(f"Failed to analyze keyword '{keyword}': {e}")
                    continue
            
            # Sort by opportunity score
            opportunities.sort(key=lambda x: x['opportunity_score'], reverse=True)
            
            return opportunities[:15]  # Return top 15 opportunities
            
        except Exception as e:
            logger.error(f"Failed to identify high opportunity keywords: {e}")
            return []
    
    async def _identify_content_gap_opportunities(self, 
                                                competitor_keywords: List[str],
                                                our_keywords: List[str],
                                                industry: str) -> List[Dict[str, Any]]:
        """Identify content gap opportunities."""
        try:
            content_gaps = []
            
            # Group keywords by semantic similarity
            keyword_groups = await self._group_keywords_by_topic(competitor_keywords)
            
            for group_topic, group_keywords in keyword_groups.items():
                # Check if we have content in this topic area
                our_coverage = len([kw for kw in our_keywords if kw in group_keywords])
                competitor_coverage = len(group_keywords)
                
                if competitor_coverage > our_coverage * 2:  # Significant gap
                    content_gaps.append({
                        'topic': group_topic,
                        'competitor_keywords': group_keywords,
                        'our_coverage': our_coverage,
                        'competitor_coverage': competitor_coverage,
                        'gap_size': competitor_coverage - our_coverage,
                        'opportunity_type': 'content_gap',
                        'recommended_action': f"Create {group_topic} content to address gap"
                    })
            
            # Sort by gap size
            content_gaps.sort(key=lambda x: x['gap_size'], reverse=True)
            
            return content_gaps[:10]  # Return top 10 gaps
            
        except Exception as e:
            logger.error(f"Failed to identify content gaps: {e}")
            return []
    
    async def _group_keywords_by_topic(self, keywords: List[str]) -> Dict[str, List[str]]:
        """Group keywords by semantic topic."""
        try:
            # Simple topic grouping based on word stems
            topic_groups = {}
            
            for keyword in keywords:
                # Simple topic assignment based on keyword
                if any(word in keyword for word in ['marketing', 'advertis', 'promo']):
                    topic = 'marketing'
                elif any(word in keyword for word in ['tech', 'software', 'digital']):
                    topic = 'technology'
                elif any(word in keyword for word in ['business', 'strategy', 'manage']):
                    topic = 'business'
                elif any(word in keyword for word in ['content', 'blog', 'article']):
                    topic = 'content'
                else:
                    topic = 'general'
                
                if topic not in topic_groups:
                    topic_groups[topic] = []
                topic_groups[topic].append(keyword)
            
            return topic_groups
            
        except Exception as e:
            logger.error(f"Failed to group keywords by topic: {e}")
            return {}
    
    async def _generate_keyword_recommendations(self, 
                                             competitor_only: List[str],
                                             our_only: List[str],
                                             shared: List[str],
                                             high_opportunity: List[Dict[str, Any]]) -> List[str]:
        """Generate strategic keyword recommendations."""
        try:
            recommendations = []
            
            # Recommendations based on competitor-only keywords
            if competitor_only:
                recommendations.append(f"Target {len(competitor_only)} keywords competitor ranks for but we don't")
                
                if len(competitor_only) > 20:
                    recommendations.append("Prioritize high-volume competitor keywords for content creation")
                else:
                    recommendations.append("Consider all competitor keywords for content opportunities")
            
            # Recommendations based on our unique keywords
            if our_only:
                recommendations.append(f"Defend {len(our_only)} keywords where we have advantage")
                recommendations.append("Create more content around our unique keyword strengths")
            
            # Recommendations based on shared keywords
            if shared:
                recommendations.append(f"Compete directly on {len(shared)} shared keywords")
                recommendations.append("Improve content quality for shared keywords to outrank competitor")
            
            # Recommendations based on high opportunities
            if high_opportunity:
                top_opportunities = [opp['keyword'] for opp in high_opportunity[:5]]
                recommendations.append(f"Immediate focus on high-opportunity keywords: {', '.join(top_opportunities)}")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate keyword recommendations: {e}")
            return []
    
    async def _generate_priority_actions(self, 
                                       high_opportunity: List[Dict[str, Any]],
                                       content_gaps: List[Dict[str, Any]]) -> List[str]:
        """Generate priority actions based on analysis."""
        try:
            actions = []
            
            # Actions based on high opportunity keywords
            if high_opportunity:
                top_keyword = high_opportunity[0]['keyword']
                actions.append(f"Create comprehensive content targeting '{top_keyword}'")
                
                if len(high_opportunity) > 5:
                    actions.append("Develop content calendar addressing top 5 opportunity keywords")
            
            # Actions based on content gaps
            if content_gaps:
                top_gap = content_gaps[0]['topic']
                actions.append(f"Address content gap in '{top_gap}' topic area")
                
                if len(content_gaps) > 3:
                    actions.append("Audit content strategy to address multiple topic gaps")
            
            # General strategic actions
            if high_opportunity or content_gaps:
                actions.append("Monitor competitor content updates for new opportunities")
                actions.append("Set up alerts for competitor ranking changes")
            
            return actions
            
        except Exception as e:
            logger.error(f"Failed to generate priority actions: {e}")
            return []