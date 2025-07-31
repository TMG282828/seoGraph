"""
Enhanced Gap Analysis Engine for the SEO Content Knowledge Graph System.

This module provides advanced content gap analysis with competitor intelligence,
trend correlation, and opportunity scoring capabilities.
"""

import asyncio
import hashlib
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from cachetools import TTLCache
import structlog

from database.neo4j_client import Neo4jClient
from database.qdrant_client import QdrantClient
from services.embedding_service import EmbeddingService
from services.searxng_service import SearXNGService
from models.seo_models import (
    ContentGapOpportunity,
    GapAnalysisResult,
    CompetitorContentAnalysis,
    KeywordData,
    TrendDirection,
    CompetitionLevel,
    SearchIntent
)
from models.content_models import ContentItem
from config.settings import get_settings

logger = structlog.get_logger(__name__)


class GapAnalysisError(Exception):
    """Raised when gap analysis operations fail."""
    pass


class TrendAnalyzer:
    """Analyzes trending topics and keyword patterns."""
    
    def __init__(self, searxng_service: SearXNGService):
        self.searxng_service = searxng_service
        self.cache = TTLCache(maxsize=500, ttl=3600)  # 1 hour cache
    
    async def get_trending_topics(self, industry: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get trending topics for specified industry."""
        cache_key = f"trending_{industry}_{limit}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Search for trending topics in the industry
            queries = [
                f"{industry} trends 2024",
                f"{industry} latest news",
                f"{industry} emerging topics",
                f"what's new in {industry}",
                f"{industry} market trends"
            ]
            
            trending_data = []
            for query in queries:
                try:
                    results = await self.searxng_service.search(
                        query=query,
                        engines=['google', 'bing'],
                        safesearch=1,
                        time_range='month'
                    )
                    
                    # Extract trending topics from search results
                    topics = await self._extract_topics_from_results(results, industry)
                    trending_data.extend(topics)
                    
                    # Add delay to respect rate limits
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.warning(f"Failed to get trends for query '{query}': {e}")
                    continue
            
            # Deduplicate and rank topics
            unique_topics = await self._deduplicate_and_rank_topics(trending_data)
            
            # Cache results
            self.cache[cache_key] = unique_topics[:limit]
            return unique_topics[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get trending topics for industry '{industry}': {e}")
            raise GapAnalysisError(f"Trending topics analysis failed: {e}")
    
    async def _extract_topics_from_results(self, results: List[Dict], industry: str) -> List[Dict[str, Any]]:
        """Extract topics from search results."""
        topics = []
        
        for result in results:
            title = result.get('title', '')
            snippet = result.get('snippet', '')
            
            # Simple keyword extraction (in production, use more sophisticated NLP)
            combined_text = f"{title} {snippet}".lower()
            
            # Extract potential topics (simple approach)
            words = combined_text.split()
            relevant_phrases = []
            
            # Look for 2-3 word phrases
            for i in range(len(words) - 1):
                phrase = ' '.join(words[i:i+2])
                if len(phrase) > 10 and industry.lower() in phrase:
                    relevant_phrases.append(phrase)
            
            for phrase in relevant_phrases:
                topics.append({
                    'topic': phrase,
                    'source': result.get('url', ''),
                    'relevance': 1.0,  # Basic relevance score
                    'trend_strength': 0.7  # Default trend strength
                })
        
        return topics
    
    async def _deduplicate_and_rank_topics(self, topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate and rank topics by relevance."""
        # Group similar topics
        topic_groups = {}
        for topic_data in topics:
            topic = topic_data['topic']
            key = hashlib.md5(topic.encode()).hexdigest()[:8]
            
            if key not in topic_groups:
                topic_groups[key] = []
            topic_groups[key].append(topic_data)
        
        # Aggregate and rank
        ranked_topics = []
        for group in topic_groups.values():
            if len(group) > 0:
                # Use the most common topic text
                topic_text = max(group, key=lambda x: x['relevance'])['topic']
                
                # Aggregate metrics
                total_relevance = sum(t['relevance'] for t in group)
                avg_trend_strength = sum(t['trend_strength'] for t in group) / len(group)
                
                ranked_topics.append({
                    'topic': topic_text,
                    'relevance': total_relevance,
                    'trend_strength': avg_trend_strength,
                    'mentions': len(group)
                })
        
        # Sort by combined score
        ranked_topics.sort(
            key=lambda x: x['relevance'] * x['trend_strength'] * x['mentions'],
            reverse=True
        )
        
        return ranked_topics


class CompetitorAnalyzer:
    """Analyzes competitor content and identifies gaps."""
    
    def __init__(self, neo4j_client: Neo4jClient, searxng_service: SearXNGService):
        self.neo4j_client = neo4j_client
        self.searxng_service = searxng_service
        self.cache = TTLCache(maxsize=200, ttl=7200)  # 2 hour cache
    
    async def analyze_competitor_content(self, 
                                       competitor_domains: List[str],
                                       industry: str,
                                       tenant_id: str) -> List[CompetitorContentAnalysis]:
        """Analyze competitor content coverage."""
        results = []
        
        for domain in competitor_domains:
            try:
                analysis = await self._analyze_single_competitor(domain, industry, tenant_id)
                results.append(analysis)
                
                # Add delay to respect rate limits
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.warning(f"Failed to analyze competitor {domain}: {e}")
                continue
        
        return results
    
    async def _analyze_single_competitor(self, 
                                       domain: str,
                                       industry: str,
                                       tenant_id: str) -> CompetitorContentAnalysis:
        """Analyze single competitor's content."""
        cache_key = f"competitor_{domain}_{industry}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Search for competitor content
            search_queries = [
                f"site:{domain} {industry}",
                f"site:{domain} blog",
                f"site:{domain} guides",
                f"site:{domain} resources"
            ]
            
            all_content = []
            for query in search_queries:
                try:
                    results = await self.searxng_service.search(
                        query=query,
                        engines=['google'],
                        safesearch=1,
                        time_range='year'
                    )
                    all_content.extend(results)
                    await asyncio.sleep(0.5)
                except Exception as e:
                    logger.warning(f"Search failed for query '{query}': {e}")
                    continue
            
            # Analyze content
            analysis = CompetitorContentAnalysis(
                competitor_domain=domain,
                competitor_name=domain.replace('.com', '').replace('.', ' ').title(),
                total_content_pieces=len(all_content),
                tenant_id=tenant_id
            )
            
            # Extract topics and keywords
            topics = await self._extract_competitor_topics(all_content)
            analysis.topics_covered = topics
            analysis.topic_distribution = self._calculate_topic_distribution(topics)
            
            # Calculate content metrics
            analysis.content_frequency = await self._calculate_content_frequency(all_content)
            analysis.average_word_count = await self._estimate_average_word_count(all_content)
            
            # Cache results
            self.cache[cache_key] = analysis
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze competitor {domain}: {e}")
            raise GapAnalysisError(f"Competitor analysis failed: {e}")
    
    async def _extract_competitor_topics(self, content: List[Dict]) -> List[str]:
        """Extract topics from competitor content."""
        topics = set()
        
        for item in content:
            title = item.get('title', '').lower()
            snippet = item.get('snippet', '').lower()
            
            # Simple topic extraction
            combined_text = f"{title} {snippet}"
            words = combined_text.split()
            
            # Extract meaningful phrases
            for i in range(len(words) - 1):
                phrase = ' '.join(words[i:i+2])
                if len(phrase) > 8 and any(char.isalpha() for char in phrase):
                    topics.add(phrase)
        
        return list(topics)[:50]  # Limit to top 50 topics
    
    def _calculate_topic_distribution(self, topics: List[str]) -> Dict[str, int]:
        """Calculate topic distribution."""
        distribution = {}
        for topic in topics:
            # Simple categorization
            if any(word in topic for word in ['strategy', 'planning', 'management']):
                category = 'strategy'
            elif any(word in topic for word in ['guide', 'how', 'tutorial']):
                category = 'educational'
            elif any(word in topic for word in ['news', 'update', 'latest']):
                category = 'news'
            else:
                category = 'other'
            
            distribution[category] = distribution.get(category, 0) + 1
        
        return distribution
    
    async def _calculate_content_frequency(self, content: List[Dict]) -> float:
        """Calculate content publishing frequency."""
        if not content:
            return 0.0
        
        # Rough estimate based on search results
        # In production, would parse dates from content
        return len(content) / 12.0  # Assume content spread over 12 months
    
    async def _estimate_average_word_count(self, content: List[Dict]) -> int:
        """Estimate average word count."""
        if not content:
            return 0
        
        total_words = 0
        for item in content:
            snippet = item.get('snippet', '')
            # Rough estimate: snippet is ~10% of full content
            estimated_words = len(snippet.split()) * 10
            total_words += estimated_words
        
        return total_words // len(content) if content else 0


class GapAnalysisEngine:
    """
    Advanced gap analysis engine with competitor intelligence and trend correlation.
    
    Provides comprehensive content gap analysis by combining:
    - Existing content analysis
    - Competitor content coverage
    - Trending topic identification
    - Opportunity scoring and prioritization
    """
    
    def __init__(self, 
                 neo4j_client: Neo4jClient,
                 qdrant_client: QdrantClient,
                 embedding_service: EmbeddingService,
                 searxng_service: SearXNGService):
        self.neo4j_client = neo4j_client
        self.qdrant_client = qdrant_client
        self.embedding_service = embedding_service
        self.searxng_service = searxng_service
        
        # Initialize analyzers
        self.trend_analyzer = TrendAnalyzer(searxng_service)
        self.competitor_analyzer = CompetitorAnalyzer(neo4j_client, searxng_service)
        
        # Cache for expensive operations
        self.cache = TTLCache(maxsize=1000, ttl=3600)
        
        # Settings
        self.settings = get_settings()
        
        logger.info("Gap Analysis Engine initialized")
    
    async def analyze_content_gaps(self,
                                 tenant_id: str,
                                 industry: str,
                                 competitor_domains: Optional[List[str]] = None,
                                 existing_topics: Optional[List[str]] = None,
                                 analysis_type: str = "comprehensive") -> GapAnalysisResult:
        """
        Analyze content gaps with competitor intelligence.
        
        Args:
            tenant_id: Tenant identifier
            industry: Industry to analyze
            competitor_domains: List of competitor domains
            existing_topics: Existing content topics (if known)
            analysis_type: Type of analysis (comprehensive, quick, trending)
            
        Returns:
            GapAnalysisResult with opportunities and insights
        """
        start_time = time.time()
        
        try:
            logger.info(
                "Starting content gap analysis",
                tenant_id=tenant_id,
                industry=industry,
                analysis_type=analysis_type
            )
            
            # Get existing content topics
            if existing_topics is None:
                existing_topics = await self._get_existing_topics(tenant_id)
            
            # Get trending topics
            trending_topics = await self.trend_analyzer.get_trending_topics(industry)
            
            # Analyze competitors if provided
            competitor_analyses = []
            if competitor_domains:
                competitor_analyses = await self.competitor_analyzer.analyze_competitor_content(
                    competitor_domains, industry, tenant_id
                )
            
            # Identify content gaps
            opportunities = await self._identify_content_gaps(
                existing_topics,
                trending_topics,
                competitor_analyses,
                industry,
                tenant_id
            )
            
            # Score and rank opportunities
            scored_opportunities = await self._score_opportunities(opportunities, industry)
            
            # Create analysis result
            result = GapAnalysisResult(
                tenant_id=tenant_id,
                industry=industry,
                analysis_type=analysis_type,
                opportunities=scored_opportunities,
                total_opportunities=len(scored_opportunities),
                top_topics=self._extract_top_topics(scored_opportunities),
                trending_themes=self._extract_trending_themes(trending_topics),
                competitor_insights=self._extract_competitor_insights(competitor_analyses),
                confidence_score=self._calculate_confidence_score(scored_opportunities),
                analysis_duration=time.time() - start_time,
                analyzed_by="gap_analysis_engine"
            )
            
            logger.info(
                "Content gap analysis completed",
                tenant_id=tenant_id,
                opportunities_found=len(scored_opportunities),
                analysis_duration=result.analysis_duration
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Content gap analysis failed",
                tenant_id=tenant_id,
                industry=industry,
                error=str(e)
            )
            raise GapAnalysisError(f"Gap analysis failed: {e}")
    
    async def _get_existing_topics(self, tenant_id: str) -> List[str]:
        """Get existing content topics from knowledge graph."""
        try:
            # Cache existing topics
            cache_key = f"existing_topics_{tenant_id}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Query Neo4j for existing topics
            query = """
            MATCH (c:Content {tenant_id: $tenant_id})
            RETURN DISTINCT c.title as title, c.topics as topics
            LIMIT 1000
            """
            
            results = await self.neo4j_client.run_query(query, {"tenant_id": tenant_id})
            
            topics = set()
            for record in results:
                # Extract topics from titles and topic fields
                title = record.get('title', '')
                content_topics = record.get('topics', [])
                
                # Add title as topic
                if title:
                    topics.add(title.lower())
                
                # Add explicit topics
                if isinstance(content_topics, list):
                    topics.update(topic.lower() for topic in content_topics)
            
            topic_list = list(topics)[:500]  # Limit to avoid memory issues
            
            # Cache results
            self.cache[cache_key] = topic_list
            return topic_list
            
        except Exception as e:
            logger.error(f"Failed to get existing topics: {e}")
            return []
    
    async def _identify_content_gaps(self,
                                   existing_topics: List[str],
                                   trending_topics: List[Dict[str, Any]],
                                   competitor_analyses: List[CompetitorContentAnalysis],
                                   industry: str,
                                   tenant_id: str) -> List[ContentGapOpportunity]:
        """Identify content gaps using multiple data sources."""
        gaps = []
        
        # Convert existing topics to set for faster lookup
        existing_set = set(existing_topics)
        
        # Process trending topics
        for trend_data in trending_topics:
            topic = trend_data['topic']
            
            # Check if we already have content on this topic
            if not any(topic in existing for existing in existing_set):
                gap = ContentGapOpportunity(
                    topic=topic,
                    priority_score=trend_data['relevance'],
                    search_volume=int(trend_data.get('mentions', 0) * 100),  # Estimate
                    trend_direction=TrendDirection.RISING,
                    confidence_score=trend_data['trend_strength'],
                    data_sources=['trends'],
                    tenant_id=tenant_id
                )
                gaps.append(gap)
        
        # Process competitor gaps
        for competitor_analysis in competitor_analyses:
            competitor_topics = competitor_analysis.topics_covered
            
            for topic in competitor_topics:
                # Check if competitor has content we don't
                if not any(topic in existing for existing in existing_set):
                    # Check if already identified from trends
                    if not any(gap.topic == topic for gap in gaps):
                        gap = ContentGapOpportunity(
                            topic=topic,
                            priority_score=0.6,  # Default competitive priority
                            search_volume=0,  # Will be enhanced in scoring
                            trend_direction=TrendDirection.STABLE,
                            competitor_coverage={competitor_analysis.competitor_domain: 1},
                            confidence_score=0.7,
                            data_sources=['competitor_analysis'],
                            tenant_id=tenant_id
                        )
                        gaps.append(gap)
        
        return gaps
    
    async def _score_opportunities(self,
                                 opportunities: List[ContentGapOpportunity],
                                 industry: str) -> List[ContentGapOpportunity]:
        """Score and rank content opportunities."""
        scored_opportunities = []
        
        for opportunity in opportunities:
            try:
                # Enhance opportunity with additional data
                enhanced_opportunity = await self._enhance_opportunity(opportunity, industry)
                
                # Calculate final scores
                enhanced_opportunity.calculate_opportunity_score()
                
                scored_opportunities.append(enhanced_opportunity)
                
            except Exception as e:
                logger.warning(f"Failed to score opportunity '{opportunity.topic}': {e}")
                # Include opportunity with basic scoring
                opportunity.calculate_opportunity_score()
                scored_opportunities.append(opportunity)
        
        # Sort by priority score
        scored_opportunities.sort(key=lambda x: x.priority_score, reverse=True)
        
        return scored_opportunities
    
    async def _enhance_opportunity(self,
                                 opportunity: ContentGapOpportunity,
                                 industry: str) -> ContentGapOpportunity:
        """Enhance opportunity with additional search and competition data."""
        try:
            # Search for keyword data
            search_query = f"{opportunity.topic} {industry}"
            search_results = await self.searxng_service.search(
                query=search_query,
                engines=['google'],
                safesearch=1
            )
            
            # Estimate search volume from results
            if search_results:
                # Simple heuristic: more results = higher volume
                estimated_volume = min(len(search_results) * 50, 10000)
                opportunity.search_volume = max(opportunity.search_volume, estimated_volume)
            
            # Analyze competition
            competition = await self._analyze_competition(opportunity.topic, search_results)
            opportunity.difficulty_score = competition
            
            # Set competition level
            if competition < 0.3:
                opportunity.competition_level = CompetitionLevel.LOW
            elif competition < 0.6:
                opportunity.competition_level = CompetitionLevel.MEDIUM
            elif competition < 0.8:
                opportunity.competition_level = CompetitionLevel.HIGH
            else:
                opportunity.competition_level = CompetitionLevel.VERY_HIGH
            
            # Generate content suggestions
            opportunity.content_suggestions = await self._generate_content_suggestions(
                opportunity.topic, industry
            )
            
            return opportunity
            
        except Exception as e:
            logger.warning(f"Failed to enhance opportunity '{opportunity.topic}': {e}")
            return opportunity
    
    async def _analyze_competition(self, topic: str, search_results: List[Dict]) -> float:
        """Analyze competition level for a topic."""
        if not search_results:
            return 0.5  # Default medium competition
        
        # Analyze domain authority of top results (simplified)
        high_authority_domains = {
            'wikipedia.org', 'forbes.com', 'entrepreneur.com',
            'harvard.edu', 'mit.edu', 'stanford.edu'
        }
        
        high_authority_count = 0
        for result in search_results[:10]:  # Check top 10 results
            url = result.get('url', '')
            domain = url.split('/')[2] if '/' in url else url
            
            if any(auth_domain in domain for auth_domain in high_authority_domains):
                high_authority_count += 1
        
        # Competition score based on high authority presence
        competition = high_authority_count / 10.0
        
        return min(competition, 1.0)
    
    async def _generate_content_suggestions(self, topic: str, industry: str) -> List[str]:
        """Generate content suggestions for a topic."""
        suggestions = [
            f"Complete guide to {topic}",
            f"How to implement {topic} in {industry}",
            f"{topic} best practices for {industry}",
            f"Common mistakes with {topic}",
            f"{topic} case studies and examples",
            f"Future trends in {topic}",
            f"{topic} vs alternatives comparison"
        ]
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def _extract_top_topics(self, opportunities: List[ContentGapOpportunity]) -> List[str]:
        """Extract top topics from opportunities."""
        return [opp.topic for opp in opportunities[:10]]
    
    def _extract_trending_themes(self, trending_topics: List[Dict[str, Any]]) -> List[str]:
        """Extract trending themes."""
        return [topic['topic'] for topic in trending_topics[:5]]
    
    def _extract_competitor_insights(self, 
                                   competitor_analyses: List[CompetitorContentAnalysis]) -> Dict[str, Any]:
        """Extract competitor insights."""
        if not competitor_analyses:
            return {}
        
        insights = {
            'total_competitors_analyzed': len(competitor_analyses),
            'average_content_pieces': sum(c.total_content_pieces for c in competitor_analyses) / len(competitor_analyses),
            'most_active_competitor': max(competitor_analyses, key=lambda c: c.total_content_pieces).competitor_domain,
            'common_topics': self._find_common_topics(competitor_analyses)
        }
        
        return insights
    
    def _find_common_topics(self, competitor_analyses: List[CompetitorContentAnalysis]) -> List[str]:
        """Find topics common across competitors."""
        if not competitor_analyses:
            return []
        
        # Find intersection of topics
        common_topics = set(competitor_analyses[0].topics_covered)
        for analysis in competitor_analyses[1:]:
            common_topics = common_topics.intersection(set(analysis.topics_covered))
        
        return list(common_topics)[:10]
    
    def _calculate_confidence_score(self, opportunities: List[ContentGapOpportunity]) -> float:
        """Calculate overall confidence score for analysis."""
        if not opportunities:
            return 0.0
        
        # Average confidence of individual opportunities
        avg_confidence = sum(opp.confidence_score for opp in opportunities) / len(opportunities)
        
        # Bonus for having multiple data sources
        data_sources = set()
        for opp in opportunities:
            data_sources.update(opp.data_sources)
        
        source_bonus = min(len(data_sources) * 0.1, 0.3)
        
        return min(avg_confidence + source_bonus, 1.0)
    
    async def get_gap_analysis_stats(self, tenant_id: str) -> Dict[str, Any]:
        """Get gap analysis statistics for a tenant."""
        try:
            # Query recent analyses
            query = """
            MATCH (a:GapAnalysis {tenant_id: $tenant_id})
            WHERE a.analyzed_at > datetime() - duration('P30D')
            RETURN count(a) as total_analyses,
                   avg(a.total_opportunities) as avg_opportunities,
                   max(a.analyzed_at) as last_analysis
            """
            
            results = await self.neo4j_client.run_query(query, {"tenant_id": tenant_id})
            
            if results:
                record = results[0]
                return {
                    'total_analyses_last_30_days': record.get('total_analyses', 0),
                    'average_opportunities_per_analysis': record.get('avg_opportunities', 0),
                    'last_analysis_date': record.get('last_analysis'),
                    'cache_hit_rate': len(self.cache) / self.cache.maxsize if self.cache.maxsize else 0
                }
            
            return {
                'total_analyses_last_30_days': 0,
                'average_opportunities_per_analysis': 0,
                'last_analysis_date': None,
                'cache_hit_rate': 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get gap analysis stats: {e}")
            return {}


# =============================================================================
# Utility Functions
# =============================================================================

async def analyze_content_gaps_simple(tenant_id: str, 
                                    industry: str,
                                    competitor_domains: Optional[List[str]] = None) -> GapAnalysisResult:
    """
    Simple content gap analysis function.
    
    Args:
        tenant_id: Tenant identifier
        industry: Industry to analyze
        competitor_domains: Optional competitor domains
        
    Returns:
        GapAnalysisResult with opportunities
    """
    # Initialize required services
    settings = get_settings()
    
    neo4j_client = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_username,
        password=settings.neo4j_password
    )
    
    qdrant_client = QdrantClient(settings.qdrant_url)
    embedding_service = EmbeddingService()
    searxng_service = SearXNGService(settings.searxng_url)
    
    # Create gap analysis engine
    engine = GapAnalysisEngine(
        neo4j_client=neo4j_client,
        qdrant_client=qdrant_client,
        embedding_service=embedding_service,
        searxng_service=searxng_service
    )
    
    # Perform analysis
    result = await engine.analyze_content_gaps(
        tenant_id=tenant_id,
        industry=industry,
        competitor_domains=competitor_domains
    )
    
    return result


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        # Test gap analysis
        result = await analyze_content_gaps_simple(
            tenant_id="test-tenant",
            industry="content marketing",
            competitor_domains=["hubspot.com", "contentmarketinginstitute.com"]
        )
        
        print(f"Analysis completed: {result.total_opportunities} opportunities found")
        print(f"Top topics: {result.top_topics}")
        print(f"Confidence score: {result.confidence_score:.2f}")
        
        # Show top opportunities
        top_opportunities = result.get_high_priority_opportunities(5)
        for i, opp in enumerate(top_opportunities, 1):
            print(f"{i}. {opp.topic} (Score: {opp.priority_score:.2f})")

    asyncio.run(main())