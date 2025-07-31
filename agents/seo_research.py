"""
SEO Research Agent for the SEO Content Knowledge Graph System.

This agent performs SEO research including keyword analysis, competitor research,
trend analysis, and content gap identification using SearXNG.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import structlog
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from config.settings import get_settings
from database.neo4j_client import Neo4jClient
from database.qdrant_client import QdrantClient
from database.supabase_client import SupabaseClient
from models.seo_models import (
    KeywordData, SEOMetrics, CompetitorData, 
    SearchIntent, TrendDirection, SERPFeature
)
from services.searxng_service import SearXNGService
from services.embedding_service import EmbeddingService

logger = structlog.get_logger(__name__)


class SEOResearchDependencies:
    """Dependencies for the SEO Research Agent."""
    
    def __init__(
        self,
        neo4j_client: Neo4jClient,
        qdrant_client: QdrantClient,
        supabase_client: SupabaseClient,
        searxng_service: SearXNGService,
        embedding_service: EmbeddingService,
        tenant_id: str
    ):
        self.neo4j_client = neo4j_client
        self.qdrant_client = qdrant_client
        self.supabase_client = supabase_client
        self.searxng_service = searxng_service
        self.embedding_service = embedding_service
        self.tenant_id = tenant_id


class KeywordResearch(BaseModel):
    """Structured output for keyword research."""
    
    primary_keywords: List[Dict[str, Any]] = Field(..., description="Primary target keywords with metrics")
    secondary_keywords: List[Dict[str, Any]] = Field(..., description="Secondary keywords to target")
    long_tail_keywords: List[Dict[str, Any]] = Field(..., description="Long-tail keyword opportunities")
    
    search_intent_analysis: Dict[str, Any] = Field(..., description="Analysis of search intent patterns")
    keyword_difficulty: Dict[str, float] = Field(..., description="Keyword difficulty scores")
    opportunity_score: float = Field(..., description="Overall keyword opportunity score")
    
    recommended_strategy: str = Field(..., description="Recommended keyword strategy")
    content_suggestions: List[str] = Field(..., description="Content creation suggestions")


class CompetitorAnalysis(BaseModel):
    """Structured output for competitor analysis."""
    
    top_competitors: List[Dict[str, Any]] = Field(..., description="Identified top competitors")
    content_gaps: List[Dict[str, Any]] = Field(..., description="Content gaps vs competitors")
    keyword_opportunities: List[Dict[str, Any]] = Field(..., description="Keyword opportunities")
    
    competitor_strengths: Dict[str, List[str]] = Field(..., description="Competitor strengths by domain")
    competitor_weaknesses: Dict[str, List[str]] = Field(..., description="Competitor weaknesses")
    
    market_positioning: Dict[str, Any] = Field(..., description="Market positioning analysis")
    differentiation_opportunities: List[str] = Field(..., description="Differentiation opportunities")


class TrendAnalysis(BaseModel):
    """Structured output for trend analysis."""
    
    trending_topics: List[Dict[str, Any]] = Field(..., description="Currently trending topics")
    emerging_keywords: List[Dict[str, Any]] = Field(..., description="Emerging keyword opportunities")
    seasonal_patterns: Dict[str, Any] = Field(..., description="Seasonal search patterns")
    
    trend_direction: TrendDirection = Field(..., description="Overall trend direction")
    trend_confidence: float = Field(..., description="Confidence in trend analysis")
    
    content_opportunities: List[Dict[str, Any]] = Field(..., description="Content opportunities based on trends")
    timing_recommendations: List[str] = Field(..., description="Timing recommendations for content")


class SEOResearchResult(BaseModel):
    """Complete SEO research result."""
    
    research_id: str = Field(..., description="Research session ID")
    research_type: str = Field(..., description="Type of research performed")
    target_topic: str = Field(..., description="Target topic or keyword")
    
    timestamp: datetime = Field(..., description="Research timestamp")
    tenant_id: str = Field(..., description="Tenant identifier")
    
    keyword_research: Optional[KeywordResearch] = None
    competitor_analysis: Optional[CompetitorAnalysis] = None
    trend_analysis: Optional[TrendAnalysis] = None
    
    processing_time: float = Field(..., description="Research processing time")
    success: bool = Field(..., description="Research success status")
    warnings: List[str] = Field(default_factory=list, description="Research warnings")
    
    actionable_insights: List[str] = Field(..., description="Key actionable insights")
    next_steps: List[str] = Field(..., description="Recommended next steps")


# Initialize the SEO Research Agent
seo_research_agent = Agent(
    "openai:gpt-4o-mini",
    deps_type=SEOResearchDependencies,
    result_type=SEOResearchResult,
    system_prompt="""
    You are a specialized SEO Research Agent focused on comprehensive search engine optimization research.
    
    Your capabilities include:
    1. Keyword research and analysis with search intent classification
    2. Competitor analysis and content gap identification
    3. Trend analysis and emerging opportunity detection
    4. Content strategy recommendations based on research findings
    
    Research Guidelines:
    - Focus on data-driven insights from search results
    - Identify both short-term and long-term opportunities
    - Consider search intent (informational, navigational, transactional, commercial)
    - Analyze competitor strengths and weaknesses objectively
    - Provide specific, actionable recommendations
    
    Search Intent Classification:
    - Informational: Users seeking information (how-to, what is, etc.)
    - Navigational: Users looking for specific websites or pages
    - Transactional: Users ready to make a purchase or take action
    - Commercial: Users researching products/services before buying
    
    Opportunity Scoring (0-100):
    - 90-100: High-value, low-competition opportunities
    - 70-89: Good opportunities with moderate competition
    - 50-69: Average opportunities requiring significant effort
    - 30-49: Challenging opportunities with high competition
    - 0-29: Low-value or extremely competitive opportunities
    
    Always provide specific, actionable recommendations with clear next steps.
    """,
)


class SEOResearchAgent:
    """
    SEO Research Agent for keyword research, competitor analysis,
    and trend identification using SearXNG.
    """
    
    def __init__(
        self,
        neo4j_client: Neo4jClient,
        qdrant_client: QdrantClient,
        supabase_client: SupabaseClient,
        searxng_service: SearXNGService,
        embedding_service: EmbeddingService,
        tenant_id: str
    ):
        """
        Initialize the SEO Research Agent.
        
        Args:
            neo4j_client: Neo4j database client
            qdrant_client: Qdrant vector database client
            supabase_client: Supabase database client
            searxng_service: SearXNG search service
            embedding_service: Embedding generation service
            tenant_id: Tenant identifier
        """
        self.neo4j_client = neo4j_client
        self.qdrant_client = qdrant_client
        self.supabase_client = supabase_client
        self.searxng_service = searxng_service
        self.embedding_service = embedding_service
        self.tenant_id = tenant_id
        
        # Initialize dependencies
        self.deps = SEOResearchDependencies(
            neo4j_client=neo4j_client,
            qdrant_client=qdrant_client,
            supabase_client=supabase_client,
            searxng_service=searxng_service,
            embedding_service=embedding_service,
            tenant_id=tenant_id
        )
        
        logger.info(
            "SEO Research Agent initialized",
            tenant_id=tenant_id
        )
    
    async def research_keywords(
        self,
        target_topic: str,
        include_competitors: bool = True,
        max_competitors: int = 5,
        competitor_domains: Optional[List[str]] = None
    ) -> SEOResearchResult:
        """
        Perform comprehensive keyword research.
        
        Args:
            target_topic: Target topic or seed keyword
            include_competitors: Whether to include competitor analysis
            max_competitors: Maximum competitors to analyze
            competitor_domains: Specific competitor domains to analyze
            
        Returns:
            SEO research result with keyword analysis
        """
        start_time = asyncio.get_event_loop().time()
        
        logger.info(
            "Starting keyword research",
            target_topic=target_topic,
            tenant_id=self.tenant_id
        )
        
        try:
            # Collect search data
            search_data = await self._collect_search_data(target_topic)
            
            # Perform AI analysis
            research_input = self._prepare_keyword_research_input(
                target_topic, search_data, include_competitors
            )
            
            result = await seo_research_agent.run(
                research_input,
                deps=self.deps
            )
            
            # Enhance with competitor analysis if requested
            if include_competitors:
                await self._enhance_with_competitor_analysis(
                    result, target_topic, competitor_domains, max_competitors
                )
            
            # Store research results
            await self._store_research_results(result)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            result.processing_time = processing_time
            
            logger.info(
                "Keyword research completed",
                target_topic=target_topic,
                processing_time=processing_time,
                keywords_found=len(result.keyword_research.primary_keywords) if result.keyword_research else 0
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Keyword research failed",
                target_topic=target_topic,
                error=str(e)
            )
            raise
    
    async def analyze_trends(
        self,
        topic: str,
        time_periods: List[str] = None,
        languages: List[str] = None
    ) -> SEOResearchResult:
        """
        Analyze search trends for a topic.
        
        Args:
            topic: Topic to analyze
            time_periods: Time periods to analyze
            languages: Languages to analyze
            
        Returns:
            SEO research result with trend analysis
        """
        start_time = asyncio.get_event_loop().time()
        
        logger.info(
            "Starting trend analysis",
            topic=topic,
            tenant_id=self.tenant_id
        )
        
        try:
            # Collect trend data
            trend_data = await self.searxng_service.search_trends(
                topic=topic,
                time_periods=time_periods,
                languages=languages
            )
            
            # Get related searches
            related_searches = await self.searxng_service.get_related_searches(
                query=topic,
                max_suggestions=20
            )
            
            # Prepare analysis input
            research_input = self._prepare_trend_analysis_input(
                topic, trend_data, related_searches
            )
            
            # Perform AI analysis
            result = await seo_research_agent.run(
                research_input,
                deps=self.deps
            )
            
            # Store results
            await self._store_research_results(result)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            result.processing_time = processing_time
            
            logger.info(
                "Trend analysis completed",
                topic=topic,
                processing_time=processing_time,
                trend_direction=result.trend_analysis.trend_direction if result.trend_analysis else "unknown"
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Trend analysis failed",
                topic=topic,
                error=str(e)
            )
            raise
    
    async def analyze_competitors(
        self,
        topic: str,
        competitor_domains: List[str],
        max_results_per_competitor: int = 10
    ) -> SEOResearchResult:
        """
        Analyze competitor content for a topic.
        
        Args:
            topic: Topic to analyze
            competitor_domains: List of competitor domains
            max_results_per_competitor: Max results per competitor
            
        Returns:
            SEO research result with competitor analysis
        """
        start_time = asyncio.get_event_loop().time()
        
        logger.info(
            "Starting competitor analysis",
            topic=topic,
            competitors=len(competitor_domains),
            tenant_id=self.tenant_id
        )
        
        try:
            # Analyze each competitor
            competitor_analyses = []
            for domain in competitor_domains:
                analysis = await self.searxng_service.analyze_competitor_content(
                    competitor_domain=domain,
                    topic_keywords=[topic],
                    max_results_per_keyword=max_results_per_competitor
                )
                competitor_analyses.append(analysis)
            
            # Prepare analysis input
            research_input = self._prepare_competitor_analysis_input(
                topic, competitor_analyses
            )
            
            # Perform AI analysis
            result = await seo_research_agent.run(
                research_input,
                deps=self.deps
            )
            
            # Store results
            await self._store_research_results(result)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            result.processing_time = processing_time
            
            logger.info(
                "Competitor analysis completed",
                topic=topic,
                processing_time=processing_time,
                competitors_analyzed=len(competitor_domains)
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Competitor analysis failed",
                topic=topic,
                error=str(e)
            )
            raise
    
    async def _collect_search_data(self, topic: str) -> Dict[str, Any]:
        """Collect search data for keyword research."""
        search_data = {}
        
        try:
            # Basic search
            search_results = await self.searxng_service.search(
                query=topic,
                max_results=20
            )
            search_data["search_results"] = search_results.get("results", [])
            
            # Related searches
            related_searches = await self.searxng_service.get_related_searches(
                query=topic,
                max_suggestions=15
            )
            search_data["related_searches"] = related_searches
            
            # Extract keywords from results
            keywords = await self.searxng_service.extract_keywords_from_results(
                search_results.get("results", []),
                min_frequency=2
            )
            search_data["extracted_keywords"] = keywords
            
        except Exception as e:
            logger.warning(
                "Failed to collect some search data",
                topic=topic,
                error=str(e)
            )
            search_data["error"] = str(e)
        
        return search_data
    
    def _prepare_keyword_research_input(
        self,
        topic: str,
        search_data: Dict[str, Any],
        include_competitors: bool
    ) -> str:
        """Prepare input for keyword research analysis."""
        input_parts = [
            f"KEYWORD RESEARCH FOR: {topic}",
            f"RESEARCH TYPE: keyword_research",
            f"INCLUDE COMPETITORS: {include_competitors}",
            "",
            "SEARCH RESULTS:"
        ]
        
        # Add search results
        for i, result in enumerate(search_data.get("search_results", [])[:10]):
            input_parts.append(f"{i+1}. {result.get('title', 'No Title')}")
            if result.get("content"):
                input_parts.append(f"   {result['content'][:200]}...")
        
        input_parts.extend([
            "",
            "RELATED SEARCHES:",
            ", ".join(search_data.get("related_searches", [])),
            "",
            "EXTRACTED KEYWORDS:",
            ", ".join(search_data.get("extracted_keywords", [])[:20])
        ])
        
        return "\n".join(input_parts)
    
    def _prepare_trend_analysis_input(
        self,
        topic: str,
        trend_data: Dict[str, Any],
        related_searches: List[str]
    ) -> str:
        """Prepare input for trend analysis."""
        input_parts = [
            f"TREND ANALYSIS FOR: {topic}",
            f"RESEARCH TYPE: trend_analysis",
            "",
            "TREND DATA:",
            f"Topic: {trend_data.get('topic', 'Unknown')}",
            f"Trend Direction: {trend_data.get('trend_direction', 'Unknown')}",
            f"Confidence: {trend_data.get('confidence', 0.0)}",
            ""
        ]
        
        # Add time period data
        for period, data in trend_data.get("time_periods", {}).items():
            input_parts.append(f"{period.upper()} PERIOD:")
            input_parts.append(f"  Results Count: {data.get('results_count', 0)}")
            input_parts.append(f"  Search Volume Indicator: {data.get('search_volume_indicator', 0.0)}")
            
            # Add sample results
            for result in data.get("results", [])[:3]:
                input_parts.append(f"  - {result.get('title', 'No Title')}")
        
        input_parts.extend([
            "",
            "RELATED SEARCHES:",
            ", ".join(related_searches)
        ])
        
        return "\n".join(input_parts)
    
    def _prepare_competitor_analysis_input(
        self,
        topic: str,
        competitor_analyses: List[Dict[str, Any]]
    ) -> str:
        """Prepare input for competitor analysis."""
        input_parts = [
            f"COMPETITOR ANALYSIS FOR: {topic}",
            f"RESEARCH TYPE: competitor_analysis",
            "",
            "COMPETITOR DATA:"
        ]
        
        for analysis in competitor_analyses:
            domain = analysis.get("competitor_domain", "Unknown")
            input_parts.append(f"\nCOMPETITOR: {domain}")
            input_parts.append(f"Total Content: {analysis.get('total_content_pieces', 0)}")
            input_parts.append(f"Content Gaps: {len(analysis.get('content_gaps', []))}")
            
            # Add sample content
            for keyword, content_list in analysis.get("content_found", {}).items():
                if content_list:
                    input_parts.append(f"  {keyword}: {len(content_list)} pieces")
                    for content in content_list[:2]:  # Show first 2 pieces
                        input_parts.append(f"    - {content.get('title', 'No Title')}")
        
        return "\n".join(input_parts)
    
    async def _enhance_with_competitor_analysis(
        self,
        result: SEOResearchResult,
        topic: str,
        competitor_domains: Optional[List[str]],
        max_competitors: int
    ) -> None:
        """Enhance research with competitor analysis."""
        try:
            if not competitor_domains:
                # Try to identify competitors from search results
                competitor_domains = await self._identify_competitors(topic, max_competitors)
            
            if competitor_domains:
                competitor_result = await self.analyze_competitors(
                    topic=topic,
                    competitor_domains=competitor_domains[:max_competitors]
                )
                
                if competitor_result.competitor_analysis:
                    result.competitor_analysis = competitor_result.competitor_analysis
                    
        except Exception as e:
            logger.warning(
                "Failed to enhance with competitor analysis",
                topic=topic,
                error=str(e)
            )
            result.warnings.append(f"Competitor analysis failed: {str(e)}")
    
    async def _identify_competitors(self, topic: str, max_competitors: int) -> List[str]:
        """Identify potential competitors from search results."""
        try:
            search_results = await self.searxng_service.search(
                query=topic,
                max_results=20
            )
            
            domains = set()
            for result in search_results.get("results", []):
                url = result.get("url", "")
                if url:
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc
                    if domain and not domain.startswith("www."):
                        domains.add(domain)
                    elif domain and domain.startswith("www."):
                        domains.add(domain[4:])  # Remove www.
            
            # Filter out common non-competitor domains
            exclude_domains = {
                "wikipedia.org", "youtube.com", "linkedin.com",
                "twitter.com", "facebook.com", "instagram.com",
                "reddit.com", "quora.com", "github.com"
            }
            
            competitor_domains = [
                domain for domain in domains
                if domain not in exclude_domains
            ]
            
            return competitor_domains[:max_competitors]
            
        except Exception as e:
            logger.warning(
                "Failed to identify competitors",
                topic=topic,
                error=str(e)
            )
            return []
    
    async def _store_research_results(self, result: SEOResearchResult) -> None:
        """Store research results in the database."""
        try:
            # Store in Supabase for structured data
            research_data = {
                "id": result.research_id,
                "research_type": result.research_type,
                "target_topic": result.target_topic,
                "tenant_id": result.tenant_id,
                "timestamp": result.timestamp.isoformat(),
                "processing_time": result.processing_time,
                "success": result.success,
                "actionable_insights": result.actionable_insights,
                "next_steps": result.next_steps,
                "result_data": result.dict()
            }
            
            # Store in Supabase
            await self.supabase_client.create_seo_research_record(research_data)
            
            # Store keywords in Neo4j if available
            if result.keyword_research:
                await self._store_keywords_in_graph(result)
            
        except Exception as e:
            logger.error(
                "Failed to store research results",
                research_id=result.research_id,
                error=str(e)
            )
    
    async def _store_keywords_in_graph(self, result: SEOResearchResult) -> None:
        """Store keywords in the knowledge graph."""
        try:
            keyword_research = result.keyword_research
            if not keyword_research:
                return
            
            # Store primary keywords
            for keyword_data in keyword_research.primary_keywords:
                keyword_text = keyword_data.get("keyword", "")
                if keyword_text:
                    # Create keyword node
                    await self.neo4j_client.create_keyword_node(
                        keyword_id=f"kw_{keyword_text.lower().replace(' ', '_')}",
                        text=keyword_text,
                        search_volume=keyword_data.get("search_volume", 0),
                        difficulty=keyword_data.get("difficulty", 0),
                        tenant_id=self.tenant_id
                    )
            
        except Exception as e:
            logger.warning(
                "Failed to store keywords in graph",
                research_id=result.research_id,
                error=str(e)
            )
    
    async def get_research_history(
        self,
        research_type: Optional[str] = None,
        days_back: int = 30,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get research history for the tenant.
        
        Args:
            research_type: Optional research type filter
            days_back: Days to look back
            limit: Maximum results
            
        Returns:
            List of research records
        """
        try:
            # Query research history from Supabase
            filters = {"tenant_id": self.tenant_id}
            if research_type:
                filters["research_type"] = research_type
            
            # Calculate date threshold
            from datetime import timedelta
            date_threshold = datetime.now(timezone.utc) - timedelta(days=days_back)
            
            research_records = await self.supabase_client.query_seo_research_records(
                filters=filters,
                date_threshold=date_threshold,
                limit=limit
            )
            
            return research_records
            
        except Exception as e:
            logger.error(
                "Failed to get research history",
                tenant_id=self.tenant_id,
                error=str(e)
            )
            return []
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "agent_type": "seo_research",
            "tenant_id": self.tenant_id,
            "initialized_at": datetime.now(timezone.utc),
            "dependencies": {
                "neo4j_client": bool(self.neo4j_client),
                "qdrant_client": bool(self.qdrant_client),
                "supabase_client": bool(self.supabase_client),
                "searxng_service": bool(self.searxng_service),
                "embedding_service": bool(self.embedding_service)
            }
        }


# =============================================================================
# Utility Functions
# =============================================================================

async def create_seo_research_agent(
    tenant_id: str,
    neo4j_client: Optional[Neo4jClient] = None,
    qdrant_client: Optional[QdrantClient] = None,
    supabase_client: Optional[SupabaseClient] = None,
    searxng_service: Optional[SearXNGService] = None,
    embedding_service: Optional[EmbeddingService] = None
) -> SEOResearchAgent:
    """
    Create a configured SEO Research Agent.
    
    Args:
        tenant_id: Tenant identifier
        neo4j_client: Neo4j client (will create if not provided)
        qdrant_client: Qdrant client (will create if not provided)
        supabase_client: Supabase client (will create if not provided)
        searxng_service: SearXNG service (will create if not provided)
        embedding_service: Embedding service (will create if not provided)
        
    Returns:
        SEOResearchAgent instance
    """
    settings = get_settings()
    
    # Initialize clients if not provided
    if neo4j_client is None:
        neo4j_client = Neo4jClient(
            uri=settings.neo4j_uri,
            user=settings.neo4j_username,
            password=settings.neo4j_password
        )
        await neo4j_client.connect()
    
    if qdrant_client is None:
        qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )
    
    if supabase_client is None:
        supabase_client = SupabaseClient(
            url=settings.supabase_url,
            key=settings.supabase_key
        )
    
    if searxng_service is None:
        searxng_service = SearXNGService(
            base_url=settings.searxng_url,
            api_key=settings.searxng_api_key
        )
    
    if embedding_service is None:
        embedding_service = EmbeddingService(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model
        )
    
    return SEOResearchAgent(
        neo4j_client=neo4j_client,
        qdrant_client=qdrant_client,
        supabase_client=supabase_client,
        searxng_service=searxng_service,
        embedding_service=embedding_service,
        tenant_id=tenant_id
    )


if __name__ == "__main__":
    # Example usage
    async def main():
        # Create agent
        agent = await create_seo_research_agent(tenant_id="test-tenant")
        
        # Test keyword research
        keyword_result = await agent.research_keywords(
            target_topic="content marketing strategies",
            include_competitors=True
        )
        
        print(f"Keyword research completed for: {keyword_result.target_topic}")
        if keyword_result.keyword_research:
            print(f"Primary keywords found: {len(keyword_result.keyword_research.primary_keywords)}")
            print(f"Opportunity score: {keyword_result.keyword_research.opportunity_score}")
        
        # Test trend analysis
        trend_result = await agent.analyze_trends(
            topic="AI content creation",
            time_periods=["day", "week", "month"]
        )
        
        print(f"Trend analysis completed for: {trend_result.target_topic}")
        if trend_result.trend_analysis:
            print(f"Trend direction: {trend_result.trend_analysis.trend_direction}")
            print(f"Trending topics: {len(trend_result.trend_analysis.trending_topics)}")
        
        # Get agent stats
        stats = agent.get_agent_stats()
        print(f"Agent stats: {stats}")
        
        # Close SearXNG service
        await agent.searxng_service.close()
    
    asyncio.run(main())