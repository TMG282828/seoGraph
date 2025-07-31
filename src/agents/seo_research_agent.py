"""
SEO Research Agent for SEO Content Knowledge Graph System.

This agent conducts comprehensive SEO research including keyword discovery,
competitor analysis, trend identification, and search intent analysis.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from pydantic_ai import tool
import json
import re
import aiohttp

from .base_agent import BaseAgent, AgentContext, AgentResult, agent_registry

logger = logging.getLogger(__name__)


class SEOResearchRequest(BaseModel):
    """Request model for SEO research tasks."""
    primary_keyword: Optional[str] = None
    target_keywords: List[str] = Field(default_factory=list)
    competitor_urls: List[str] = Field(default_factory=list)
    research_type: str = Field(default="comprehensive")  # comprehensive, keywords, competitors, trends, intent
    search_volume_threshold: int = 100
    competition_level: str = "medium"  # low, medium, high, any
    location: str = "US"
    language: str = "en"
    include_questions: bool = True
    include_related_topics: bool = True


class SEOResearchAgent(BaseAgent):
    """
    AI agent for comprehensive SEO research and competitive analysis.
    
    Capabilities:
    - Keyword research and opportunity identification
    - Search volume and competition analysis
    - Competitor content gap analysis
    - Search trend identification and seasonal patterns
    - Search intent classification and optimization
    - Related keyword and topic discovery
    - SERP feature analysis and optimization opportunities
    """
    
    def __init__(self):
        super().__init__(
            name="seo_research",
            description="Conducts comprehensive SEO research including keyword discovery, competitor analysis, and trend identification"
        )
        self.searxng_url = "http://localhost:8080"  # From environment
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the SEO Research Agent."""
        return """You are an expert SEO Research Agent specializing in keyword research, competitor analysis, and search trend identification.

Your role is to conduct comprehensive SEO research and provide actionable insights for:
- Keyword discovery and opportunity identification
- Search volume estimation and competition analysis
- Competitor content gap analysis
- Search intent classification and optimization strategies
- Trending topics and seasonal keyword patterns
- Related keyword and topic cluster development
- SERP feature optimization opportunities

Always consider:
1. Search intent behind keywords (informational, navigational, transactional, commercial)
2. Keyword difficulty vs. opportunity balance
3. Long-tail keyword variations and semantic relationships
4. Content format preferences for different keyword types
5. Geographic and demographic targeting considerations
6. Seasonal trends and timing opportunities

Provide specific, data-driven recommendations with clear prioritization and implementation strategies."""
    
    def _register_tools(self) -> None:
        """Register tools specific to SEO research."""
        
        @self._agent.tool
        async def search_keyword_opportunities(seed_keywords: List[str], volume_threshold: int) -> Dict[str, Any]:
            """Search for keyword opportunities and variations."""
            return await self._search_keyword_opportunities(seed_keywords, volume_threshold)
        
        @self._agent.tool
        async def analyze_competitor_content(competitor_urls: List[str]) -> Dict[str, Any]:
            """Analyze competitor content for keyword and topic gaps."""
            return await self._analyze_competitor_content(competitor_urls)
        
        @self._agent.tool
        async def research_search_trends(keywords: List[str], timeframe: str) -> Dict[str, Any]:
            """Research search trends and seasonal patterns."""
            return await self._research_search_trends(keywords, timeframe)
        
        @self._agent.tool
        async def classify_search_intent(keywords: List[str]) -> Dict[str, Any]:
            """Classify search intent for keywords."""
            return await self._classify_search_intent(keywords)
        
        @self._agent.tool
        async def find_related_keywords(primary_keyword: str, depth: int) -> Dict[str, Any]:
            """Find related keywords and semantic variations."""
            return await self._find_related_keywords(primary_keyword, depth)
        
        @self._agent.tool
        async def analyze_serp_features(keywords: List[str]) -> Dict[str, Any]:
            """Analyze SERP features and optimization opportunities."""
            return await self._analyze_serp_features(keywords)
        
        @self._agent.tool
        async def estimate_keyword_difficulty(keywords: List[str]) -> Dict[str, Any]:
            """Estimate keyword difficulty and ranking opportunities."""
            return await self._estimate_keyword_difficulty(keywords)
    
    async def _execute_task(self, task_data: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        """Execute SEO research task."""
        request = SEOResearchRequest(**task_data)
        
        # Get industry context and preferences
        brand_voice = await self._get_brand_voice_config()
        seo_preferences = await self._get_seo_preferences()
        industry_context = context.industry_context or brand_voice.get('industryContext', '')
        
        # Perform research based on type
        if request.research_type == "comprehensive":
            return await self._comprehensive_research(request, context, industry_context, seo_preferences)
        elif request.research_type == "keywords":
            return await self._keyword_research(request, context, industry_context)
        elif request.research_type == "competitors":
            return await self._competitor_research(request, context)
        elif request.research_type == "trends":
            return await self._trend_research(request, context)
        elif request.research_type == "intent":
            return await self._intent_research(request, context)
        else:
            raise ValueError(f"Unknown research type: {request.research_type}")
    
    async def _comprehensive_research(self, request: SEOResearchRequest, context: AgentContext,
                                    industry_context: str, seo_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive SEO research."""
        
        # Prepare research keywords
        all_keywords = []
        if request.primary_keyword:
            all_keywords.append(request.primary_keyword)
        all_keywords.extend(request.target_keywords)
        
        if not all_keywords and industry_context:
            # Generate seed keywords from industry context
            all_keywords = await self._generate_seed_keywords(industry_context)
        
        # Run comprehensive research using Pydantic AI agent
        research_prompt = f"""
        Conduct comprehensive SEO research for the following:

        PRIMARY KEYWORD: {request.primary_keyword}
        TARGET KEYWORDS: {', '.join(request.target_keywords)}
        INDUSTRY CONTEXT: {industry_context}
        COMPETITOR URLS: {', '.join(request.competitor_urls)}
        
        RESEARCH CRITERIA:
        - Minimum search volume: {request.search_volume_threshold}
        - Competition level preference: {request.competition_level}
        - Location: {request.location}
        - Language: {request.language}
        - Include questions: {request.include_questions}
        - Include related topics: {request.include_related_topics}

        SEO PREFERENCES:
        {json.dumps(seo_preferences, indent=2)}

        Use the available tools to conduct thorough research including:
        1. Keyword opportunity identification
        2. Search volume and competition analysis
        3. Competitor content gap analysis
        4. Search intent classification
        5. Related keyword discovery
        6. SERP feature analysis
        7. Trend identification

        Provide prioritized recommendations with implementation strategies.
        """
        
        # Execute AI research
        ai_result = await self._agent.run(research_prompt)
        
        # Combine with programmatic research
        keyword_opportunities = await self._search_keyword_opportunities(all_keywords, request.search_volume_threshold)
        competitor_analysis = await self._analyze_competitor_content(request.competitor_urls)
        intent_analysis = await self._classify_search_intent(all_keywords)
        serp_analysis = await self._analyze_serp_features(all_keywords[:5])  # Limit for performance
        
        return {
            "research_type": "comprehensive",
            "primary_keyword": request.primary_keyword,
            "keyword_count": len(all_keywords),
            "ai_research": ai_result.data if hasattr(ai_result, 'data') else str(ai_result),
            "keyword_opportunities": keyword_opportunities,
            "competitor_analysis": competitor_analysis,
            "intent_analysis": intent_analysis,
            "serp_analysis": serp_analysis,
            "content_recommendations": await self._generate_content_recommendations(keyword_opportunities, intent_analysis),
            "priority_keywords": await self._prioritize_keywords(keyword_opportunities, seo_preferences),
            "confidence_score": 0.85
        }
    
    async def _keyword_research(self, request: SEOResearchRequest, context: AgentContext,
                              industry_context: str) -> Dict[str, Any]:
        """Perform keyword-focused research."""
        seed_keywords = [request.primary_keyword] if request.primary_keyword else []
        seed_keywords.extend(request.target_keywords)
        
        if not seed_keywords and industry_context:
            seed_keywords = await self._generate_seed_keywords(industry_context)
        
        keyword_opportunities = await self._search_keyword_opportunities(seed_keywords, request.search_volume_threshold)
        related_keywords = {}
        
        for keyword in seed_keywords[:3]:  # Limit for performance
            related = await self._find_related_keywords(keyword, depth=2)
            related_keywords[keyword] = related
        
        difficulty_analysis = await self._estimate_keyword_difficulty(seed_keywords)
        
        return {
            "research_type": "keywords",
            "seed_keywords": seed_keywords,
            "keyword_opportunities": keyword_opportunities,
            "related_keywords": related_keywords,
            "difficulty_analysis": difficulty_analysis,
            "long_tail_opportunities": await self._find_long_tail_opportunities(seed_keywords),
            "question_keywords": await self._find_question_keywords(seed_keywords) if request.include_questions else [],
            "confidence_score": 0.9
        }
    
    async def _competitor_research(self, request: SEOResearchRequest, context: AgentContext) -> Dict[str, Any]:
        """Perform competitor-focused research."""
        if not request.competitor_urls:
            return {
                "research_type": "competitors",
                "error": "No competitor URLs provided",
                "confidence_score": 0
            }
        
        competitor_analysis = await self._analyze_competitor_content(request.competitor_urls)
        gap_analysis = await self._identify_content_gaps(request.competitor_urls, request.target_keywords)
        
        return {
            "research_type": "competitors",
            "competitor_count": len(request.competitor_urls),
            "competitor_analysis": competitor_analysis,
            "content_gaps": gap_analysis,
            "competitive_opportunities": await self._find_competitive_opportunities(competitor_analysis),
            "confidence_score": 0.8
        }
    
    async def _trend_research(self, request: SEOResearchRequest, context: AgentContext) -> Dict[str, Any]:
        """Perform trend-focused research."""
        keywords_to_research = [request.primary_keyword] if request.primary_keyword else []
        keywords_to_research.extend(request.target_keywords)
        
        trend_analysis = await self._research_search_trends(keywords_to_research, "12_months")
        seasonal_patterns = await self._identify_seasonal_patterns(keywords_to_research)
        
        return {
            "research_type": "trends",
            "trend_analysis": trend_analysis,
            "seasonal_patterns": seasonal_patterns,
            "emerging_keywords": await self._identify_emerging_keywords(keywords_to_research),
            "trend_recommendations": await self._generate_trend_recommendations(trend_analysis),
            "confidence_score": 0.75
        }
    
    async def _intent_research(self, request: SEOResearchRequest, context: AgentContext) -> Dict[str, Any]:
        """Perform search intent focused research."""
        keywords_to_analyze = [request.primary_keyword] if request.primary_keyword else []
        keywords_to_analyze.extend(request.target_keywords)
        
        intent_analysis = await self._classify_search_intent(keywords_to_analyze)
        content_mapping = await self._map_content_to_intent(intent_analysis)
        
        return {
            "research_type": "intent",
            "intent_analysis": intent_analysis,
            "content_mapping": content_mapping,
            "optimization_strategies": await self._generate_intent_strategies(intent_analysis),
            "confidence_score": 0.85
        }
    
    # Tool implementation methods
    
    async def _search_keyword_opportunities(self, seed_keywords: List[str], volume_threshold: int) -> Dict[str, Any]:
        """Search for keyword opportunities and variations."""
        opportunities = []
        
        for seed in seed_keywords:
            # Generate variations
            variations = await self._generate_keyword_variations(seed)
            
            # Mock search volume and competition data (in production, use real APIs)
            for variation in variations:
                estimated_volume = self._estimate_search_volume(variation)
                if estimated_volume >= volume_threshold:
                    opportunities.append({
                        "keyword": variation,
                        "estimated_volume": estimated_volume,
                        "competition": self._estimate_competition(variation),
                        "difficulty": self._estimate_difficulty(variation),
                        "opportunity_score": self._calculate_opportunity_score(estimated_volume, self._estimate_competition(variation)),
                        "source_keyword": seed
                    })
        
        # Sort by opportunity score
        opportunities.sort(key=lambda x: x["opportunity_score"], reverse=True)
        
        return {
            "total_opportunities": len(opportunities),
            "high_opportunity": [opp for opp in opportunities if opp["opportunity_score"] > 70],
            "medium_opportunity": [opp for opp in opportunities if 40 <= opp["opportunity_score"] <= 70],
            "low_opportunity": [opp for opp in opportunities if opp["opportunity_score"] < 40],
            "all_opportunities": opportunities[:50]  # Limit results
        }
    
    async def _analyze_competitor_content(self, competitor_urls: List[str]) -> Dict[str, Any]:
        """Analyze competitor content for keyword and topic gaps."""
        competitor_data = []
        
        for url in competitor_urls[:5]:  # Limit for performance
            try:
                content_analysis = await self._analyze_competitor_url(url)
                competitor_data.append({
                    "url": url,
                    "domain": self._extract_domain(url),
                    "content_analysis": content_analysis,
                    "estimated_traffic": self._estimate_traffic(url),
                    "content_gaps": await self._identify_content_gaps_for_url(url)
                })
            except Exception as e:
                logger.warning(f"Failed to analyze competitor URL {url}: {e}")
                competitor_data.append({
                    "url": url,
                    "error": str(e)
                })
        
        # Aggregate insights
        all_keywords = []
        all_topics = []
        
        for comp in competitor_data:
            if "content_analysis" in comp:
                all_keywords.extend(comp["content_analysis"].get("keywords", []))
                all_topics.extend(comp["content_analysis"].get("topics", []))
        
        return {
            "competitors_analyzed": len(competitor_data),
            "competitor_data": competitor_data,
            "common_keywords": self._find_common_elements(all_keywords),
            "common_topics": self._find_common_elements(all_topics),
            "content_format_analysis": await self._analyze_content_formats(competitor_data),
            "competitive_landscape": await self._assess_competitive_landscape(competitor_data)
        }
    
    async def _research_search_trends(self, keywords: List[str], timeframe: str) -> Dict[str, Any]:
        """Research search trends and seasonal patterns."""
        trends_data = {}
        
        for keyword in keywords:
            # Mock trend data (in production, use Google Trends API or similar)
            trends_data[keyword] = {
                "trend_direction": self._calculate_trend_direction(keyword),
                "search_volume_trend": self._generate_trend_data(keyword, timeframe),
                "seasonal_score": self._calculate_seasonal_score(keyword),
                "growth_rate": self._calculate_growth_rate(keyword),
                "peak_months": self._identify_peak_months(keyword),
                "related_queries": await self._find_trending_related_queries(keyword)
            }
        
        return {
            "timeframe": timeframe,
            "keywords_analyzed": len(keywords),
            "trends_data": trends_data,
            "overall_trend": self._calculate_overall_trend(trends_data),
            "recommendations": await self._generate_trend_based_recommendations(trends_data)
        }
    
    async def _classify_search_intent(self, keywords: List[str]) -> Dict[str, Any]:
        """Classify search intent for keywords."""
        intent_classification = {}
        
        for keyword in keywords:
            intent = self._determine_search_intent(keyword)
            intent_classification[keyword] = {
                "primary_intent": intent["primary"],
                "secondary_intent": intent.get("secondary"),
                "confidence_score": intent["confidence"],
                "content_type_recommendation": self._get_content_type_for_intent(intent["primary"]),
                "optimization_focus": self._get_optimization_focus(intent["primary"]),
                "user_journey_stage": self._map_to_user_journey(intent["primary"])
            }
        
        # Aggregate insights
        intent_distribution = self._calculate_intent_distribution(intent_classification)
        
        return {
            "keywords_analyzed": len(keywords),
            "intent_classification": intent_classification,
            "intent_distribution": intent_distribution,
            "content_strategy": await self._generate_content_strategy_for_intents(intent_distribution),
            "priority_intents": self._prioritize_intents(intent_distribution)
        }
    
    async def _find_related_keywords(self, primary_keyword: str, depth: int) -> Dict[str, Any]:
        """Find related keywords and semantic variations."""
        related_keywords = {
            "synonyms": await self._find_synonyms(primary_keyword),
            "semantic_variations": await self._find_semantic_variations(primary_keyword),
            "topic_related": await self._find_topic_related_keywords(primary_keyword),
            "long_tail": await self._generate_long_tail_variations(primary_keyword),
            "question_based": await self._generate_question_keywords(primary_keyword),
            "modifier_variations": await self._generate_modifier_variations(primary_keyword)
        }
        
        if depth > 1:
            # Second level expansion
            second_level = {}
            for category, keywords in related_keywords.items():
                second_level[category] = []
                for keyword in keywords[:3]:  # Limit expansion
                    sub_related = await self._find_related_keywords(keyword, depth - 1)
                    second_level[category].extend(sub_related.get("synonyms", [])[:2])
            
            related_keywords["second_level"] = second_level
        
        return {
            "primary_keyword": primary_keyword,
            "expansion_depth": depth,
            "related_keywords": related_keywords,
            "total_keywords_found": sum(len(keywords) for keywords in related_keywords.values() if isinstance(keywords, list)),
            "keyword_clusters": await self._cluster_related_keywords(related_keywords)
        }
    
    async def _analyze_serp_features(self, keywords: List[str]) -> Dict[str, Any]:
        """Analyze SERP features and optimization opportunities."""
        serp_analysis = {}
        
        for keyword in keywords:
            # Mock SERP analysis (in production, use real SERP API)
            serp_analysis[keyword] = {
                "featured_snippet_opportunity": self._assess_featured_snippet_opportunity(keyword),
                "people_also_ask": await self._extract_people_also_ask(keyword),
                "related_searches": await self._extract_related_searches(keyword),
                "image_results": self._assess_image_optimization_opportunity(keyword),
                "video_results": self._assess_video_optimization_opportunity(keyword),
                "local_results": self._assess_local_seo_opportunity(keyword),
                "knowledge_panel": self._assess_knowledge_panel_opportunity(keyword),
                "shopping_results": self._assess_shopping_opportunity(keyword)
            }
        
        # Aggregate opportunities
        optimization_opportunities = self._aggregate_serp_opportunities(serp_analysis)
        
        return {
            "keywords_analyzed": len(keywords),
            "serp_analysis": serp_analysis,
            "optimization_opportunities": optimization_opportunities,
            "priority_features": self._prioritize_serp_features(optimization_opportunities),
            "implementation_recommendations": await self._generate_serp_recommendations(optimization_opportunities)
        }
    
    async def _estimate_keyword_difficulty(self, keywords: List[str]) -> Dict[str, Any]:
        """Estimate keyword difficulty and ranking opportunities."""
        difficulty_analysis = {}
        
        for keyword in keywords:
            difficulty_score = self._calculate_keyword_difficulty(keyword)
            difficulty_analysis[keyword] = {
                "difficulty_score": difficulty_score,
                "difficulty_level": self._get_difficulty_level(difficulty_score),
                "ranking_opportunity": self._assess_ranking_opportunity(keyword, difficulty_score),
                "content_requirements": self._estimate_content_requirements(keyword, difficulty_score),
                "estimated_time_to_rank": self._estimate_time_to_rank(difficulty_score),
                "competition_analysis": await self._analyze_keyword_competition(keyword)
            }
        
        return {
            "keywords_analyzed": len(keywords),
            "difficulty_analysis": difficulty_analysis,
            "average_difficulty": sum(data["difficulty_score"] for data in difficulty_analysis.values()) / len(difficulty_analysis),
            "quick_wins": [kw for kw, data in difficulty_analysis.items() if data["difficulty_score"] < 30],
            "long_term_targets": [kw for kw, data in difficulty_analysis.items() if data["difficulty_score"] > 70]
        }
    
    # Helper methods (many would be implemented with real APIs in production)
    
    async def _generate_seed_keywords(self, industry_context: str) -> List[str]:
        """Generate seed keywords from industry context."""
        # Simple keyword extraction from industry context
        words = re.findall(r'\b[a-zA-Z]{3,}\b', industry_context.lower())
        return list(set(words))[:10]
    
    async def _generate_keyword_variations(self, seed: str) -> List[str]:
        """Generate keyword variations."""
        variations = [seed]
        
        # Add common modifiers
        modifiers = ["best", "how to", "what is", "guide", "tips", "review", "comparison", "vs", "free", "online"]
        for modifier in modifiers:
            variations.extend([f"{modifier} {seed}", f"{seed} {modifier}"])
        
        # Add plural/singular variations
        if seed.endswith('s'):
            variations.append(seed[:-1])
        else:
            variations.append(f"{seed}s")
        
        return list(set(variations))
    
    def _estimate_search_volume(self, keyword: str) -> int:
        """Estimate search volume (mock implementation)."""
        # Mock estimation based on keyword length and common words
        base_volume = max(100, 2000 - len(keyword) * 50)
        return int(base_volume * (0.5 + hash(keyword) % 100 / 100))
    
    def _estimate_competition(self, keyword: str) -> float:
        """Estimate competition score (mock implementation)."""
        return min(1.0, max(0.1, (hash(keyword) % 100) / 100))
    
    def _estimate_difficulty(self, keyword: str) -> str:
        """Estimate keyword difficulty."""
        competition = self._estimate_competition(keyword)
        if competition < 0.3:
            return "easy"
        elif competition < 0.7:
            return "medium"
        else:
            return "hard"
    
    def _calculate_opportunity_score(self, volume: int, competition: float) -> float:
        """Calculate keyword opportunity score."""
        # Higher volume, lower competition = higher opportunity
        return min(100, (volume / 100) * (1 - competition) * 10)
    
    async def _analyze_competitor_url(self, url: str) -> Dict[str, Any]:
        """Analyze a competitor URL (mock implementation)."""
        return {
            "title": f"Title for {url}",
            "word_count": 1500,
            "keywords": ["keyword1", "keyword2", "keyword3"],
            "topics": ["topic1", "topic2"],
            "headers": ["H1", "H2", "H3"],
            "estimated_seo_score": 75
        }
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        import urllib.parse
        return urllib.parse.urlparse(url).netloc
    
    def _estimate_traffic(self, url: str) -> int:
        """Estimate traffic for URL (mock implementation)."""
        return hash(url) % 10000 + 1000
    
    async def _identify_content_gaps_for_url(self, url: str) -> List[str]:
        """Identify content gaps for a specific URL."""
        return ["gap1", "gap2", "gap3"]
    
    def _find_common_elements(self, elements: List[str]) -> List[str]:
        """Find common elements in a list."""
        from collections import Counter
        counter = Counter(elements)
        return [element for element, count in counter.most_common(10)]
    
    async def _analyze_content_formats(self, competitor_data: List[Dict]) -> Dict[str, Any]:
        """Analyze content formats used by competitors."""
        return {
            "common_formats": ["blog_post", "guide", "listicle"],
            "word_count_average": 1500,
            "header_structure": "H1-H2-H3 hierarchy"
        }
    
    async def _assess_competitive_landscape(self, competitor_data: List[Dict]) -> Dict[str, Any]:
        """Assess the overall competitive landscape."""
        return {
            "competition_level": "medium",
            "market_saturation": 0.7,
            "differentiation_opportunities": ["unique_angle1", "unique_angle2"]
        }
    
    def _calculate_trend_direction(self, keyword: str) -> str:
        """Calculate trend direction (mock implementation)."""
        directions = ["rising", "stable", "declining"]
        return directions[hash(keyword) % 3]
    
    def _generate_trend_data(self, keyword: str, timeframe: str) -> List[int]:
        """Generate mock trend data."""
        import random
        return [random.randint(50, 150) for _ in range(12)]
    
    def _calculate_seasonal_score(self, keyword: str) -> float:
        """Calculate seasonal score."""
        return (hash(keyword) % 100) / 100
    
    def _calculate_growth_rate(self, keyword: str) -> float:
        """Calculate growth rate."""
        return ((hash(keyword) % 40) - 20) / 100  # -20% to +20%
    
    def _identify_peak_months(self, keyword: str) -> List[str]:
        """Identify peak months."""
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        peak_count = (hash(keyword) % 3) + 1
        return months[:peak_count]
    
    async def _find_trending_related_queries(self, keyword: str) -> List[str]:
        """Find trending related queries."""
        return [f"{keyword} trending", f"best {keyword}", f"{keyword} 2024"]
    
    def _calculate_overall_trend(self, trends_data: Dict) -> str:
        """Calculate overall trend across all keywords."""
        rising_count = sum(1 for data in trends_data.values() if data["trend_direction"] == "rising")
        total_count = len(trends_data)
        
        if rising_count / total_count > 0.6:
            return "positive"
        elif rising_count / total_count < 0.3:
            return "negative"
        else:
            return "stable"
    
    async def _generate_trend_based_recommendations(self, trends_data: Dict) -> List[str]:
        """Generate recommendations based on trend analysis."""
        return [
            "Focus on rising trend keywords",
            "Create seasonal content calendar",
            "Monitor declining keywords for pivot opportunities"
        ]
    
    def _determine_search_intent(self, keyword: str) -> Dict[str, Any]:
        """Determine search intent for keyword."""
        # Simple intent classification based on keyword patterns
        keyword_lower = keyword.lower()
        
        if any(word in keyword_lower for word in ["how to", "what is", "why", "guide", "tutorial"]):
            return {"primary": "informational", "confidence": 0.9}
        elif any(word in keyword_lower for word in ["buy", "price", "cost", "cheap", "deal", "review"]):
            return {"primary": "commercial", "confidence": 0.8}
        elif any(word in keyword_lower for word in ["best", "top", "compare", "vs"]):
            return {"primary": "commercial", "secondary": "informational", "confidence": 0.7}
        else:
            return {"primary": "informational", "confidence": 0.6}
    
    def _get_content_type_for_intent(self, intent: str) -> str:
        """Get recommended content type for search intent."""
        intent_mapping = {
            "informational": "blog_post",
            "commercial": "comparison_page",
            "transactional": "product_page",
            "navigational": "landing_page"
        }
        return intent_mapping.get(intent, "blog_post")
    
    def _get_optimization_focus(self, intent: str) -> str:
        """Get optimization focus for search intent."""
        focus_mapping = {
            "informational": "content_depth",
            "commercial": "trust_signals",
            "transactional": "conversion_optimization",
            "navigational": "brand_clarity"
        }
        return focus_mapping.get(intent, "content_quality")
    
    def _map_to_user_journey(self, intent: str) -> str:
        """Map search intent to user journey stage."""
        journey_mapping = {
            "informational": "awareness",
            "commercial": "consideration",
            "transactional": "decision",
            "navigational": "retention"
        }
        return journey_mapping.get(intent, "awareness")
    
    def _calculate_intent_distribution(self, intent_classification: Dict) -> Dict[str, float]:
        """Calculate distribution of search intents."""
        intents = [data["primary_intent"] for data in intent_classification.values()]
        total = len(intents)
        
        distribution = {}
        for intent in set(intents):
            distribution[intent] = intents.count(intent) / total
        
        return distribution
    
    async def _generate_content_strategy_for_intents(self, intent_distribution: Dict) -> Dict[str, Any]:
        """Generate content strategy based on intent distribution."""
        return {
            "primary_focus": max(intent_distribution, key=intent_distribution.get),
            "content_mix": intent_distribution,
            "recommendations": [
                f"Prioritize {intent} content ({percentage:.1%})" 
                for intent, percentage in intent_distribution.items()
            ]
        }
    
    def _prioritize_intents(self, intent_distribution: Dict) -> List[str]:
        """Prioritize intents by distribution."""
        return sorted(intent_distribution.keys(), key=intent_distribution.get, reverse=True)
    
    # Additional helper methods for comprehensive functionality
    
    async def _find_synonyms(self, keyword: str) -> List[str]:
        """Find synonyms for keyword."""
        # Mock synonym finding
        return [f"{keyword}_synonym1", f"{keyword}_synonym2"]
    
    async def _find_semantic_variations(self, keyword: str) -> List[str]:
        """Find semantic variations."""
        return [f"{keyword}_variation1", f"{keyword}_variation2"]
    
    async def _find_topic_related_keywords(self, keyword: str) -> List[str]:
        """Find topic-related keywords."""
        return [f"{keyword}_related1", f"{keyword}_related2"]
    
    async def _generate_long_tail_variations(self, keyword: str) -> List[str]:
        """Generate long-tail keyword variations."""
        return [f"best {keyword} for beginners", f"{keyword} step by step guide"]
    
    async def _generate_question_keywords(self, keyword: str) -> List[str]:
        """Generate question-based keywords."""
        return [f"what is {keyword}", f"how to use {keyword}", f"why {keyword}"]
    
    async def _generate_modifier_variations(self, keyword: str) -> List[str]:
        """Generate modifier variations."""
        modifiers = ["free", "best", "cheap", "professional", "advanced"]
        return [f"{modifier} {keyword}" for modifier in modifiers]
    
    async def _cluster_related_keywords(self, related_keywords: Dict) -> List[Dict[str, Any]]:
        """Cluster related keywords by topic."""
        return [
            {"cluster": "main_topic", "keywords": ["keyword1", "keyword2"]},
            {"cluster": "sub_topic", "keywords": ["keyword3", "keyword4"]}
        ]
    
    def _assess_featured_snippet_opportunity(self, keyword: str) -> Dict[str, Any]:
        """Assess featured snippet opportunity."""
        return {
            "opportunity_score": (hash(keyword) % 100) / 100,
            "current_snippet_type": "paragraph",
            "optimization_recommendation": "Create structured content with clear answers"
        }
    
    async def _extract_people_also_ask(self, keyword: str) -> List[str]:
        """Extract People Also Ask questions."""
        return [f"What is {keyword}?", f"How does {keyword} work?", f"Why use {keyword}?"]
    
    async def _extract_related_searches(self, keyword: str) -> List[str]:
        """Extract related searches."""
        return [f"{keyword} guide", f"{keyword} tips", f"{keyword} benefits"]
    
    def _assess_image_optimization_opportunity(self, keyword: str) -> Dict[str, Any]:
        """Assess image optimization opportunity."""
        return {"opportunity_score": 0.7, "recommended_images": 3}
    
    def _assess_video_optimization_opportunity(self, keyword: str) -> Dict[str, Any]:
        """Assess video optimization opportunity."""
        return {"opportunity_score": 0.5, "recommended_video_type": "tutorial"}
    
    def _assess_local_seo_opportunity(self, keyword: str) -> Dict[str, Any]:
        """Assess local SEO opportunity."""
        return {"opportunity_score": 0.3, "local_intent": False}
    
    def _assess_knowledge_panel_opportunity(self, keyword: str) -> Dict[str, Any]:
        """Assess knowledge panel opportunity."""
        return {"opportunity_score": 0.2, "entity_optimization_needed": True}
    
    def _assess_shopping_opportunity(self, keyword: str) -> Dict[str, Any]:
        """Assess shopping results opportunity."""
        return {"opportunity_score": 0.4, "commercial_intent": True}
    
    def _aggregate_serp_opportunities(self, serp_analysis: Dict) -> Dict[str, Any]:
        """Aggregate SERP optimization opportunities."""
        return {
            "featured_snippets": sum(1 for data in serp_analysis.values() if data["featured_snippet_opportunity"]["opportunity_score"] > 0.7),
            "image_optimization": sum(1 for data in serp_analysis.values() if data["image_results"]["opportunity_score"] > 0.6),
            "video_optimization": sum(1 for data in serp_analysis.values() if data["video_results"]["opportunity_score"] > 0.6)
        }
    
    def _prioritize_serp_features(self, opportunities: Dict) -> List[str]:
        """Prioritize SERP features by opportunity."""
        return ["featured_snippets", "image_optimization", "video_optimization"]
    
    async def _generate_serp_recommendations(self, opportunities: Dict) -> List[str]:
        """Generate SERP optimization recommendations."""
        return [
            "Optimize for featured snippets with structured content",
            "Add relevant images with optimized alt text",
            "Consider creating video content for high-opportunity keywords"
        ]
    
    def _calculate_keyword_difficulty(self, keyword: str) -> int:
        """Calculate keyword difficulty score (0-100)."""
        return hash(keyword) % 100
    
    def _get_difficulty_level(self, score: int) -> str:
        """Get difficulty level from score."""
        if score < 30:
            return "easy"
        elif score < 60:
            return "medium"
        else:
            return "hard"
    
    def _assess_ranking_opportunity(self, keyword: str, difficulty: int) -> str:
        """Assess ranking opportunity."""
        if difficulty < 40:
            return "high"
        elif difficulty < 70:
            return "medium"
        else:
            return "low"
    
    def _estimate_content_requirements(self, keyword: str, difficulty: int) -> Dict[str, Any]:
        """Estimate content requirements for ranking."""
        return {
            "min_word_count": 500 + (difficulty * 10),
            "recommended_headers": max(3, difficulty // 20),
            "backlinks_needed": difficulty // 10
        }
    
    def _estimate_time_to_rank(self, difficulty: int) -> str:
        """Estimate time to rank based on difficulty."""
        if difficulty < 30:
            return "1-3 months"
        elif difficulty < 60:
            return "3-6 months"
        else:
            return "6-12 months"
    
    async def _analyze_keyword_competition(self, keyword: str) -> Dict[str, Any]:
        """Analyze competition for keyword."""
        return {
            "top_competitor_domains": ["competitor1.com", "competitor2.com"],
            "average_domain_authority": 60,
            "content_type_dominance": "blog_posts"
        }
    
    async def _prioritize_keywords(self, opportunities: Dict, seo_preferences: Dict) -> List[Dict[str, Any]]:
        """Prioritize keywords based on opportunities and preferences."""
        high_opp = opportunities.get("high_opportunity", [])
        return sorted(high_opp, key=lambda x: x["opportunity_score"], reverse=True)[:10]
    
    async def _generate_content_recommendations(self, keyword_opportunities: Dict, intent_analysis: Dict) -> List[str]:
        """Generate content recommendations based on research."""
        return [
            "Create comprehensive pillar content for high-volume keywords",
            "Develop cluster content for long-tail variations",
            "Optimize existing content for featured snippet opportunities"
        ]
    
    async def _identify_content_gaps(self, competitor_urls: List[str], target_keywords: List[str]) -> Dict[str, Any]:
        """Identify content gaps compared to competitors."""
        return {
            "missing_topics": ["topic1", "topic2"],
            "underoptimized_keywords": target_keywords[:3],
            "content_format_gaps": ["video", "infographic"]
        }
    
    async def _find_competitive_opportunities(self, competitor_analysis: Dict) -> List[str]:
        """Find competitive opportunities."""
        return [
            "Target competitor weak points",
            "Create better content on shared topics",
            "Identify uncontested keyword opportunities"
        ]
    
    async def _identify_seasonal_patterns(self, keywords: List[str]) -> Dict[str, Any]:
        """Identify seasonal patterns in keywords."""
        return {
            "seasonal_keywords": keywords[:2],
            "peak_seasons": ["Q4", "Summer"],
            "content_calendar_recommendations": ["Plan holiday content", "Prepare summer guides"]
        }
    
    async def _identify_emerging_keywords(self, keywords: List[str]) -> List[str]:
        """Identify emerging keyword opportunities."""
        return [f"{keyword} 2024" for keyword in keywords[:3]]
    
    async def _generate_trend_recommendations(self, trend_analysis: Dict) -> List[str]:
        """Generate recommendations based on trend analysis."""
        return [
            "Capitalize on rising trend keywords",
            "Create evergreen content for stable trends",
            "Monitor and pivot on declining trends"
        ]
    
    async def _map_content_to_intent(self, intent_analysis: Dict) -> Dict[str, List[str]]:
        """Map content types to search intents."""
        content_mapping = {}
        for keyword, data in intent_analysis["intent_classification"].items():
            intent = data["primary_intent"]
            content_type = data["content_type_recommendation"]
            
            if intent not in content_mapping:
                content_mapping[intent] = []
            content_mapping[intent].append(f"{keyword} -> {content_type}")
        
        return content_mapping
    
    async def _generate_intent_strategies(self, intent_analysis: Dict) -> Dict[str, List[str]]:
        """Generate optimization strategies for each intent."""
        return {
            "informational": ["Create comprehensive guides", "Optimize for featured snippets"],
            "commercial": ["Add comparison tables", "Include trust signals"],
            "transactional": ["Optimize for conversions", "Add clear CTAs"]
        }
    
    async def _find_long_tail_opportunities(self, seed_keywords: List[str]) -> List[Dict[str, Any]]:
        """Find long-tail keyword opportunities."""
        long_tail = []
        for keyword in seed_keywords:
            long_tail.extend([
                {"keyword": f"best {keyword} for beginners", "estimated_volume": 200, "difficulty": "easy"},
                {"keyword": f"how to choose {keyword}", "estimated_volume": 150, "difficulty": "easy"},
                {"keyword": f"{keyword} vs alternatives", "estimated_volume": 100, "difficulty": "medium"}
            ])
        return long_tail
    
    async def _find_question_keywords(self, seed_keywords: List[str]) -> List[Dict[str, Any]]:
        """Find question-based keywords."""
        questions = []
        question_words = ["what", "how", "why", "when", "where", "which"]
        
        for keyword in seed_keywords:
            for qword in question_words:
                questions.append({
                    "keyword": f"{qword} is {keyword}",
                    "intent": "informational",
                    "estimated_volume": 100
                })
        
        return questions[:20]  # Limit results


# Register the agent
seo_research_agent = SEOResearchAgent()
agent_registry.register(seo_research_agent)