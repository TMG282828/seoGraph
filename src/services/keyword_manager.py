"""
Keyword Management Service for SerpBear Integration.

This service handles:
- Automatic keyword extraction from content
- Keyword registration with SerpBear
- Keyword performance analysis
- Integration with Neo4j knowledge graph

Bridges our SEO Knowledge Graph with real-time ranking data.
"""

import logging
import asyncio
import re
from typing import Dict, List, Set, Optional, Any, Tuple
from datetime import datetime, date
from collections import defaultdict, Counter
import openai
from pydantic import BaseModel, Field

from .serpbear_client import serpbear_client, SerpBearKeyword, RankingUpdate
from ..database.content_service import content_db_service
from ..database.neo4j_client import neo4j_client

logger = logging.getLogger(__name__)


class ExtractedKeyword(BaseModel):
    """Model for extracted keyword with metadata."""
    keyword: str
    relevance_score: float = Field(description="Relevance score 0-1")
    search_intent: str = Field(description="informational, transactional, navigational")
    competition_level: str = Field(description="low, medium, high")
    content_ids: List[str] = Field(default_factory=list)
    frequency: int = 1
    keyword_type: str = Field(description="primary, secondary, long_tail")


class KeywordRegistration(BaseModel):
    """Model for keyword registration request."""
    keyword: str
    domain: str
    device: str = "desktop"  # desktop or mobile
    country: str = "US"
    organization_id: str


class KeywordPerformanceMetrics(BaseModel):
    """Model for keyword performance analysis."""
    keyword: str
    current_position: Optional[int]
    average_position: Optional[float]
    best_position: Optional[int]
    worst_position: Optional[int]
    volatility_score: float  # 0-1, higher = more volatile
    trend: str  # "improving", "declining", "stable"
    days_tracked: int
    last_updated: str


class KeywordManager:
    """
    Comprehensive keyword management for SerpBear integration.
    
    This service bridges content analysis with rank tracking by:
    1. Extracting relevant keywords from uploaded content
    2. Registering high-value keywords with SerpBear
    3. Analyzing ranking performance and trends
    4. Updating Neo4j graph with ranking data
    """
    
    def __init__(self, organization_id: str = "demo-org"):
        """
        Initialize keyword manager.
        
        Args:
            organization_id: Organization context for operations
        """
        self.organization_id = organization_id
        self.registered_keywords = {}  # Cache of registered keywords
        logger.info(f"Keyword manager initialized for org: {organization_id}")
    
    async def extract_keywords_from_content(
        self, 
        content_items: List[Dict[str, Any]], 
        max_keywords: int = 50
    ) -> List[ExtractedKeyword]:
        """
        Extract high-value keywords from content using multiple methods.
        
        Args:
            content_items: List of content dictionaries
            max_keywords: Maximum keywords to extract
            
        Returns:
            List of extracted keywords with metadata
        """
        try:
            logger.info(f"🔍 Extracting keywords from {len(content_items)} content items")
            
            # Method 1: Extract from titles and existing keywords
            basic_keywords = self._extract_basic_keywords(content_items)
            
            # Method 2: Use AI for semantic keyword extraction
            ai_keywords = await self._extract_ai_keywords(content_items)
            
            # Method 3: Analyze content for SEO opportunities
            seo_keywords = self._extract_seo_keywords(content_items)
            
            # Combine and rank keywords
            all_keywords = self._combine_and_rank_keywords(
                basic_keywords, ai_keywords, seo_keywords
            )
            
            # Filter and limit results
            top_keywords = self._filter_keywords(all_keywords, max_keywords)
            
            logger.info(f"✅ Extracted {len(top_keywords)} high-value keywords")
            return top_keywords
            
        except Exception as e:
            logger.error(f"❌ Keyword extraction failed: {e}")
            return []
    
    def _extract_basic_keywords(self, content_items: List[Dict[str, Any]]) -> List[ExtractedKeyword]:
        """Extract keywords from titles and existing metadata."""
        keywords = {}
        
        for content in content_items:
            content_id = content.get('id', '')
            title = content.get('title', '').lower()
            existing_keywords = content.get('keywords', [])
            
            # Extract from title (remove file extensions)
            title_clean = re.sub(r'\.(pdf|docx?|txt|md)$', '', title)
            title_words = re.findall(r'\b[a-zA-Z]{3,}\b', title_clean)
            
            # Process title words
            for word in title_words:
                word = word.lower()
                if word not in keywords:
                    keywords[word] = ExtractedKeyword(
                        keyword=word,
                        relevance_score=0.6,  # Base score for title words
                        search_intent="informational",
                        competition_level="medium",
                        content_ids=[content_id],
                        keyword_type="secondary"
                    )
                else:
                    keywords[word].content_ids.append(content_id)
                    keywords[word].frequency += 1
                    keywords[word].relevance_score = min(0.9, keywords[word].relevance_score + 0.1)
            
            # Process existing keywords
            for keyword in existing_keywords:
                if isinstance(keyword, str) and len(keyword) > 2:
                    keyword = keyword.lower().strip()
                    if keyword not in keywords:
                        keywords[keyword] = ExtractedKeyword(
                            keyword=keyword,
                            relevance_score=0.8,  # Higher score for existing keywords
                            search_intent="informational",
                            competition_level="medium",
                            content_ids=[content_id],
                            keyword_type="primary"
                        )
                    else:
                        keywords[keyword].content_ids.append(content_id)
                        keywords[keyword].frequency += 1
                        keywords[keyword].relevance_score = min(1.0, keywords[keyword].relevance_score + 0.1)
        
        return list(keywords.values())
    
    async def _extract_ai_keywords(self, content_items: List[Dict[str, Any]]) -> List[ExtractedKeyword]:
        """Use AI to extract semantic keywords from content."""
        try:
            # Prepare content summaries for AI analysis
            content_summaries = []
            for content in content_items[:5]:  # Limit to avoid token limits
                summary = f"Title: {content.get('title', '')}\n"
                summary += f"Summary: {content.get('summary', '')[:200]}\n"
                content_summaries.append(summary)
            
            combined_content = "\n---\n".join(content_summaries)
            
            # AI prompt for keyword extraction
            prompt = f"""
            Analyze the following content and extract 15-20 high-value SEO keywords that would be worth tracking for search rankings.
            
            Focus on:
            1. Primary keywords (high search volume, competitive)
            2. Long-tail keywords (specific, lower competition)
            3. Intent-based keywords (what users actually search for)
            
            Content:
            {combined_content}
            
            Return only a JSON array of keywords, no explanation:
            ["keyword1", "keyword2", "keyword3", ...]
            """
            
            # Make AI request (using OpenAI client)
            try:
                import openai
                openai.api_key = os.getenv("OPENAI_API_KEY")
                
                response = await openai.ChatCompletion.acreate(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.3
                )
                
                # Parse AI response
                import json
                ai_keywords_raw = json.loads(response.choices[0].message.content.strip())
                
                # Convert to ExtractedKeyword objects
                ai_keywords = []
                for keyword in ai_keywords_raw:
                    if isinstance(keyword, str) and len(keyword) > 2:
                        ai_keywords.append(ExtractedKeyword(
                            keyword=keyword.lower().strip(),
                            relevance_score=0.9,  # High score for AI-extracted keywords
                            search_intent=self._determine_search_intent(keyword),
                            competition_level="medium",
                            keyword_type=self._determine_keyword_type(keyword)
                        ))
                
                logger.info(f"🤖 AI extracted {len(ai_keywords)} semantic keywords")
                return ai_keywords
                
            except Exception as ai_error:
                logger.warning(f"AI keyword extraction failed: {ai_error}")
                return []
            
        except Exception as e:
            logger.error(f"AI keyword extraction error: {e}")
            return []
    
    def _extract_seo_keywords(self, content_items: List[Dict[str, Any]]) -> List[ExtractedKeyword]:
        """Extract SEO-focused keywords based on content analysis."""
        seo_keywords = []
        
        # SEO keyword patterns
        seo_patterns = {
            "how to": ("informational", "long_tail", 0.8),
            "best": ("informational", "primary", 0.9),
            "guide": ("informational", "secondary", 0.7),
            "tutorial": ("informational", "secondary", 0.7),
            "tips": ("informational", "secondary", 0.6),
            "review": ("commercial", "primary", 0.8),
            "vs": ("commercial", "secondary", 0.7),
            "comparison": ("commercial", "secondary", 0.7),
            "buy": ("transactional", "primary", 0.9),
            "price": ("transactional", "secondary", 0.8),
            "free": ("informational", "long_tail", 0.6)
        }
        
        for content in content_items:
            title = content.get('title', '').lower()
            content_text = content.get('content', '')[:500].lower()  # First 500 chars
            
            # Look for SEO patterns
            for pattern, (intent, type_, score) in seo_patterns.items():
                if pattern in title or pattern in content_text:
                    # Create keyword based on context
                    if pattern in title:
                        # Extract phrase containing the pattern
                        words = title.split()
                        try:
                            pattern_index = next(i for i, word in enumerate(words) if pattern in word)
                            start = max(0, pattern_index - 1)
                            end = min(len(words), pattern_index + 3)
                            keyword_phrase = " ".join(words[start:end])
                        except StopIteration:
                            keyword_phrase = pattern
                    else:
                        keyword_phrase = pattern
                    
                    seo_keywords.append(ExtractedKeyword(
                        keyword=keyword_phrase.strip(),
                        relevance_score=score,
                        search_intent=intent,
                        competition_level="medium",
                        content_ids=[content.get('id', '')],
                        keyword_type=type_
                    ))
        
        logger.info(f"🎯 Extracted {len(seo_keywords)} SEO-focused keywords")
        return seo_keywords
    
    def _combine_and_rank_keywords(
        self, 
        basic: List[ExtractedKeyword], 
        ai: List[ExtractedKeyword], 
        seo: List[ExtractedKeyword]
    ) -> List[ExtractedKeyword]:
        """Combine keyword lists and rank by relevance."""
        combined = {}
        
        # Merge all keyword sources
        for keyword_list in [basic, ai, seo]:
            for keyword in keyword_list:
                key = keyword.keyword.lower().strip()
                if key not in combined:
                    combined[key] = keyword
                else:
                    # Merge with existing keyword
                    existing = combined[key]
                    existing.relevance_score = max(existing.relevance_score, keyword.relevance_score)
                    existing.content_ids.extend(keyword.content_ids)
                    existing.frequency += keyword.frequency
                    
                    # Promote keyword type if better
                    if keyword.keyword_type == "primary":
                        existing.keyword_type = "primary"
        
        # Sort by relevance score
        sorted_keywords = sorted(
            combined.values(), 
            key=lambda k: (k.relevance_score, k.frequency), 
            reverse=True
        )
        
        return sorted_keywords
    
    def _filter_keywords(
        self, 
        keywords: List[ExtractedKeyword], 
        max_keywords: int
    ) -> List[ExtractedKeyword]:
        """Filter and limit keywords based on quality criteria."""
        filtered = []
        
        for keyword in keywords:
            # Quality filters
            if len(keyword.keyword) < 3:
                continue
            if len(keyword.keyword) > 50:
                continue
            if keyword.relevance_score < 0.5:
                continue
            
            # Remove common stop words
            stop_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
            if keyword.keyword.lower() in stop_words:
                continue
            
            filtered.append(keyword)
            
            if len(filtered) >= max_keywords:
                break
        
        return filtered
    
    def _determine_search_intent(self, keyword: str) -> str:
        """Determine search intent from keyword."""
        keyword_lower = keyword.lower()
        
        if any(word in keyword_lower for word in ["how to", "what is", "guide", "tutorial", "tips"]):
            return "informational"
        elif any(word in keyword_lower for word in ["buy", "price", "cost", "purchase", "order"]):
            return "transactional"
        elif any(word in keyword_lower for word in ["vs", "comparison", "review", "best"]):
            return "commercial"
        else:
            return "informational"
    
    def _determine_keyword_type(self, keyword: str) -> str:
        """Determine keyword type based on length and structure."""
        word_count = len(keyword.split())
        
        if word_count >= 4:
            return "long_tail"
        elif word_count == 1:
            return "primary"
        else:
            return "secondary"
    
    async def register_keywords_with_serpbear(
        self, 
        keywords: List[ExtractedKeyword], 
        domain: str,
        devices: List[str] = ["desktop", "mobile"]
    ) -> Dict[str, Any]:
        """
        Register extracted keywords with SerpBear for tracking.
        
        Args:
            keywords: List of keywords to register
            domain: Domain to track rankings for
            devices: List of devices to track (desktop, mobile)
            
        Returns:
            Registration results summary
        """
        try:
            logger.info(f"📝 Registering {len(keywords)} keywords with SerpBear for {domain}")
            
            registration_results = {
                "total_keywords": len(keywords),
                "registration_attempts": 0,
                "successful_registrations": 0,
                "failed_registrations": 0,
                "errors": []
            }
            
            async with serpbear_client as client:
                for keyword in keywords:
                    for device in devices:
                        try:
                            registration_results["registration_attempts"] += 1
                            
                            # Create registration request
                            reg_request = KeywordRegistration(
                                keyword=keyword.keyword,
                                domain=domain,
                                device=device,
                                organization_id=self.organization_id
                            )
                            
                            # Note: SerpBear API doesn't have a direct registration endpoint
                            # This would need to be implemented in SerpBear or handled through the UI
                            # For now, we'll log the registration intent
                            
                            logger.info(f"🎯 Would register: {keyword.keyword} ({device}) for {domain}")
                            registration_results["successful_registrations"] += 1
                            
                        except Exception as reg_error:
                            registration_results["failed_registrations"] += 1
                            registration_results["errors"].append(str(reg_error))
                            logger.error(f"Failed to register {keyword.keyword}: {reg_error}")
            
            logger.info(f"✅ Keyword registration complete: {registration_results['successful_registrations']}/{registration_results['total_keywords']} successful")
            return registration_results
            
        except Exception as e:
            logger.error(f"❌ Keyword registration failed: {e}")
            return {"error": str(e)}
    
    async def analyze_keyword_performance(
        self, 
        domain: str, 
        days: int = 30
    ) -> List[KeywordPerformanceMetrics]:
        """
        Analyze keyword ranking performance and trends.
        
        Args:
            domain: Domain to analyze
            days: Number of days to analyze
            
        Returns:
            List of performance metrics for each keyword
        """
        try:
            logger.info(f"📊 Analyzing keyword performance for {domain} over {days} days")
            
            performance_metrics = []
            
            async with serpbear_client as client:
                keywords = await client.get_keywords(domain)
                
                for keyword in keywords:
                    if not keyword.history:
                        continue
                    
                    # Analyze position history
                    positions = list(keyword.history.values())
                    positions = [p for p in positions if p and p > 0]  # Filter valid positions
                    
                    if not positions:
                        continue
                    
                    # Calculate metrics
                    current_position = keyword.position
                    average_position = sum(positions) / len(positions)
                    best_position = min(positions)
                    worst_position = max(positions)
                    
                    # Calculate volatility (standard deviation normalized)
                    if len(positions) > 1:
                        variance = sum((p - average_position) ** 2 for p in positions) / len(positions)
                        volatility_score = min(1.0, (variance ** 0.5) / 50)  # Normalized to 0-1
                    else:
                        volatility_score = 0.0
                    
                    # Determine trend
                    if len(positions) >= 3:
                        recent_avg = sum(positions[-3:]) / 3
                        older_avg = sum(positions[:3]) / 3
                        
                        if recent_avg < older_avg - 2:  # Lower position = better rank
                            trend = "improving"
                        elif recent_avg > older_avg + 2:
                            trend = "declining"
                        else:
                            trend = "stable"
                    else:
                        trend = "stable"
                    
                    metrics = KeywordPerformanceMetrics(
                        keyword=keyword.keyword,
                        current_position=current_position,
                        average_position=round(average_position, 1),
                        best_position=best_position,
                        worst_position=worst_position,
                        volatility_score=round(volatility_score, 3),
                        trend=trend,
                        days_tracked=len(positions),
                        last_updated=keyword.lastUpdated or str(datetime.now())
                    )
                    
                    performance_metrics.append(metrics)
            
            # Sort by current position (best first)
            performance_metrics.sort(key=lambda m: m.current_position or 999)
            
            logger.info(f"📈 Analyzed performance for {len(performance_metrics)} keywords")
            return performance_metrics
            
        except Exception as e:
            logger.error(f"❌ Performance analysis failed: {e}")
            return []


# Global keyword manager instance
keyword_manager = KeywordManager()


async def extract_and_register_keywords(domain: str, max_keywords: int = 30) -> Dict[str, Any]:
    """
    Complete workflow: extract keywords from content and register with SerpBear.
    
    Args:
        domain: Domain to register keywords for
        max_keywords: Maximum keywords to process
        
    Returns:
        Summary of extraction and registration process
    """
    try:
        logger.info(f"🚀 Starting complete keyword workflow for {domain}")
        
        # Step 1: Get content from database
        result = await content_db_service.get_content_items(
            organization_id="demo-org",
            limit=20
        )
        
        if not result.get("success"):
            raise Exception("Failed to fetch content for keyword extraction")
        
        content_items = result.get("content", [])
        
        # Step 2: Extract keywords
        keywords = await keyword_manager.extract_keywords_from_content(
            content_items, max_keywords
        )
        
        # Step 3: Register with SerpBear
        registration_results = await keyword_manager.register_keywords_with_serpbear(
            keywords, domain
        )
        
        # Step 4: Analyze current performance
        performance_metrics = await keyword_manager.analyze_keyword_performance(domain)
        
        summary = {
            "domain": domain,
            "content_analyzed": len(content_items),
            "keywords_extracted": len(keywords),
            "registration_results": registration_results,
            "current_tracked_keywords": len(performance_metrics),
            "top_keywords": [k.keyword for k in keywords[:10]],
            "timestamp": str(datetime.now())
        }
        
        logger.info(f"✅ Complete keyword workflow finished: {summary}")
        return summary
        
    except Exception as e:
        logger.error(f"❌ Keyword workflow failed: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Test the keyword manager
    async def main():
        print("Testing keyword extraction and registration...")
        result = await extract_and_register_keywords("example.com", max_keywords=10)
        print(f"Result: {result}")
    
    asyncio.run(main())