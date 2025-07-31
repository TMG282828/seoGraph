"""
Content analysis functionality for competitor analysis.

Analyzes competitor content strategies, quality patterns, engagement metrics,
and publishing behaviors to extract strategic insights.
"""

import asyncio
import re
from datetime import datetime
from typing import Any, Dict, List
from urllib.parse import urlparse

import structlog
from cachetools import TTLCache

from ...services.searxng_service import SearXNGService
from ...services.embedding_service import EmbeddingService
from .models import CompetitorAnalysisError

logger = structlog.get_logger(__name__)


class ContentAnalyzer:
    """
    Analyzes competitor content for strategic insights.
    
    Provides comprehensive analysis of competitor content strategies including:
    - Content type distribution and patterns
    - Topic analysis and focus areas
    - Content quality assessment
    - Publishing patterns and frequency
    - SEO optimization patterns
    - Engagement indicators
    """
    
    def __init__(self, searxng_service: SearXNGService, embedding_service: EmbeddingService):
        """
        Initialize ContentAnalyzer.
        
        Args:
            searxng_service: Service for web search operations
            embedding_service: Service for text embeddings and similarity
        """
        self.searxng_service = searxng_service
        self.embedding_service = embedding_service
        self.cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour cache
        
    async def analyze_competitor_content_strategy(self, 
                                                domain: str,
                                                sample_size: int = 50) -> Dict[str, Any]:
        """
        Analyze competitor's content strategy.
        
        Args:
            domain: Competitor domain to analyze
            sample_size: Number of content pieces to analyze
            
        Returns:
            Comprehensive content strategy analysis
        """
        try:
            cache_key = f"content_strategy_{domain}_{sample_size}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Get competitor content sample
            content_sample = await self._get_competitor_content_sample(domain, sample_size)
            
            if not content_sample:
                return {'error': f'No content found for {domain}'}
            
            # Analyze content patterns
            content_types = await self._analyze_content_types(content_sample)
            topic_distribution = await self._analyze_topic_distribution(content_sample)
            content_quality = await self._analyze_content_quality(content_sample)
            publishing_patterns = await self._analyze_publishing_patterns(content_sample)
            
            # SEO analysis
            seo_patterns = await self._analyze_seo_patterns(content_sample)
            
            # Engagement patterns
            engagement_patterns = await self._analyze_engagement_patterns(content_sample)
            
            strategy_analysis = {
                'domain': domain,
                'content_sample_size': len(content_sample),
                'content_types': content_types,
                'topic_distribution': topic_distribution,
                'content_quality': content_quality,
                'publishing_patterns': publishing_patterns,
                'seo_patterns': seo_patterns,
                'engagement_patterns': engagement_patterns,
                'strategic_insights': await self._generate_strategy_insights(
                    content_types, topic_distribution, content_quality, publishing_patterns
                )
            }
            
            # Cache results
            self.cache[cache_key] = strategy_analysis
            
            return strategy_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze content strategy for {domain}: {e}")
            raise CompetitorAnalysisError(f"Content strategy analysis failed: {e}")
    
    async def _get_competitor_content_sample(self, domain: str, sample_size: int) -> List[Dict[str, Any]]:
        """Get a sample of competitor content for analysis."""
        try:
            # Search for content on competitor domain
            queries = [
                f"site:{domain} blog",
                f"site:{domain} articles",
                f"site:{domain} news",
                f"site:{domain} resources",
                f"site:{domain} guides"
            ]
            
            all_content = []
            
            for query in queries:
                try:
                    results = await self.searxng_service.search(
                        query=query,
                        engines=['google', 'bing'],
                        safesearch=1,
                        time_range='month'
                    )
                    
                    for result in results[:20]:  # Limit per query
                        content_item = {
                            'url': result.get('url', ''),
                            'title': result.get('title', ''),
                            'snippet': result.get('snippet', ''),
                            'source_query': query
                        }
                        all_content.append(content_item)
                    
                    # Rate limiting
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.warning(f"Failed to search with query '{query}': {e}")
                    continue
            
            # Remove duplicates and limit to sample size
            unique_content = []
            seen_urls = set()
            
            for item in all_content:
                if item['url'] not in seen_urls:
                    unique_content.append(item)
                    seen_urls.add(item['url'])
                    
                    if len(unique_content) >= sample_size:
                        break
            
            return unique_content
            
        except Exception as e:
            logger.error(f"Failed to get competitor content sample: {e}")
            return []
    
    async def _analyze_content_types(self, content_sample: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze content types from sample."""
        try:
            content_types = {}
            
            for item in content_sample:
                url = item.get('url', '').lower()
                title = item.get('title', '').lower()
                
                # Simple content type classification
                if 'blog' in url or 'blog' in title:
                    content_type = 'blog_post'
                elif 'news' in url or 'news' in title:
                    content_type = 'news'
                elif 'guide' in url or 'guide' in title or 'tutorial' in title:
                    content_type = 'guide'
                elif 'case-study' in url or 'case study' in title:
                    content_type = 'case_study'
                elif 'whitepaper' in url or 'whitepaper' in title:
                    content_type = 'whitepaper'
                elif 'product' in url or 'product' in title:
                    content_type = 'product_page'
                else:
                    content_type = 'article'
                
                content_types[content_type] = content_types.get(content_type, 0) + 1
            
            # Calculate percentages
            total_content = len(content_sample)
            content_type_distribution = {
                ct: {'count': count, 'percentage': (count / total_content) * 100}
                for ct, count in content_types.items()
            }
            
            # Identify primary content types
            primary_types = sorted(content_types.items(), key=lambda x: x[1], reverse=True)[:3]
            
            return {
                'distribution': content_type_distribution,
                'primary_types': [ct[0] for ct in primary_types],
                'diversity_score': len(content_types) / 7.0  # Normalize by max expected types
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze content types: {e}")
            return {}
    
    async def _analyze_topic_distribution(self, content_sample: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze topic distribution from content sample."""
        try:
            # Extract keywords from titles and snippets
            all_text = []
            for item in content_sample:
                title = item.get('title', '')
                snippet = item.get('snippet', '')
                all_text.append(f"{title} {snippet}")
            
            # Simple topic extraction (in production, use more sophisticated NLP)
            topic_keywords = {}
            
            for text in all_text:
                # Extract meaningful words
                words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
                
                # Filter common words
                stop_words = {
                    'the', 'and', 'are', 'was', 'will', 'been', 'have', 'has', 'had',
                    'this', 'that', 'these', 'those', 'with', 'for', 'from', 'not',
                    'but', 'can', 'could', 'would', 'should', 'may', 'might', 'must',
                    'your', 'you', 'our', 'how', 'what', 'when', 'where', 'why'
                }
                
                for word in words:
                    if word not in stop_words and len(word) > 3:
                        topic_keywords[word] = topic_keywords.get(word, 0) + 1
            
            # Get top topics
            top_topics = sorted(topic_keywords.items(), key=lambda x: x[1], reverse=True)[:20]
            
            # Calculate topic concentration
            total_mentions = sum(topic_keywords.values())
            topic_concentration = (top_topics[0][1] / total_mentions) if top_topics else 0
            
            return {
                'top_topics': [{'topic': topic, 'mentions': count} for topic, count in top_topics],
                'topic_diversity': len(topic_keywords),
                'topic_concentration': topic_concentration,
                'primary_focus': top_topics[0][0] if top_topics else None
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze topic distribution: {e}")
            return {}
    
    async def _analyze_content_quality(self, content_sample: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze content quality indicators."""
        try:
            # Quality indicators from titles and snippets
            quality_scores = []
            
            for item in content_sample:
                title = item.get('title', '')
                snippet = item.get('snippet', '')
                
                # Simple quality scoring
                title_score = 0.0
                snippet_score = 0.0
                
                # Title quality factors
                if title:
                    title_length = len(title)
                    if 30 <= title_length <= 60:  # Good title length
                        title_score += 0.3
                    if any(word in title.lower() for word in ['how', 'what', 'why', 'guide', 'best']):
                        title_score += 0.2
                    if not title.isupper():  # Not all caps
                        title_score += 0.1
                
                # Snippet quality factors
                if snippet:
                    snippet_length = len(snippet)
                    if 100 <= snippet_length <= 300:  # Good snippet length
                        snippet_score += 0.2
                    if snippet.count('.') >= 2:  # Multiple sentences
                        snippet_score += 0.1
                    if any(word in snippet.lower() for word in ['benefits', 'advantages', 'solutions']):
                        snippet_score += 0.1
                
                overall_score = min((title_score + snippet_score), 1.0)
                quality_scores.append(overall_score)
            
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            
            return {
                'average_quality_score': avg_quality,
                'quality_distribution': {
                    'high_quality': len([s for s in quality_scores if s > 0.7]),
                    'medium_quality': len([s for s in quality_scores if 0.3 <= s <= 0.7]),
                    'low_quality': len([s for s in quality_scores if s < 0.3])
                },
                'content_optimization_level': 'high' if avg_quality > 0.7 else 'medium' if avg_quality > 0.4 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze content quality: {e}")
            return {}
    
    async def _analyze_publishing_patterns(self, content_sample: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze publishing patterns and frequency."""
        try:
            # This is a simplified analysis based on search results
            # In production, would extract actual publish dates
            
            # Estimate based on search result freshness and URL patterns
            estimated_frequency = len(content_sample) / 30  # Assume 30-day window
            
            # Analyze URL patterns for content organization
            url_patterns = {}
            for item in content_sample:
                url = item.get('url', '')
                if url:
                    # Extract path patterns
                    path = urlparse(url).path
                    path_segments = [seg for seg in path.split('/') if seg]
                    
                    if len(path_segments) > 1:
                        pattern = f"/{path_segments[1]}"
                        url_patterns[pattern] = url_patterns.get(pattern, 0) + 1
            
            # Identify primary content paths
            primary_paths = sorted(url_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                'estimated_monthly_frequency': estimated_frequency,
                'content_paths': url_patterns,
                'primary_content_paths': primary_paths,
                'content_organization': 'structured' if len(primary_paths) > 2 else 'simple'
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze publishing patterns: {e}")
            return {}
    
    async def _analyze_seo_patterns(self, content_sample: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze SEO patterns from content sample."""
        try:
            seo_indicators = {
                'keyword_optimized_titles': 0,
                'descriptive_urls': 0,
                'meta_descriptions': 0,
                'structured_content': 0
            }
            
            for item in content_sample:
                title = item.get('title', '')
                url = item.get('url', '')
                snippet = item.get('snippet', '')
                
                # Check for keyword optimization in titles
                if title and any(word in title.lower() for word in ['how', 'what', 'best', 'guide', 'tips']):
                    seo_indicators['keyword_optimized_titles'] += 1
                
                # Check for descriptive URLs
                if url and len(urlparse(url).path.split('/')) > 2:
                    seo_indicators['descriptive_urls'] += 1
                
                # Check for meta descriptions (from snippets)
                if snippet and len(snippet) > 80:
                    seo_indicators['meta_descriptions'] += 1
                
                # Check for structured content indicators
                if snippet and ('step' in snippet.lower() or 'section' in snippet.lower() or snippet.count(':') > 1):
                    seo_indicators['structured_content'] += 1
            
            total_content = len(content_sample)
            seo_scores = {
                indicator: (count / total_content) * 100
                for indicator, count in seo_indicators.items()
            } if total_content > 0 else {}
            
            # Calculate overall SEO optimization level
            avg_seo_score = sum(seo_scores.values()) / len(seo_scores) if seo_scores else 0
            
            return {
                'seo_scores': seo_scores,
                'overall_seo_optimization': avg_seo_score,
                'seo_maturity': 'advanced' if avg_seo_score > 70 else 'intermediate' if avg_seo_score > 40 else 'basic'
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze SEO patterns: {e}")
            return {}
    
    async def _analyze_engagement_patterns(self, content_sample: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze engagement patterns from content sample."""
        try:
            # Engagement indicators from titles and snippets
            engagement_indicators = {
                'question_based_content': 0,
                'actionable_content': 0,
                'list_based_content': 0,
                'how_to_content': 0
            }
            
            for item in content_sample:
                title = item.get('title', '').lower()
                snippet = item.get('snippet', '').lower()
                combined_text = f"{title} {snippet}"
                
                # Question-based content
                if any(word in combined_text for word in ['?', 'how', 'what', 'why', 'when', 'where']):
                    engagement_indicators['question_based_content'] += 1
                
                # Actionable content
                if any(word in combined_text for word in ['tips', 'steps', 'ways', 'methods', 'strategies']):
                    engagement_indicators['actionable_content'] += 1
                
                # List-based content
                if any(word in combined_text for word in ['top', 'best', 'list', 'reasons', 'benefits']):
                    engagement_indicators['list_based_content'] += 1
                
                # How-to content
                if 'how to' in combined_text or 'how-to' in combined_text:
                    engagement_indicators['how_to_content'] += 1
            
            total_content = len(content_sample)
            engagement_scores = {
                indicator: (count / total_content) * 100
                for indicator, count in engagement_indicators.items()
            } if total_content > 0 else {}
            
            # Calculate engagement optimization level
            avg_engagement_score = sum(engagement_scores.values()) / len(engagement_scores) if engagement_scores else 0
            
            return {
                'engagement_scores': engagement_scores,
                'engagement_optimization': avg_engagement_score,
                'content_style': 'highly_engaging' if avg_engagement_score > 60 else 'moderately_engaging' if avg_engagement_score > 30 else 'low_engagement'
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze engagement patterns: {e}")
            return {}
    
    async def _generate_strategy_insights(self, 
                                        content_types: Dict[str, Any],
                                        topic_distribution: Dict[str, Any],
                                        content_quality: Dict[str, Any],
                                        publishing_patterns: Dict[str, Any]) -> List[str]:
        """Generate strategic insights from content analysis."""
        try:
            insights = []
            
            # Content type insights
            if content_types.get('primary_types'):
                primary_type = content_types['primary_types'][0]
                insights.append(f"Primary content focus: {primary_type.replace('_', ' ').title()}")
                
                diversity_score = content_types.get('diversity_score', 0)
                if diversity_score > 0.7:
                    insights.append("High content diversity - competitor uses varied content formats")
                elif diversity_score < 0.3:
                    insights.append("Low content diversity - opportunity to differentiate with varied formats")
            
            # Topic insights
            if topic_distribution.get('primary_focus'):
                primary_focus = topic_distribution['primary_focus']
                insights.append(f"Primary topic focus: {primary_focus}")
                
                concentration = topic_distribution.get('topic_concentration', 0)
                if concentration > 0.3:
                    insights.append("High topic concentration - competitor has narrow focus")
                else:
                    insights.append("Broad topic coverage - competitor targets wide audience")
            
            # Quality insights
            quality_level = content_quality.get('content_optimization_level', 'unknown')
            if quality_level == 'high':
                insights.append("High content quality - strong competitor with optimized content")
            elif quality_level == 'low':
                insights.append("Low content quality - opportunity to outperform with better content")
            
            # Publishing insights
            frequency = publishing_patterns.get('estimated_monthly_frequency', 0)
            if frequency > 20:
                insights.append("High publishing frequency - competitor prioritizes content volume")
            elif frequency < 5:
                insights.append("Low publishing frequency - opportunity to gain share with more content")
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate strategy insights: {e}")
            return []