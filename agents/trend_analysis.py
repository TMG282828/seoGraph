"""
Trend Analysis Agent for the SEO Content Knowledge Graph System.

This module provides a Pydantic AI agent specialized in trend analysis with
Google Trends integration, social media trend detection, and prediction algorithms.
"""

import asyncio
import json
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import aiohttp
import structlog
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from cachetools import TTLCache
from pytrends.request import TrendReq
import pandas as pd
import numpy as np

from database.neo4j_client import Neo4jClient
from database.qdrant_client import QdrantClient
from services.searxng_service import SearXNGService
from models.seo_models import TrendAnalysis, TrendDirection, KeywordData
from models.analytics_models import TimeSeriesData, TimeGranularity
from config.settings import get_settings

logger = structlog.get_logger(__name__)


class TrendAnalysisError(Exception):
    """Raised when trend analysis operations fail."""
    pass


@dataclass
class TrendAnalysisDeps:
    """Dependencies for trend analysis agent."""
    neo4j_client: Neo4jClient
    qdrant_client: QdrantClient
    searxng_service: SearXNGService
    tenant_id: str
    industry: Optional[str] = None


class TrendData(BaseModel):
    """Structured trend data for analysis."""
    
    topic: str = Field(..., description="Topic or keyword")
    trend_score: float = Field(..., ge=0.0, le=1.0, description="Trend strength score")
    search_volume: int = Field(0, ge=0, description="Search volume")
    growth_rate: float = Field(0.0, description="Growth rate percentage")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in trend data")
    
    # Temporal data
    historical_data: List[Dict[str, Union[str, int]]] = Field(default_factory=list)
    seasonal_pattern: Optional[str] = Field(None, description="Seasonal pattern if detected")
    
    # Sources
    data_sources: List[str] = Field(default_factory=list, description="Data sources used")
    geographic_data: Dict[str, float] = Field(default_factory=dict, description="Geographic distribution")
    
    # Related information
    related_topics: List[str] = Field(default_factory=list, description="Related trending topics")
    competitor_activity: Dict[str, Any] = Field(default_factory=dict, description="Competitor activity data")


class GoogleTrendsService:
    """Service for Google Trends integration."""
    
    def __init__(self):
        self.pytrends = None
        self.cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour cache
        self.rate_limit_delay = 2.0  # seconds between requests
        self.last_request_time = 0.0
        
    async def initialize(self) -> None:
        """Initialize Google Trends service."""
        try:
            # Initialize pytrends with rate limiting
            self.pytrends = TrendReq(
                hl='en-US',
                tz=360,
                timeout=(10, 25),
                retries=2,
                backoff_factor=0.1
            )
            logger.info("Google Trends service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Google Trends service: {e}")
            raise TrendAnalysisError(f"Google Trends initialization failed: {e}")
    
    async def get_trending_topics(self, 
                                category: str = '',
                                geo: str = 'US',
                                timeframe: str = 'today 3-m') -> List[Dict[str, Any]]:
        """Get trending topics from Google Trends."""
        try:
            cache_key = f"trending_{category}_{geo}_{timeframe}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Rate limiting
            await self._apply_rate_limit()
            
            if not self.pytrends:
                await self.initialize()
            
            # Get trending searches
            trending_searches = self.pytrends.trending_searches(pn=geo)
            
            trends = []
            for topic in trending_searches[0][:20]:  # Top 20 trends
                trend_data = {
                    'topic': topic,
                    'category': category,
                    'geo': geo,
                    'trend_score': 1.0,  # Default high score for trending searches
                    'source': 'google_trends_trending'
                }
                trends.append(trend_data)
            
            # Cache results
            self.cache[cache_key] = trends
            
            return trends
            
        except Exception as e:
            logger.error(f"Failed to get trending topics: {e}")
            return []
    
    async def get_keyword_trends(self, 
                               keywords: List[str],
                               timeframe: str = 'today 3-m',
                               geo: str = 'US') -> Dict[str, TrendData]:
        """Get trend data for specific keywords."""
        try:
            cache_key = f"keywords_{hash(tuple(keywords))}_{timeframe}_{geo}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Rate limiting
            await self._apply_rate_limit()
            
            if not self.pytrends:
                await self.initialize()
            
            # Limit to 5 keywords per request (Google Trends limitation)
            keyword_batches = [keywords[i:i+5] for i in range(0, len(keywords), 5)]
            
            all_trends = {}
            
            for batch in keyword_batches:
                try:
                    # Build payload
                    self.pytrends.build_payload(
                        batch,
                        cat=0,
                        timeframe=timeframe,
                        geo=geo,
                        gprop=''
                    )
                    
                    # Get interest over time
                    interest_over_time = self.pytrends.interest_over_time()
                    
                    # Get related topics
                    related_topics = self.pytrends.related_topics()
                    
                    # Get interest by region
                    interest_by_region = self.pytrends.interest_by_region(resolution='COUNTRY')
                    
                    # Process each keyword
                    for keyword in batch:
                        if keyword in interest_over_time.columns:
                            trend_data = await self._process_keyword_trend_data(
                                keyword,
                                interest_over_time,
                                related_topics.get(keyword, {}),
                                interest_by_region
                            )
                            all_trends[keyword] = trend_data
                    
                    # Rate limiting between batches
                    if len(keyword_batches) > 1:
                        await asyncio.sleep(self.rate_limit_delay)
                    
                except Exception as e:
                    logger.warning(f"Failed to process keyword batch {batch}: {e}")
                    continue
            
            # Cache results
            self.cache[cache_key] = all_trends
            
            return all_trends
            
        except Exception as e:
            logger.error(f"Failed to get keyword trends: {e}")
            return {}
    
    async def _process_keyword_trend_data(self,
                                        keyword: str,
                                        interest_over_time: pd.DataFrame,
                                        related_topics: Dict,
                                        interest_by_region: pd.DataFrame) -> TrendData:
        """Process raw trend data into structured format."""
        try:
            # Get time series data
            time_series = interest_over_time[keyword].dropna()
            
            # Calculate trend metrics
            if len(time_series) > 1:
                # Calculate growth rate
                first_value = time_series.iloc[0]
                last_value = time_series.iloc[-1]
                growth_rate = ((last_value - first_value) / first_value * 100) if first_value > 0 else 0
                
                # Calculate trend score based on recent activity
                recent_values = time_series.tail(4).values  # Last 4 data points
                trend_score = min(np.mean(recent_values) / 100.0, 1.0)
                
                # Detect seasonal patterns
                seasonal_pattern = await self._detect_seasonal_pattern(time_series)
                
                # Calculate confidence based on data consistency
                confidence = min(1.0 - (np.std(time_series) / (np.mean(time_series) + 1)), 1.0)
            else:
                growth_rate = 0.0
                trend_score = 0.0
                seasonal_pattern = None
                confidence = 0.0
            
            # Process historical data
            historical_data = []
            for date, value in time_series.items():
                historical_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'value': int(value)
                })
            
            # Process related topics
            related_topics_list = []
            if 'top' in related_topics and related_topics['top'] is not None:
                for _, topic in related_topics['top'].head(5).iterrows():
                    related_topics_list.append(topic['topic_title'])
            
            # Process geographic data
            geographic_data = {}
            if keyword in interest_by_region.columns:
                region_data = interest_by_region[keyword].dropna().head(10)
                for region, value in region_data.items():
                    geographic_data[region] = float(value)
            
            return TrendData(
                topic=keyword,
                trend_score=trend_score,
                search_volume=int(time_series.sum()) if len(time_series) > 0 else 0,
                growth_rate=growth_rate,
                confidence=confidence,
                historical_data=historical_data,
                seasonal_pattern=seasonal_pattern,
                data_sources=['google_trends'],
                geographic_data=geographic_data,
                related_topics=related_topics_list
            )
            
        except Exception as e:
            logger.error(f"Failed to process trend data for {keyword}: {e}")
            return TrendData(
                topic=keyword,
                trend_score=0.0,
                confidence=0.0,
                data_sources=['google_trends']
            )
    
    async def _detect_seasonal_pattern(self, time_series: pd.Series) -> Optional[str]:
        """Detect seasonal patterns in time series data."""
        try:
            if len(time_series) < 12:  # Need at least 12 data points
                return None
            
            # Simple seasonal detection using autocorrelation
            values = time_series.values
            
            # Check for weekly pattern (if daily data)
            if len(values) >= 14:
                weekly_corr = np.corrcoef(values[:-7], values[7:])[0, 1]
                if weekly_corr > 0.7:
                    return 'weekly'
            
            # Check for monthly pattern
            if len(values) >= 24:
                monthly_corr = np.corrcoef(values[:-12], values[12:])[0, 1]
                if monthly_corr > 0.7:
                    return 'monthly'
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to detect seasonal pattern: {e}")
            return None
    
    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting to Google Trends requests."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = asyncio.get_event_loop().time()


class SocialTrendsService:
    """Service for social media trend analysis."""
    
    def __init__(self):
        self.cache = TTLCache(maxsize=500, ttl=1800)  # 30 minutes cache
        
    async def get_social_trends(self, platform: str = 'twitter') -> List[Dict[str, Any]]:
        """Get trending topics from social media platforms."""
        try:
            cache_key = f"social_trends_{platform}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            trends = []
            
            if platform == 'twitter':
                # Placeholder for Twitter API integration
                # In production, use Twitter API v2
                trends = await self._get_twitter_trends()
            elif platform == 'reddit':
                # Placeholder for Reddit API integration
                trends = await self._get_reddit_trends()
            elif platform == 'linkedin':
                # Placeholder for LinkedIn API integration
                trends = await self._get_linkedin_trends()
            
            # Cache results
            self.cache[cache_key] = trends
            
            return trends
            
        except Exception as e:
            logger.error(f"Failed to get social trends from {platform}: {e}")
            return []
    
    async def _get_twitter_trends(self) -> List[Dict[str, Any]]:
        """Get Twitter trending topics (placeholder)."""
        # Placeholder implementation
        # In production, integrate with Twitter API v2
        return [
            {'topic': 'AI content creation', 'mentions': 15000, 'sentiment': 0.7},
            {'topic': 'SEO automation', 'mentions': 8000, 'sentiment': 0.8},
            {'topic': 'content marketing', 'mentions': 12000, 'sentiment': 0.6}
        ]
    
    async def _get_reddit_trends(self) -> List[Dict[str, Any]]:
        """Get Reddit trending topics (placeholder)."""
        # Placeholder implementation
        # In production, integrate with Reddit API
        return [
            {'topic': 'digital marketing trends', 'upvotes': 2500, 'comments': 450},
            {'topic': 'AI writing tools', 'upvotes': 1800, 'comments': 320}
        ]
    
    async def _get_linkedin_trends(self) -> List[Dict[str, Any]]:
        """Get LinkedIn trending topics (placeholder)."""
        # Placeholder implementation
        # In production, integrate with LinkedIn API
        return [
            {'topic': 'B2B content strategy', 'engagement': 5000, 'shares': 800},
            {'topic': 'professional development', 'engagement': 7000, 'shares': 1200}
        ]


# Create the trend analysis agent
trend_analysis_agent = Agent(
    'openai:gpt-4o',
    deps_type=TrendAnalysisDeps,
    system_prompt="""
You are a specialized Trend Analysis Agent for the SEO Content Knowledge Graph System.

Your primary responsibilities:
1. Analyze trending topics and keywords across multiple data sources
2. Identify emerging content opportunities and market shifts
3. Provide predictive insights for content strategy
4. Validate trends against competitor activity and market context
5. Generate actionable recommendations for content creation

Key capabilities:
- Google Trends integration for search trend analysis
- Social media trend monitoring across platforms
- Competitor trend analysis and benchmarking
- Seasonal pattern detection and forecasting
- Industry-specific trend identification
- Cross-platform trend correlation analysis

When analyzing trends, consider:
- Search volume trajectory and growth patterns
- Geographic distribution and market penetration
- Seasonal variations and cyclical patterns
- Competitor activity and market saturation
- Content gap opportunities
- Audience engagement potential

Always provide specific, actionable insights with confidence scores and supporting data.
Be analytical but accessible, focusing on business value and strategic implications.
""",
    result_type=TrendAnalysis
)


@trend_analysis_agent.tool
async def analyze_trending_topics(
    ctx: RunContext[TrendAnalysisDeps],
    industry: str,
    timeframe: str = 'today 3-m',
    include_social: bool = True,
    geo: str = 'US'
) -> Dict[str, Any]:
    """
    Analyze trending topics for a specific industry.
    
    Args:
        industry: Industry to analyze (e.g., 'technology', 'healthcare')
        timeframe: Analysis timeframe ('today 1-m', 'today 3-m', 'today 12-m')
        include_social: Whether to include social media trends
        geo: Geographic region for analysis
        
    Returns:
        Comprehensive trend analysis with insights
    """
    try:
        logger.info(f"Analyzing trending topics for {industry}")
        
        # Initialize services
        google_trends = GoogleTrendsService()
        await google_trends.initialize()
        
        social_trends = SocialTrendsService()
        
        # Get trending topics from Google Trends
        google_trending = await google_trends.get_trending_topics(
            category=industry,
            geo=geo,
            timeframe=timeframe
        )
        
        # Get social media trends if requested
        social_trending = []
        if include_social:
            platforms = ['twitter', 'reddit', 'linkedin']
            for platform in platforms:
                platform_trends = await social_trends.get_social_trends(platform)
                social_trending.extend(platform_trends)
        
        # Combine and analyze trends
        all_trends = []
        
        # Process Google Trends
        for trend in google_trending:
            all_trends.append({
                'topic': trend['topic'],
                'source': 'google_trends',
                'score': trend['trend_score'],
                'confidence': 0.8,  # High confidence for Google Trends
                'metadata': trend
            })
        
        # Process social trends
        for trend in social_trending:
            all_trends.append({
                'topic': trend['topic'],
                'source': 'social_media',
                'score': min(trend.get('mentions', trend.get('upvotes', 0)) / 10000, 1.0),
                'confidence': 0.6,  # Lower confidence for social trends
                'metadata': trend
            })
        
        # Rank and filter trends
        ranked_trends = sorted(all_trends, key=lambda x: x['score'] * x['confidence'], reverse=True)
        top_trends = ranked_trends[:20]
        
        # Analyze trend patterns
        trend_categories = await _categorize_trends(top_trends)
        emerging_themes = await _identify_emerging_themes(top_trends)
        
        return {
            'total_trends_analyzed': len(all_trends),
            'top_trends': top_trends,
            'trend_categories': trend_categories,
            'emerging_themes': emerging_themes,
            'analysis_timeframe': timeframe,
            'geographic_scope': geo,
            'confidence_score': sum(t['confidence'] for t in top_trends) / len(top_trends) if top_trends else 0
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze trending topics: {e}")
        raise TrendAnalysisError(f"Trend analysis failed: {e}")


@trend_analysis_agent.tool
async def analyze_keyword_trends(
    ctx: RunContext[TrendAnalysisDeps],
    keywords: List[str],
    competitor_keywords: Optional[List[str]] = None,
    timeframe: str = 'today 3-m'
) -> Dict[str, Any]:
    """
    Analyze trend data for specific keywords.
    
    Args:
        keywords: List of keywords to analyze
        competitor_keywords: Optional competitor keywords for comparison
        timeframe: Analysis timeframe
        
    Returns:
        Detailed keyword trend analysis
    """
    try:
        logger.info(f"Analyzing trends for {len(keywords)} keywords")
        
        # Initialize Google Trends service
        google_trends = GoogleTrendsService()
        await google_trends.initialize()
        
        # Get trend data for keywords
        keyword_trends = await google_trends.get_keyword_trends(keywords, timeframe)
        
        # Get competitor trend data if provided
        competitor_trends = {}
        if competitor_keywords:
            competitor_trends = await google_trends.get_keyword_trends(competitor_keywords, timeframe)
        
        # Analyze trends
        trend_analysis = {}
        opportunities = []
        threats = []
        
        for keyword, trend_data in keyword_trends.items():
            # Classify trend direction
            if trend_data.growth_rate > 20:
                direction = TrendDirection.RISING
            elif trend_data.growth_rate < -20:
                direction = TrendDirection.DECLINING
            else:
                direction = TrendDirection.STABLE
            
            # Identify opportunities
            if direction == TrendDirection.RISING and trend_data.confidence > 0.6:
                opportunities.append({
                    'keyword': keyword,
                    'growth_rate': trend_data.growth_rate,
                    'trend_score': trend_data.trend_score,
                    'opportunity_type': 'growing_interest'
                })
            
            # Identify threats
            if direction == TrendDirection.DECLINING and keyword in [kw.lower() for kw in keywords]:
                threats.append({
                    'keyword': keyword,
                    'decline_rate': abs(trend_data.growth_rate),
                    'risk_level': 'high' if trend_data.growth_rate < -50 else 'medium'
                })
            
            trend_analysis[keyword] = {
                'trend_direction': direction,
                'growth_rate': trend_data.growth_rate,
                'trend_score': trend_data.trend_score,
                'search_volume': trend_data.search_volume,
                'confidence': trend_data.confidence,
                'seasonal_pattern': trend_data.seasonal_pattern,
                'related_topics': trend_data.related_topics,
                'geographic_data': trend_data.geographic_data
            }
        
        # Compare with competitors
        competitive_insights = []
        if competitor_trends:
            for comp_keyword, comp_trend in competitor_trends.items():
                # Find similar keywords in our list
                similar_keywords = [kw for kw in keywords if _calculate_keyword_similarity(kw, comp_keyword) > 0.7]
                
                if similar_keywords:
                    for our_keyword in similar_keywords:
                        our_trend = keyword_trends.get(our_keyword)
                        if our_trend:
                            competitive_insights.append({
                                'our_keyword': our_keyword,
                                'competitor_keyword': comp_keyword,
                                'our_growth': our_trend.growth_rate,
                                'competitor_growth': comp_trend.growth_rate,
                                'gap': comp_trend.growth_rate - our_trend.growth_rate,
                                'recommendation': 'monitor' if abs(comp_trend.growth_rate - our_trend.growth_rate) > 30 else 'stable'
                            })
        
        return {
            'keyword_trends': trend_analysis,
            'opportunities': opportunities,
            'threats': threats,
            'competitive_insights': competitive_insights,
            'summary': {
                'total_keywords': len(keywords),
                'rising_trends': len([k for k, v in trend_analysis.items() if v['trend_direction'] == TrendDirection.RISING]),
                'declining_trends': len([k for k, v in trend_analysis.items() if v['trend_direction'] == TrendDirection.DECLINING]),
                'stable_trends': len([k for k, v in trend_analysis.items() if v['trend_direction'] == TrendDirection.STABLE]),
                'avg_confidence': sum(v['confidence'] for v in trend_analysis.values()) / len(trend_analysis) if trend_analysis else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze keyword trends: {e}")
        raise TrendAnalysisError(f"Keyword trend analysis failed: {e}")


@trend_analysis_agent.tool
async def predict_trend_trajectory(
    ctx: RunContext[TrendAnalysisDeps],
    topic: str,
    prediction_horizon: str = '3m',
    include_seasonality: bool = True
) -> Dict[str, Any]:
    """
    Predict future trend trajectory for a topic.
    
    Args:
        topic: Topic to predict trends for
        prediction_horizon: How far to predict ('1m', '3m', '6m', '12m')
        include_seasonality: Whether to include seasonal adjustments
        
    Returns:
        Trend prediction with confidence intervals
    """
    try:
        logger.info(f"Predicting trend trajectory for: {topic}")
        
        # Initialize Google Trends service
        google_trends = GoogleTrendsService()
        await google_trends.initialize()
        
        # Get historical trend data (longer timeframe for better prediction)
        historical_trends = await google_trends.get_keyword_trends(
            [topic], 
            timeframe='today 12-m'  # 12 months of data
        )
        
        if not historical_trends or topic not in historical_trends:
            return {
                'error': f'Insufficient data for trend prediction of {topic}',
                'confidence': 0.0
            }
        
        trend_data = historical_trends[topic]
        
        # Extract time series data
        if not trend_data.historical_data:
            return {
                'error': f'No historical data available for {topic}',
                'confidence': 0.0
            }
        
        # Prepare data for prediction
        dates = [pd.to_datetime(d['date']) for d in trend_data.historical_data]
        values = [d['value'] for d in trend_data.historical_data]
        
        if len(values) < 8:  # Need minimum data points
            return {
                'error': f'Insufficient data points for prediction ({len(values)} < 8)',
                'confidence': 0.0
            }
        
        # Simple linear trend prediction
        x = np.arange(len(values))
        z = np.polyfit(x, values, 1)
        trend_line = np.poly1d(z)
        
        # Calculate prediction steps
        horizon_months = int(prediction_horizon.replace('m', ''))
        prediction_steps = min(horizon_months * 4, 24)  # Weekly data points, max 6 months
        
        # Generate predictions
        future_x = np.arange(len(values), len(values) + prediction_steps)
        predictions = trend_line(future_x)
        
        # Apply seasonal adjustment if requested
        if include_seasonality and trend_data.seasonal_pattern:
            seasonal_factor = await _calculate_seasonal_factor(values, trend_data.seasonal_pattern)
            predictions = predictions * seasonal_factor
        
        # Calculate confidence intervals
        residuals = values - trend_line(x)
        mse = np.mean(residuals ** 2)
        std_error = np.sqrt(mse)
        
        confidence_upper = predictions + 1.96 * std_error
        confidence_lower = predictions - 1.96 * std_error
        
        # Generate future dates
        last_date = dates[-1]
        future_dates = [last_date + timedelta(weeks=i+1) for i in range(prediction_steps)]
        
        # Calculate prediction confidence
        trend_consistency = 1.0 - (np.std(residuals) / (np.mean(values) + 1))
        data_quality = min(len(values) / 20, 1.0)  # More data = higher confidence
        prediction_confidence = (trend_consistency * 0.7) + (data_quality * 0.3)
        
        # Determine trend classification
        avg_slope = z[0]  # Slope of trend line
        if avg_slope > 2:
            trend_classification = 'strongly_rising'
        elif avg_slope > 0.5:
            trend_classification = 'rising'
        elif avg_slope > -0.5:
            trend_classification = 'stable'
        elif avg_slope > -2:
            trend_classification = 'declining'
        else:
            trend_classification = 'strongly_declining'
        
        return {
            'topic': topic,
            'prediction_horizon': prediction_horizon,
            'trend_classification': trend_classification,
            'predictions': [
                {
                    'date': date.strftime('%Y-%m-%d'),
                    'predicted_value': float(pred),
                    'confidence_upper': float(upper),
                    'confidence_lower': float(lower)
                }
                for date, pred, upper, lower in zip(future_dates, predictions, confidence_upper, confidence_lower)
            ],
            'historical_trend_slope': float(avg_slope),
            'prediction_confidence': float(max(0.0, min(1.0, prediction_confidence))),
            'seasonal_adjustment_applied': include_seasonality and trend_data.seasonal_pattern is not None,
            'data_points_used': len(values),
            'prediction_summary': {
                'expected_direction': trend_classification,
                'confidence_level': 'high' if prediction_confidence > 0.7 else 'medium' if prediction_confidence > 0.4 else 'low',
                'key_insights': await _generate_prediction_insights(trend_classification, avg_slope, prediction_confidence)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to predict trend trajectory: {e}")
        raise TrendAnalysisError(f"Trend prediction failed: {e}")


@trend_analysis_agent.tool
async def validate_trends_with_competitors(
    ctx: RunContext[TrendAnalysisDeps],
    topics: List[str],
    competitor_domains: List[str]
) -> Dict[str, Any]:
    """
    Validate trends against competitor activity and market context.
    
    Args:
        topics: Topics to validate
        competitor_domains: Competitor domains to analyze
        
    Returns:
        Trend validation with competitive context
    """
    try:
        logger.info(f"Validating {len(topics)} trends against {len(competitor_domains)} competitors")
        
        # Use SearXNG to analyze competitor content for trend validation
        searxng_service = ctx.deps.searxng_service
        
        validation_results = {}
        
        for topic in topics:
            topic_validation = {
                'topic': topic,
                'competitor_coverage': {},
                'market_validation': 'unknown',
                'recommendation': 'monitor'
            }
            
            # Check competitor coverage for each topic
            for domain in competitor_domains:
                try:
                    # Search for topic on competitor domain
                    query = f"site:{domain} {topic}"
                    results = await searxng_service.search(
                        query=query,
                        engines=['google'],
                        safesearch=1
                    )
                    
                    coverage_score = min(len(results) / 10.0, 1.0)  # Normalize to 0-1
                    topic_validation['competitor_coverage'][domain] = coverage_score
                    
                    await asyncio.sleep(0.5)  # Rate limiting
                    
                except Exception as e:
                    logger.warning(f"Failed to check competitor coverage for {domain}: {e}")
                    topic_validation['competitor_coverage'][domain] = 0.0
            
            # Calculate overall market validation
            avg_coverage = sum(topic_validation['competitor_coverage'].values()) / len(competitor_domains) if competitor_domains else 0
            
            if avg_coverage > 0.7:
                topic_validation['market_validation'] = 'high_competition'
                topic_validation['recommendation'] = 'differentiate'
            elif avg_coverage > 0.3:
                topic_validation['market_validation'] = 'moderate_competition'
                topic_validation['recommendation'] = 'compete'
            else:
                topic_validation['market_validation'] = 'low_competition'
                topic_validation['recommendation'] = 'opportunity'
            
            validation_results[topic] = topic_validation
        
        # Generate overall insights
        high_opportunity_topics = [
            topic for topic, data in validation_results.items()
            if data['market_validation'] == 'low_competition'
        ]
        
        high_competition_topics = [
            topic for topic, data in validation_results.items()
            if data['market_validation'] == 'high_competition'
        ]
        
        return {
            'validation_results': validation_results,
            'summary': {
                'total_topics_analyzed': len(topics),
                'high_opportunity_topics': high_opportunity_topics,
                'high_competition_topics': high_competition_topics,
                'avg_market_coverage': sum(
                    sum(data['competitor_coverage'].values()) / len(data['competitor_coverage'])
                    for data in validation_results.values()
                ) / len(validation_results) if validation_results else 0
            },
            'recommendations': {
                'immediate_opportunities': high_opportunity_topics[:5],
                'competitive_analysis_needed': high_competition_topics,
                'strategic_insights': await _generate_strategic_insights(validation_results)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to validate trends with competitors: {e}")
        raise TrendAnalysisError(f"Trend validation failed: {e}")


# =============================================================================
# Helper Functions
# =============================================================================

async def _categorize_trends(trends: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Categorize trends by type/theme."""
    categories = {
        'technology': [],
        'business': [],
        'marketing': [],
        'lifestyle': [],
        'other': []
    }
    
    tech_keywords = ['ai', 'machine learning', 'automation', 'digital', 'tech', 'software', 'app']
    business_keywords = ['business', 'strategy', 'management', 'finance', 'revenue', 'growth']
    marketing_keywords = ['marketing', 'seo', 'content', 'social media', 'advertising', 'brand']
    lifestyle_keywords = ['health', 'fitness', 'lifestyle', 'travel', 'food', 'fashion']
    
    for trend in trends:
        topic = trend['topic'].lower()
        categorized = False
        
        if any(keyword in topic for keyword in tech_keywords):
            categories['technology'].append(trend['topic'])
            categorized = True
        elif any(keyword in topic for keyword in business_keywords):
            categories['business'].append(trend['topic'])
            categorized = True
        elif any(keyword in topic for keyword in marketing_keywords):
            categories['marketing'].append(trend['topic'])
            categorized = True
        elif any(keyword in topic for keyword in lifestyle_keywords):
            categories['lifestyle'].append(trend['topic'])
            categorized = True
        
        if not categorized:
            categories['other'].append(trend['topic'])
    
    return categories


async def _identify_emerging_themes(trends: List[Dict[str, Any]]) -> List[str]:
    """Identify emerging themes from trends."""
    # Simple theme identification based on common words
    word_freq = {}
    
    for trend in trends:
        words = re.findall(r'\b\w+\b', trend['topic'].lower())
        for word in words:
            if len(word) > 3:  # Filter short words
                word_freq[word] = word_freq.get(word, 0) + 1
    
    # Get most common themes
    common_themes = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [theme[0] for theme in common_themes[:10]]


def _calculate_keyword_similarity(keyword1: str, keyword2: str) -> float:
    """Calculate similarity between two keywords."""
    words1 = set(keyword1.lower().split())
    words2 = set(keyword2.lower().split())
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


async def _calculate_seasonal_factor(values: List[float], seasonal_pattern: str) -> float:
    """Calculate seasonal adjustment factor."""
    if seasonal_pattern == 'weekly':
        # Simple weekly seasonality adjustment
        return 1.1 if datetime.now().weekday() < 5 else 0.9  # Weekday vs weekend
    elif seasonal_pattern == 'monthly':
        # Simple monthly seasonality adjustment
        month = datetime.now().month
        if month in [11, 12, 1]:  # Holiday season
            return 1.2
        elif month in [6, 7, 8]:  # Summer
            return 0.9
        else:
            return 1.0
    
    return 1.0  # No adjustment


async def _generate_prediction_insights(trend_classification: str, slope: float, confidence: float) -> List[str]:
    """Generate insights for trend predictions."""
    insights = []
    
    if trend_classification == 'strongly_rising':
        insights.append("Strong upward momentum detected - consider immediate content creation")
        insights.append("High growth potential - prioritize resource allocation")
    elif trend_classification == 'rising':
        insights.append("Positive trend trajectory - good opportunity for content investment")
    elif trend_classification == 'stable':
        insights.append("Stable interest levels - consistent content strategy recommended")
    elif trend_classification == 'declining':
        insights.append("Declining interest - consider pivoting or improving content quality")
    else:
        insights.append("Strong downward trend - may need to reconsider topic relevance")
    
    if confidence > 0.7:
        insights.append("High confidence prediction - reliable for strategic planning")
    elif confidence > 0.4:
        insights.append("Moderate confidence - monitor closely for validation")
    else:
        insights.append("Low confidence prediction - requires additional data validation")
    
    return insights


async def _generate_strategic_insights(validation_results: Dict[str, Any]) -> List[str]:
    """Generate strategic insights from trend validation."""
    insights = []
    
    # Analyze competition patterns
    low_competition_count = sum(1 for data in validation_results.values() if data['market_validation'] == 'low_competition')
    high_competition_count = sum(1 for data in validation_results.values() if data['market_validation'] == 'high_competition')
    
    if low_competition_count > high_competition_count:
        insights.append("Market shows significant opportunity gaps - consider aggressive content expansion")
    elif high_competition_count > low_competition_count:
        insights.append("Highly competitive market - focus on differentiation and unique angles")
    else:
        insights.append("Balanced competitive landscape - selective topic prioritization recommended")
    
    # Add timing insights
    insights.append("Trends validated against real competitor activity - higher reliability for decision making")
    
    return insights


# =============================================================================
# Utility Functions
# =============================================================================

async def analyze_trends_simple(topics: List[str], 
                              industry: str = "technology",
                              tenant_id: str = "default") -> Optional[TrendAnalysis]:
    """
    Simple function to analyze trends for given topics.
    
    Args:
        topics: List of topics to analyze
        industry: Industry context
        tenant_id: Tenant identifier
        
    Returns:
        TrendAnalysis result if successful
    """
    # Initialize required dependencies
    settings = get_settings()
    
    # Placeholder for dependency initialization
    deps = TrendAnalysisDeps(
        neo4j_client=None,  # Would be initialized in production
        qdrant_client=None,  # Would be initialized in production
        searxng_service=None,  # Would be initialized in production
        tenant_id=tenant_id,
        industry=industry
    )
    
    try:
        # Run trend analysis
        result = await trend_analysis_agent.run(
            f"Analyze trends for these topics in {industry}: {', '.join(topics)}",
            deps=deps
        )
        
        return result.data if hasattr(result, 'data') else None
        
    except Exception as e:
        logger.error(f"Failed to analyze trends: {e}")
        return None


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        # Test trend analysis
        topics = ["AI content creation", "SEO automation", "voice search optimization"]
        
        analysis = await analyze_trends_simple(
            topics=topics,
            industry="digital marketing",
            tenant_id="test-tenant"
        )
        
        if analysis:
            print(f"Trend analysis completed for: {analysis.topic}")
            print(f"Trend direction: {analysis.trend_direction}")
            print(f"Trend strength: {analysis.trend_strength}")
        else:
            print("Trend analysis failed")

    asyncio.run(main())