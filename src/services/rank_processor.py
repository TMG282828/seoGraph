"""
Rank Processing Service for SerpBear Integration.

This service handles:
- Daily rank data fetching from SerpBear
- Ranking trend analysis and insights
- Performance alerts and notifications
- Data transformation for dashboard consumption

Processes raw ranking data into actionable SEO insights.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date, timedelta
from collections import defaultdict
from dataclasses import dataclass
from pydantic import BaseModel, Field
import statistics

from .serpbear_client import serpbear_client, SerpBearKeyword, RankingUpdate
from .keyword_manager import KeywordPerformanceMetrics
from ..database.neo4j_client import neo4j_client

logger = logging.getLogger(__name__)


class RankingTrend(BaseModel):
    """Model for ranking trend analysis."""
    keyword: str
    domain: str
    device: str
    current_position: Optional[int]
    previous_position: Optional[int]
    change: int = 0  # Positive = improvement
    change_percentage: float = 0.0
    trend_direction: str = "stable"  # improving, declining, stable
    trend_strength: str = "weak"  # weak, moderate, strong
    days_analyzed: int = 7


class PerformanceAlert(BaseModel):
    """Model for ranking performance alerts."""
    alert_type: str = Field(description="improvement, decline, volatility, milestone")
    keyword: str
    domain: str
    device: str
    current_position: Optional[int]
    previous_position: Optional[int]
    change: int
    severity: str = Field(description="low, medium, high, critical")
    message: str
    timestamp: str
    action_required: bool = False


class DomainRankingSummary(BaseModel):
    """Model for domain-level ranking summary."""
    domain: str
    total_keywords: int
    average_position: float
    keywords_top_10: int
    keywords_top_3: int
    keywords_improved: int
    keywords_declined: int
    keywords_stable: int
    visibility_score: float  # Calculated visibility metric
    last_updated: str


class RankProcessor:
    """
    Comprehensive rank data processing for SEO insights.
    
    This service processes raw SerpBear ranking data into:
    1. Trend analysis and performance metrics
    2. Automated alerts for significant changes
    3. Domain-level performance summaries
    4. Historical performance insights
    """
    
    def __init__(self, organization_id: str = "demo-org"):
        """
        Initialize rank processor.
        
        Args:
            organization_id: Organization context for operations
        """
        self.organization_id = organization_id
        logger.info(f"Rank processor initialized for org: {organization_id}")
    
    async def fetch_daily_rankings(self, domains: List[str]) -> Dict[str, List[SerpBearKeyword]]:
        """
        Fetch latest ranking data for all domains.
        
        Args:
            domains: List of domains to fetch rankings for
            
        Returns:
            Dictionary mapping domain to keyword rankings
        """
        try:
            logger.info(f"📥 Fetching daily rankings for {len(domains)} domains")
            
            domain_rankings = {}
            
            async with serpbear_client as client:
                for domain in domains:
                    try:
                        keywords = await client.get_keywords(domain)
                        domain_rankings[domain] = keywords
                        logger.info(f"✅ Fetched {len(keywords)} keywords for {domain}")
                        
                    except Exception as domain_error:
                        logger.error(f"❌ Failed to fetch rankings for {domain}: {domain_error}")
                        domain_rankings[domain] = []
            
            total_keywords = sum(len(keywords) for keywords in domain_rankings.values())
            logger.info(f"📊 Fetched {total_keywords} total keyword rankings")
            
            return domain_rankings
            
        except Exception as e:
            logger.error(f"❌ Daily rankings fetch failed: {e}")
            return {}
    
    async def analyze_ranking_trends(
        self, 
        domain: str, 
        days: int = 7
    ) -> List[RankingTrend]:
        """
        Analyze ranking trends over specified period.
        
        Args:
            domain: Domain to analyze
            days: Number of days to analyze
            
        Returns:
            List of ranking trends for keywords
        """
        try:
            logger.info(f"📈 Analyzing ranking trends for {domain} over {days} days")
            
            trends = []
            
            async with serpbear_client as client:
                ranking_updates = await client.get_ranking_updates(domain, days)
                
                for update in ranking_updates:
                    # Calculate trend metrics
                    change = update.change
                    change_percentage = 0.0
                    
                    if update.previous_position and update.previous_position > 0:
                        change_percentage = (change / update.previous_position) * 100
                    
                    # Determine trend direction and strength
                    if change > 0:
                        trend_direction = "improving"
                        if change >= 10:
                            trend_strength = "strong"
                        elif change >= 5:
                            trend_strength = "moderate"
                        else:
                            trend_strength = "weak"
                    elif change < 0:
                        trend_direction = "declining"
                        if abs(change) >= 10:
                            trend_strength = "strong"
                        elif abs(change) >= 5:
                            trend_strength = "moderate"
                        else:
                            trend_strength = "weak"
                    else:
                        trend_direction = "stable"
                        trend_strength = "weak"
                    
                    trend = RankingTrend(
                        keyword=update.keyword,
                        domain=domain,
                        device=update.device,
                        current_position=update.position,
                        previous_position=update.previous_position,
                        change=change,
                        change_percentage=round(change_percentage, 2),
                        trend_direction=trend_direction,
                        trend_strength=trend_strength,
                        days_analyzed=days
                    )
                    
                    trends.append(trend)
            
            logger.info(f"📊 Analyzed {len(trends)} ranking trends")
            return trends
            
        except Exception as e:
            logger.error(f"❌ Trend analysis failed: {e}")
            return []
    
    async def generate_performance_alerts(
        self, 
        trends: List[RankingTrend],
        alert_thresholds: Dict[str, int] = None
    ) -> List[PerformanceAlert]:
        """
        Generate performance alerts based on ranking changes.
        
        Args:
            trends: List of ranking trends to analyze
            alert_thresholds: Custom thresholds for alerts
            
        Returns:
            List of performance alerts
        """
        try:
            # Default alert thresholds
            if not alert_thresholds:
                alert_thresholds = {
                    "major_improvement": 10,    # Moved up 10+ positions
                    "major_decline": -10,       # Dropped 10+ positions
                    "moderate_improvement": 5,   # Moved up 5+ positions
                    "moderate_decline": -5,      # Dropped 5+ positions
                    "top_10_entry": 10,         # Entered top 10
                    "top_3_entry": 3,           # Entered top 3
                    "page_1_loss": 10           # Dropped below position 10
                }
            
            alerts = []
            
            for trend in trends:
                current_pos = trend.current_position
                previous_pos = trend.previous_position
                change = trend.change
                
                if not current_pos or not previous_pos:
                    continue
                
                # Major improvement alert
                if change >= alert_thresholds["major_improvement"]:
                    alerts.append(PerformanceAlert(
                        alert_type="improvement",
                        keyword=trend.keyword,
                        domain=trend.domain,
                        device=trend.device,
                        current_position=current_pos,
                        previous_position=previous_pos,
                        change=change,
                        severity="high",
                        message=f"🚀 Major improvement: '{trend.keyword}' jumped {change} positions to #{current_pos}",
                        timestamp=str(datetime.now()),
                        action_required=False
                    ))
                
                # Major decline alert
                elif change <= alert_thresholds["major_decline"]:
                    alerts.append(PerformanceAlert(
                        alert_type="decline",
                        keyword=trend.keyword,
                        domain=trend.domain,
                        device=trend.device,
                        current_position=current_pos,
                        previous_position=previous_pos,
                        change=change,
                        severity="critical",
                        message=f"⚠️ Major decline: '{trend.keyword}' dropped {abs(change)} positions to #{current_pos}",
                        timestamp=str(datetime.now()),
                        action_required=True
                    ))
                
                # Top 10 entry
                elif previous_pos > 10 and current_pos <= 10:
                    alerts.append(PerformanceAlert(
                        alert_type="milestone",
                        keyword=trend.keyword,
                        domain=trend.domain,
                        device=trend.device,
                        current_position=current_pos,
                        previous_position=previous_pos,
                        change=change,
                        severity="medium",
                        message=f"🎯 Milestone achieved: '{trend.keyword}' entered top 10 at position #{current_pos}",
                        timestamp=str(datetime.now()),
                        action_required=False
                    ))
                
                # Top 3 entry
                elif previous_pos > 3 and current_pos <= 3:
                    alerts.append(PerformanceAlert(
                        alert_type="milestone",
                        keyword=trend.keyword,
                        domain=trend.domain,
                        device=trend.device,
                        current_position=current_pos,
                        previous_position=previous_pos,
                        change=change,
                        severity="high",
                        message=f"🏆 Elite performance: '{trend.keyword}' reached top 3 at position #{current_pos}",
                        timestamp=str(datetime.now()),
                        action_required=False
                    ))
                
                # Page 1 loss
                elif previous_pos <= 10 and current_pos > 10:
                    alerts.append(PerformanceAlert(
                        alert_type="decline",
                        keyword=trend.keyword,
                        domain=trend.domain,
                        device=trend.device,
                        current_position=current_pos,
                        previous_position=previous_pos,
                        change=change,
                        severity="high",
                        message=f"📉 Page 1 loss: '{trend.keyword}' dropped from #{previous_pos} to #{current_pos}",
                        timestamp=str(datetime.now()),
                        action_required=True
                    ))
            
            # Sort alerts by severity
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            alerts.sort(key=lambda a: severity_order.get(a.severity, 4))
            
            logger.info(f"🚨 Generated {len(alerts)} performance alerts")
            return alerts
            
        except Exception as e:
            logger.error(f"❌ Alert generation failed: {e}")
            return []
    
    async def calculate_domain_summary(self, domain: str) -> DomainRankingSummary:
        """
        Calculate comprehensive domain ranking summary.
        
        Args:
            domain: Domain to summarize
            
        Returns:
            Domain ranking summary with key metrics
        """
        try:
            logger.info(f"📊 Calculating domain summary for {domain}")
            
            async with serpbear_client as client:
                keywords = await client.get_keywords(domain)
                trends = await self.analyze_ranking_trends(domain, days=7)
            
            if not keywords:
                return DomainRankingSummary(
                    domain=domain,
                    total_keywords=0,
                    average_position=0.0,
                    keywords_top_10=0,
                    keywords_top_3=0,
                    keywords_improved=0,
                    keywords_declined=0,
                    keywords_stable=0,
                    visibility_score=0.0,
                    last_updated=str(datetime.now())
                )
            
            # Calculate position metrics
            positions = [k.position for k in keywords if k.position and k.position > 0]
            average_position = statistics.mean(positions) if positions else 0.0
            
            keywords_top_10 = len([p for p in positions if p <= 10])
            keywords_top_3 = len([p for p in positions if p <= 3])
            
            # Calculate trend metrics
            keywords_improved = len([t for t in trends if t.change > 0])
            keywords_declined = len([t for t in trends if t.change < 0])
            keywords_stable = len([t for t in trends if t.change == 0])
            
            # Calculate visibility score (weighted by position quality)
            visibility_score = 0.0
            if positions:
                # Visibility formula: weighted score based on position quality
                for pos in positions:
                    if pos <= 3:
                        visibility_score += 10  # Top 3 gets full points
                    elif pos <= 10:
                        visibility_score += 5   # Top 10 gets half points
                    elif pos <= 20:
                        visibility_score += 2   # Top 20 gets quarter points
                    else:
                        visibility_score += 0.5 # Beyond page 2 gets minimal points
                
                # Normalize to 0-100 scale
                visibility_score = min(100, (visibility_score / len(positions)) * 5)
            
            summary = DomainRankingSummary(
                domain=domain,
                total_keywords=len(keywords),
                average_position=round(average_position, 1),
                keywords_top_10=keywords_top_10,
                keywords_top_3=keywords_top_3,
                keywords_improved=keywords_improved,
                keywords_declined=keywords_declined,
                keywords_stable=keywords_stable,
                visibility_score=round(visibility_score, 1),
                last_updated=str(datetime.now())
            )
            
            logger.info(f"✅ Domain summary calculated: {summary.total_keywords} keywords, {summary.visibility_score}% visibility")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Domain summary calculation failed: {e}")
            return DomainRankingSummary(
                domain=domain,
                total_keywords=0,
                average_position=0.0,
                keywords_top_10=0,
                keywords_top_3=0,
                keywords_improved=0,
                keywords_declined=0,
                keywords_stable=0,
                visibility_score=0.0,
                last_updated=str(datetime.now())
            )
    
    async def process_daily_rankings(self, domains: List[str]) -> Dict[str, Any]:
        """
        Complete daily ranking processing workflow.
        
        Args:
            domains: List of domains to process
            
        Returns:
            Summary of processing results
        """
        try:
            logger.info(f"🚀 Starting daily ranking processing for {len(domains)} domains")
            
            processing_results = {
                "timestamp": str(datetime.now()),
                "domains_processed": 0,
                "total_keywords": 0,
                "alerts_generated": 0,
                "domain_summaries": {},
                "top_performers": [],
                "attention_needed": [],
                "processing_errors": []
            }
            
            # Fetch all rankings
            domain_rankings = await self.fetch_daily_rankings(domains)
            
            for domain, keywords in domain_rankings.items():
                try:
                    # Analyze trends
                    trends = await self.analyze_ranking_trends(domain)
                    
                    # Generate alerts
                    alerts = await self.generate_performance_alerts(trends)
                    
                    # Calculate summary
                    summary = await self.calculate_domain_summary(domain)
                    
                    # Update processing results
                    processing_results["domains_processed"] += 1
                    processing_results["total_keywords"] += len(keywords)
                    processing_results["alerts_generated"] += len(alerts)
                    processing_results["domain_summaries"][domain] = summary.dict()
                    
                    # Categorize performance
                    if summary.visibility_score >= 70:
                        processing_results["top_performers"].append({
                            "domain": domain,
                            "visibility_score": summary.visibility_score,
                            "keywords_top_10": summary.keywords_top_10
                        })
                    elif summary.keywords_declined > summary.keywords_improved:
                        processing_results["attention_needed"].append({
                            "domain": domain,
                            "declined_keywords": summary.keywords_declined,
                            "critical_alerts": len([a for a in alerts if a.severity == "critical"])
                        })
                    
                    logger.info(f"✅ Processed {domain}: {len(keywords)} keywords, {len(alerts)} alerts")
                    
                except Exception as domain_error:
                    error_msg = f"Processing failed for {domain}: {domain_error}"
                    logger.error(error_msg)
                    processing_results["processing_errors"].append(error_msg)
            
            logger.info(f"🏁 Daily ranking processing complete: {processing_results}")
            return processing_results
            
        except Exception as e:
            logger.error(f"❌ Daily ranking processing failed: {e}")
            return {"error": str(e)}
    
    async def get_ranking_insights(self, domain: str, days: int = 30) -> Dict[str, Any]:
        """
        Generate comprehensive ranking insights for dashboard.
        
        Args:
            domain: Domain to analyze
            days: Number of days to analyze
            
        Returns:
            Comprehensive insights dictionary
        """
        try:
            logger.info(f"🔍 Generating ranking insights for {domain}")
            
            # Gather all data
            trends = await self.analyze_ranking_trends(domain, days)
            alerts = await self.generate_performance_alerts(trends)
            summary = await self.calculate_domain_summary(domain)
            
            # Calculate additional insights
            insights = {
                "domain": domain,
                "analysis_period": f"{days} days",
                "summary": summary.dict(),
                "trends": {
                    "total_keywords": len(trends),
                    "improving": len([t for t in trends if t.trend_direction == "improving"]),
                    "declining": len([t for t in trends if t.trend_direction == "declining"]),
                    "stable": len([t for t in trends if t.trend_direction == "stable"]),
                    "strong_trends": len([t for t in trends if t.trend_strength == "strong"])
                },
                "alerts": {
                    "total": len(alerts),
                    "critical": len([a for a in alerts if a.severity == "critical"]),
                    "high": len([a for a in alerts if a.severity == "high"]),
                    "recent_alerts": [a.dict() for a in alerts[:5]]  # Top 5 alerts
                },
                "top_opportunities": [
                    {
                        "keyword": t.keyword,
                        "current_position": t.current_position,
                        "potential": f"Could reach top 10" if t.current_position and t.current_position <= 20 else "Monitor closely"
                    }
                    for t in sorted(trends, key=lambda x: x.current_position or 999)[:5]
                ],
                "performance_grade": self._calculate_performance_grade(summary),
                "recommendations": self._generate_recommendations(trends, alerts, summary),
                "timestamp": str(datetime.now())
            }
            
            logger.info(f"📈 Generated comprehensive insights for {domain}")
            return insights
            
        except Exception as e:
            logger.error(f"❌ Insights generation failed: {e}")
            return {"error": str(e)}
    
    def _calculate_performance_grade(self, summary: DomainRankingSummary) -> str:
        """Calculate overall performance grade A-F."""
        score = summary.visibility_score
        
        if score >= 85:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 55:
            return "C"
        elif score >= 40:
            return "D"
        else:
            return "F"
    
    def _generate_recommendations(
        self, 
        trends: List[RankingTrend], 
        alerts: List[PerformanceAlert], 
        summary: DomainRankingSummary
    ) -> List[str]:
        """Generate actionable SEO recommendations."""
        recommendations = []
        
        # Based on alerts
        critical_alerts = [a for a in alerts if a.severity == "critical"]
        if critical_alerts:
            recommendations.append(f"🚨 Immediate attention: {len(critical_alerts)} keywords dropped significantly - review content and technical SEO")
        
        # Based on trends
        declining_trends = [t for t in trends if t.trend_direction == "declining"]
        if len(declining_trends) > len(trends) * 0.3:  # More than 30% declining
            recommendations.append("📉 High decline rate detected - audit recent site changes and competitor activity")
        
        # Based on summary
        if summary.keywords_top_10 / max(summary.total_keywords, 1) < 0.2:  # Less than 20% in top 10
            recommendations.append("🎯 Focus on improving top 20 keywords to reach page 1 - optimize content and build authority")
        
        if summary.visibility_score < 50:
            recommendations.append("🔍 Low visibility score - consider expanding keyword portfolio and improving on-page SEO")
        
        # Opportunity-based recommendations
        near_top_10 = [t for t in trends if t.current_position and 11 <= t.current_position <= 20]
        if near_top_10:
            recommendations.append(f"⭐ Quick wins available: {len(near_top_10)} keywords on page 2 could reach page 1 with optimization")
        
        return recommendations[:5]  # Limit to top 5 recommendations


# Global rank processor instance
rank_processor = RankProcessor()


async def daily_ranking_workflow(domains: List[str] = None) -> Dict[str, Any]:
    """
    Execute complete daily ranking workflow.
    
    Args:
        domains: List of domains to process (auto-detect if None)
        
    Returns:
        Workflow execution summary
    """
    try:
        if not domains:
            # Auto-detect domains from SerpBear
            async with serpbear_client as client:
                serpbear_domains = await client.get_domains()
                domains = [d.domain for d in serpbear_domains]
        
        if not domains:
            logger.warning("No domains found for ranking processing")
            return {"error": "No domains configured"}
        
        logger.info(f"🌅 Starting daily ranking workflow for domains: {domains}")
        
        # Execute complete processing
        results = await rank_processor.process_daily_rankings(domains)
        
        logger.info(f"✅ Daily ranking workflow completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"❌ Daily ranking workflow failed: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Test the rank processor
    async def main():
        print("Testing rank processing...")
        result = await daily_ranking_workflow(["example.com"])
        print(f"Result: {result}")
    
    asyncio.run(main())