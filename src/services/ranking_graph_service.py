"""
Ranking Graph Service for Neo4j Integration.

This service handles:
- Storing SerpBear ranking data in Neo4j graph
- Creating relationships between keywords, content, and rankings
- Historical ranking data management
- Graph-based ranking analytics and insights

Integrates ranking data with our SEO Knowledge Graph structure.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date, timedelta
from pydantic import BaseModel, Field

from .serpbear_client import SerpBearKeyword, RankingUpdate
from .rank_processor import RankingTrend, PerformanceAlert, DomainRankingSummary
from ..database.neo4j_client import neo4j_client

logger = logging.getLogger(__name__)


class RankingNode(BaseModel):
    """Model for ranking node in Neo4j."""
    keyword_id: int
    keyword: str
    domain: str
    device: str
    country: str
    position: Optional[int]
    url: Optional[str]
    date: str
    search_volume: Optional[int] = None
    competition: Optional[float] = None


class RankingRelationship(BaseModel):
    """Model for ranking relationships in Neo4j."""
    source_node: str
    target_node: str
    relationship_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class RankingGraphService:
    """
    Neo4j integration service for ranking data.
    
    This service extends our SEO Knowledge Graph with:
    1. Keyword ranking nodes and historical data
    2. Relationships between content, keywords, and rankings
    3. Performance trend analysis in graph structure
    4. Competitive landscape insights
    """
    
    def __init__(self, organization_id: str = "demo-org"):
        """
        Initialize ranking graph service.
        
        Args:
            organization_id: Organization context for operations
        """
        self.organization_id = organization_id
        logger.info(f"Ranking graph service initialized for org: {organization_id}")
    
    async def initialize_ranking_schema(self):
        """Initialize Neo4j schema for ranking data."""
        try:
            logger.info("🔧 Initializing ranking data schema in Neo4j")
            
            # Create ranking-specific constraints and indexes
            schema_queries = [
                # Ranking node constraints
                """
                CREATE CONSTRAINT ranking_unique IF NOT EXISTS 
                FOR (r:Ranking) REQUIRE (r.id, r.organization_id) IS UNIQUE
                """,
                
                # Keyword ranking constraints
                """
                CREATE CONSTRAINT keyword_ranking_unique IF NOT EXISTS 
                FOR (kr:KeywordRanking) REQUIRE (kr.keyword_id, kr.date, kr.device, kr.organization_id) IS UNIQUE
                """,
                
                # Domain constraints
                """
                CREATE CONSTRAINT domain_unique IF NOT EXISTS 
                FOR (d:Domain) REQUIRE (d.domain, d.organization_id) IS UNIQUE
                """,
                
                # Indexes for performance
                """
                CREATE INDEX ranking_date_idx IF NOT EXISTS 
                FOR (r:Ranking) ON (r.date)
                """,
                
                """
                CREATE INDEX keyword_ranking_position_idx IF NOT EXISTS 
                FOR (kr:KeywordRanking) ON (kr.position)
                """,
                
                """
                CREATE INDEX domain_organization_idx IF NOT EXISTS 
                FOR (d:Domain) ON (d.organization_id)
                """
            ]
            
            for query in schema_queries:
                try:
                    await neo4j_client._execute_query(query, {})
                    logger.debug(f"✅ Schema query executed: {query[:50]}...")
                except Exception as query_error:
                    logger.warning(f"Schema query failed (may already exist): {query_error}")
            
            logger.info("✅ Ranking schema initialization complete")
            
        except Exception as e:
            logger.error(f"❌ Ranking schema initialization failed: {e}")
            raise
    
    async def store_keyword_rankings(self, rankings: List[SerpBearKeyword]) -> Dict[str, Any]:
        """
        Store keyword ranking data in Neo4j graph.
        
        Args:
            rankings: List of SerpBear keyword rankings
            
        Returns:
            Storage operation summary
        """
        try:
            logger.info(f"💾 Storing {len(rankings)} keyword rankings in Neo4j")
            
            storage_results = {
                "total_rankings": len(rankings),
                "stored_successfully": 0,
                "updated_existing": 0,
                "failed_storage": 0,
                "errors": []
            }
            
            # Set organization context
            neo4j_client.set_organization_context(self.organization_id)
            
            for ranking in rankings:
                try:
                    # Create/update keyword ranking node
                    ranking_query = """
                    MERGE (kr:KeywordRanking {
                        keyword_id: $keyword_id,
                        date: $date,
                        device: $device,
                        organization_id: $org_id
                    })
                    SET kr.keyword = $keyword,
                        kr.domain = $domain,
                        kr.country = $country,
                        kr.position = $position,
                        kr.url = $url,
                        kr.updated_at = datetime()
                    
                    // Create or update domain node
                    MERGE (d:Domain {domain: $domain, organization_id: $org_id})
                    SET d.last_updated = datetime()
                    
                    // Create or update keyword node
                    MERGE (k:Keyword {keyword: $keyword, organization_id: $org_id})
                    SET k.last_tracked = datetime()
                    
                    // Create relationships
                    MERGE (d)-[:TRACKS_KEYWORD]->(k)
                    MERGE (k)-[:HAS_RANKING]->(kr)
                    MERGE (kr)-[:BELONGS_TO_DOMAIN]->(d)
                    
                    RETURN kr.keyword_id as stored_id
                    """
                    
                    # Prepare parameters
                    today = str(date.today())
                    params = {
                        "keyword_id": ranking.id,
                        "keyword": ranking.keyword,
                        "domain": ranking.domain,
                        "device": ranking.device,
                        "country": ranking.country,
                        "position": ranking.position,
                        "url": ranking.url,
                        "date": today,
                        "org_id": self.organization_id
                    }
                    
                    # Execute query
                    result = await neo4j_client._execute_query(ranking_query, params)
                    
                    if result:
                        storage_results["stored_successfully"] += 1
                        logger.debug(f"✅ Stored ranking: {ranking.keyword} #{ranking.position}")
                    else:
                        storage_results["updated_existing"] += 1
                        logger.debug(f"🔄 Updated ranking: {ranking.keyword} #{ranking.position}")
                
                except Exception as ranking_error:
                    storage_results["failed_storage"] += 1
                    error_msg = f"Failed to store {ranking.keyword}: {ranking_error}"
                    storage_results["errors"].append(error_msg)
                    logger.error(error_msg)
            
            logger.info(f"💾 Ranking storage complete: {storage_results['stored_successfully']} stored, {storage_results['failed_storage']} failed")
            return storage_results
            
        except Exception as e:
            logger.error(f"❌ Keyword ranking storage failed: {e}")
            return {"error": str(e)}
    
    async def store_ranking_history(self, keyword_id: int, history: Dict[str, int]) -> bool:
        """
        Store historical ranking data for a keyword.
        
        Args:
            keyword_id: SerpBear keyword ID
            history: Dictionary mapping date strings to positions
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"📚 Storing ranking history for keyword {keyword_id}")
            
            # Set organization context
            neo4j_client.set_organization_context(self.organization_id)
            
            for date_str, position in history.items():
                try:
                    # Validate date format
                    date.fromisoformat(date_str)
                    
                    history_query = """
                    MATCH (k:Keyword)-[:HAS_RANKING]->(kr:KeywordRanking {keyword_id: $keyword_id})
                    WHERE kr.organization_id = $org_id
                    
                    MERGE (kr)-[:HAS_HISTORY]->(h:RankingHistory {
                        keyword_id: $keyword_id,
                        date: $date,
                        organization_id: $org_id
                    })
                    SET h.position = $position,
                        h.updated_at = datetime()
                    
                    RETURN h.date as stored_date
                    """
                    
                    params = {
                        "keyword_id": keyword_id,
                        "date": date_str,
                        "position": position,
                        "org_id": self.organization_id
                    }
                    
                    await neo4j_client._execute_query(history_query, params)
                    
                except Exception as date_error:
                    logger.warning(f"Failed to store history for {date_str}: {date_error}")
                    continue
            
            logger.info(f"✅ Stored {len(history)} historical data points")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ranking history storage failed: {e}")
            return False
    
    async def link_rankings_to_content(self) -> Dict[str, Any]:
        """
        Create relationships between rankings and content based on keyword matching.
        
        Returns:
            Linking operation summary
        """
        try:
            logger.info("🔗 Linking ranking data to content nodes")
            
            # Query to find and link related content
            linking_query = """
            MATCH (kr:KeywordRanking {organization_id: $org_id})
            MATCH (c:Content {organization_id: $org_id})
            
            // Link if keyword appears in content title or keywords
            WHERE toLower(c.title) CONTAINS toLower(kr.keyword)
               OR any(kw IN c.keywords WHERE toLower(kw) CONTAINS toLower(kr.keyword))
            
            MERGE (c)-[r:TARGETS_KEYWORD]->(kr)
            SET r.relevance_score = CASE
                WHEN toLower(c.title) CONTAINS toLower(kr.keyword) THEN 0.9
                ELSE 0.7
            END,
            r.linked_at = datetime()
            
            RETURN count(r) as links_created
            """
            
            result = await neo4j_client._execute_query(
                linking_query, 
                {"org_id": self.organization_id}
            )
            
            links_created = result[0]["links_created"] if result else 0
            
            logger.info(f"🔗 Created {links_created} content-ranking links")
            return {
                "links_created": links_created,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ Content-ranking linking failed: {e}")
            return {"error": str(e)}
    
    async def get_ranking_insights(self, domain: str, days: int = 30) -> Dict[str, Any]:
        """
        Get comprehensive ranking insights from Neo4j graph.
        
        Args:
            domain: Domain to analyze
            days: Number of days to analyze
            
        Returns:
            Graph-based ranking insights
        """
        try:
            logger.info(f"📊 Generating graph-based ranking insights for {domain}")
            
            # Set organization context
            neo4j_client.set_organization_context(self.organization_id)
            
            # Complex insight query
            insights_query = """
            MATCH (d:Domain {domain: $domain, organization_id: $org_id})
            MATCH (d)-[:TRACKS_KEYWORD]->(k:Keyword)-[:HAS_RANKING]->(kr:KeywordRanking)
            
            // Get recent rankings
            WHERE kr.date >= date() - duration({days: $days})
            
            // Calculate aggregated metrics
            WITH d, k, kr,
                 collect(kr.position) as positions,
                 min(kr.position) as best_position,
                 max(kr.position) as worst_position,
                 avg(kr.position) as avg_position
            
            // Find related content
            OPTIONAL MATCH (c:Content)-[:TARGETS_KEYWORD]->(kr)
            
            // Calculate insights
            RETURN {
                domain: d.domain,
                keyword: k.keyword,
                current_position: head(positions),
                best_position: best_position,
                worst_position: worst_position,
                average_position: avg_position,
                position_count: size(positions),
                volatility: toFloat(worst_position - best_position),
                related_content: collect(DISTINCT c.title)[0..3],
                trend: CASE
                    WHEN size(positions) >= 3 AND positions[-1] < positions[0] THEN 'improving'
                    WHEN size(positions) >= 3 AND positions[-1] > positions[0] THEN 'declining'
                    ELSE 'stable'
                END
            } as insight
            ORDER BY insight.current_position
            LIMIT 50
            """
            
            result = await neo4j_client._execute_query(
                insights_query,
                {
                    "domain": domain,
                    "days": days,
                    "org_id": self.organization_id
                }
            )
            
            insights = [record["insight"] for record in result]
            
            # Calculate summary metrics
            if insights:
                positions = [i["current_position"] for i in insights if i["current_position"]]
                summary = {
                    "total_keywords": len(insights),
                    "average_position": sum(positions) / len(positions) if positions else 0,
                    "top_10_count": len([p for p in positions if p <= 10]),
                    "top_3_count": len([p for p in positions if p <= 3]),
                    "improving_trends": len([i for i in insights if i["trend"] == "improving"]),
                    "declining_trends": len([i for i in insights if i["trend"] == "declining"])
                }
            else:
                summary = {
                    "total_keywords": 0,
                    "average_position": 0,
                    "top_10_count": 0,
                    "top_3_count": 0,
                    "improving_trends": 0,
                    "declining_trends": 0
                }
            
            graph_insights = {
                "domain": domain,
                "analysis_period": f"{days} days",
                "summary": summary,
                "keyword_insights": insights[:20],  # Top 20 insights
                "recommendations": self._generate_graph_recommendations(insights),
                "timestamp": str(datetime.now())
            }
            
            logger.info(f"📈 Generated graph insights: {summary['total_keywords']} keywords analyzed")
            return graph_insights
            
        except Exception as e:
            logger.error(f"❌ Graph insights generation failed: {e}")
            return {"error": str(e)}
    
    async def get_competitive_landscape(self, keywords: List[str]) -> Dict[str, Any]:
        """
        Analyze competitive landscape using graph data.
        
        Args:
            keywords: List of keywords to analyze
            
        Returns:
            Competitive analysis based on graph relationships
        """
        try:
            logger.info(f"🥊 Analyzing competitive landscape for {len(keywords)} keywords")
            
            # Competitive landscape query
            landscape_query = """
            UNWIND $keywords as target_keyword
            
            MATCH (k:Keyword {keyword: target_keyword, organization_id: $org_id})
            MATCH (k)-[:HAS_RANKING]->(kr:KeywordRanking)
            
            // Find domains competing for same keywords
            MATCH (other_kr:KeywordRanking)-[:BELONGS_TO_DOMAIN]->(other_d:Domain)
            WHERE other_kr.keyword = target_keyword 
              AND other_kr.organization_id = $org_id
              AND other_kr.position <= 20  // Top 20 only
            
            // Calculate competitive metrics
            WITH target_keyword, 
                 collect(DISTINCT {
                     domain: other_d.domain,
                     position: other_kr.position,
                     device: other_kr.device
                 }) as competitors
            
            RETURN {
                keyword: target_keyword,
                total_competitors: size(competitors),
                top_3_competitors: [c IN competitors WHERE c.position <= 3],
                our_position: head([c.position IN competitors WHERE c.domain CONTAINS $our_domain]),
                competition_level: CASE
                    WHEN size(competitors) >= 15 THEN 'high'
                    WHEN size(competitors) >= 8 THEN 'medium'
                    ELSE 'low'
                END
            } as landscape
            """
            
            # Note: This would need the actual domain parameter
            our_domain = "example.com"  # This should be passed as parameter
            
            result = await neo4j_client._execute_query(
                landscape_query,
                {
                    "keywords": keywords,
                    "org_id": self.organization_id,
                    "our_domain": our_domain
                }
            )
            
            landscape_data = [record["landscape"] for record in result]
            
            # Aggregate competitive insights
            competitive_insights = {
                "keywords_analyzed": len(keywords),
                "high_competition": len([l for l in landscape_data if l["competition_level"] == "high"]),
                "medium_competition": len([l for l in landscape_data if l["competition_level"] == "medium"]),
                "low_competition": len([l for l in landscape_data if l["competition_level"] == "low"]),
                "keyword_details": landscape_data,
                "opportunities": [
                    l for l in landscape_data 
                    if l["competition_level"] == "low" and (l["our_position"] or 999) > 10
                ],
                "timestamp": str(datetime.now())
            }
            
            logger.info(f"🥊 Competitive analysis complete: {len(landscape_data)} keywords analyzed")
            return competitive_insights
            
        except Exception as e:
            logger.error(f"❌ Competitive landscape analysis failed: {e}")
            return {"error": str(e)}
    
    def _generate_graph_recommendations(self, insights: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on graph insights."""
        recommendations = []
        
        if not insights:
            return ["No ranking data available for analysis"]
        
        # Analyze patterns in insights
        high_volatility = [i for i in insights if i.get("volatility", 0) > 20]
        improving_trends = [i for i in insights if i.get("trend") == "improving"]
        declining_trends = [i for i in insights if i.get("trend") == "declining"]
        near_top_10 = [i for i in insights if i.get("current_position", 999) in range(11, 21)]
        
        # Generate specific recommendations
        if high_volatility:
            recommendations.append(f"🎯 {len(high_volatility)} keywords show high volatility - focus on content stability and technical SEO")
        
        if len(declining_trends) > len(improving_trends):
            recommendations.append("📉 More keywords declining than improving - audit recent changes and competitor activity")
        
        if near_top_10:
            recommendations.append(f"⭐ {len(near_top_10)} keywords on page 2 - potential quick wins with targeted optimization")
        
        # Content-based recommendations
        content_linked = [i for i in insights if i.get("related_content")]
        if len(content_linked) < len(insights) * 0.5:
            recommendations.append("🔗 Many keywords lack supporting content - create targeted content for better rankings")
        
        return recommendations[:5]
    
    async def cleanup_old_rankings(self, days_to_keep: int = 90) -> Dict[str, Any]:
        """
        Clean up old ranking data to maintain graph performance.
        
        Args:
            days_to_keep: Number of days of data to retain
            
        Returns:
            Cleanup operation summary
        """
        try:
            logger.info(f"🧹 Cleaning up ranking data older than {days_to_keep} days")
            
            cleanup_query = """
            MATCH (kr:KeywordRanking {organization_id: $org_id})
            WHERE kr.date < date() - duration({days: $days_to_keep})
            
            OPTIONAL MATCH (kr)-[r]-()
            
            WITH kr, collect(r) as relationships, count(r) as rel_count
            
            // Delete relationships first
            FOREACH (rel IN relationships | DELETE rel)
            
            // Delete the ranking node
            DELETE kr
            
            RETURN count(kr) as deleted_rankings
            """
            
            result = await neo4j_client._execute_query(
                cleanup_query,
                {
                    "org_id": self.organization_id,
                    "days_to_keep": days_to_keep
                }
            )
            
            deleted_count = result[0]["deleted_rankings"] if result else 0
            
            logger.info(f"🧹 Cleaned up {deleted_count} old ranking records")
            return {
                "deleted_rankings": deleted_count,
                "days_retained": days_to_keep,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ Ranking cleanup failed: {e}")
            return {"error": str(e)}


# Global ranking graph service instance
ranking_graph_service = RankingGraphService()


async def sync_rankings_to_graph(domains: List[str] = None) -> Dict[str, Any]:
    """
    Complete workflow to sync SerpBear rankings to Neo4j graph.
    
    Args:
        domains: List of domains to sync (auto-detect if None)
        
    Returns:
        Sync operation summary
    """
    try:
        logger.info("🔄 Starting complete ranking sync to graph")
        
        from .serpbear_client import serpbear_client
        
        # Initialize schema
        await ranking_graph_service.initialize_ranking_schema()
        
        sync_results = {
            "domains_processed": 0,
            "total_rankings_stored": 0,
            "content_links_created": 0,
            "errors": []
        }
        
        async with serpbear_client as client:
            # Auto-detect domains if not provided
            if not domains:
                serpbear_domains = await client.get_domains()
                domains = [d.domain for d in serpbear_domains]
            
            for domain in domains:
                try:
                    # Get keyword rankings
                    keywords = await client.get_keywords(domain)
                    
                    if keywords:
                        # Store rankings in graph
                        storage_result = await ranking_graph_service.store_keyword_rankings(keywords)
                        sync_results["total_rankings_stored"] += storage_result.get("stored_successfully", 0)
                        
                        # Store historical data
                        for keyword in keywords:
                            if keyword.history:
                                await ranking_graph_service.store_ranking_history(
                                    keyword.id, keyword.history
                                )
                    
                    sync_results["domains_processed"] += 1
                    logger.info(f"✅ Synced {domain}: {len(keywords)} keywords")
                    
                except Exception as domain_error:
                    error_msg = f"Failed to sync {domain}: {domain_error}"
                    logger.error(error_msg)
                    sync_results["errors"].append(error_msg)
        
        # Link rankings to content
        link_result = await ranking_graph_service.link_rankings_to_content()
        sync_results["content_links_created"] = link_result.get("links_created", 0)
        
        logger.info(f"✅ Ranking sync complete: {sync_results}")
        return sync_results
        
    except Exception as e:
        logger.error(f"❌ Ranking sync failed: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Test the ranking graph service
    async def main():
        print("Testing ranking graph integration...")
        result = await sync_rankings_to_graph(["example.com"])
        print(f"Result: {result}")
    
    asyncio.run(main())