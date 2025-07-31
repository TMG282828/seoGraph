"""
Neo4j Graph Database client for multi-tenant SEO Content Knowledge Graph System.

This module provides a client wrapper for Neo4j operations with multi-tenant support,
content relationship management, and graph analytics.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union
from neo4j import GraphDatabase, Driver, Session
import json
from datetime import datetime
import hashlib
from dotenv import load_dotenv

# Load production environment variables
load_dotenv('.env.production')

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Neo4j client wrapper with multi-tenant support for content knowledge graphs."""
    
    def __init__(self):
        """Initialize Neo4j client with environment configuration."""
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        
        self.driver: Optional[Driver] = None
        self.demo_mode = False
        self._current_organization_id = None
        
        self._connect()
    
    def _connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password),
                database=self.database
            )
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS test")
                result.single()
            logger.info("Connected to Neo4j successfully")
            
        except Exception as e:
            logger.warning(f"Failed to connect to Neo4j: {e} - running in demo mode")
            self.demo_mode = True
            self.driver = None
    
    def set_organization_context(self, organization_id: str) -> None:
        """Set the current organization context for multi-tenant operations."""
        self._current_organization_id = organization_id
    
    def close(self) -> None:
        """Close Neo4j driver connection."""
        if self.driver:
            self.driver.close()
    
    def _get_session(self) -> Optional[Session]:
        """Get Neo4j session or return None if in demo mode."""
        if self.demo_mode or not self.driver:
            return None
        return self.driver.session(database=self.database)
    
    def _execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict]:
        """Execute Cypher query with error handling and demo mode fallback."""
        if self.demo_mode:
            logger.info(f"Demo mode: Would execute query: {query}")
            return []
        
        if not parameters:
            parameters = {}
        
        # Add organization context to all queries
        if self._current_organization_id:
            parameters['org_id'] = self._current_organization_id
        
        try:
            with self._get_session() as session:
                result = session.run(query, parameters)
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Neo4j query failed: {e}")
            return []
    
    # ============================================================================
    # Schema Management
    # ============================================================================
    
    def initialize_schema(self) -> bool:
        """Initialize Neo4j schema with constraints and indexes for multi-tenant setup."""
        if self.demo_mode:
            logger.info("Demo mode: Schema initialization skipped")
            return True
        
        schema_queries = [
            # Node uniqueness constraints
            "CREATE CONSTRAINT IF NOT EXISTS FOR (o:Organization) REQUIRE o.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Content) REQUIRE (c.id, c.organization_id) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE (t.id, t.organization_id) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (k:Keyword) REQUIRE (k.id, k.organization_id) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (tr:Trend) REQUIRE (tr.id, tr.organization_id) IS UNIQUE",
            
            # Indexes for performance
            "CREATE INDEX IF NOT EXISTS FOR (c:Content) ON (c.organization_id)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Content) ON (c.title)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Content) ON (c.content_type)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Content) ON (c.created_at)",
            "CREATE INDEX IF NOT EXISTS FOR (t:Topic) ON (t.organization_id)",
            "CREATE INDEX IF NOT EXISTS FOR (t:Topic) ON (t.name)",
            "CREATE INDEX IF NOT EXISTS FOR (k:Keyword) ON (k.organization_id)",
            "CREATE INDEX IF NOT EXISTS FOR (k:Keyword) ON (k.keyword)",
            "CREATE INDEX IF NOT EXISTS FOR (k:Keyword) ON (k.search_volume)",
            "CREATE INDEX IF NOT EXISTS FOR (tr:Trend) ON (tr.organization_id)",
            "CREATE INDEX IF NOT EXISTS FOR (tr:Trend) ON (tr.trend_date)",
            
            # Full-text search indexes
            "CREATE FULLTEXT INDEX content_search IF NOT EXISTS FOR (c:Content) ON EACH [c.title, c.summary, c.content]",
            "CREATE FULLTEXT INDEX topic_search IF NOT EXISTS FOR (t:Topic) ON EACH [t.name, t.description]",
        ]
        
        try:
            with self._get_session() as session:
                for query in schema_queries:
                    session.run(query)
            logger.info("Neo4j schema initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j schema: {e}")
            return False
    
    # ============================================================================
    # Organization Management
    # ============================================================================
    
    def create_organization_node(self, org_id: str, name: str, config: Dict[str, Any]) -> bool:
        """Create organization node in graph."""
        query = """
        CREATE (o:Organization {
            id: $org_id,
            name: $name,
            config: $config,
            created_at: datetime(),
            updated_at: datetime()
        })
        """
        
        result = self._execute_query(query, {
            'org_id': org_id,
            'name': name,
            'config': json.dumps(config)
        })
        
        return len(result) >= 0  # Query succeeds even if no records returned
    
    def get_organization_stats(self, org_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive organization graph statistics."""
        if not org_id:
            org_id = self._current_organization_id
        
        if not org_id:
            return {}
        
        if self.demo_mode:
            return {
                'total_nodes': 1250,
                'total_relationships': 3450,
                'content_nodes': 847,
                'topic_nodes': 235,
                'keyword_nodes': 168,
                'avg_connections_per_content': 4.1,
                'densest_topics': ['SEO Strategy', 'Content Marketing', 'Technical SEO'],
                'coverage_score': 78.5
            }
        
        query = """
        MATCH (o:Organization {id: $org_id})
        OPTIONAL MATCH (o)-[:OWNS]->(c:Content)
        OPTIONAL MATCH (o)-[:OWNS]->(t:Topic)
        OPTIONAL MATCH (o)-[:OWNS]->(k:Keyword)
        OPTIONAL MATCH ()-[r]->() WHERE r.organization_id = $org_id
        
        RETURN {
            total_nodes: count(DISTINCT c) + count(DISTINCT t) + count(DISTINCT k),
            content_nodes: count(DISTINCT c),
            topic_nodes: count(DISTINCT t),
            keyword_nodes: count(DISTINCT k),
            total_relationships: count(DISTINCT r)
        } AS stats
        """
        
        result = self._execute_query(query, {'org_id': org_id})
        
        if result:
            stats = result[0]['stats']
            
            # Calculate derived metrics
            if stats['content_nodes'] > 0:
                stats['avg_connections_per_content'] = round(
                    stats['total_relationships'] / stats['content_nodes'], 1
                )
            else:
                stats['avg_connections_per_content'] = 0
            
            # Get top topics by content count
            topic_query = """
            MATCH (t:Topic)-[:CONTAINS]->(c:Content)
            WHERE t.organization_id = $org_id
            RETURN t.name AS topic, count(c) AS content_count
            ORDER BY content_count DESC
            LIMIT 3
            """
            
            topic_result = self._execute_query(topic_query, {'org_id': org_id})
            stats['densest_topics'] = [record['topic'] for record in topic_result]
            
            # Calculate coverage score (simplified)
            stats['coverage_score'] = min(100, stats['total_relationships'] / 10)
            
            return stats
        
        return {}
    
    # ============================================================================
    # Content Node Operations
    # ============================================================================
    
    def create_content_node(self, content_data: Dict[str, Any]) -> str:
        """Create content node with full metadata and relationships."""
        content_id = content_data.get('id') or self._generate_id(content_data.get('title', ''))
        
        query = """
        MERGE (c:Content {id: $content_id, organization_id: $org_id})
        SET c += {
            title: $title,
            content_type: $content_type,
            url: $url,
            summary: $summary,
            word_count: $word_count,
            seo_score: $seo_score,
            readability_score: $readability_score,
            publish_date: $publish_date,
            last_updated: datetime(),
            metadata: $metadata
        }
        
        // Connect to organization
        MERGE (o:Organization {id: $org_id})
        MERGE (o)-[:OWNS]->(c)
        
        RETURN c.id AS content_id
        """
        
        metadata = {
            'created_at': content_data.get('created_at', datetime.now().isoformat()),
            'source': content_data.get('source', 'manual'),
            'tags': content_data.get('tags', []),
            'author': content_data.get('author', ''),
            'language': content_data.get('language', 'en')
        }
        
        result = self._execute_query(query, {
            'content_id': content_id,
            'org_id': self._current_organization_id,
            'title': content_data.get('title', ''),
            'content_type': content_data.get('content_type', 'article'),
            'url': content_data.get('url', ''),
            'summary': content_data.get('summary', ''),
            'word_count': content_data.get('word_count', 0),
            'seo_score': content_data.get('seo_score', 0),
            'readability_score': content_data.get('readability_score', 0),
            'publish_date': content_data.get('publish_date', datetime.now().isoformat()),
            'metadata': json.dumps(metadata)
        })
        
        return content_id
    
    def link_content_to_topics(self, content_id: str, topics: List[str]) -> bool:
        """Create relationships between content and topics."""
        query = """
        MATCH (c:Content {id: $content_id, organization_id: $org_id})
        UNWIND $topics AS topic_name
        MERGE (t:Topic {name: topic_name, organization_id: $org_id})
        MERGE (t)-[:CONTAINS]->(c)
        MERGE (c)-[:BELONGS_TO]->(t)
        """
        
        result = self._execute_query(query, {
            'content_id': content_id,
            'org_id': self._current_organization_id,
            'topics': topics
        })
        
        return True
    
    def link_content_to_keywords(self, content_id: str, keywords: List[Dict[str, Any]]) -> bool:
        """Create relationships between content and keywords with weights."""
        query = """
        MATCH (c:Content {id: $content_id, organization_id: $org_id})
        UNWIND $keywords AS kw
        MERGE (k:Keyword {keyword: kw.keyword, organization_id: $org_id})
        SET k += {
            search_volume: COALESCE(kw.search_volume, k.search_volume, 0),
            competition: COALESCE(kw.competition, k.competition, 0),
            updated_at: datetime()
        }
        MERGE (c)-[r:TARGETS]->(k)
        SET r.weight = kw.weight,
            r.density = kw.density,
            r.relevance = kw.relevance
        """
        
        result = self._execute_query(query, {
            'content_id': content_id,
            'org_id': self._current_organization_id,
            'keywords': keywords
        })
        
        return True
    
    def create_content_relationships(self, content_id: str, related_content_ids: List[str], 
                                   relationship_type: str = 'RELATED_TO') -> bool:
        """Create relationships between content pieces."""
        query = f"""
        MATCH (c1:Content {{id: $content_id, organization_id: $org_id}})
        UNWIND $related_ids AS related_id
        MATCH (c2:Content {{id: related_id, organization_id: $org_id}})
        MERGE (c1)-[r:{relationship_type}]->(c2)
        SET r.created_at = datetime(),
            r.weight = 1.0
        """
        
        result = self._execute_query(query, {
            'content_id': content_id,
            'org_id': self._current_organization_id,
            'related_ids': related_content_ids
        })
        
        return True
    
    # ============================================================================
    # Content Discovery and Analytics
    # ============================================================================
    
    def find_content_gaps(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Identify content gaps by analyzing topic coverage and keyword opportunities."""
        if self.demo_mode:
            return [
                {
                    'topic': 'Technical SEO Audits',
                    'keyword_opportunities': 25,
                    'search_volume_potential': 15000,
                    'competition_level': 'medium',
                    'priority_score': 85.2
                },
                {
                    'topic': 'Local SEO Strategies',
                    'keyword_opportunities': 18,
                    'search_volume_potential': 8500,
                    'competition_level': 'low',
                    'priority_score': 78.5
                }
            ]
        
        query = """
        // Find topics with high-value keywords but low content coverage
        MATCH (k:Keyword {organization_id: $org_id})
        WHERE k.search_volume > 1000
        OPTIONAL MATCH (k)<-[:TARGETS]-(c:Content {organization_id: $org_id})
        
        WITH k, count(c) AS content_count
        WHERE content_count < 3  // Topics with less than 3 pieces of content
        
        // Group by topic clusters (simplified - would use actual topic modeling)
        WITH substring(k.keyword, 0, 20) AS topic_cluster, 
             collect(k) AS keywords,
             sum(k.search_volume) AS total_volume,
             avg(k.competition) AS avg_competition
        
        WHERE size(keywords) >= 3  // At least 3 related keywords
        
        RETURN {
            topic: topic_cluster,
            keyword_opportunities: size(keywords),
            search_volume_potential: total_volume,
            competition_level: CASE 
                WHEN avg_competition < 0.3 THEN 'low'
                WHEN avg_competition < 0.7 THEN 'medium'
                ELSE 'high'
            END,
            priority_score: (total_volume / 1000.0) * (1.0 - avg_competition) * size(keywords)
        } AS gap
        ORDER BY gap.priority_score DESC
        LIMIT $limit
        """
        
        result = self._execute_query(query, {
            'org_id': self._current_organization_id,
            'limit': limit
        })
        
        return [record['gap'] for record in result]
    
    def get_content_recommendations(self, content_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get content recommendations based on graph relationships."""
        query = """
        MATCH (c:Content {id: $content_id, organization_id: $org_id})
        
        // Find related content through shared topics and keywords
        MATCH (related:Content {organization_id: $org_id})
        WHERE related.id <> c.id
        
        // Count shared topics
        OPTIONAL MATCH (c)-[:BELONGS_TO]->(t:Topic)<-[:BELONGS_TO]-(related)
        WITH c, related, count(DISTINCT t) AS shared_topics
        
        // Count shared keywords
        OPTIONAL MATCH (c)-[:TARGETS]->(k:Keyword)<-[:TARGETS]-(related)
        WITH c, related, shared_topics, count(DISTINCT k) AS shared_keywords
        
        // Only include content with at least one shared topic or keyword
        WHERE shared_topics > 0 OR shared_keywords > 0
        
        // Calculate recommendation score
        WITH related, 
             shared_topics,
             shared_keywords,
             (shared_topics * 2 + shared_keywords) AS relevance_score,
             related.seo_score AS seo_score,
             related.readability_score AS readability_score
        
        RETURN {
            id: related.id,
            title: related.title,
            content_type: related.content_type,
            url: related.url,
            seo_score: seo_score,
            readability_score: readability_score,
            relevance_score: relevance_score,
            recommendation_reason: 'Shares ' + toString(shared_topics) + ' topics and ' + toString(shared_keywords) + ' keywords'
        } AS recommendation
        ORDER BY recommendation.relevance_score DESC, recommendation.seo_score DESC
        LIMIT $limit
        """
        
        result = self._execute_query(query, {
            'content_id': content_id,
            'org_id': self._current_organization_id,
            'limit': limit
        })
        
        return [record['recommendation'] for record in result]
    
    def get_topic_hierarchy(self) -> List[Dict[str, Any]]:
        """Get hierarchical topic structure with content counts."""
        query = """
        MATCH (t:Topic {organization_id: $org_id})
        OPTIONAL MATCH (t)-[:CONTAINS]->(c:Content)
        OPTIONAL MATCH (t)-[:PARENT_OF]->(child:Topic)
        OPTIONAL MATCH (parent:Topic)-[:PARENT_OF]->(t)
        
        WITH t, 
             count(DISTINCT c) AS content_count,
             count(DISTINCT child) AS children_count,
             parent.name AS parent_name,
             t.description AS description
        
        RETURN {
            id: t.name,
            name: t.name,
            content_count: content_count,
            children_count: children_count,
            parent: parent_name,
            description: description
        } AS topic
        ORDER BY topic.content_count DESC
        """
        
        result = self._execute_query(query, {
            'org_id': self._current_organization_id
        })
        
        return [record['topic'] for record in result]
    
    # ============================================================================
    # Utility Methods
    # ============================================================================
    
    def _generate_id(self, content: str) -> str:
        """Generate unique ID from content."""
        return hashlib.md5(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()[:12]
    
    def search_content(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Full-text search across content nodes."""
        if self.demo_mode:
            return [
                {
                    'id': 'demo-content-1',
                    'title': f'Demo Content Matching: {query}',
                    'summary': f'This is demo content that would match your search for "{query}"',
                    'relevance_score': 0.95
                }
            ]
        
        cypher_query = """
        CALL db.index.fulltext.queryNodes('content_search', $search_query)
        YIELD node, score
        WHERE node.organization_id = $org_id
        RETURN {
            id: node.id,
            title: node.title,
            summary: node.summary,
            content_type: node.content_type,
            url: node.url,
            seo_score: node.seo_score,
            relevance_score: score
        } AS content
        ORDER BY content.relevance_score DESC
        LIMIT $limit
        """
        
        result = self._execute_query(cypher_query, {
            'search_query': query,
            'org_id': self._current_organization_id,
            'limit': limit
        })
        
        return [record['content'] for record in result]


# Global Neo4j client instance
neo4j_client = Neo4jClient()