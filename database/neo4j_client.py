"""
Neo4j database client for the SEO Content Knowledge Graph System.

This module provides an async Neo4j client with connection management,
graph schema operations, and content relationship management.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

import structlog
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession, AsyncTransaction
from neo4j.exceptions import ServiceUnavailable, TransientError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from config.settings import get_settings

# Import OpenTelemetry tracing
try:
    from monitoring.otel_monitor import trace_db
    OTEL_TRACING_AVAILABLE = True
except ImportError:
    OTEL_TRACING_AVAILABLE = False
    # No-op decorator if monitoring not available
    def trace_db(database: str, operation: str):
        def decorator(func):
            return func
        return decorator

logger = structlog.get_logger(__name__)


class Neo4jConnectionError(Exception):
    """Raised when Neo4j connection fails."""
    pass


class Neo4jClient:
    """
    Async Neo4j client with connection pooling and retry logic.
    
    Provides methods for graph operations, schema management,
    and content relationship handling with proper error handling
    and connection management.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        max_connection_pool_size: int = 50,
        max_transaction_retry_time: int = 30,
    ):
        """
        Initialize Neo4j client.

        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
            database: Neo4j database name
            max_connection_pool_size: Maximum connection pool size
            max_transaction_retry_time: Maximum transaction retry time in seconds
        """
        settings = get_settings()
        
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_username
        self.password = password or settings.neo4j_password
        self.database = database or settings.neo4j_database
        self.max_connection_pool_size = max_connection_pool_size
        self.max_transaction_retry_time = max_transaction_retry_time
        
        self._driver: Optional[AsyncDriver] = None
        self._is_connected = False

    async def connect(self) -> None:
        """
        Establish connection to Neo4j database.
        
        Raises:
            Neo4jConnectionError: If connection fails
        """
        try:
            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_pool_size=self.max_connection_pool_size,
                max_transaction_retry_time=self.max_transaction_retry_time,
            )
            
            # Verify connectivity
            await self.health_check()
            self._is_connected = True
            
            logger.info(
                "Neo4j connection established",
                uri=self.uri,
                database=self.database,
            )
            
        except Exception as e:
            logger.error(
                "Failed to connect to Neo4j",
                uri=self.uri,
                error=str(e),
            )
            raise Neo4jConnectionError(f"Failed to connect to Neo4j: {e}") from e

    async def close(self) -> None:
        """Close Neo4j connection."""
        if self._driver:
            await self._driver.close()
            self._is_connected = False
            logger.info("Neo4j connection closed")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ServiceUnavailable, TransientError)),
    )
    async def health_check(self) -> bool:
        """
        Check Neo4j database health.
        
        Returns:
            True if database is healthy
            
        Raises:
            Neo4jConnectionError: If health check fails
        """
        if not self._driver:
            raise Neo4jConnectionError("Driver not initialized")
            
        try:
            async with self._driver.session(database=self.database) as session:
                result = await session.run("RETURN 1 as health")
                record = await result.single()
                return record["health"] == 1
                
        except Exception as e:
            logger.error("Neo4j health check failed", error=str(e))
            raise Neo4jConnectionError(f"Health check failed: {e}") from e

    @asynccontextmanager
    async def session(self, **kwargs) -> AsyncSession:
        """
        Get a Neo4j session with proper context management.
        
        Args:
            **kwargs: Additional session parameters
            
        Yields:
            AsyncSession: Neo4j async session
        """
        if not self._driver:
            raise Neo4jConnectionError("Driver not initialized. Call connect() first.")
            
        async with self._driver.session(database=self.database, **kwargs) as session:
            try:
                yield session
            except Exception as e:
                logger.error("Session error occurred", error=str(e))
                raise

    @asynccontextmanager
    async def transaction(self, **kwargs) -> AsyncTransaction:
        """
        Get a Neo4j transaction with proper context management.
        
        Args:
            **kwargs: Additional transaction parameters
            
        Yields:
            AsyncTransaction: Neo4j async transaction
        """
        async with self.session(**kwargs) as session:
            async with session.begin_transaction() as tx:
                try:
                    yield tx
                    await tx.commit()
                except Exception as e:
                    await tx.rollback()
                    logger.error("Transaction rolled back", error=str(e))
                    raise

    @trace_db("neo4j", "execute_query")
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return results.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            **kwargs: Additional session parameters
            
        Returns:
            List of result records as dictionaries
        """
        parameters = parameters or {}
        
        async with self.session(**kwargs) as session:
            result = await session.run(query, parameters)
            records = await result.data()
            
            logger.debug(
                "Query executed",
                query=query[:100] + "..." if len(query) > 100 else query,
                record_count=len(records),
            )
            
            return records

    async def execute_write_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Execute a write query in a transaction.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            **kwargs: Additional transaction parameters
            
        Returns:
            List of result records as dictionaries
        """
        parameters = parameters or {}
        
        async with self.transaction(**kwargs) as tx:
            result = await tx.run(query, parameters)
            records = await result.data()
            
            logger.debug(
                "Write query executed",
                query=query[:100] + "..." if len(query) > 100 else query,
                record_count=len(records),
            )
            
            return records

    # =============================================================================
    # Schema Management
    # =============================================================================

    async def initialize_schema(self) -> None:
        """Initialize Neo4j schema with constraints and indexes."""
        schema_queries = [
            # Constraints for uniqueness
            "CREATE CONSTRAINT content_id_unique IF NOT EXISTS FOR (c:Content) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT topic_name_unique IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT keyword_text_unique IF NOT EXISTS FOR (k:Keyword) REQUIRE k.text IS UNIQUE",
            "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
            "CREATE CONSTRAINT tenant_id_unique IF NOT EXISTS FOR (t:Tenant) REQUIRE t.id IS UNIQUE",
            
            # Indexes for performance
            "CREATE INDEX content_tenant_idx IF NOT EXISTS FOR (c:Content) ON (c.tenant_id)",
            "CREATE INDEX content_status_idx IF NOT EXISTS FOR (c:Content) ON (c.status)",
            "CREATE INDEX content_type_idx IF NOT EXISTS FOR (c:Content) ON (c.content_type)",
            "CREATE INDEX content_created_idx IF NOT EXISTS FOR (c:Content) ON (c.created_at)",
            "CREATE INDEX topic_tenant_idx IF NOT EXISTS FOR (t:Topic) ON (t.tenant_id)",
            "CREATE INDEX keyword_tenant_idx IF NOT EXISTS FOR (k:Keyword) ON (k.tenant_id)",
            "CREATE INDEX keyword_search_volume_idx IF NOT EXISTS FOR (k:Keyword) ON (k.search_volume)",
            
            # Full-text search indexes
            "CREATE FULLTEXT INDEX content_fulltext IF NOT EXISTS FOR (c:Content) ON EACH [c.title, c.content]",
            "CREATE FULLTEXT INDEX topic_fulltext IF NOT EXISTS FOR (t:Topic) ON EACH [t.name, t.description]",
        ]
        
        for query in schema_queries:
            try:
                await self.execute_write_query(query)
                logger.debug("Schema query executed", query=query)
            except Exception as e:
                # Log but don't fail if constraint/index already exists
                logger.warning("Schema query failed (may already exist)", query=query, error=str(e))
        
        logger.info("Neo4j schema initialization completed")

    # =============================================================================
    # Content Operations
    # =============================================================================

    async def create_content_node(
        self,
        content_data: Dict[str, Any],
        tenant_id: str
    ) -> str:
        """
        Create a content node in the graph.
        
        Args:
            content_data: Content node properties
            tenant_id: Tenant identifier for multi-tenancy
            
        Returns:
            Created node ID
        """
        content_data["tenant_id"] = tenant_id
        
        query = """
        CREATE (c:Content $content_data)
        RETURN c.id as node_id
        """
        
        result = await self.execute_write_query(
            query,
            {"content_data": content_data}
        )
        
        if not result:
            raise ValueError("Failed to create content node")
            
        node_id = result[0]["node_id"]
        logger.info("Content node created", node_id=node_id, tenant_id=tenant_id)
        
        return node_id

    async def create_topic_node(
        self,
        topic_name: str,
        topic_data: Optional[Dict[str, Any]] = None,
        tenant_id: str = None
    ) -> str:
        """
        Create or get existing topic node.
        
        Args:
            topic_name: Topic name
            topic_data: Additional topic properties
            tenant_id: Tenant identifier
            
        Returns:
            Topic node ID
        """
        topic_data = topic_data or {}
        topic_data.update({
            "name": topic_name,
            "tenant_id": tenant_id
        })
        
        query = """
        MERGE (t:Topic {name: $topic_name, tenant_id: $tenant_id})
        ON CREATE SET t += $topic_data
        RETURN t.name as topic_name
        """
        
        result = await self.execute_write_query(
            query,
            {
                "topic_name": topic_name,
                "tenant_id": tenant_id,
                "topic_data": topic_data
            }
        )
        
        logger.debug("Topic node created/found", topic_name=topic_name, tenant_id=tenant_id)
        return result[0]["topic_name"]

    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
        source_label: str = "Content",
        target_label: str = "Content"
    ) -> bool:
        """
        Create a relationship between two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship_type: Type of relationship
            properties: Relationship properties
            source_label: Source node label
            target_label: Target node label
            
        Returns:
            True if relationship was created
        """
        properties = properties or {}
        
        query = f"""
        MATCH (source:{source_label} {{id: $source_id}})
        MATCH (target:{target_label} {{id: $target_id}})
        MERGE (source)-[r:{relationship_type}]->(target)
        ON CREATE SET r += $properties
        RETURN r
        """
        
        result = await self.execute_write_query(
            query,
            {
                "source_id": source_id,
                "target_id": target_id,
                "properties": properties
            }
        )
        
        success = len(result) > 0
        logger.debug(
            "Relationship created",
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            success=success
        )
        
        return success

    async def get_content_relationships(
        self,
        content_id: str,
        tenant_id: str,
        relationship_types: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get relationships for a content node.
        
        Args:
            content_id: Content node ID
            tenant_id: Tenant identifier
            relationship_types: Filter by relationship types
            limit: Maximum number of relationships to return
            
        Returns:
            List of relationship data
        """
        if relationship_types:
            rel_filter = "|".join(relationship_types)
            rel_pattern = f"[r:{rel_filter}]"
        else:
            rel_pattern = "[r]"
        
        query = f"""
        MATCH (c:Content {{id: $content_id, tenant_id: $tenant_id}})
        MATCH (c)-{rel_pattern}-(related)
        RETURN type(r) as relationship_type,
               r as relationship_properties,
               related as related_node
        LIMIT $limit
        """
        
        result = await self.execute_query(
            query,
            {
                "content_id": content_id,
                "tenant_id": tenant_id,
                "limit": limit
            }
        )
        
        logger.debug(
            "Retrieved content relationships",
            content_id=content_id,
            tenant_id=tenant_id,
            count=len(result)
        )
        
        return result

    async def get_topic_coverage(
        self,
        topics: List[str],
        tenant_id: str
    ) -> Dict[str, int]:
        """
        Get content coverage for specific topics.
        
        Args:
            topics: List of topic names
            tenant_id: Tenant identifier
            
        Returns:
            Dictionary mapping topic names to content counts
        """
        query = """
        UNWIND $topics as topic_name
        MATCH (t:Topic {name: topic_name, tenant_id: $tenant_id})
        OPTIONAL MATCH (t)<-[:RELATES_TO]-(c:Content {tenant_id: $tenant_id})
        RETURN topic_name, count(c) as content_count
        """
        
        result = await self.execute_query(
            query,
            {"topics": topics, "tenant_id": tenant_id}
        )
        
        coverage = {row["topic_name"]: row["content_count"] for row in result}
        
        logger.debug(
            "Retrieved topic coverage",
            tenant_id=tenant_id,
            coverage=coverage
        )
        
        return coverage

    async def search_content(
        self,
        search_text: str,
        tenant_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Full-text search for content.
        
        Args:
            search_text: Search query
            tenant_id: Tenant identifier
            limit: Maximum number of results
            
        Returns:
            List of matching content nodes
        """
        query = """
        CALL db.index.fulltext.queryNodes("content_fulltext", $search_text)
        YIELD node, score
        WHERE node.tenant_id = $tenant_id
        RETURN node, score
        ORDER BY score DESC
        LIMIT $limit
        """
        
        result = await self.execute_query(
            query,
            {
                "search_text": search_text,
                "tenant_id": tenant_id,
                "limit": limit
            }
        )
        
        logger.debug(
            "Content search completed",
            search_text=search_text,
            tenant_id=tenant_id,
            result_count=len(result)
        )
        
        return result

    # =============================================================================
    # Graph Analytics
    # =============================================================================

    async def get_graph_statistics(self, tenant_id: str) -> Dict[str, Any]:
        """
        Get graph statistics for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Dictionary containing graph statistics
        """
        queries = {
            "content_count": "MATCH (c:Content {tenant_id: $tenant_id}) RETURN count(c) as count",
            "topic_count": "MATCH (t:Topic {tenant_id: $tenant_id}) RETURN count(t) as count",
            "keyword_count": "MATCH (k:Keyword {tenant_id: $tenant_id}) RETURN count(k) as count",
            "relationship_count": """
                MATCH (c:Content {tenant_id: $tenant_id})-[r]-()
                RETURN count(r) as count
            """,
        }
        
        stats = {}
        for stat_name, query in queries.items():
            result = await self.execute_query(query, {"tenant_id": tenant_id})
            stats[stat_name] = result[0]["count"] if result else 0
        
        logger.debug("Retrieved graph statistics", tenant_id=tenant_id, stats=stats)
        return stats

    def __repr__(self) -> str:
        """String representation of Neo4j client."""
        return f"Neo4jClient(uri={self.uri}, database={self.database}, connected={self._is_connected})"


# =============================================================================
# Utility Functions
# =============================================================================

async def get_neo4j_client() -> Neo4jClient:
    """
    Get a configured Neo4j client instance.
    
    Returns:
        Configured Neo4jClient instance
    """
    client = Neo4jClient()
    await client.connect()
    return client


async def initialize_neo4j_schema() -> None:
    """Initialize Neo4j database schema."""
    client = await get_neo4j_client()
    try:
        await client.initialize_schema()
        logger.info("Neo4j schema initialization completed")
    finally:
        await client.close()


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        client = Neo4jClient()
        try:
            await client.connect()
            await client.health_check()
            await client.initialize_schema()
            print("Neo4j client test completed successfully")
        except Exception as e:
            print(f"Neo4j client test failed: {e}")
        finally:
            await client.close()

    asyncio.run(main())