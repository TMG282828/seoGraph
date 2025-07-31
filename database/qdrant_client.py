"""
Qdrant vector database client for the SEO Content Knowledge Graph System.

This module provides an async Qdrant client with collection management,
vector operations, and semantic search capabilities.
"""

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional, Union

import structlog
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    CollectionStatus,
    Distance,
    PointStruct,
    VectorParams,
    CreateCollection,
    UpdateCollection,
    SearchRequest,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from config.settings import get_settings

logger = structlog.get_logger(__name__)


class QdrantConnectionError(Exception):
    """Raised when Qdrant connection fails."""
    pass


class QdrantCollectionError(Exception):
    """Raised when Qdrant collection operations fail."""
    pass


class QdrantClient:
    """
    Async Qdrant client with collection management and vector operations.
    
    Provides methods for vector storage, similarity search, and collection
    management with proper error handling and connection management.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
        vector_size: int = 1536,
        distance_metric: Distance = Distance.COSINE,
        timeout: int = 60,
    ):
        """
        Initialize Qdrant client.

        Args:
            url: Qdrant server URL
            api_key: Qdrant API key (optional)
            collection_name: Default collection name
            vector_size: Vector dimension size
            distance_metric: Distance metric for similarity search
            timeout: Request timeout in seconds
        """
        settings = get_settings()
        
        self.url = url or settings.qdrant_url
        self.api_key = api_key or settings.qdrant_api_key
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.vector_size = vector_size
        self.distance_metric = distance_metric
        self.timeout = timeout
        
        self._client: Optional[AsyncQdrantClient] = None
        self._is_connected = False

    async def connect(self) -> None:
        """
        Establish connection to Qdrant database.
        
        Raises:
            QdrantConnectionError: If connection fails
        """
        try:
            self._client = AsyncQdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
            
            # Verify connectivity
            await self.health_check()
            self._is_connected = True
            
            logger.info(
                "Qdrant connection established",
                url=self.url,
                collection=self.collection_name,
            )
            
        except Exception as e:
            logger.error(
                "Failed to connect to Qdrant",
                url=self.url,
                error=str(e),
            )
            raise QdrantConnectionError(f"Failed to connect to Qdrant: {e}") from e

    async def close(self) -> None:
        """Close Qdrant connection."""
        if self._client:
            await self._client.close()
            self._is_connected = False
            logger.info("Qdrant connection closed")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
    )
    async def health_check(self) -> bool:
        """
        Check Qdrant database health.
        
        Returns:
            True if database is healthy
            
        Raises:
            QdrantConnectionError: If health check fails
        """
        if not self._client:
            raise QdrantConnectionError("Client not initialized")
            
        try:
            # Get cluster info as health check
            cluster_info = await self._client.get_cluster_info()
            logger.debug("Qdrant health check passed", cluster_info=cluster_info)
            return True
            
        except Exception as e:
            logger.error("Qdrant health check failed", error=str(e))
            raise QdrantConnectionError(f"Health check failed: {e}") from e

    def _ensure_connected(self) -> None:
        """Ensure client is connected."""
        if not self._client:
            raise QdrantConnectionError("Client not initialized. Call connect() first.")

    # =============================================================================
    # Collection Management
    # =============================================================================

    async def create_collection(
        self,
        collection_name: Optional[str] = None,
        vector_size: Optional[int] = None,
        distance_metric: Optional[Distance] = None,
        force_recreate: bool = False,
    ) -> bool:
        """
        Create a new collection.
        
        Args:
            collection_name: Collection name
            vector_size: Vector dimension size
            distance_metric: Distance metric for similarity search
            force_recreate: Whether to recreate if collection exists
            
        Returns:
            True if collection was created
            
        Raises:
            QdrantCollectionError: If collection creation fails
        """
        self._ensure_connected()
        
        collection_name = collection_name or self.collection_name
        vector_size = vector_size or self.vector_size
        distance_metric = distance_metric or self.distance_metric
        
        try:
            # Check if collection exists
            exists = await self.collection_exists(collection_name)
            
            if exists and force_recreate:
                await self.delete_collection(collection_name)
                exists = False
            elif exists:
                logger.info("Collection already exists", collection=collection_name)
                return False
            
            # Create collection
            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance_metric,
                ),
            )
            
            logger.info(
                "Collection created",
                collection=collection_name,
                vector_size=vector_size,
                distance_metric=distance_metric.value,
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to create collection",
                collection=collection_name,
                error=str(e),
            )
            raise QdrantCollectionError(f"Failed to create collection: {e}") from e

    async def collection_exists(self, collection_name: str) -> bool:
        """
        Check if collection exists.
        
        Args:
            collection_name: Collection name
            
        Returns:
            True if collection exists
        """
        self._ensure_connected()
        
        try:
            collections = await self._client.get_collections()
            collection_names = [col.name for col in collections.collections]
            return collection_name in collection_names
            
        except Exception as e:
            logger.error("Failed to check collection existence", error=str(e))
            return False

    async def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            collection_name: Collection name
            
        Returns:
            True if collection was deleted
        """
        self._ensure_connected()
        
        try:
            await self._client.delete_collection(collection_name)
            logger.info("Collection deleted", collection=collection_name)
            return True
            
        except Exception as e:
            logger.error(
                "Failed to delete collection",
                collection=collection_name,
                error=str(e),
            )
            return False

    async def get_collection_info(
        self,
        collection_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get collection information.
        
        Args:
            collection_name: Collection name
            
        Returns:
            Collection information dictionary
        """
        self._ensure_connected()
        
        collection_name = collection_name or self.collection_name
        
        try:
            info = await self._client.get_collection(collection_name)
            return {
                "name": collection_name,
                "status": info.status.value,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "config": {
                    "vector_size": info.config.params.vectors.size,
                    "distance": info.config.params.vectors.distance.value,
                },
            }
            
        except Exception as e:
            logger.error(
                "Failed to get collection info",
                collection=collection_name,
                error=str(e),
            )
            return None

    # =============================================================================
    # Vector Operations
    # =============================================================================

    async def add_vectors(
        self,
        vectors: List[List[float]],
        payloads: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[Union[str, int]]] = None,
        collection_name: Optional[str] = None,
        batch_size: int = 100,
    ) -> List[str]:
        """
        Add vectors to collection.
        
        Args:
            vectors: List of vector embeddings
            payloads: List of metadata payloads
            ids: List of point IDs (auto-generated if None)
            collection_name: Collection name
            batch_size: Batch size for bulk operations
            
        Returns:
            List of added point IDs
        """
        self._ensure_connected()
        
        collection_name = collection_name or self.collection_name
        payloads = payloads or [{}] * len(vectors)
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]
        
        # Ensure all lists have same length
        if not (len(vectors) == len(payloads) == len(ids)):
            raise ValueError("Vectors, payloads, and IDs must have the same length")
        
        try:
            # Create points
            points = []
            for i, (vector, payload, point_id) in enumerate(zip(vectors, payloads, ids)):
                # Add metadata to payload
                payload.update({
                    "vector_id": point_id,
                    "created_at": asyncio.get_event_loop().time(),
                })
                
                points.append(PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                ))
            
            # Upload in batches
            added_ids = []
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                
                await self._client.upsert(
                    collection_name=collection_name,
                    points=batch,
                )
                
                batch_ids = [point.id for point in batch]
                added_ids.extend(batch_ids)
                
                logger.debug(
                    "Vector batch uploaded",
                    collection=collection_name,
                    batch_size=len(batch),
                    total_uploaded=len(added_ids),
                )
            
            logger.info(
                "Vectors added to collection",
                collection=collection_name,
                count=len(added_ids),
            )
            
            return added_ids
            
        except Exception as e:
            logger.error(
                "Failed to add vectors",
                collection=collection_name,
                count=len(vectors),
                error=str(e),
            )
            raise

    async def search_similar(
        self,
        query_vector: List[float],
        collection_name: Optional[str] = None,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector for similarity search
            collection_name: Collection name
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filter_conditions: Additional filter conditions
            tenant_id: Tenant ID for multi-tenant filtering
            
        Returns:
            List of similar vectors with scores and payloads
        """
        self._ensure_connected()
        
        collection_name = collection_name or self.collection_name
        
        try:
            # Build filter conditions
            query_filter = None
            if filter_conditions or tenant_id:
                conditions = []
                
                # Add tenant filtering
                if tenant_id:
                    conditions.append(
                        FieldCondition(
                            key="tenant_id",
                            match=MatchValue(value=tenant_id),
                        )
                    )
                
                # Add custom filters
                if filter_conditions:
                    for key, value in filter_conditions.items():
                        if isinstance(value, (str, int, bool)):
                            conditions.append(
                                FieldCondition(
                                    key=key,
                                    match=MatchValue(value=value),
                                )
                            )
                        elif isinstance(value, dict) and "gte" in value or "lte" in value:
                            # Range filter
                            conditions.append(
                                FieldCondition(
                                    key=key,
                                    range=Range(
                                        gte=value.get("gte"),
                                        lte=value.get("lte"),
                                    )
                                )
                            )
                
                if conditions:
                    query_filter = Filter(must=conditions)
            
            # Perform search
            search_result = await self._client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,  # Don't return vectors to save bandwidth
            )
            
            # Format results
            results = []
            for point in search_result:
                results.append({
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload or {},
                })
            
            logger.debug(
                "Vector similarity search completed",
                collection=collection_name,
                query_size=len(query_vector),
                result_count=len(results),
                tenant_id=tenant_id,
            )
            
            return results
            
        except Exception as e:
            logger.error(
                "Failed to search similar vectors",
                collection=collection_name,
                error=str(e),
            )
            raise

    async def get_vector_by_id(
        self,
        vector_id: Union[str, int],
        collection_name: Optional[str] = None,
        with_vector: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Get vector by ID.
        
        Args:
            vector_id: Vector ID
            collection_name: Collection name
            with_vector: Include vector data in response
            
        Returns:
            Vector data with payload
        """
        self._ensure_connected()
        
        collection_name = collection_name or self.collection_name
        
        try:
            points = await self._client.retrieve(
                collection_name=collection_name,
                ids=[vector_id],
                with_payload=True,
                with_vectors=with_vector,
            )
            
            if not points:
                return None
            
            point = points[0]
            result = {
                "id": point.id,
                "payload": point.payload or {},
            }
            
            if with_vector and point.vector:
                result["vector"] = point.vector
            
            return result
            
        except Exception as e:
            logger.error(
                "Failed to get vector by ID",
                vector_id=vector_id,
                collection=collection_name,
                error=str(e),
            )
            return None

    async def delete_vectors(
        self,
        vector_ids: List[Union[str, int]],
        collection_name: Optional[str] = None,
    ) -> bool:
        """
        Delete vectors by IDs.
        
        Args:
            vector_ids: List of vector IDs to delete
            collection_name: Collection name
            
        Returns:
            True if deletion was successful
        """
        self._ensure_connected()
        
        collection_name = collection_name or self.collection_name
        
        try:
            await self._client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=vector_ids,
                ),
            )
            
            logger.info(
                "Vectors deleted",
                collection=collection_name,
                count=len(vector_ids),
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to delete vectors",
                collection=collection_name,
                count=len(vector_ids),
                error=str(e),
            )
            return False

    async def update_vector_payload(
        self,
        vector_id: Union[str, int],
        payload_update: Dict[str, Any],
        collection_name: Optional[str] = None,
    ) -> bool:
        """
        Update vector payload.
        
        Args:
            vector_id: Vector ID
            payload_update: Payload updates
            collection_name: Collection name
            
        Returns:
            True if update was successful
        """
        self._ensure_connected()
        
        collection_name = collection_name or self.collection_name
        
        try:
            await self._client.set_payload(
                collection_name=collection_name,
                payload=payload_update,
                points=[vector_id],
            )
            
            logger.debug(
                "Vector payload updated",
                vector_id=vector_id,
                collection=collection_name,
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to update vector payload",
                vector_id=vector_id,
                collection=collection_name,
                error=str(e),
            )
            return False

    # =============================================================================
    # Content-specific Operations
    # =============================================================================

    async def add_content_embedding(
        self,
        content_id: str,
        embedding: List[float],
        content_metadata: Dict[str, Any],
        tenant_id: str,
        collection_name: Optional[str] = None,
    ) -> str:
        """
        Add content embedding with metadata.
        
        Args:
            content_id: Content identifier
            embedding: Content embedding vector
            content_metadata: Content metadata
            tenant_id: Tenant identifier
            collection_name: Collection name
            
        Returns:
            Vector point ID
        """
        payload = {
            "content_id": content_id,
            "tenant_id": tenant_id,
            "content_type": content_metadata.get("content_type"),
            "title": content_metadata.get("title"),
            "topics": content_metadata.get("topics", []),
            "keywords": content_metadata.get("keywords", []),
            "created_at": content_metadata.get("created_at"),
            **content_metadata,
        }
        
        vector_ids = await self.add_vectors(
            vectors=[embedding],
            payloads=[payload],
            ids=[content_id],
            collection_name=collection_name,
        )
        
        logger.info(
            "Content embedding added",
            content_id=content_id,
            tenant_id=tenant_id,
            vector_id=vector_ids[0],
        )
        
        return vector_ids[0]

    async def search_similar_content(
        self,
        query_embedding: List[float],
        tenant_id: str,
        content_type: Optional[str] = None,
        topics: Optional[List[str]] = None,
        limit: int = 10,
        score_threshold: float = 0.7,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for semantically similar content.
        
        Args:
            query_embedding: Query embedding vector
            tenant_id: Tenant identifier
            content_type: Filter by content type
            topics: Filter by topics
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            collection_name: Collection name
            
        Returns:
            List of similar content with metadata
        """
        filter_conditions = {}
        
        if content_type:
            filter_conditions["content_type"] = content_type
        
        # Note: Qdrant doesn't directly support array filtering
        # This would need custom logic for topic filtering
        
        results = await self.search_similar(
            query_vector=query_embedding,
            collection_name=collection_name,
            limit=limit,
            score_threshold=score_threshold,
            filter_conditions=filter_conditions,
            tenant_id=tenant_id,
        )
        
        # Post-process topic filtering if needed
        if topics:
            filtered_results = []
            for result in results:
                content_topics = result["payload"].get("topics", [])
                if any(topic in content_topics for topic in topics):
                    filtered_results.append(result)
            results = filtered_results
        
        logger.debug(
            "Similar content search completed",
            tenant_id=tenant_id,
            content_type=content_type,
            topics=topics,
            result_count=len(results),
        )
        
        return results

    def __repr__(self) -> str:
        """String representation of Qdrant client."""
        return f"QdrantClient(url={self.url}, collection={self.collection_name}, connected={self._is_connected})"


# =============================================================================
# Utility Functions
# =============================================================================

async def get_qdrant_client() -> QdrantClient:
    """
    Get a configured Qdrant client instance.
    
    Returns:
        Configured QdrantClient instance
    """
    client = QdrantClient()
    await client.connect()
    return client


async def initialize_qdrant_collections() -> None:
    """Initialize Qdrant collections for the system."""
    client = await get_qdrant_client()
    try:
        # Create main content collection
        await client.create_collection(
            collection_name="seo_content_embeddings",
            vector_size=1536,  # OpenAI text-embedding-ada-002 size
            distance_metric=Distance.COSINE,
            force_recreate=False,
        )
        
        logger.info("Qdrant collections initialization completed")
    finally:
        await client.close()


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        client = QdrantClient()
        try:
            await client.connect()
            await client.health_check()
            
            # Test collection creation
            await client.create_collection(
                collection_name="test_collection",
                vector_size=128,
                force_recreate=True,
            )
            
            print("Qdrant client test completed successfully")
        except Exception as e:
            print(f"Qdrant client test failed: {e}")
        finally:
            await client.close()

    asyncio.run(main())