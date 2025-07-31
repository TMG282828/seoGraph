"""
Embedding generation service for the SEO Content Knowledge Graph System.

This module provides async embedding generation with caching, batch processing,
and integration with various embedding models.
"""

import asyncio
import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import structlog
import tiktoken
from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from config.settings import get_settings

logger = structlog.get_logger(__name__)


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass


class EmbeddingCache:
    """Cache for storing and retrieving embeddings."""
    
    def __init__(self, redis_client=None, ttl: int = 86400):
        """
        Initialize embedding cache.
        
        Args:
            redis_client: Redis client for caching
            ttl: Time to live for cache entries in seconds
        """
        self.redis_client = redis_client
        self.ttl = ttl
        self._memory_cache: Dict[str, Tuple[List[float], float]] = {}
        self._max_memory_cache_size = 1000
    
    def _generate_cache_key(self, text: str, model: str) -> str:
        """Generate cache key for text and model combination."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return f"embedding:{model}:{text_hash}"
    
    async def get(self, text: str, model: str) -> Optional[List[float]]:
        """
        Get embedding from cache.
        
        Args:
            text: Input text
            model: Embedding model name
            
        Returns:
            Cached embedding or None if not found
        """
        cache_key = self._generate_cache_key(text, model)
        
        # Try Redis cache first
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    embedding = json.loads(cached_data)
                    logger.debug("Embedding cache hit (Redis)", cache_key=cache_key)
                    return embedding
            except Exception as e:
                logger.warning("Redis cache read failed", error=str(e))
        
        # Try memory cache
        if cache_key in self._memory_cache:
            embedding, timestamp = self._memory_cache[cache_key]
            if time.time() - timestamp < self.ttl:
                logger.debug("Embedding cache hit (memory)", cache_key=cache_key)
                return embedding
            else:
                # Remove expired entry
                del self._memory_cache[cache_key]
        
        return None
    
    async def set(self, text: str, model: str, embedding: List[float]) -> None:
        """
        Store embedding in cache.
        
        Args:
            text: Input text
            model: Embedding model name
            embedding: Generated embedding
        """
        cache_key = self._generate_cache_key(text, model)
        
        # Store in Redis cache
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    cache_key,
                    self.ttl,
                    json.dumps(embedding)
                )
                logger.debug("Embedding cached (Redis)", cache_key=cache_key)
            except Exception as e:
                logger.warning("Redis cache write failed", error=str(e))
        
        # Store in memory cache
        if len(self._memory_cache) >= self._max_memory_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]
        
        self._memory_cache[cache_key] = (embedding, time.time())
        logger.debug("Embedding cached (memory)", cache_key=cache_key)


class EmbeddingService:
    """
    Service for generating text embeddings with caching and batch processing.
    
    Supports multiple embedding models with async processing and caching.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-ada-002",
        max_batch_size: int = 100,
        max_tokens_per_request: int = 8000,
        cache_ttl: int = 86400,
        redis_client=None
    ):
        """
        Initialize embedding service.
        
        Args:
            api_key: OpenAI API key
            model: Embedding model name
            max_batch_size: Maximum batch size for processing
            max_tokens_per_request: Maximum tokens per API request
            cache_ttl: Cache time to live in seconds
            redis_client: Redis client for caching
        """
        settings = get_settings()
        
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_embedding_model
        self.max_batch_size = max_batch_size
        self.max_tokens_per_request = max_tokens_per_request
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=self.api_key)
        
        # Initialize cache
        self.cache = EmbeddingCache(redis_client=redis_client, ttl=cache_ttl)
        
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        except Exception:
            # Fallback to cl100k_base encoding
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Model configuration
        self.model_dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        
        logger.info(
            "Embedding service initialized",
            model=self.model,
            max_batch_size=self.max_batch_size,
            dimensions=self.model_dimensions.get(self.model, "unknown")
        )
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning("Token counting failed", error=str(e))
            # Fallback estimation: roughly 4 characters per token
            return len(text) // 4
    
    def get_model_dimensions(self) -> int:
        """
        Get embedding dimensions for current model.
        
        Returns:
            Number of dimensions
        """
        return self.model_dimensions.get(self.model, 1536)
    
    def truncate_text(self, text: str, max_tokens: Optional[int] = None) -> str:
        """
        Truncate text to fit within token limits.
        
        Args:
            text: Input text
            max_tokens: Maximum tokens (defaults to model limit)
            
        Returns:
            Truncated text
        """
        max_tokens = max_tokens or (self.max_tokens_per_request - 100)  # Leave buffer
        
        try:
            tokens = self.tokenizer.encode(text)
            if len(tokens) <= max_tokens:
                return text
            
            # Truncate tokens and decode back to text
            truncated_tokens = tokens[:max_tokens]
            truncated_text = self.tokenizer.decode(truncated_tokens)
            
            logger.debug(
                "Text truncated",
                original_tokens=len(tokens),
                truncated_tokens=len(truncated_tokens)
            )
            
            return truncated_text
            
        except Exception as e:
            logger.warning("Text truncation failed", error=str(e))
            # Fallback: truncate by character count
            estimated_chars = max_tokens * 4
            return text[:estimated_chars]
    
    def prepare_text_for_embedding(self, text: str) -> str:
        """
        Prepare text for embedding generation.
        
        Args:
            text: Raw input text
            
        Returns:
            Processed text ready for embedding
        """
        # Clean and normalize text
        text = text.strip()
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Remove very long words (likely corrupted data)
        words = text.split()
        words = [word for word in words if len(word) <= 50]
        text = ' '.join(words)
        
        # Truncate if necessary
        text = self.truncate_text(text)
        
        return text
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception,)),
    )
    async def _generate_embeddings_api(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using OpenAI API.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts,
                encoding_format="float"
            )
            
            # Extract embeddings in correct order
            embeddings = []
            for i in range(len(texts)):
                embedding_data = next(
                    (item for item in response.data if item.index == i),
                    None
                )
                if embedding_data:
                    embeddings.append(embedding_data.embedding)
                else:
                    raise EmbeddingError(f"Missing embedding for index {i}")
            
            logger.debug(
                "Embeddings generated",
                count=len(embeddings),
                model=self.model,
                total_tokens=response.usage.total_tokens if hasattr(response, 'usage') else None
            )
            
            return embeddings
            
        except Exception as e:
            logger.error("Embedding generation failed", error=str(e), model=self.model)
            raise EmbeddingError(f"Embedding generation failed: {e}") from e
    
    async def generate_embedding(
        self,
        text: str,
        use_cache: bool = True,
        preprocess: bool = True
    ) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            use_cache: Whether to use caching
            preprocess: Whether to preprocess text
            
        Returns:
            Embedding vector
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Preprocess text if requested
        if preprocess:
            text = self.prepare_text_for_embedding(text)
        
        # Check cache first
        if use_cache:
            cached_embedding = await self.cache.get(text, self.model)
            if cached_embedding:
                return cached_embedding
        
        # Generate new embedding
        embeddings = await self._generate_embeddings_api([text])
        embedding = embeddings[0]
        
        # Cache the result
        if use_cache:
            await self.cache.set(text, self.model, embedding)
        
        return embedding
    
    async def generate_embeddings_batch(
        self,
        texts: List[str],
        use_cache: bool = True,
        preprocess: bool = True,
        progress_callback: Optional[callable] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with batching.
        
        Args:
            texts: List of input texts
            use_cache: Whether to use caching
            preprocess: Whether to preprocess texts
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Preprocess texts if requested
        processed_texts = []
        for text in texts:
            if not text or not text.strip():
                logger.warning("Empty text in batch, skipping")
                processed_texts.append("")
                continue
            
            if preprocess:
                text = self.prepare_text_for_embedding(text)
            processed_texts.append(text)
        
        # Check cache for existing embeddings
        embeddings = [None] * len(processed_texts)
        texts_to_generate = []
        text_indices = []
        
        if use_cache:
            for i, text in enumerate(processed_texts):
                if not text:
                    continue
                    
                cached_embedding = await self.cache.get(text, self.model)
                if cached_embedding:
                    embeddings[i] = cached_embedding
                else:
                    texts_to_generate.append(text)
                    text_indices.append(i)
        else:
            texts_to_generate = [text for text in processed_texts if text]
            text_indices = [i for i, text in enumerate(processed_texts) if text]
        
        logger.info(
            "Batch embedding generation",
            total_texts=len(texts),
            cached=len(texts) - len(texts_to_generate),
            to_generate=len(texts_to_generate)
        )
        
        # Generate embeddings in batches
        generated_embeddings = []
        for i in range(0, len(texts_to_generate), self.max_batch_size):
            batch_texts = texts_to_generate[i:i + self.max_batch_size]
            
            # Check token limits for batch
            total_tokens = sum(self.count_tokens(text) for text in batch_texts)
            if total_tokens > self.max_tokens_per_request:
                # Process individually to handle token limits
                batch_embeddings = []
                for text in batch_texts:
                    embedding = await self.generate_embedding(
                        text, use_cache=use_cache, preprocess=False
                    )
                    batch_embeddings.append(embedding)
            else:
                # Process as batch
                batch_embeddings = await self._generate_embeddings_api(batch_texts)
            
            generated_embeddings.extend(batch_embeddings)
            
            # Update progress
            if progress_callback:
                progress = (i + len(batch_texts)) / len(texts_to_generate)
                await progress_callback(progress)
            
            # Small delay to respect rate limits
            await asyncio.sleep(0.1)
        
        # Fill in generated embeddings
        for i, embedding in enumerate(generated_embeddings):
            original_index = text_indices[i]
            embeddings[original_index] = embedding
            
            # Cache the result
            if use_cache:
                await self.cache.set(
                    processed_texts[original_index],
                    self.model,
                    embedding
                )
        
        # Handle empty texts
        zero_embedding = [0.0] * self.get_model_dimensions()
        for i, embedding in enumerate(embeddings):
            if embedding is None:
                embeddings[i] = zero_embedding
        
        logger.info(
            "Batch embedding generation completed",
            total_texts=len(texts),
            generated=len(generated_embeddings)
        )
        
        return embeddings
    
    async def embed_content_items(
        self,
        content_items: List[Dict[str, Any]],
        text_field: str = "content",
        include_title: bool = True,
        title_field: str = "title"
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings for content items.
        
        Args:
            content_items: List of content item dictionaries
            text_field: Field name containing main text
            include_title: Whether to include title in embedding
            title_field: Field name containing title
            
        Returns:
            List of content items with embeddings added
        """
        # Prepare texts for embedding
        texts_to_embed = []
        for item in content_items:
            text_parts = []
            
            # Add title if requested
            if include_title and title_field in item:
                title = item[title_field]
                if title:
                    text_parts.append(f"Title: {title}")
            
            # Add main content
            if text_field in item:
                content = item[text_field]
                if content:
                    text_parts.append(content)
            
            # Combine parts
            combined_text = "\n\n".join(text_parts)
            texts_to_embed.append(combined_text)
        
        # Generate embeddings
        embeddings = await self.generate_embeddings_batch(texts_to_embed)
        
        # Add embeddings to content items
        enriched_items = []
        for item, embedding in zip(content_items, embeddings):
            enriched_item = item.copy()
            enriched_item["embedding"] = embedding
            enriched_item["embedding_model"] = self.model
            enriched_item["embedding_dimensions"] = len(embedding)
            enriched_items.append(enriched_item)
        
        return enriched_items
    
    async def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (-1 to 1)
        """
        if len(embedding1) != len(embedding2):
            raise ValueError("Embeddings must have same dimensions")
        
        # Compute dot product
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        
        # Compute magnitudes
        magnitude1 = sum(a * a for a in embedding1) ** 0.5
        magnitude2 = sum(b * b for b in embedding2) ** 0.5
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = dot_product / (magnitude1 * magnitude2)
        
        return similarity
    
    async def find_similar_content(
        self,
        query_embedding: List[float],
        content_embeddings: List[Tuple[str, List[float]]],
        top_k: int = 10,
        min_similarity: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Find similar content based on embedding similarity.
        
        Args:
            query_embedding: Query embedding vector
            content_embeddings: List of (content_id, embedding) tuples
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (content_id, similarity_score) tuples
        """
        similarities = []
        
        for content_id, embedding in content_embeddings:
            similarity = await self.compute_similarity(query_embedding, embedding)
            if similarity >= min_similarity:
                similarities.append((content_id, similarity))
        
        # Sort by similarity (descending) and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_service_stats(self) -> Dict[str, Any]:
        """
        Get service statistics.
        
        Returns:
            Dictionary containing service statistics
        """
        return {
            "model": self.model,
            "dimensions": self.get_model_dimensions(),
            "max_batch_size": self.max_batch_size,
            "max_tokens_per_request": self.max_tokens_per_request,
            "cache_size": len(self.cache._memory_cache),
            "cache_max_size": self.cache._max_memory_cache_size,
        }


# =============================================================================
# Utility Functions
# =============================================================================

async def get_embedding_service(
    redis_client=None,
    **kwargs
) -> EmbeddingService:
    """
    Get configured embedding service instance.
    
    Args:
        redis_client: Redis client for caching
        **kwargs: Additional service parameters
        
    Returns:
        EmbeddingService instance
    """
    return EmbeddingService(redis_client=redis_client, **kwargs)


async def embed_text_simple(text: str, model: str = "text-embedding-ada-002") -> List[float]:
    """
    Simple function to embed a single text.
    
    Args:
        text: Text to embed
        model: Embedding model to use
        
    Returns:
        Embedding vector
    """
    service = EmbeddingService(model=model)
    return await service.generate_embedding(text)


# Global service instance
embedding_service = EmbeddingService()


# Add the missing method to the main class
async def generate_content_embedding_method(self, content_text: str) -> List[float]:
    """
    Generate embedding for content text.
    
    Args:
        content_text: The content text to embed.
        
    Returns:
        The embedding vector, or empty list on failure.
    """
    return await self.generate_embedding(content_text)

# Add the method to the EmbeddingService class
EmbeddingService.generate_content_embedding = generate_content_embedding_method


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        service = EmbeddingService()
        
        # Test single embedding
        text = "This is a test document about SEO and content marketing."
        embedding = await service.generate_embedding(text)
        print(f"Generated embedding with {len(embedding)} dimensions")
        
        # Test batch embeddings
        texts = [
            "SEO optimization techniques",
            "Content marketing strategies",
            "Keyword research methods"
        ]
        embeddings = await service.generate_embeddings_batch(texts)
        print(f"Generated {len(embeddings)} embeddings")
        
        # Test similarity
        similarity = await service.compute_similarity(embeddings[0], embeddings[1])
        print(f"Similarity between first two texts: {similarity:.3f}")
        
        # Service stats
        stats = service.get_service_stats()
        print(f"Service stats: {stats}")

    asyncio.run(main())