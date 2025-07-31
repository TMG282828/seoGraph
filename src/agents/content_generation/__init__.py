"""
Content Generation Agent Package.

This package contains the modular content generation agent components.
"""

from .agent import ContentGenerationAgent, ContentGenerationRequest
from typing import Dict, Any, Optional

# Import dependencies for compatibility function
try:
    from ...database.neo4j_client import Neo4jClient
    from ...database.qdrant_client import QdrantClient
    from ...database.supabase_client import SupabaseClient
    from ...services.embedding_service import EmbeddingService
except ImportError:
    # Fallback imports from legacy structure
    try:
        from database.neo4j_client import Neo4jClient
        from database.qdrant_client import QdrantClient
        from database.supabase_client import SupabaseClient
        from services.embedding_service import EmbeddingService
    except ImportError:
        # Define dummy classes if imports fail
        Neo4jClient = None
        QdrantClient = None
        SupabaseClient = None
        EmbeddingService = None


async def create_content_generation_agent(
    tenant_id: str,
    brand_voice: Optional[Dict[str, Any]] = None,
    neo4j_client: Optional[Neo4jClient] = None,
    qdrant_client: Optional[QdrantClient] = None,
    supabase_client: Optional[SupabaseClient] = None,
    embedding_service: Optional[EmbeddingService] = None
) -> ContentGenerationAgent:
    """
    Create a configured Content Generation Agent.
    
    Backward compatibility function for legacy imports.
    
    Args:
        tenant_id: Tenant identifier
        brand_voice: Brand voice configuration
        neo4j_client: Neo4j client instance
        qdrant_client: Qdrant client instance
        supabase_client: Supabase client instance
        embedding_service: Embedding service instance
        
    Returns:
        Configured ContentGenerationAgent instance
    """
    agent = ContentGenerationAgent()
    
    # Initialize the agent with provided dependencies
    await agent.initialize(
        tenant_id=tenant_id,
        brand_voice=brand_voice,
        neo4j_client=neo4j_client,
        qdrant_client=qdrant_client,
        supabase_client=supabase_client,
        embedding_service=embedding_service
    )
    
    return agent


__all__ = ['ContentGenerationAgent', 'ContentGenerationRequest', 'create_content_generation_agent']