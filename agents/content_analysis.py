"""
Content Analysis Agent for the SEO Content Knowledge Graph System.

This agent analyzes content to extract topics, entities, and relationships,
and assesses content quality for SEO optimization.
"""

import asyncio
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import structlog
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from config.settings import get_settings
from database.neo4j_client import Neo4jClient
from database.qdrant_client import QdrantClient
from database.supabase_client import SupabaseClient
from models.content_models import ContentItem, ContentMetrics, ContentType
from models.graph_models import (
    ContentNode, TopicNode, EntityNode, KeywordNode,
    RelatestoRelationship, SimilarityRelationship,
    create_content_node, create_topic_node, create_topic_relationship
)
from services.embedding_service import EmbeddingService

# Import OpenTelemetry tracing
try:
    from monitoring.otel_monitor import trace_agent
    OTEL_TRACING_AVAILABLE = True
except ImportError:
    OTEL_TRACING_AVAILABLE = False
    # No-op decorator if monitoring not available
    def trace_agent(agent_type: str, operation: str):
        def decorator(func):
            return func
        return decorator

logger = structlog.get_logger(__name__)


class ContentAnalysisDependencies:
    """Dependencies for the Content Analysis Agent."""
    
    def __init__(
        self,
        neo4j_client: Neo4jClient,
        qdrant_client: QdrantClient,
        supabase_client: SupabaseClient,
        embedding_service: EmbeddingService,
        tenant_id: str
    ):
        self.neo4j_client = neo4j_client
        self.qdrant_client = qdrant_client
        self.supabase_client = supabase_client
        self.embedding_service = embedding_service
        self.tenant_id = tenant_id


class TopicExtraction(BaseModel):
    """Structured output for topic extraction."""
    
    main_topics: List[str] = Field(..., description="Primary topics covered in the content")
    secondary_topics: List[str] = Field(..., description="Secondary topics mentioned")
    entities: List[Dict[str, str]] = Field(..., description="Named entities found (name, type)")
    keywords: List[Dict[str, Any]] = Field(..., description="Important keywords with relevance scores")
    content_category: str = Field(..., description="Overall content category")
    topic_confidence: float = Field(..., description="Confidence score for topic extraction")


class ContentQualityAssessment(BaseModel):
    """Structured output for content quality assessment."""
    
    overall_score: float = Field(..., description="Overall content quality score (0-100)")
    readability_score: float = Field(..., description="Readability assessment score")
    seo_score: float = Field(..., description="SEO optimization score")
    
    # Detailed assessments
    structure_score: float = Field(..., description="Content structure quality")
    depth_score: float = Field(..., description="Content depth and comprehensiveness")
    uniqueness_score: float = Field(..., description="Content uniqueness assessment")
    
    # Specific recommendations
    improvements: List[str] = Field(..., description="Specific improvement recommendations")
    strengths: List[str] = Field(..., description="Content strengths identified")
    
    # Metrics
    word_count: int = Field(..., description="Total word count")
    sentence_count: int = Field(..., description="Total sentence count")
    paragraph_count: int = Field(..., description="Total paragraph count")
    avg_sentence_length: float = Field(..., description="Average sentence length")
    
    # SEO specifics
    title_optimization: Dict[str, Any] = Field(..., description="Title optimization analysis")
    meta_description_analysis: Dict[str, Any] = Field(..., description="Meta description analysis")
    keyword_density: Dict[str, float] = Field(..., description="Keyword density analysis")


class RelationshipMapping(BaseModel):
    """Structured output for relationship mapping."""
    
    similar_content: List[Dict[str, Any]] = Field(..., description="Similar content items found")
    topic_relationships: List[Dict[str, Any]] = Field(..., description="Topic relationships to create")
    entity_relationships: List[Dict[str, Any]] = Field(..., description="Entity relationships to create")
    keyword_relationships: List[Dict[str, Any]] = Field(..., description="Keyword relationships to create")
    
    content_gaps: List[str] = Field(..., description="Identified content gaps")
    recommended_connections: List[Dict[str, Any]] = Field(..., description="Recommended content connections")


class ContentAnalysisResult(BaseModel):
    """Complete content analysis result."""
    
    content_id: str = Field(..., description="Analyzed content ID")
    analysis_timestamp: datetime = Field(..., description="Analysis timestamp")
    
    topic_extraction: TopicExtraction
    quality_assessment: ContentQualityAssessment
    relationship_mapping: RelationshipMapping
    
    processing_time: float = Field(..., description="Analysis processing time in seconds")
    success: bool = Field(..., description="Analysis success status")
    warnings: List[str] = Field(default_factory=list, description="Analysis warnings")


# Initialize the Content Analysis Agent
content_analysis_agent = Agent(
    "openai:gpt-4o-mini",
    deps_type=ContentAnalysisDependencies,
    result_type=ContentAnalysisResult,
    system_prompt="""
    You are a specialized Content Analysis Agent for SEO content optimization.
    
    Your responsibilities include:
    1. Extracting main topics, secondary topics, entities, and keywords from content
    2. Assessing content quality across multiple dimensions (SEO, readability, structure, depth)
    3. Identifying relationships between content pieces and topics
    4. Recommending improvements and optimizations
    
    Analysis Guidelines:
    - Focus on SEO-relevant topics and keywords
    - Assess content depth and comprehensiveness
    - Identify named entities (people, organizations, locations, etc.)
    - Evaluate content structure and readability
    - Consider search intent and user value
    - Recommend specific, actionable improvements
    
    Quality Scoring (0-100):
    - 90-100: Exceptional content, comprehensive and well-optimized
    - 70-89: Good content with minor optimization opportunities
    - 50-69: Average content needing significant improvements
    - 30-49: Below average content requiring major revisions
    - 0-29: Poor content needing complete rewrite
    
    Always provide specific, actionable recommendations for improvement.
    """,
)


class ContentAnalysisAgent:
    """
    Content Analysis Agent for extracting topics, assessing quality,
    and mapping relationships in the knowledge graph.
    """
    
    def __init__(
        self,
        neo4j_client: Neo4jClient,
        qdrant_client: QdrantClient,
        supabase_client: SupabaseClient,
        embedding_service: EmbeddingService,
        tenant_id: str
    ):
        """
        Initialize the Content Analysis Agent.
        
        Args:
            neo4j_client: Neo4j database client
            qdrant_client: Qdrant vector database client
            supabase_client: Supabase database client
            embedding_service: Embedding generation service
            tenant_id: Tenant identifier
        """
        self.neo4j_client = neo4j_client
        self.qdrant_client = qdrant_client
        self.supabase_client = supabase_client
        self.embedding_service = embedding_service
        self.tenant_id = tenant_id
        
        # Initialize dependencies
        self.deps = ContentAnalysisDependencies(
            neo4j_client=neo4j_client,
            qdrant_client=qdrant_client,
            supabase_client=supabase_client,
            embedding_service=embedding_service,
            tenant_id=tenant_id
        )
        
        logger.info(
            "Content Analysis Agent initialized",
            tenant_id=tenant_id
        )
    
    @trace_agent("content_analysis", "analyze_content")
    async def analyze_content(
        self,
        content_item: ContentItem,
        include_similarity_analysis: bool = True,
        min_similarity_threshold: float = 0.7
    ) -> ContentAnalysisResult:
        """
        Perform comprehensive content analysis.
        
        Args:
            content_item: Content item to analyze
            include_similarity_analysis: Whether to include similarity analysis
            min_similarity_threshold: Minimum similarity threshold for relationships
            
        Returns:
            Complete content analysis result
        """
        start_time = asyncio.get_event_loop().time()
        
        logger.info(
            "Starting content analysis",
            content_id=content_item.id,
            title=content_item.title,
            tenant_id=self.tenant_id
        )
        
        try:
            # Prepare analysis input
            analysis_input = self._prepare_analysis_input(content_item)
            
            # Generate content embedding for similarity analysis
            content_embedding = None
            if include_similarity_analysis:
                content_embedding = await self.embedding_service.generate_embedding(
                    f"Title: {content_item.title}\n\nContent: {content_item.content}",
                    use_cache=True
                )
            
            # Run AI analysis
            result = await content_analysis_agent.run(
                analysis_input,
                deps=self.deps
            )
            
            # Enhance with similarity analysis
            if include_similarity_analysis and content_embedding:
                await self._enhance_with_similarity_analysis(
                    result,
                    content_item,
                    content_embedding,
                    min_similarity_threshold
                )
            
            # Store analysis results
            await self._store_analysis_results(content_item, result)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            result.processing_time = processing_time
            
            logger.info(
                "Content analysis completed",
                content_id=content_item.id,
                processing_time=processing_time,
                overall_score=result.quality_assessment.overall_score
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Content analysis failed",
                content_id=content_item.id,
                error=str(e)
            )
            raise
    
    def _prepare_analysis_input(self, content_item: ContentItem) -> str:
        """Prepare input text for AI analysis."""
        input_parts = [
            f"TITLE: {content_item.title}",
            f"CONTENT TYPE: {content_item.content_type.value}",
            f"STATUS: {content_item.status}",
            "",
            "CONTENT:",
            content_item.content
        ]
        
        # Add metadata if available
        if content_item.metadata:
            if content_item.metadata.meta_description:
                input_parts.insert(-2, f"META DESCRIPTION: {content_item.metadata.meta_description}")
            if content_item.metadata.meta_keywords:
                input_parts.insert(-2, f"META KEYWORDS: {', '.join(content_item.metadata.meta_keywords)}")
        
        return "\n".join(input_parts)
    
    async def _enhance_with_similarity_analysis(
        self,
        result: ContentAnalysisResult,
        content_item: ContentItem,
        content_embedding: List[float],
        min_similarity_threshold: float
    ) -> None:
        """Enhance analysis with similarity analysis."""
        try:
            # Search for similar content
            similar_items = await self.qdrant_client.search_similar(
                collection_name=f"content_{self.tenant_id}",
                query_vector=content_embedding,
                limit=10,
                score_threshold=min_similarity_threshold
            )
            
            # Process similar content
            similar_content = []
            for item in similar_items:
                if item.id != content_item.id:  # Exclude self
                    similar_content.append({
                        "content_id": item.id,
                        "similarity_score": item.score,
                        "title": item.payload.get("title", "Unknown"),
                        "content_type": item.payload.get("content_type", "Unknown")
                    })
            
            result.relationship_mapping.similar_content = similar_content
            
        except Exception as e:
            logger.warning(
                "Similarity analysis failed",
                content_id=content_item.id,
                error=str(e)
            )
            result.warnings.append(f"Similarity analysis failed: {str(e)}")
    
    async def _store_analysis_results(
        self,
        content_item: ContentItem,
        result: ContentAnalysisResult
    ) -> None:
        """Store analysis results in the knowledge graph."""
        try:
            # Create content node
            content_node = create_content_node(
                content_id=content_item.id,
                title=content_item.title,
                content_type=content_item.content_type.value,
                status=content_item.status,
                author_id=content_item.author_id,
                tenant_id=self.tenant_id,
                word_count=result.quality_assessment.word_count,
                seo_score=result.quality_assessment.seo_score,
                readability_score=result.quality_assessment.readability_score
            )
            
            # Store content node
            await self.neo4j_client.create_content_node(content_node)
            
            # Create and store topic nodes
            for topic_name in result.topic_extraction.main_topics:
                topic_node = create_topic_node(
                    topic_id=f"topic_{topic_name.lower().replace(' ', '_')}",
                    name=topic_name,
                    tenant_id=self.tenant_id,
                    category="main",
                    importance_score=0.8
                )
                
                await self.neo4j_client.create_topic_node(topic_node)
                
                # Create topic relationship
                relationship = create_topic_relationship(
                    content_id=content_item.id,
                    topic_id=topic_node.id,
                    relevance_score=0.8
                )
                
                await self.neo4j_client.create_relationship(relationship)
            
            # Store secondary topics
            for topic_name in result.topic_extraction.secondary_topics:
                topic_node = create_topic_node(
                    topic_id=f"topic_{topic_name.lower().replace(' ', '_')}",
                    name=topic_name,
                    tenant_id=self.tenant_id,
                    category="secondary",
                    importance_score=0.5
                )
                
                await self.neo4j_client.create_topic_node(topic_node)
                
                # Create topic relationship
                relationship = create_topic_relationship(
                    content_id=content_item.id,
                    topic_id=topic_node.id,
                    relevance_score=0.5
                )
                
                await self.neo4j_client.create_relationship(relationship)
            
            # Store content in vector database
            await self.qdrant_client.add_content_embedding(
                content_id=content_item.id,
                embedding=await self.embedding_service.generate_embedding(
                    f"Title: {content_item.title}\n\nContent: {content_item.content}"
                ),
                metadata={
                    "title": content_item.title,
                    "content_type": content_item.content_type.value,
                    "tenant_id": self.tenant_id,
                    "seo_score": result.quality_assessment.seo_score,
                    "main_topics": result.topic_extraction.main_topics
                }
            )
            
        except Exception as e:
            logger.error(
                "Failed to store analysis results",
                content_id=content_item.id,
                error=str(e)
            )
            raise
    
    @trace_agent("content_analysis", "analyze_content_batch")
    async def analyze_content_batch(
        self,
        content_items: List[ContentItem],
        max_concurrent: int = 5,
        progress_callback: Optional[callable] = None
    ) -> List[ContentAnalysisResult]:
        """
        Analyze multiple content items concurrently.
        
        Args:
            content_items: List of content items to analyze
            max_concurrent: Maximum concurrent analyses
            progress_callback: Optional progress callback
            
        Returns:
            List of analysis results
        """
        logger.info(
            "Starting batch content analysis",
            batch_size=len(content_items),
            tenant_id=self.tenant_id
        )
        
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []
        
        async def analyze_single(item: ContentItem, index: int) -> ContentAnalysisResult:
            async with semaphore:
                result = await self.analyze_content(item)
                
                if progress_callback:
                    progress = (index + 1) / len(content_items)
                    await progress_callback(progress, item.id, result)
                
                return result
        
        # Execute analyses concurrently
        tasks = [
            analyze_single(item, i)
            for i, item in enumerate(content_items)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Separate successful results from errors
        successful_results = []
        errors = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append({
                    "content_id": content_items[i].id,
                    "error": str(result)
                })
            else:
                successful_results.append(result)
        
        if errors:
            logger.warning(
                "Some content analyses failed",
                error_count=len(errors),
                success_count=len(successful_results)
            )
        
        logger.info(
            "Batch content analysis completed",
            total_items=len(content_items),
            successful=len(successful_results),
            errors=len(errors)
        )
        
        return successful_results
    
    async def get_content_analytics(
        self,
        content_id: Optional[str] = None,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Get content analytics for a specific content item or all content.
        
        Args:
            content_id: Optional content ID to filter by
            days_back: Number of days to look back
            
        Returns:
            Content analytics data
        """
        try:
            # Query graph for content analytics
            if content_id:
                query = """
                MATCH (c:Content {id: $content_id, tenant_id: $tenant_id})
                OPTIONAL MATCH (c)-[:RELATES_TO]->(t:Topic)
                OPTIONAL MATCH (c)-[:SIMILAR_TO]->(similar:Content)
                RETURN c, collect(DISTINCT t.name) as topics, 
                       collect(DISTINCT similar.title) as similar_content,
                       size((c)-[:RELATES_TO]->()) as topic_count,
                       size((c)-[:SIMILAR_TO]->()) as similarity_count
                """
                params = {"content_id": content_id, "tenant_id": self.tenant_id}
            else:
                query = """
                MATCH (c:Content {tenant_id: $tenant_id})
                WHERE c.created_at >= datetime() - duration('P${days_back}D')
                OPTIONAL MATCH (c)-[:RELATES_TO]->(t:Topic)
                RETURN c, collect(DISTINCT t.name) as topics,
                       size((c)-[:RELATES_TO]->()) as topic_count,
                       avg(c.seo_score) as avg_seo_score
                """
                params = {"tenant_id": self.tenant_id, "days_back": days_back}
            
            result = await self.neo4j_client.execute_query(query, params)
            
            return {
                "analytics": result,
                "generated_at": datetime.now(timezone.utc),
                "tenant_id": self.tenant_id
            }
            
        except Exception as e:
            logger.error(
                "Failed to get content analytics",
                content_id=content_id,
                error=str(e)
            )
            raise
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "agent_type": "content_analysis",
            "tenant_id": self.tenant_id,
            "initialized_at": datetime.now(timezone.utc),
            "dependencies": {
                "neo4j_client": bool(self.neo4j_client),
                "qdrant_client": bool(self.qdrant_client),
                "supabase_client": bool(self.supabase_client),
                "embedding_service": bool(self.embedding_service)
            }
        }


# =============================================================================
# Utility Functions
# =============================================================================

async def create_content_analysis_agent(
    tenant_id: str,
    neo4j_client: Optional[Neo4jClient] = None,
    qdrant_client: Optional[QdrantClient] = None,
    supabase_client: Optional[SupabaseClient] = None,
    embedding_service: Optional[EmbeddingService] = None
) -> ContentAnalysisAgent:
    """
    Create a configured Content Analysis Agent.
    
    Args:
        tenant_id: Tenant identifier
        neo4j_client: Neo4j client (will create if not provided)
        qdrant_client: Qdrant client (will create if not provided)
        supabase_client: Supabase client (will create if not provided)
        embedding_service: Embedding service (will create if not provided)
        
    Returns:
        ContentAnalysisAgent instance
    """
    settings = get_settings()
    
    # Initialize clients if not provided
    if neo4j_client is None:
        neo4j_client = Neo4jClient(
            uri=settings.neo4j_uri,
            user=settings.neo4j_username,
            password=settings.neo4j_password
        )
        await neo4j_client.connect()
    
    if qdrant_client is None:
        qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )
    
    if supabase_client is None:
        supabase_client = SupabaseClient(
            url=settings.supabase_url,
            key=settings.supabase_key
        )
    
    if embedding_service is None:
        embedding_service = EmbeddingService(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model
        )
    
    return ContentAnalysisAgent(
        neo4j_client=neo4j_client,
        qdrant_client=qdrant_client,
        supabase_client=supabase_client,
        embedding_service=embedding_service,
        tenant_id=tenant_id
    )


if __name__ == "__main__":
    # Example usage
    async def main():
        # Create agent
        agent = await create_content_analysis_agent(tenant_id="test-tenant")
        
        # Create sample content
        content = ContentItem(
            title="SEO Best Practices for 2024",
            content="""
            Search Engine Optimization continues to evolve rapidly. 
            In 2024, focus on user experience, quality content, and technical SEO.
            
            Key areas include:
            1. Core Web Vitals optimization
            2. Mobile-first indexing
            3. AI-generated content guidelines
            4. Voice search optimization
            5. Local SEO improvements
            
            Quality content remains the foundation of good SEO.
            """,
            content_type=ContentType.ARTICLE,
            status="published",
            tenant_id="test-tenant",
            author_id="test-author"
        )
        
        # Analyze content
        result = await agent.analyze_content(content)
        
        print(f"Analysis completed for: {content.title}")
        print(f"Overall quality score: {result.quality_assessment.overall_score}")
        print(f"Main topics: {result.topic_extraction.main_topics}")
        print(f"Processing time: {result.processing_time:.2f}s")
        
        # Get agent stats
        stats = agent.get_agent_stats()
        print(f"Agent stats: {stats}")
    
    asyncio.run(main())