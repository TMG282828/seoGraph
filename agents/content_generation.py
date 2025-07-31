"""
Content Generation Agent for the SEO Content Knowledge Graph System.

This agent generates high-quality, SEO-optimized content aligned with brand voice
and guidelines using AI and knowledge graph insights.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import structlog
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from config.settings import get_settings
from database.neo4j_client import Neo4jClient
from database.qdrant_client import QdrantClient
from database.supabase_client import SupabaseClient
from models.content_models import ContentItem, ContentType, ContentMetadata, ContentMetrics
from models.seo_models import KeywordData, SearchIntent
from services.embedding_service import EmbeddingService

logger = structlog.get_logger(__name__)


class ContentGenerationDependencies:
    """Dependencies for the Content Generation Agent."""
    
    def __init__(
        self,
        neo4j_client: Neo4jClient,
        qdrant_client: QdrantClient,
        supabase_client: SupabaseClient,
        embedding_service: EmbeddingService,
        tenant_id: str,
        brand_voice: Optional[Dict[str, Any]] = None
    ):
        self.neo4j_client = neo4j_client
        self.qdrant_client = qdrant_client
        self.supabase_client = supabase_client
        self.embedding_service = embedding_service
        self.tenant_id = tenant_id
        self.brand_voice = brand_voice or {}


class ContentBrief(BaseModel):
    """Content brief specification."""
    
    title: str = Field(..., description="Content title")
    content_type: ContentType = Field(..., description="Type of content to generate")
    target_audience: str = Field(..., description="Target audience description")
    
    primary_keywords: List[str] = Field(..., description="Primary keywords to target")
    secondary_keywords: List[str] = Field(default_factory=list, description="Secondary keywords")
    
    content_goals: List[str] = Field(..., description="Content objectives and goals")
    search_intent: SearchIntent = Field(SearchIntent.INFORMATIONAL, description="Target search intent")
    
    word_count_target: Optional[int] = Field(None, description="Target word count")
    tone: Optional[str] = Field(None, description="Content tone (professional, casual, etc.)")
    
    include_sections: List[str] = Field(default_factory=list, description="Required sections")
    related_topics: List[str] = Field(default_factory=list, description="Related topics to cover")
    
    seo_requirements: Dict[str, Any] = Field(default_factory=dict, description="SEO requirements")
    brand_guidelines: Dict[str, Any] = Field(default_factory=dict, description="Brand guidelines")


class GeneratedContent(BaseModel):
    """Generated content output."""
    
    title: str = Field(..., description="Generated title")
    content: str = Field(..., description="Generated content body")
    meta_description: str = Field(..., description="Generated meta description")
    
    outline: List[Dict[str, str]] = Field(..., description="Content outline with sections")
    headings: List[str] = Field(..., description="Generated headings (H1, H2, H3)")
    
    keyword_integration: Dict[str, Any] = Field(..., description="Keyword usage analysis")
    seo_elements: Dict[str, Any] = Field(..., description="SEO elements included")
    
    quality_score: float = Field(..., description="Content quality assessment")
    readability_score: float = Field(..., description="Readability assessment")
    brand_alignment_score: float = Field(..., description="Brand alignment assessment")


class ContentSuggestions(BaseModel):
    """Content improvement suggestions."""
    
    seo_improvements: List[str] = Field(..., description="SEO optimization suggestions")
    content_enhancements: List[str] = Field(..., description="Content quality improvements")
    brand_alignment_tips: List[str] = Field(..., description="Brand alignment suggestions")
    
    additional_keywords: List[str] = Field(..., description="Additional keyword opportunities")
    related_content_ideas: List[str] = Field(..., description="Related content suggestions")
    
    call_to_action_suggestions: List[str] = Field(..., description="CTA recommendations")


class ContentGenerationResult(BaseModel):
    """Complete content generation result."""
    
    generation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tenant_id: str = Field(..., description="Tenant identifier")
    
    brief: ContentBrief = Field(..., description="Original content brief")
    generated_content: GeneratedContent = Field(..., description="Generated content")
    suggestions: ContentSuggestions = Field(..., description="Improvement suggestions")
    
    knowledge_graph_insights: Dict[str, Any] = Field(default_factory=dict, description="Insights from knowledge graph")
    similar_content_references: List[Dict[str, Any]] = Field(default_factory=list, description="Similar content found")
    
    processing_time: float = Field(..., description="Generation processing time")
    success: bool = Field(..., description="Generation success status")
    warnings: List[str] = Field(default_factory=list, description="Generation warnings")


# Initialize the Content Generation Agent (lazy initialization)
content_generation_agent = None

def get_content_generation_agent() -> Agent:
    """Get or create the content generation agent."""
    global content_generation_agent
    if content_generation_agent is None:
        from config.settings import get_settings
        settings = get_settings()
        
        content_generation_agent = Agent(
            f"openai:{settings.openai_model}",  # Use configured model from settings
            deps_type=ContentGenerationDependencies,
            result_type=ContentGenerationResult,
            system_prompt="""
You are a specialized Content Generation Agent for creating high-quality, SEO-optimized content.

Your responsibilities include:
1. Creating compelling, well-structured content that aligns with brand voice
2. Optimizing content for target keywords while maintaining natural flow
3. Ensuring content serves the target search intent and audience needs
4. Generating appropriate meta descriptions and SEO elements
5. Providing actionable improvement suggestions

Content Generation Guidelines:
- Write for humans first, search engines second
- Maintain natural keyword integration (avoid keyword stuffing)
- Structure content with clear headings and logical flow
- Include actionable insights and valuable information
- Align with specified brand voice and tone
- Consider search intent (informational, transactional, commercial, navigational)

Quality Standards:
- Content should be comprehensive and in-depth
- Include relevant examples and practical advice
- Maintain consistent tone throughout
- Ensure proper grammar and readability
- Meet or exceed target word count when specified

SEO Optimization:
- Include primary keywords in title, headings, and naturally throughout content
- Use secondary keywords for semantic richness
- Create compelling meta descriptions under 160 characters
- Structure content with proper heading hierarchy (H1, H2, H3)
- Include relevant internal linking opportunities

Brand Alignment:
- Follow brand voice guidelines strictly
- Use appropriate terminology and language style
- Maintain consistency with brand values and messaging
- Adapt tone based on target audience and content type

Always provide specific, actionable suggestions for content improvement and optimization.
            """,
        )
    
    return content_generation_agent


class ContentGenerationAgent:
    """
    Content Generation Agent for creating SEO-optimized,
    brand-aligned content using AI and knowledge graph insights.
    """
    
    def __init__(
        self,
        neo4j_client: Neo4jClient,
        qdrant_client: QdrantClient,
        supabase_client: SupabaseClient,
        embedding_service: EmbeddingService,
        tenant_id: str,
        brand_voice: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Content Generation Agent.
        
        Args:
            neo4j_client: Neo4j database client
            qdrant_client: Qdrant vector database client
            supabase_client: Supabase database client
            embedding_service: Embedding generation service
            tenant_id: Tenant identifier
            brand_voice: Brand voice configuration
        """
        self.neo4j_client = neo4j_client
        self.qdrant_client = qdrant_client
        self.supabase_client = supabase_client
        self.embedding_service = embedding_service
        self.tenant_id = tenant_id
        self.brand_voice = brand_voice or {}
        
        # Initialize dependencies
        self.deps = ContentGenerationDependencies(
            neo4j_client=neo4j_client,
            qdrant_client=qdrant_client,
            supabase_client=supabase_client,
            embedding_service=embedding_service,
            tenant_id=tenant_id,
            brand_voice=brand_voice
        )
        
        logger.info(
            "Content Generation Agent initialized",
            tenant_id=tenant_id,
            brand_voice_configured=bool(brand_voice)
        )
    
    async def generate_content(
        self,
        brief: ContentBrief,
        use_knowledge_graph: bool = True,
        include_similar_content_analysis: bool = True
    ) -> ContentGenerationResult:
        """
        Generate content based on the provided brief.
        
        Args:
            brief: Content brief with requirements
            use_knowledge_graph: Whether to use knowledge graph insights
            include_similar_content_analysis: Whether to analyze similar content
            
        Returns:
            Content generation result
        """
        start_time = asyncio.get_event_loop().time()
        
        logger.info(
            "Starting content generation",
            title=brief.title,
            content_type=brief.content_type,
            tenant_id=self.tenant_id
        )
        
        try:
            # Gather knowledge graph insights
            knowledge_insights = {}
            similar_content = []
            
            if use_knowledge_graph:
                knowledge_insights = await self._gather_knowledge_insights(brief)
            
            if include_similar_content_analysis:
                similar_content = await self._find_similar_content(brief)
            
            # Prepare generation input
            generation_input = self._prepare_generation_input(
                brief, knowledge_insights, similar_content
            )
            
            # Generate content using AI
            agent = get_content_generation_agent()
            result = await agent.run(
                generation_input,
                deps=self.deps
            )
            
            # Enhance result with additional data
            result.knowledge_graph_insights = knowledge_insights
            result.similar_content_references = similar_content
            
            # Create ContentItem for storage
            content_item = self._create_content_item(result)
            
            # Store generated content
            await self._store_generated_content(content_item, result)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            result.processing_time = processing_time
            
            logger.info(
                "Content generation completed",
                title=brief.title,
                processing_time=processing_time,
                quality_score=result.generated_content.quality_score
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Content generation failed",
                title=brief.title,
                error=str(e)
            )
            raise
    
    async def generate_content_variations(
        self,
        brief: ContentBrief,
        variation_count: int = 3,
        variation_types: List[str] = None
    ) -> List[ContentGenerationResult]:
        """
        Generate multiple content variations.
        
        Args:
            brief: Base content brief
            variation_count: Number of variations to generate
            variation_types: Types of variations (tone, structure, focus)
            
        Returns:
            List of content generation results
        """
        variation_types = variation_types or ["tone", "structure", "focus"]
        
        logger.info(
            "Generating content variations",
            title=brief.title,
            variation_count=variation_count,
            tenant_id=self.tenant_id
        )
        
        variations = []
        
        for i in range(variation_count):
            # Modify brief for variation
            variation_brief = self._create_variation_brief(brief, i, variation_types)
            
            # Generate variation
            try:
                variation_result = await self.generate_content(
                    variation_brief,
                    use_knowledge_graph=True,
                    include_similar_content_analysis=i == 0  # Only for first variation
                )
                variations.append(variation_result)
                
            except Exception as e:
                logger.warning(
                    "Failed to generate content variation",
                    variation_number=i,
                    error=str(e)
                )
        
        logger.info(
            "Content variations generated",
            requested=variation_count,
            successful=len(variations)
        )
        
        return variations
    
    async def optimize_existing_content(
        self,
        content_item: ContentItem,
        optimization_goals: List[str],
        preserve_original: bool = True
    ) -> ContentGenerationResult:
        """
        Optimize existing content for better performance.
        
        Args:
            content_item: Existing content to optimize
            optimization_goals: Goals for optimization (seo, readability, engagement)
            preserve_original: Whether to preserve original structure
            
        Returns:
            Optimized content generation result
        """
        logger.info(
            "Starting content optimization",
            content_id=content_item.id,
            goals=optimization_goals,
            tenant_id=self.tenant_id
        )
        
        try:
            # Create optimization brief
            optimization_brief = self._create_optimization_brief(
                content_item, optimization_goals, preserve_original
            )
            
            # Generate optimized content
            result = await self.generate_content(
                optimization_brief,
                use_knowledge_graph=True,
                include_similar_content_analysis=True
            )
            
            # Mark as optimization
            result.brief.content_goals.append("content_optimization")
            
            logger.info(
                "Content optimization completed",
                content_id=content_item.id,
                quality_improvement=result.generated_content.quality_score - (content_item.metrics.quality_score if content_item.metrics else 0)
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Content optimization failed",
                content_id=content_item.id,
                error=str(e)
            )
            raise
    
    async def _gather_knowledge_insights(self, brief: ContentBrief) -> Dict[str, Any]:
        """Gather insights from the knowledge graph."""
        insights = {}
        
        try:
            # Get related topics from graph
            related_topics_query = """
            MATCH (t:Topic {tenant_id: $tenant_id})
            WHERE any(keyword IN $keywords WHERE t.name CONTAINS keyword)
            OPTIONAL MATCH (t)<-[:RELATES_TO]-(c:Content)
            RETURN t.name as topic, t.importance_score as importance, 
                   count(c) as content_count
            ORDER BY importance DESC, content_count DESC
            LIMIT 10
            """
            
            all_keywords = brief.primary_keywords + brief.secondary_keywords
            related_topics = await self.neo4j_client.execute_query(
                related_topics_query,
                {"tenant_id": self.tenant_id, "keywords": all_keywords}
            )
            
            insights["related_topics"] = related_topics
            
            # Get keyword information
            keyword_query = """
            MATCH (k:Keyword {tenant_id: $tenant_id})
            WHERE k.text IN $keywords
            RETURN k.text as keyword, k.search_volume as volume, 
                   k.difficulty as difficulty, k.competition as competition
            """
            
            keyword_data = await self.neo4j_client.execute_query(
                keyword_query,
                {"tenant_id": self.tenant_id, "keywords": all_keywords}
            )
            
            insights["keyword_data"] = keyword_data
            
            # Get content coverage analysis
            coverage_query = """
            MATCH (c:Content {tenant_id: $tenant_id})-[:RELATES_TO]->(t:Topic)
            WHERE any(keyword IN $keywords WHERE t.name CONTAINS keyword)
            RETURN c.title as title, c.content_type as type, 
                   c.seo_score as seo_score, collect(t.name) as topics
            ORDER BY c.seo_score DESC
            LIMIT 5
            """
            
            existing_coverage = await self.neo4j_client.execute_query(
                coverage_query,
                {"tenant_id": self.tenant_id, "keywords": all_keywords}
            )
            
            insights["existing_coverage"] = existing_coverage
            
        except Exception as e:
            logger.warning(
                "Failed to gather knowledge insights",
                error=str(e)
            )
            insights["error"] = str(e)
        
        return insights
    
    async def _find_similar_content(self, brief: ContentBrief) -> List[Dict[str, Any]]:
        """Find similar content in the vector database."""
        similar_content = []
        
        try:
            # Create search vector from brief
            search_text = f"Title: {brief.title}\nKeywords: {', '.join(brief.primary_keywords)}\nGoals: {', '.join(brief.content_goals)}"
            search_embedding = await self.embedding_service.generate_embedding(search_text)
            
            # Search for similar content
            similar_items = await self.qdrant_client.search_similar(
                collection_name=f"content_{self.tenant_id}",
                query_vector=search_embedding,
                limit=5,
                score_threshold=0.6
            )
            
            for item in similar_items:
                similar_content.append({
                    "content_id": item.id,
                    "similarity_score": item.score,
                    "title": item.payload.get("title", "Unknown"),
                    "content_type": item.payload.get("content_type", "Unknown"),
                    "seo_score": item.payload.get("seo_score", 0)
                })
            
        except Exception as e:
            logger.warning(
                "Failed to find similar content",
                error=str(e)
            )
        
        return similar_content
    
    def _prepare_generation_input(
        self,
        brief: ContentBrief,
        knowledge_insights: Dict[str, Any],
        similar_content: List[Dict[str, Any]]
    ) -> str:
        """Prepare input for content generation."""
        input_parts = [
            f"CONTENT BRIEF:",
            f"Title: {brief.title}",
            f"Content Type: {brief.content_type.value}",
            f"Target Audience: {brief.target_audience}",
            f"Search Intent: {brief.search_intent.value}",
            "",
            f"PRIMARY KEYWORDS: {', '.join(brief.primary_keywords)}",
            f"SECONDARY KEYWORDS: {', '.join(brief.secondary_keywords)}",
            "",
            f"CONTENT GOALS:",
        ]
        
        for goal in brief.content_goals:
            input_parts.append(f"- {goal}")
        
        if brief.word_count_target:
            input_parts.append(f"\nTARGET WORD COUNT: {brief.word_count_target}")
        
        if brief.tone:
            input_parts.append(f"TONE: {brief.tone}")
        
        if brief.include_sections:
            input_parts.extend([
                "",
                "REQUIRED SECTIONS:",
            ])
            for section in brief.include_sections:
                input_parts.append(f"- {section}")
        
        if brief.related_topics:
            input_parts.extend([
                "",
                f"RELATED TOPICS: {', '.join(brief.related_topics)}"
            ])
        
        # Add brand voice information
        if self.brand_voice:
            input_parts.extend([
                "",
                "BRAND VOICE GUIDELINES:",
                f"Tone: {self.brand_voice.get('tone', 'Professional')}",
                f"Style: {self.brand_voice.get('style', 'Clear and informative')}",
                f"Values: {', '.join(self.brand_voice.get('values', []))}"
            ])
        
        # Add knowledge graph insights
        if knowledge_insights.get("related_topics"):
            input_parts.extend([
                "",
                "KNOWLEDGE GRAPH INSIGHTS:",
                "Related Topics:"
            ])
            for topic in knowledge_insights["related_topics"][:5]:
                input_parts.append(f"- {topic.get('topic', '')} (importance: {topic.get('importance', 0)})")
        
        # Add similar content references
        if similar_content:
            input_parts.extend([
                "",
                "SIMILAR EXISTING CONTENT (for reference, avoid duplication):"
            ])
            for content in similar_content[:3]:
                input_parts.append(f"- {content['title']} (SEO score: {content['seo_score']})")
        
        return "\n".join(input_parts)
    
    def _create_content_item(self, result: ContentGenerationResult) -> ContentItem:
        """Create ContentItem from generation result."""
        # Create metadata
        metadata = ContentMetadata(
            meta_description=result.generated_content.meta_description,
            meta_keywords=result.brief.primary_keywords + result.brief.secondary_keywords
        )
        
        # Create metrics
        metrics = ContentMetrics(
            quality_score=result.generated_content.quality_score,
            seo_score=result.generated_content.seo_elements.get("seo_score", 0),
            readability_score=result.generated_content.readability_score
        )
        
        # Create content item
        content_item = ContentItem(
            title=result.generated_content.title,
            content=result.generated_content.content,
            content_type=result.brief.content_type,
            status="draft",
            tenant_id=result.tenant_id,
            author_id="ai_generated",
            metadata=metadata,
            metrics=metrics,
            custom_fields={
                "generation_id": result.generation_id,
                "primary_keywords": result.brief.primary_keywords,
                "target_audience": result.brief.target_audience,
                "content_goals": result.brief.content_goals,
                "brand_alignment_score": result.generated_content.brand_alignment_score
            }
        )
        
        return content_item
    
    def _create_variation_brief(
        self,
        base_brief: ContentBrief,
        variation_index: int,
        variation_types: List[str]
    ) -> ContentBrief:
        """Create a variation of the content brief."""
        variation_brief = base_brief.copy(deep=True)
        
        # Modify based on variation type
        variation_type = variation_types[variation_index % len(variation_types)]
        
        if variation_type == "tone":
            tones = ["professional", "conversational", "authoritative", "friendly"]
            variation_brief.tone = tones[variation_index % len(tones)]
            
        elif variation_type == "structure":
            structures = ["step-by-step guide", "comprehensive overview", "practical tips", "case study analysis"]
            variation_brief.content_goals.append(f"structure_as_{structures[variation_index % len(structures)]}")
            
        elif variation_type == "focus":
            focuses = ["beginner-friendly", "advanced insights", "practical application", "theoretical foundation"]
            variation_brief.target_audience = f"{variation_brief.target_audience} - {focuses[variation_index % len(focuses)]}"
        
        # Add variation identifier
        variation_brief.title = f"{variation_brief.title} (Variation {variation_index + 1})"
        
        return variation_brief
    
    def _create_optimization_brief(
        self,
        content_item: ContentItem,
        optimization_goals: List[str],
        preserve_original: bool
    ) -> ContentBrief:
        """Create a brief for content optimization."""
        # Extract existing keywords from content
        existing_keywords = content_item.custom_fields.get("primary_keywords", [])
        if content_item.metadata and content_item.metadata.meta_keywords:
            existing_keywords.extend(content_item.metadata.meta_keywords)
        
        optimization_brief = ContentBrief(
            title=f"Optimized: {content_item.title}",
            content_type=content_item.content_type,
            target_audience=content_item.custom_fields.get("target_audience", "General audience"),
            primary_keywords=existing_keywords[:5] if existing_keywords else ["content optimization"],
            content_goals=optimization_goals + (["preserve_structure"] if preserve_original else []),
            word_count_target=len(content_item.content.split()) if preserve_original else None
        )
        
        # Add original content as context
        optimization_brief.brand_guidelines["original_content"] = content_item.content[:1000] + "..."
        
        return optimization_brief
    
    async def _store_generated_content(
        self,
        content_item: ContentItem,
        result: ContentGenerationResult
    ) -> None:
        """Store generated content in databases."""
        try:
            # Store in Supabase
            content_data = {
                "id": content_item.id,
                "title": content_item.title,
                "content": content_item.content,
                "content_type": content_item.content_type.value,
                "status": content_item.status,
                "tenant_id": content_item.tenant_id,
                "author_id": content_item.author_id,
                "metadata": content_item.metadata.dict() if content_item.metadata else {},
                "metrics": content_item.metrics.dict() if content_item.metrics else {},
                "custom_fields": content_item.custom_fields,
                "created_at": content_item.created_at.isoformat()
            }
            
            await self.supabase_client.create_content_record(content_data)
            
            # Store generation result
            generation_data = {
                "id": result.generation_id,
                "content_id": content_item.id,
                "tenant_id": result.tenant_id,
                "brief": result.brief.dict(),
                "generated_content": result.generated_content.dict(),
                "suggestions": result.suggestions.dict(),
                "processing_time": result.processing_time,
                "timestamp": result.timestamp.isoformat()
            }
            
            await self.supabase_client.create_generation_record(generation_data)
            
            # Generate and store embedding
            content_embedding = await self.embedding_service.generate_embedding(
                f"Title: {content_item.title}\n\nContent: {content_item.content}"
            )
            
            await self.qdrant_client.add_content_embedding(
                content_id=content_item.id,
                embedding=content_embedding,
                metadata={
                    "title": content_item.title,
                    "content_type": content_item.content_type.value,
                    "tenant_id": content_item.tenant_id,
                    "seo_score": result.generated_content.seo_elements.get("seo_score", 0),
                    "quality_score": result.generated_content.quality_score,
                    "generated": True
                }
            )
            
        except Exception as e:
            logger.error(
                "Failed to store generated content",
                content_id=content_item.id,
                error=str(e)
            )
            raise
    
    async def get_generation_history(
        self,
        content_type: Optional[ContentType] = None,
        days_back: int = 30,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get content generation history.
        
        Args:
            content_type: Optional content type filter
            days_back: Days to look back
            limit: Maximum results
            
        Returns:
            List of generation records
        """
        try:
            filters = {"tenant_id": self.tenant_id}
            if content_type:
                filters["content_type"] = content_type.value
            
            # Calculate date threshold
            from datetime import timedelta
            date_threshold = datetime.now(timezone.utc) - timedelta(days=days_back)
            
            generation_records = await self.supabase_client.query_generation_records(
                filters=filters,
                date_threshold=date_threshold,
                limit=limit
            )
            
            return generation_records
            
        except Exception as e:
            logger.error(
                "Failed to get generation history",
                tenant_id=self.tenant_id,
                error=str(e)
            )
            return []
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "agent_type": "content_generation",
            "tenant_id": self.tenant_id,
            "initialized_at": datetime.now(timezone.utc),
            "brand_voice_configured": bool(self.brand_voice),
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
    
    Args:
        tenant_id: Tenant identifier
        brand_voice: Brand voice configuration
        neo4j_client: Neo4j client (will create if not provided)
        qdrant_client: Qdrant client (will create if not provided)
        supabase_client: Supabase client (will create if not provided)
        embedding_service: Embedding service (will create if not provided)
        
    Returns:
        ContentGenerationAgent instance
    """
    try:
        from config.settings import get_settings
        settings = get_settings()
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        # Use mock clients for testing
        return ContentGenerationAgent(
            neo4j_client=neo4j_client or MockNeo4jClient(),
            qdrant_client=qdrant_client or MockQdrantClient(),
            supabase_client=supabase_client or MockSupabaseClient(),
            embedding_service=embedding_service or MockEmbeddingService(),
            tenant_id=tenant_id,
            brand_voice=brand_voice
        )
    
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
    
    return ContentGenerationAgent(
        neo4j_client=neo4j_client,
        qdrant_client=qdrant_client,
        supabase_client=supabase_client,
        embedding_service=embedding_service,
        tenant_id=tenant_id,
        brand_voice=brand_voice
    )


def create_content_brief(
    title: str,
    content_type: ContentType,
    target_audience: str,
    primary_keywords: List[str],
    content_goals: List[str],
    **kwargs
) -> ContentBrief:
    """
    Create a content brief with standard parameters.
    
    Args:
        title: Content title
        content_type: Type of content
        target_audience: Target audience description
        primary_keywords: Primary keywords to target
        content_goals: Content objectives
        **kwargs: Additional brief parameters
        
    Returns:
        ContentBrief instance
    """
    return ContentBrief(
        title=title,
        content_type=content_type,
        target_audience=target_audience,
        primary_keywords=primary_keywords,
        content_goals=content_goals,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    async def main():
        # Brand voice configuration
        brand_voice = {
            "tone": "professional",
            "style": "clear and informative",
            "values": ["innovation", "quality", "customer-focus"]
        }
        
        # Create agent
        agent = await create_content_generation_agent(
            tenant_id="test-tenant",
            brand_voice=brand_voice
        )
        
        # Create content brief
        brief = create_content_brief(
            title="Ultimate Guide to SEO Content Optimization",
            content_type=ContentType.ARTICLE,
            target_audience="Digital marketers and content creators",
            primary_keywords=["seo content optimization", "content seo", "search optimization"],
            content_goals=["educate", "provide actionable tips", "improve search rankings"],
            word_count_target=2000,
            tone="professional",
            include_sections=["Introduction", "Key Strategies", "Best Practices", "Conclusion"]
        )
        
        # Generate content
        result = await agent.generate_content(
            brief=brief,
            use_knowledge_graph=True,
            include_similar_content_analysis=True
        )
        
        print(f"Content generated: {result.generated_content.title}")
        print(f"Quality score: {result.generated_content.quality_score}")
        print(f"Brand alignment: {result.generated_content.brand_alignment_score}")
        print(f"Processing time: {result.processing_time:.2f}s")
        
        # Get agent stats
        stats = agent.get_agent_stats()
        print(f"Agent stats: {stats}")
    
    asyncio.run(main())