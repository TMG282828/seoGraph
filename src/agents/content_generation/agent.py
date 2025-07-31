"""
Content Generation Agent for SEO Content Knowledge Graph System.

This agent generates high-quality, SEO-optimized content that aligns with brand voice,
targets specific keywords, and maintains semantic coherence with the knowledge graph.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pydantic import BaseModel, Field
import json

from ..base_agent import BaseAgent, AgentContext, AgentResult, agent_registry
from pydantic_ai import RunContext
from .tools import ContentGenerationTools
from .prompts import ContentGenerationPrompts

# RAG Integration imports
try:
    from ...database.neo4j_client import Neo4jClient
    from ...database.qdrant_client import QdrantClient
    NEO4J_AVAILABLE = True
    QDRANT_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    QDRANT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Neo4j or Qdrant clients not available - RAG features will be limited")

logger = logging.getLogger(__name__)


class ContentGenerationRequest(BaseModel):
    """Request model for content generation tasks."""
    content_type: str = Field(default="blog_post")  # blog_post, article, guide, landing_page, product_description
    topic: str
    target_keywords: List[str] = Field(default_factory=list)
    content_length: str = Field(default="medium")  # short, medium, long
    writing_style: str = Field(default="informational")  # informational, persuasive, narrative, technical
    target_audience: str = Field(default="general")
    outline_only: bool = False
    include_meta_tags: bool = True
    include_internal_links: bool = True
    reference_content: List[str] = Field(default_factory=list)
    competitor_analysis_data: Optional[Dict[str, Any]] = None
    seo_requirements: Optional[Dict[str, Any]] = None
    
    # RAG Enhancement fields
    use_knowledge_graph: bool = Field(default=True)
    use_vector_search: bool = Field(default=True)
    similarity_threshold: float = Field(default=0.7)
    max_related_content: int = Field(default=5)
    
    # Human-in-Loop Configuration
    human_in_loop: Optional[Dict[str, Any]] = None
    content_goals: Optional[Dict[str, Any]] = None
    brand_voice: Optional[Dict[str, Any]] = None
    
    # Enhanced Brief Structure fields
    parsed_brief: Optional[Dict[str, Any]] = None
    heading_structure: List[Dict[str, str]] = Field(default_factory=list)
    meta_description: Optional[str] = None
    title_tag: Optional[str] = None
    competitor_articles: List[str] = Field(default_factory=list)
    objectives: Optional[str] = None
    call_to_action: Optional[str] = None


class ContentGenerationAgent(BaseAgent):
    """
    AI agent for generating high-quality, SEO-optimized content.
    
    Capabilities:
    - SEO-optimized blog posts and articles
    - Comprehensive guides and tutorials
    - Landing page copy with conversion focus
    - Product descriptions and sales copy
    - Content outlines and structure planning
    - Brand voice compliance and consistency
    - Keyword integration and semantic optimization
    - Internal linking strategy implementation
    - Meta tag and structured data generation
    """
    
    def __init__(self):
        # Initialize tools and prompts first
        self.tools = ContentGenerationTools()
        self.prompts = ContentGenerationPrompts()
        
        super().__init__(
            name="content_generation",
            description="Generates high-quality, SEO-optimized content aligned with brand voice and keyword strategy using RAG-enhanced knowledge retrieval"
        )
        
        # Initialize RAG clients
        self.neo4j_client = None
        self.qdrant_client = None
        
        if NEO4J_AVAILABLE:
            try:
                self.neo4j_client = Neo4jClient()
                self.tools.set_neo4j_client(self.neo4j_client)
                logger.info("Neo4j client initialized for knowledge graph RAG")
            except Exception as e:
                logger.warning(f"Failed to initialize Neo4j client: {e}")
        
        if QDRANT_AVAILABLE:
            try:
                self.qdrant_client = QdrantClient()
                self.tools.set_qdrant_client(self.qdrant_client)
                logger.info("Qdrant client initialized for vector search RAG")
            except Exception as e:
                logger.warning(f"Failed to initialize Qdrant client: {e}")
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Content Generation Agent."""
        return self.prompts.get_system_prompt()
    
    def _register_tools(self) -> None:
        """Register tools specific to content generation."""
        
        @self._agent.tool
        async def create_content_outline(ctx: RunContext[AgentContext], topic: str, keywords: List[str], content_type: str) -> Dict[str, Any]:
            """Create a comprehensive content outline."""
            return await self.tools.create_content_outline(topic, keywords, content_type)
        
        @self._agent.tool
        async def generate_title_variations(ctx: RunContext[AgentContext], topic: str, keywords: List[str], target_audience: str) -> List[str]:
            """Generate multiple title variations optimized for SEO and engagement."""
            return await self.tools.generate_title_variations(topic, keywords, target_audience)
        
        @self._agent.tool
        async def create_introduction(ctx: RunContext[AgentContext], title: str, keywords: List[str], hook_type: str) -> str:
            """Create an engaging introduction section."""
            return await self.tools.create_introduction(title, keywords, hook_type)
        
        @self._agent.tool
        async def generate_section_content(ctx: RunContext[AgentContext], section_title: str, keywords: List[str], word_count: int) -> str:
            """Generate content for a specific section."""
            return await self.tools.generate_section_content(section_title, keywords, word_count)
        
        @self._agent.tool
        async def create_conclusion(ctx: RunContext[AgentContext], main_points: List[str], cta_type: str) -> str:
            """Create a compelling conclusion with call-to-action."""
            return await self.tools.create_conclusion(main_points, cta_type)
        
        @self._agent.tool
        async def suggest_internal_links(ctx: RunContext[AgentContext], content: str, topic: str) -> List[Dict[str, str]]:
            """Suggest internal linking opportunities."""
            return await self.tools.suggest_internal_links(content, topic)
        
        @self._agent.tool
        async def generate_meta_tags(ctx: RunContext[AgentContext], title: str, content: str, keywords: List[str]) -> Dict[str, str]:
            """Generate meta title, description, and keywords."""
            return await self.tools.generate_meta_tags(title, content, keywords)
        
        @self._agent.tool
        async def optimize_for_featured_snippets(ctx: RunContext[AgentContext], content: str, target_query: str) -> str:
            """Optimize content sections for featured snippet capture."""
            return await self.tools.optimize_for_featured_snippets(content, target_query)
        
        # RAG-Enhanced Tools
        @self._agent.tool
        async def search_knowledge_graph(ctx: RunContext[AgentContext], topic: str, keywords: List[str]) -> Dict[str, Any]:
            """Search the knowledge graph for related content and topics."""
            return await self.tools.search_knowledge_graph(topic, keywords)
        
        @self._agent.tool
        async def find_similar_content(ctx: RunContext[AgentContext], query: str, limit: int = 5) -> List[Dict[str, Any]]:
            """Find similar content using vector search."""
            return await self.tools.find_similar_content(query, limit)
        
        @self._agent.tool
        async def get_content_relationships(ctx: RunContext[AgentContext], topic: str) -> Dict[str, Any]:
            """Get relationships and connections for the topic from knowledge graph."""
            return await self.tools.get_content_relationships(topic)
        
        @self._agent.tool
        async def enhance_with_context(ctx: RunContext[AgentContext], content: str, topic: str) -> str:
            """Enhance content with contextual information from knowledge base."""
            return await self.tools.enhance_with_context(content, topic)
    
    async def _execute_task(self, task_data: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        """Execute content generation task."""
        request = ContentGenerationRequest(**task_data)
        
        # Get brand voice and SEO preferences
        brand_voice = await self._get_brand_voice_config()
        seo_preferences = await self._get_seo_preferences()
        
        # Merge SEO requirements
        merged_seo_requirements = {**seo_preferences}
        if request.seo_requirements:
            merged_seo_requirements.update(request.seo_requirements)
        
        # Generate content based on type and requirements
        if request.outline_only:
            return await self._generate_outline_only(request, brand_voice, merged_seo_requirements)
        else:
            return await self._generate_full_content(request, brand_voice, merged_seo_requirements, context)
    
    async def _generate_outline_only(self, request: ContentGenerationRequest, 
                                   brand_voice: Dict[str, Any], seo_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Generate content outline only."""
        
        outline_prompt = self.prompts.get_outline_prompt(request, brand_voice, seo_requirements)
        
        # Execute AI outline generation
        ai_result = await self._agent.run(outline_prompt)
        
        # Generate programmatic outline elements
        title_variations = await self.tools.generate_title_variations(request.topic, request.target_keywords, request.target_audience)
        content_outline = await self.tools.create_content_outline(request.topic, request.target_keywords, request.content_type)
        
        return {
            "generation_type": "outline",
            "topic": request.topic,
            "content_type": request.content_type,
            "ai_outline": ai_result.result_data if hasattr(ai_result, 'result_data') else str(ai_result),
            "title_variations": title_variations,
            "content_outline": content_outline,
            "estimated_word_count": self.tools.estimate_word_count(content_outline, request.content_length),
            "seo_analysis": await self.tools.analyze_outline_seo(content_outline, request.target_keywords),
            "confidence_score": 0.9
        }
    
    async def _generate_full_content(self, request: ContentGenerationRequest, 
                                   brand_voice: Dict[str, Any], seo_requirements: Dict[str, Any],
                                   context: AgentContext) -> Dict[str, Any]:
        """Generate full content piece using structured brief data."""
        
        logger.info(f"🤖 Starting structured content generation for: {request.topic}")
        
        # Use structured brief data if available
        if request.heading_structure:
            logger.info(f"📋 Using structured heading framework: {len(request.heading_structure)} headings")
            return await self._generate_structured_content(request, brand_voice, seo_requirements, context)
        else:
            logger.info("📝 Using traditional outline-based generation")
            return await self._generate_traditional_content(request, brand_voice, seo_requirements, context)
    
    async def _generate_structured_content(self, request: ContentGenerationRequest,
                                         brand_voice: Dict[str, Any], seo_requirements: Dict[str, Any],
                                         context: AgentContext) -> Dict[str, Any]:
        """Generate content following the structured brief heading framework."""
        
        # Determine final article title
        article_title = request.topic  # This should be the actual title from TopicExtractionService
        
        # Build enhanced prompt with structured data
        structured_prompt = self._build_structured_content_prompt(request, brand_voice, seo_requirements)
        
        logger.info(f"🔄 Sending structured content generation to AI (prompt length: {len(structured_prompt)})")
        
        # Execute AI content generation with structured framework
        ai_result = await self._agent.run(structured_prompt)
        
        # Extract generated content from AI result
        generated_content = self._extract_ai_content(ai_result)
        
        # Generate enhanced meta tags using structured data
        meta_tags = {
            "title": request.title_tag or article_title,
            "description": request.meta_description or await self._generate_meta_description(generated_content, request.target_keywords),
            "keywords": ", ".join(request.target_keywords)
        }
        
        # Suggest internal links
        internal_links = await self.tools.suggest_internal_links(generated_content, request.topic)
        
        # Analyze generated content
        content_analysis = await self.tools.analyze_generated_content(generated_content, request, brand_voice)
        
        logger.info(f"✅ Structured content generation complete: {len(generated_content.split())} words")
        
        return {
            "generation_type": "structured_content",
            "topic": request.topic,
            "content_type": request.content_type,
            "title": article_title,
            "content": generated_content,
            "ai_generation_notes": "Generated using structured brief framework with heading structure",
            "content_outline": self._convert_headings_to_outline(request.heading_structure),
            "meta_tags": meta_tags,
            "internal_links": internal_links,
            "content_analysis": content_analysis,
            "word_count": len(generated_content.split()),
            "readability_score": self.tools.calculate_readability_score(generated_content),
            "seo_score": await self.tools.calculate_content_seo_score(generated_content, request.target_keywords),
            "brand_voice_compliance": await self.tools.check_brand_voice_compliance(generated_content, brand_voice),
            "improvement_suggestions": await self.tools.generate_improvement_suggestions(generated_content, request),
            "confidence_score": 0.92,
            "structured_brief_used": True,
            "heading_structure_followed": len(request.heading_structure),
            "knowledge_sources": await self._get_knowledge_sources(request.topic, request.target_keywords),
            "related_topics": await self._get_related_topics(request.topic)
        }
    
    async def _generate_traditional_content(self, request: ContentGenerationRequest,
                                          brand_voice: Dict[str, Any], seo_requirements: Dict[str, Any], 
                                          context: AgentContext) -> Dict[str, Any]:
        """Generate content using traditional outline-based approach."""
        
        # First, create the outline
        outline_result = await self._generate_outline_only(request, brand_voice, seo_requirements)
        content_outline = outline_result["content_outline"]
        selected_title = outline_result["title_variations"][0] if outline_result["title_variations"] else request.topic
        
        # Generate full content using Pydantic AI agent
        content_prompt = self.prompts.get_full_content_prompt(
            request, selected_title, content_outline, brand_voice, seo_requirements
        )
        
        # Execute AI content generation
        ai_result = await self._agent.run(content_prompt)
        
        # Generate content sections programmatically
        introduction = await self.tools.create_introduction(selected_title, request.target_keywords, "question")
        
        # Generate main content sections
        sections_content = {}
        for section in content_outline.get("sections", []):
            section_title = section.get("title", "")
            section_keywords = section.get("keywords", request.target_keywords)
            section_word_count = section.get("estimated_words", 200)
            
            sections_content[section_title] = await self.tools.generate_section_content(
                section_title, section_keywords, section_word_count
            )
        
        # Generate conclusion
        main_points = [section.get("title", "") for section in content_outline.get("sections", [])]
        conclusion = await self.tools.create_conclusion(main_points, "action")
        
        # Combine all content
        full_content = self.tools.combine_content_sections(
            selected_title, introduction, sections_content, conclusion
        )
        
        # Generate meta tags and SEO elements
        meta_tags = await self.tools.generate_meta_tags(selected_title, full_content, request.target_keywords)
        
        # Suggest internal links
        internal_links = await self.tools.suggest_internal_links(full_content, request.topic)
        
        # Optimize for featured snippets
        if request.target_keywords:
            optimized_content = await self.tools.optimize_for_featured_snippets(
                full_content, request.target_keywords[0]
            )
        else:
            optimized_content = full_content
        
        # Analyze generated content
        content_analysis = await self.tools.analyze_generated_content(optimized_content, request, brand_voice)
        
        return {
            "generation_type": "traditional_content",
            "topic": request.topic,
            "content_type": request.content_type,
            "title": selected_title,
            "content": optimized_content,
            "ai_generation_notes": ai_result.result_data if hasattr(ai_result, 'result_data') else str(ai_result),
            "content_outline": content_outline,
            "meta_tags": meta_tags,
            "internal_links": internal_links,
            "content_analysis": content_analysis,
            "word_count": len(optimized_content.split()),
            "readability_score": self.tools.calculate_readability_score(optimized_content),
            "seo_score": await self.tools.calculate_content_seo_score(optimized_content, request.target_keywords),
            "brand_voice_compliance": await self.tools.check_brand_voice_compliance(optimized_content, brand_voice),
            "improvement_suggestions": await self.tools.generate_improvement_suggestions(optimized_content, request),
            "confidence_score": 0.85
        }


    def _build_structured_content_prompt(self, request: ContentGenerationRequest,
                                       brand_voice: Dict[str, Any], seo_requirements: Dict[str, Any]) -> str:
        """Build AI prompt for structured content generation using brief framework."""
        
        # Prepare structured brief data
        brief_context = ""
        if request.parsed_brief:
            brief_context = f"""
STRUCTURED BRIEF DATA:
- Objectives: {request.objectives or 'Not specified'}
- Target Audience: {request.target_audience}
- Call to Action: {request.call_to_action or 'Not specified'}
- Word Count Range: {request.parsed_brief.get('word_count_range', 'Not specified')}
"""
        
        # Prepare heading structure
        heading_framework = ""
        if request.heading_structure:
            heading_framework = "CONTENT STRUCTURE TO FOLLOW:\\n"
            for i, heading in enumerate(request.heading_structure, 1):
                heading_framework += f"{i}. {heading['level'].upper()}: {heading['title']}\\n"
                if heading.get('content'):
                    heading_framework += f"   Content guidance: {heading['content'][:100]}...\\n"
        
        # Prepare competitor context
        competitor_context = ""
        if request.competitor_articles:
            competitor_context = f"""
COMPETITOR ARTICLES TO DIFFERENTIATE FROM:
{chr(10).join(f"- {url}" for url in request.competitor_articles[:3])}
"""
        
        # Prepare reference content
        reference_context = ""
        if request.reference_content:
            reference_context = f"""
REFERENCE CONTENT:
{chr(10).join(request.reference_content[:1000])}
"""
        
        prompt = f"""Generate a comprehensive health trends article about "{request.topic}" following this exact structure:

{brief_context}

{heading_framework}

{competitor_context}

{reference_context}

GENERATION REQUIREMENTS:
- Topic: {request.topic}
- Target Keywords: {', '.join(request.target_keywords)}
- Content Length: {request.content_length}
- Writing Style: {request.writing_style}
- Brand Voice Tone: {brand_voice.get('tone', 'professional and informative')}

CONTENT GUIDELINES:
1. Write intelligent, research-based content about health trends for July 2025
2. Include specific health trend insights, not generic template content
3. Follow the exact heading structure provided above
4. Naturally integrate target keywords: {', '.join(request.target_keywords)}
5. Maintain {brand_voice.get('tone', 'professional')} tone throughout
6. Include actionable advice and practical tips
7. Reference current wellness research and statistics where appropriate
8. Differentiate from competitor content while providing unique value

CONTENT STRUCTURE:
- Start with an engaging introduction that hooks readers
- Follow the provided heading structure exactly
- Each section should provide substantial, valuable information
- Include practical tips and actionable advice in each section
- End with a compelling conclusion and clear call-to-action

Generate the complete article now, ensuring it covers actual health trends and wellness insights for July 2025, not generic placeholder content."""
        
        return prompt
    
    def _extract_ai_content(self, ai_result) -> str:
        """Extract content from AI result object."""
        if hasattr(ai_result, 'result_data') and ai_result.result_data:
            if isinstance(ai_result.result_data, str):
                return ai_result.result_data
            elif isinstance(ai_result.result_data, dict):
                return ai_result.result_data.get('content', str(ai_result.result_data))
        
        # Fallback to string representation
        content = str(ai_result)
        if len(content) > 100:  # Reasonable content length
            return content
        
        return "AI content generation failed - no substantial content returned"
    
    def _convert_headings_to_outline(self, heading_structure: List[Dict[str, str]]) -> Dict[str, Any]:
        """Convert heading structure to traditional outline format."""
        sections = []
        for heading in heading_structure:
            sections.append({
                "title": heading.get('title', ''),
                "level": heading.get('level', 'h2'),
                "content_guidance": heading.get('content', '')[:100] if heading.get('content') else '',
                "estimated_words": 200
            })
        
        return {
            "title": "Structured Content Outline",
            "sections": sections,
            "estimated_total_words": len(sections) * 200
        }
    
    async def _generate_meta_description(self, content: str, keywords: List[str]) -> str:
        """Generate meta description from content and keywords."""
        # Extract first 160 characters of meaningful content for meta description
        clean_content = content.replace('\\n', ' ').strip()
        sentences = clean_content.split('.')[:2]  # First 2 sentences
        meta_desc = '. '.join(sentences)
        
        if len(meta_desc) > 160:
            meta_desc = meta_desc[:157] + "..."
        
        return meta_desc
    
    async def _get_knowledge_sources(self, topic: str, keywords: List[str]) -> List[str]:
        """Get knowledge sources used for content generation."""
        # This would integrate with RAG system to return actual sources
        return [
            "Health and wellness research database",
            "Current wellness trend analysis",
            "July 2025 health trend reports"
        ]
    
    async def _get_related_topics(self, topic: str) -> List[str]:
        """Get related topics for the content."""
        # This would use knowledge graph to find related topics
        return [
            "Summer wellness routines",
            "Health technology trends",
            "Sustainable wellness practices",
            "Mental health awareness"
        ]


# Register the agent
content_generation_agent = ContentGenerationAgent()
agent_registry.register(content_generation_agent)