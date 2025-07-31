"""
Quality Assurance Agent for the SEO Content Knowledge Graph System.

This agent validates content quality, SEO compliance, brand alignment,
and provides comprehensive scoring and improvement recommendations.
"""

import asyncio
import re
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
from models.content_models import ContentItem, ContentType, ContentMetrics
from services.embedding_service import EmbeddingService

logger = structlog.get_logger(__name__)


class QualityAssuranceDependencies:
    """Dependencies for the Quality Assurance Agent."""
    
    def __init__(
        self,
        neo4j_client: Neo4jClient,
        qdrant_client: QdrantClient,
        supabase_client: SupabaseClient,
        embedding_service: EmbeddingService,
        tenant_id: str,
        quality_standards: Optional[Dict[str, Any]] = None
    ):
        self.neo4j_client = neo4j_client
        self.qdrant_client = qdrant_client
        self.supabase_client = supabase_client
        self.embedding_service = embedding_service
        self.tenant_id = tenant_id
        self.quality_standards = quality_standards or {}


class ContentQualityCheck(BaseModel):
    """Individual quality check result."""
    
    check_name: str = Field(..., description="Name of the quality check")
    check_type: str = Field(..., description="Type of check (seo, content, brand, technical)")
    
    passed: bool = Field(..., description="Whether the check passed")
    score: float = Field(..., description="Check score (0-100)")
    weight: float = Field(..., description="Weight of this check in overall score")
    
    issues_found: List[str] = Field(default_factory=list, description="Issues identified")
    recommendations: List[str] = Field(default_factory=list, description="Specific recommendations")
    
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional check details")


class SEOValidation(BaseModel):
    """SEO validation results."""
    
    overall_seo_score: float = Field(..., description="Overall SEO score (0-100)")
    
    title_optimization: ContentQualityCheck = Field(..., description="Title optimization check")
    meta_description: ContentQualityCheck = Field(..., description="Meta description check")
    heading_structure: ContentQualityCheck = Field(..., description="Heading structure check")
    keyword_optimization: ContentQualityCheck = Field(..., description="Keyword optimization check")
    
    content_length: ContentQualityCheck = Field(..., description="Content length check")
    readability: ContentQualityCheck = Field(..., description="Readability check")
    internal_linking: ContentQualityCheck = Field(..., description="Internal linking opportunities")
    
    technical_seo: List[ContentQualityCheck] = Field(default_factory=list, description="Technical SEO checks")


class ContentValidation(BaseModel):
    """Content quality validation results."""
    
    overall_content_score: float = Field(..., description="Overall content score (0-100)")
    
    grammar_spelling: ContentQualityCheck = Field(..., description="Grammar and spelling check")
    structure_flow: ContentQualityCheck = Field(..., description="Content structure and flow")
    depth_comprehensiveness: ContentQualityCheck = Field(..., description="Content depth and comprehensiveness")
    
    originality: ContentQualityCheck = Field(..., description="Content originality check")
    fact_accuracy: ContentQualityCheck = Field(..., description="Fact accuracy assessment")
    source_credibility: ContentQualityCheck = Field(..., description="Source credibility check")
    
    engagement_factors: List[ContentQualityCheck] = Field(default_factory=list, description="Engagement factor checks")


class BrandAlignment(BaseModel):
    """Brand alignment validation results."""
    
    overall_brand_score: float = Field(..., description="Overall brand alignment score (0-100)")
    
    tone_consistency: ContentQualityCheck = Field(..., description="Tone consistency check")
    voice_alignment: ContentQualityCheck = Field(..., description="Brand voice alignment")
    messaging_consistency: ContentQualityCheck = Field(..., description="Messaging consistency")
    
    style_guidelines: ContentQualityCheck = Field(..., description="Style guidelines compliance")
    terminology_usage: ContentQualityCheck = Field(..., description="Brand terminology usage")
    
    brand_values_reflection: ContentQualityCheck = Field(..., description="Brand values reflection")


class QualityAssuranceResult(BaseModel):
    """Complete quality assurance result."""
    
    qa_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content_id: str = Field(..., description="Content item ID")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tenant_id: str = Field(..., description="Tenant identifier")
    
    overall_quality_score: float = Field(..., description="Overall quality score (0-100)")
    quality_grade: str = Field(..., description="Quality grade (A, B, C, D, F)")
    
    seo_validation: SEOValidation = Field(..., description="SEO validation results")
    content_validation: ContentValidation = Field(..., description="Content validation results")
    brand_alignment: BrandAlignment = Field(..., description="Brand alignment results")
    
    critical_issues: List[str] = Field(default_factory=list, description="Critical issues requiring immediate attention")
    improvement_priorities: List[str] = Field(default_factory=list, description="Prioritized improvement recommendations")
    
    readiness_status: str = Field(..., description="Publication readiness status")
    approval_required: bool = Field(..., description="Whether manual approval is required")
    
    processing_time: float = Field(..., description="QA processing time")
    success: bool = Field(..., description="QA success status")
    warnings: List[str] = Field(default_factory=list, description="QA warnings")


# Initialize the Quality Assurance Agent
quality_assurance_agent = Agent(
    "openai:gpt-4o",  # Use GPT-4 for comprehensive quality assessment
    deps_type=QualityAssuranceDependencies,
    result_type=QualityAssuranceResult,
    system_prompt="""
    You are a specialized Quality Assurance Agent for comprehensive content validation and scoring.
    
    Your responsibilities include:
    1. Comprehensive SEO validation and optimization assessment
    2. Content quality evaluation across multiple dimensions
    3. Brand alignment and consistency verification
    4. Technical compliance and best practices validation
    5. Actionable improvement recommendations with priorities
    
    Quality Assessment Framework:
    
    SEO Validation (30% of overall score):
    - Title optimization: length, keywords, appeal (20% of SEO score)
    - Meta description: length, keywords, call-to-action (15% of SEO score)
    - Heading structure: H1, H2, H3 hierarchy and keyword usage (20% of SEO score)
    - Keyword optimization: primary/secondary keyword integration (25% of SEO score)
    - Content length: appropriate for content type and topic depth (10% of SEO score)
    - Readability: reading level, sentence length, paragraph structure (10% of SEO score)
    
    Content Quality (40% of overall score):
    - Grammar and spelling: language accuracy and professionalism (15% of content score)
    - Structure and flow: logical organization and smooth transitions (20% of content score)
    - Depth and comprehensiveness: topic coverage and detail level (25% of content score)
    - Originality: unique insights and avoiding duplication (20% of content score)
    - Engagement factors: hooks, storytelling, actionability (20% of content score)
    
    Brand Alignment (30% of overall score):
    - Tone consistency: matches brand voice guidelines (30% of brand score)
    - Voice alignment: personality and communication style (25% of brand score)
    - Messaging consistency: core messages and value propositions (20% of brand score)
    - Style guidelines: formatting, terminology, conventions (15% of brand score)
    - Brand values reflection: alignment with company values (10% of brand score)
    
    Scoring Scale:
    - 90-100: Exceptional quality, ready for publication
    - 80-89: High quality, minor improvements needed
    - 70-79: Good quality, moderate improvements needed
    - 60-69: Acceptable quality, significant improvements needed
    - Below 60: Poor quality, major revisions required
    
    Grade Assignment:
    - A (90-100): Publication ready
    - B (80-89): Minor revisions needed
    - C (70-79): Moderate revisions needed
    - D (60-69): Major revisions needed
    - F (Below 60): Complete rewrite recommended
    
    Critical Issues (require immediate attention):
    - Factual inaccuracies or misleading information
    - Poor grammar or spelling that affects credibility
    - Missing or poorly optimized title/meta description
    - Keyword stuffing or over-optimization
    - Brand voice inconsistencies
    - Copyright or plagiarism concerns
    
    Always provide specific, actionable recommendations with clear priorities for improvement.
    """,
)


class QualityAssuranceAgent:
    """
    Quality Assurance Agent for comprehensive content validation,
    scoring, and improvement recommendations.
    """
    
    def __init__(
        self,
        neo4j_client: Neo4jClient,
        qdrant_client: QdrantClient,
        supabase_client: SupabaseClient,
        embedding_service: EmbeddingService,
        tenant_id: str,
        quality_standards: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Quality Assurance Agent.
        
        Args:
            neo4j_client: Neo4j database client
            qdrant_client: Qdrant vector database client
            supabase_client: Supabase database client
            embedding_service: Embedding generation service
            tenant_id: Tenant identifier
            quality_standards: Custom quality standards configuration
        """
        self.neo4j_client = neo4j_client
        self.qdrant_client = qdrant_client
        self.supabase_client = supabase_client
        self.embedding_service = embedding_service
        self.tenant_id = tenant_id
        self.quality_standards = quality_standards or self._get_default_quality_standards()
        
        # Initialize dependencies
        self.deps = QualityAssuranceDependencies(
            neo4j_client=neo4j_client,
            qdrant_client=qdrant_client,
            supabase_client=supabase_client,
            embedding_service=embedding_service,
            tenant_id=tenant_id,
            quality_standards=self.quality_standards
        )
        
        logger.info(
            "Quality Assurance Agent initialized",
            tenant_id=tenant_id,
            quality_standards_configured=bool(quality_standards)
        )
    
    async def validate_content(
        self,
        content_item: ContentItem,
        validation_scope: List[str] = None,
        strictness_level: str = "standard"
    ) -> QualityAssuranceResult:
        """
        Perform comprehensive content quality validation.
        
        Args:
            content_item: Content item to validate
            validation_scope: Specific validation areas to focus on
            strictness_level: Validation strictness (lenient, standard, strict)
            
        Returns:
            Quality assurance result with scores and recommendations
        """
        start_time = asyncio.get_event_loop().time()
        
        validation_scope = validation_scope or ["seo", "content", "brand", "technical"]
        
        logger.info(
            "Starting content quality validation",
            content_id=content_item.id,
            scope=validation_scope,
            strictness=strictness_level,
            tenant_id=self.tenant_id
        )
        
        try:
            # Gather validation context
            validation_context = await self._gather_validation_context(content_item)
            
            # Perform technical validations
            technical_metrics = await self._perform_technical_validations(content_item)
            
            # Check for similar content (originality)
            similarity_analysis = await self._analyze_content_similarity(content_item)
            
            # Prepare validation input
            validation_input = self._prepare_validation_input(
                content_item, validation_context, technical_metrics,
                similarity_analysis, validation_scope, strictness_level
            )
            
            # Run AI validation
            result = await quality_assurance_agent.run(
                validation_input,
                deps=self.deps
            )
            
            # Enhance with technical metrics
            await self._enhance_with_technical_metrics(result, technical_metrics)
            
            # Determine readiness status
            result.readiness_status = self._determine_readiness_status(result)
            result.approval_required = self._requires_approval(result)
            
            # Store validation results
            await self._store_validation_results(result)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            result.processing_time = processing_time
            
            logger.info(
                "Content quality validation completed",
                content_id=content_item.id,
                overall_score=result.overall_quality_score,
                grade=result.quality_grade,
                processing_time=processing_time
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Content quality validation failed",
                content_id=content_item.id,
                error=str(e)
            )
            raise
    
    async def validate_content_batch(
        self,
        content_items: List[ContentItem],
        max_concurrent: int = 3,
        progress_callback: Optional[callable] = None
    ) -> List[QualityAssuranceResult]:
        """
        Validate multiple content items concurrently.
        
        Args:
            content_items: List of content items to validate
            max_concurrent: Maximum concurrent validations
            progress_callback: Optional progress callback
            
        Returns:
            List of quality assurance results
        """
        logger.info(
            "Starting batch content validation",
            batch_size=len(content_items),
            tenant_id=self.tenant_id
        )
        
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []
        
        async def validate_single(item: ContentItem, index: int) -> QualityAssuranceResult:
            async with semaphore:
                result = await self.validate_content(item)
                
                if progress_callback:
                    progress = (index + 1) / len(content_items)
                    await progress_callback(progress, item.id, result)
                
                return result
        
        # Execute validations concurrently
        tasks = [
            validate_single(item, i)
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
                "Some content validations failed",
                error_count=len(errors),
                success_count=len(successful_results)
            )
        
        logger.info(
            "Batch content validation completed",
            total_items=len(content_items),
            successful=len(successful_results),
            errors=len(errors)
        )
        
        return successful_results
    
    async def generate_quality_report(
        self,
        content_ids: Optional[List[str]] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        report_type: str = "summary"
    ) -> Dict[str, Any]:
        """
        Generate quality assurance report.
        
        Args:
            content_ids: Specific content IDs to include
            date_range: Date range for report
            report_type: Type of report (summary, detailed, trends)
            
        Returns:
            Quality assurance report data
        """
        logger.info(
            "Generating quality report",
            content_ids_count=len(content_ids) if content_ids else None,
            report_type=report_type,
            tenant_id=self.tenant_id
        )
        
        try:
            # Query QA results from database
            qa_results = await self._query_qa_results(content_ids, date_range)
            
            if report_type == "summary":
                report = await self._generate_summary_report(qa_results)
            elif report_type == "detailed":
                report = await self._generate_detailed_report(qa_results)
            elif report_type == "trends":
                report = await self._generate_trends_report(qa_results)
            else:
                raise ValueError(f"Unknown report type: {report_type}")
            
            report["generated_at"] = datetime.now(timezone.utc)
            report["tenant_id"] = self.tenant_id
            report["report_type"] = report_type
            
            return report
            
        except Exception as e:
            logger.error(
                "Quality report generation failed",
                report_type=report_type,
                error=str(e)
            )
            raise
    
    def _get_default_quality_standards(self) -> Dict[str, Any]:
        """Get default quality standards configuration."""
        return {
            "seo": {
                "title_min_length": 30,
                "title_max_length": 60,
                "meta_description_min_length": 120,
                "meta_description_max_length": 160,
                "min_word_count": 300,
                "max_keyword_density": 3.0,
                "min_headings": 2
            },
            "content": {
                "max_sentence_length": 25,
                "max_paragraph_length": 150,
                "min_readability_score": 60,
                "max_passive_voice_percent": 10
            },
            "brand": {
                "required_tone": "professional",
                "forbidden_words": [],
                "required_terminology": []
            },
            "technical": {
                "max_response_time": 3.0,
                "required_meta_tags": ["description"],
                "image_alt_text_required": True
            }
        }
    
    async def _gather_validation_context(self, content_item: ContentItem) -> Dict[str, Any]:
        """Gather context for content validation."""
        context = {}
        
        try:
            # Get brand voice settings
            brand_settings = await self._get_brand_settings()
            context["brand_voice"] = brand_settings
            
            # Get related content for context
            related_content = await self._get_related_content(content_item)
            context["related_content"] = related_content
            
            # Get topic coverage
            topic_coverage = await self._get_topic_coverage(content_item)
            context["topic_coverage"] = topic_coverage
            
        except Exception as e:
            logger.warning("Failed to gather validation context", error=str(e))
            context["error"] = str(e)
        
        return context
    
    async def _perform_technical_validations(self, content_item: ContentItem) -> Dict[str, Any]:
        """Perform technical validations on content."""
        metrics = {}
        
        # Basic text metrics
        content = content_item.content
        words = content.split()
        sentences = re.split(r'[.!?]+', content)
        paragraphs = content.split('\n\n')
        
        metrics["word_count"] = len(words)
        metrics["sentence_count"] = len([s for s in sentences if s.strip()])
        metrics["paragraph_count"] = len([p for p in paragraphs if p.strip()])
        
        # Average metrics
        if sentences:
            metrics["avg_sentence_length"] = len(words) / len(sentences)
        else:
            metrics["avg_sentence_length"] = 0
        
        if paragraphs:
            metrics["avg_paragraph_length"] = len(words) / len(paragraphs)
        else:
            metrics["avg_paragraph_length"] = 0
        
        # Readability (simple approximation)
        if metrics["sentence_count"] > 0:
            metrics["readability_score"] = self._calculate_readability_score(content)
        else:
            metrics["readability_score"] = 0
        
        # SEO metrics
        title = content_item.title
        if title:
            metrics["title_length"] = len(title)
            metrics["title_word_count"] = len(title.split())
        
        if content_item.metadata and content_item.metadata.meta_description:
            metrics["meta_description_length"] = len(content_item.metadata.meta_description)
        
        # Keyword density (if keywords available)
        if content_item.metadata and content_item.metadata.meta_keywords:
            metrics["keyword_density"] = self._calculate_keyword_density(
                content, content_item.metadata.meta_keywords
            )
        
        # Heading analysis
        metrics["heading_analysis"] = self._analyze_headings(content)
        
        return metrics
    
    async def _analyze_content_similarity(self, content_item: ContentItem) -> Dict[str, Any]:
        """Analyze content similarity for originality check."""
        similarity_analysis = {}
        
        try:
            # Generate embedding for content
            content_embedding = await self.embedding_service.generate_embedding(
                f"Title: {content_item.title}\n\nContent: {content_item.content}"
            )
            
            # Search for similar content
            similar_items = await self.qdrant_client.search_similar(
                collection_name=f"content_{self.tenant_id}",
                query_vector=content_embedding,
                limit=5,
                score_threshold=0.7
            )
            
            # Filter out self
            similar_items = [item for item in similar_items if item.id != content_item.id]
            
            similarity_analysis["similar_content_count"] = len(similar_items)
            similarity_analysis["max_similarity_score"] = max([item.score for item in similar_items], default=0.0)
            similarity_analysis["similar_content"] = [
                {
                    "content_id": item.id,
                    "similarity_score": item.score,
                    "title": item.payload.get("title", "Unknown")
                }
                for item in similar_items
            ]
            
        except Exception as e:
            logger.warning("Failed to analyze content similarity", error=str(e))
            similarity_analysis["error"] = str(e)
        
        return similarity_analysis
    
    def _prepare_validation_input(
        self,
        content_item: ContentItem,
        validation_context: Dict[str, Any],
        technical_metrics: Dict[str, Any],
        similarity_analysis: Dict[str, Any],
        validation_scope: List[str],
        strictness_level: str
    ) -> str:
        """Prepare input for quality validation."""
        input_parts = [
            f"CONTENT QUALITY VALIDATION",
            f"CONTENT ID: {content_item.id}",
            f"VALIDATION SCOPE: {', '.join(validation_scope)}",
            f"STRICTNESS LEVEL: {strictness_level}",
            f"TENANT: {self.tenant_id}",
            "",
            "CONTENT DETAILS:",
            f"Title: {content_item.title}",
            f"Content Type: {content_item.content_type.value}",
            f"Status: {content_item.status}",
            f"Author: {content_item.author_id}",
            "",
            "TECHNICAL METRICS:",
            f"Word Count: {technical_metrics.get('word_count', 0)}",
            f"Sentence Count: {technical_metrics.get('sentence_count', 0)}",
            f"Paragraph Count: {technical_metrics.get('paragraph_count', 0)}",
            f"Average Sentence Length: {technical_metrics.get('avg_sentence_length', 0):.1f} words",
            f"Readability Score: {technical_metrics.get('readability_score', 0):.1f}",
        ]
        
        # Add title analysis
        if technical_metrics.get("title_length"):
            input_parts.extend([
                "",
                "TITLE ANALYSIS:",
                f"Title Length: {technical_metrics['title_length']} characters",
                f"Title Word Count: {technical_metrics.get('title_word_count', 0)} words"
            ])
        
        # Add meta description analysis
        if technical_metrics.get("meta_description_length"):
            input_parts.extend([
                "",
                "META DESCRIPTION ANALYSIS:",
                f"Meta Description Length: {technical_metrics['meta_description_length']} characters"
            ])
        
        # Add heading analysis
        if technical_metrics.get("heading_analysis"):
            input_parts.extend([
                "",
                "HEADING STRUCTURE:",
                f"Headings Found: {technical_metrics['heading_analysis']}"
            ])
        
        # Add similarity analysis
        if similarity_analysis.get("similar_content_count", 0) > 0:
            input_parts.extend([
                "",
                "ORIGINALITY ANALYSIS:",
                f"Similar Content Found: {similarity_analysis['similar_content_count']} items",
                f"Maximum Similarity Score: {similarity_analysis.get('max_similarity_score', 0.0):.2f}"
            ])
        
        # Add brand voice context
        if validation_context.get("brand_voice"):
            brand_voice = validation_context["brand_voice"]
            input_parts.extend([
                "",
                "BRAND VOICE GUIDELINES:",
                f"Tone: {brand_voice.get('tone', 'Professional')}",
                f"Style: {brand_voice.get('style', 'Clear and informative')}",
                f"Values: {', '.join(brand_voice.get('values', []))}"
            ])
        
        # Add quality standards
        input_parts.extend([
            "",
            "QUALITY STANDARDS:",
            f"Minimum Word Count: {self.quality_standards['seo']['min_word_count']}",
            f"Title Length Range: {self.quality_standards['seo']['title_min_length']}-{self.quality_standards['seo']['title_max_length']} characters",
            f"Meta Description Range: {self.quality_standards['seo']['meta_description_min_length']}-{self.quality_standards['seo']['meta_description_max_length']} characters",
            f"Maximum Keyword Density: {self.quality_standards['seo']['max_keyword_density']}%",
            "",
            "CONTENT TO VALIDATE:",
            content_item.content
        ])
        
        return "\n".join(input_parts)
    
    def _calculate_readability_score(self, content: str) -> float:
        """Calculate a simple readability score."""
        words = content.split()
        sentences = re.split(r'[.!?]+', content)
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        
        # Simple readability approximation (higher is better)
        # Optimal sentence length is around 15-20 words
        if avg_sentence_length <= 20:
            return min(100.0, 80.0 + (20 - avg_sentence_length))
        else:
            return max(0.0, 80.0 - (avg_sentence_length - 20) * 2)
    
    def _calculate_keyword_density(self, content: str, keywords: List[str]) -> Dict[str, float]:
        """Calculate keyword density for given keywords."""
        content_lower = content.lower()
        words = content_lower.split()
        total_words = len(words)
        
        keyword_density = {}
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            count = content_lower.count(keyword_lower)
            density = (count / total_words) * 100 if total_words > 0 else 0
            keyword_density[keyword] = density
        
        return keyword_density
    
    def _analyze_headings(self, content: str) -> Dict[str, Any]:
        """Analyze heading structure in content."""
        headings = {
            "h1": len(re.findall(r'^# .+', content, re.MULTILINE)),
            "h2": len(re.findall(r'^## .+', content, re.MULTILINE)),
            "h3": len(re.findall(r'^### .+', content, re.MULTILINE)),
            "h4": len(re.findall(r'^#### .+', content, re.MULTILINE)),
            "h5": len(re.findall(r'^##### .+', content, re.MULTILINE)),
            "h6": len(re.findall(r'^###### .+', content, re.MULTILINE))
        }
        
        total_headings = sum(headings.values())
        
        return {
            "headings_by_level": headings,
            "total_headings": total_headings,
            "has_h1": headings["h1"] > 0,
            "has_proper_hierarchy": headings["h1"] >= 1 and headings["h2"] >= 1
        }
    
    async def _enhance_with_technical_metrics(
        self,
        result: QualityAssuranceResult,
        technical_metrics: Dict[str, Any]
    ) -> None:
        """Enhance validation result with technical metrics."""
        # Update SEO validation with technical data
        if result.seo_validation.content_length.details:
            result.seo_validation.content_length.details.update({
                "word_count": technical_metrics.get("word_count", 0),
                "sentence_count": technical_metrics.get("sentence_count", 0),
                "paragraph_count": technical_metrics.get("paragraph_count", 0)
            })
        
        if result.seo_validation.readability.details:
            result.seo_validation.readability.details.update({
                "readability_score": technical_metrics.get("readability_score", 0),
                "avg_sentence_length": technical_metrics.get("avg_sentence_length", 0)
            })
    
    def _determine_readiness_status(self, result: QualityAssuranceResult) -> str:
        """Determine publication readiness status."""
        if result.overall_quality_score >= 90:
            return "ready_for_publication"
        elif result.overall_quality_score >= 80:
            return "minor_revisions_needed"
        elif result.overall_quality_score >= 70:
            return "moderate_revisions_needed"
        elif result.overall_quality_score >= 60:
            return "major_revisions_needed"
        else:
            return "complete_rewrite_required"
    
    def _requires_approval(self, result: QualityAssuranceResult) -> bool:
        """Determine if manual approval is required."""
        # Require approval for critical issues or low scores
        if result.critical_issues:
            return True
        
        if result.overall_quality_score < 70:
            return True
        
        # Require approval for brand alignment issues
        if result.brand_alignment.overall_brand_score < 80:
            return True
        
        return False
    
    async def _get_brand_settings(self) -> Dict[str, Any]:
        """Get brand voice settings for the tenant."""
        try:
            # Query brand settings from database
            brand_query = """
            SELECT brand_voice, style_guide, tone_guidelines
            FROM tenant_settings
            WHERE tenant_id = $1
            """
            
            # This would be implemented with actual database query
            # For now, return default settings
            return {
                "tone": "professional",
                "style": "clear and informative",
                "values": ["quality", "innovation", "customer-focus"]
            }
            
        except Exception as e:
            logger.warning("Failed to get brand settings", error=str(e))
            return {}
    
    async def _get_related_content(self, content_item: ContentItem) -> List[Dict[str, Any]]:
        """Get related content for validation context."""
        try:
            # Query related content from Neo4j
            related_query = """
            MATCH (c:Content {id: $content_id, tenant_id: $tenant_id})
            MATCH (c)-[:SIMILAR_TO|RELATES_TO]-(related:Content)
            RETURN related.id as id, related.title as title, related.seo_score as seo_score
            LIMIT 5
            """
            
            related_content = await self.neo4j_client.execute_query(
                related_query,
                {"content_id": content_item.id, "tenant_id": self.tenant_id}
            )
            
            return related_content
            
        except Exception as e:
            logger.warning("Failed to get related content", error=str(e))
            return []
    
    async def _get_topic_coverage(self, content_item: ContentItem) -> Dict[str, Any]:
        """Get topic coverage analysis for content."""
        try:
            # Query topic relationships
            topic_query = """
            MATCH (c:Content {id: $content_id, tenant_id: $tenant_id})-[:RELATES_TO]->(t:Topic)
            RETURN t.name as topic, t.importance_score as importance
            """
            
            topics = await self.neo4j_client.execute_query(
                topic_query,
                {"content_id": content_item.id, "tenant_id": self.tenant_id}
            )
            
            return {
                "topics_covered": len(topics),
                "topic_list": [topic["topic"] for topic in topics],
                "avg_importance": sum(topic.get("importance", 0) for topic in topics) / len(topics) if topics else 0
            }
            
        except Exception as e:
            logger.warning("Failed to get topic coverage", error=str(e))
            return {}
    
    async def _store_validation_results(self, result: QualityAssuranceResult) -> None:
        """Store validation results in the database."""
        try:
            validation_data = {
                "id": result.qa_id,
                "content_id": result.content_id,
                "tenant_id": result.tenant_id,
                "overall_quality_score": result.overall_quality_score,
                "quality_grade": result.quality_grade,
                "seo_score": result.seo_validation.overall_seo_score,
                "content_score": result.content_validation.overall_content_score,
                "brand_score": result.brand_alignment.overall_brand_score,
                "readiness_status": result.readiness_status,
                "approval_required": result.approval_required,
                "critical_issues": result.critical_issues,
                "improvement_priorities": result.improvement_priorities,
                "processing_time": result.processing_time,
                "timestamp": result.timestamp.isoformat(),
                "result_data": result.dict()
            }
            
            await self.supabase_client.create_qa_record(validation_data)
            
        except Exception as e:
            logger.error(
                "Failed to store validation results",
                qa_id=result.qa_id,
                error=str(e)
            )
    
    async def _query_qa_results(
        self,
        content_ids: Optional[List[str]],
        date_range: Optional[Tuple[datetime, datetime]]
    ) -> List[Dict[str, Any]]:
        """Query QA results from database."""
        try:
            filters = {"tenant_id": self.tenant_id}
            
            if content_ids:
                filters["content_id"] = content_ids
            
            date_threshold = None
            if date_range:
                date_threshold = date_range[0]
            
            qa_results = await self.supabase_client.query_qa_records(
                filters=filters,
                date_threshold=date_threshold,
                limit=1000
            )
            
            return qa_results
            
        except Exception as e:
            logger.error("Failed to query QA results", error=str(e))
            return []
    
    async def _generate_summary_report(self, qa_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary quality report."""
        if not qa_results:
            return {"message": "No QA results found", "summary": {}}
        
        # Calculate summary metrics
        total_validations = len(qa_results)
        avg_quality_score = sum(result["overall_quality_score"] for result in qa_results) / total_validations
        avg_seo_score = sum(result["seo_score"] for result in qa_results) / total_validations
        avg_content_score = sum(result["content_score"] for result in qa_results) / total_validations
        avg_brand_score = sum(result["brand_score"] for result in qa_results) / total_validations
        
        # Grade distribution
        grade_distribution = {}
        for result in qa_results:
            grade = result["quality_grade"]
            grade_distribution[grade] = grade_distribution.get(grade, 0) + 1
        
        # Readiness status distribution
        readiness_distribution = {}
        for result in qa_results:
            status = result["readiness_status"]
            readiness_distribution[status] = readiness_distribution.get(status, 0) + 1
        
        return {
            "summary": {
                "total_validations": total_validations,
                "average_quality_score": round(avg_quality_score, 2),
                "average_seo_score": round(avg_seo_score, 2),
                "average_content_score": round(avg_content_score, 2),
                "average_brand_score": round(avg_brand_score, 2),
                "grade_distribution": grade_distribution,
                "readiness_distribution": readiness_distribution
            },
            "insights": {
                "quality_trend": "improving" if avg_quality_score > 75 else "needs_attention",
                "strongest_area": max([
                    ("seo", avg_seo_score),
                    ("content", avg_content_score),
                    ("brand", avg_brand_score)
                ], key=lambda x: x[1])[0],
                "improvement_needed": min([
                    ("seo", avg_seo_score),
                    ("content", avg_content_score),
                    ("brand", avg_brand_score)
                ], key=lambda x: x[1])[0]
            }
        }
    
    async def _generate_detailed_report(self, qa_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate detailed quality report."""
        summary = await self._generate_summary_report(qa_results)
        
        # Add detailed breakdowns
        detailed_analysis = {
            "content_breakdown": [],
            "common_issues": {},
            "improvement_recommendations": []
        }
        
        # Analyze common issues
        all_issues = []
        for result in qa_results:
            all_issues.extend(result.get("critical_issues", []))
        
        issue_frequency = {}
        for issue in all_issues:
            issue_frequency[issue] = issue_frequency.get(issue, 0) + 1
        
        detailed_analysis["common_issues"] = dict(sorted(
            issue_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10])
        
        return {
            **summary,
            "detailed_analysis": detailed_analysis
        }
    
    async def _generate_trends_report(self, qa_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate trends quality report."""
        # Sort results by timestamp
        qa_results_sorted = sorted(qa_results, key=lambda x: x["timestamp"])
        
        # Calculate trends over time
        trends = {
            "quality_trend": [],
            "seo_trend": [],
            "content_trend": [],
            "brand_trend": []
        }
        
        # Group by week/month for trend analysis
        from collections import defaultdict
        
        weekly_data = defaultdict(list)
        for result in qa_results_sorted:
            timestamp = datetime.fromisoformat(result["timestamp"])
            week_key = timestamp.strftime("%Y-W%U")
            weekly_data[week_key].append(result)
        
        for week, results in weekly_data.items():
            if results:
                avg_quality = sum(r["overall_quality_score"] for r in results) / len(results)
                avg_seo = sum(r["seo_score"] for r in results) / len(results)
                avg_content = sum(r["content_score"] for r in results) / len(results)
                avg_brand = sum(r["brand_score"] for r in results) / len(results)
                
                trends["quality_trend"].append({"week": week, "score": avg_quality})
                trends["seo_trend"].append({"week": week, "score": avg_seo})
                trends["content_trend"].append({"week": week, "score": avg_content})
                trends["brand_trend"].append({"week": week, "score": avg_brand})
        
        return {
            "trends": trends,
            "trend_analysis": {
                "overall_direction": "improving" if len(trends["quality_trend"]) > 1 and 
                                  trends["quality_trend"][-1]["score"] > trends["quality_trend"][0]["score"] 
                                  else "stable_or_declining",
                "most_improved_area": "quality",  # Would calculate actual most improved
                "needs_attention": "brand"  # Would calculate actual area needing attention
            }
        }
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "agent_type": "quality_assurance",
            "tenant_id": self.tenant_id,
            "initialized_at": datetime.now(timezone.utc),
            "quality_standards_configured": bool(self.quality_standards),
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

async def create_quality_assurance_agent(
    tenant_id: str,
    quality_standards: Optional[Dict[str, Any]] = None,
    neo4j_client: Optional[Neo4jClient] = None,
    qdrant_client: Optional[QdrantClient] = None,
    supabase_client: Optional[SupabaseClient] = None,
    embedding_service: Optional[EmbeddingService] = None
) -> QualityAssuranceAgent:
    """
    Create a configured Quality Assurance Agent.
    
    Args:
        tenant_id: Tenant identifier
        quality_standards: Custom quality standards configuration
        neo4j_client: Neo4j client (will create if not provided)
        qdrant_client: Qdrant client (will create if not provided)
        supabase_client: Supabase client (will create if not provided)
        embedding_service: Embedding service (will create if not provided)
        
    Returns:
        QualityAssuranceAgent instance
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
    
    return QualityAssuranceAgent(
        neo4j_client=neo4j_client,
        qdrant_client=qdrant_client,
        supabase_client=supabase_client,
        embedding_service=embedding_service,
        tenant_id=tenant_id,
        quality_standards=quality_standards
    )


if __name__ == "__main__":
    # Example usage
    async def main():
        # Custom quality standards
        quality_standards = {
            "seo": {
                "title_min_length": 40,
                "title_max_length": 65,
                "meta_description_min_length": 140,
                "meta_description_max_length": 160,
                "min_word_count": 500,
                "max_keyword_density": 2.5,
                "min_headings": 3
            },
            "content": {
                "max_sentence_length": 20,
                "max_paragraph_length": 120,
                "min_readability_score": 70,
                "max_passive_voice_percent": 8
            }
        }
        
        # Create agent
        agent = await create_quality_assurance_agent(
            tenant_id="test-tenant",
            quality_standards=quality_standards
        )
        
        # Create sample content for validation
        from models.content_models import ContentMetadata
        
        content_item = ContentItem(
            title="Complete Guide to SEO Content Optimization in 2024",
            content="""
            # Complete Guide to SEO Content Optimization in 2024
            
            Search Engine Optimization continues to evolve rapidly. In this comprehensive guide,
            we'll cover the latest strategies and best practices for optimizing your content.
            
            ## Understanding Modern SEO
            
            Modern SEO is about creating valuable content that serves user intent. It's no longer
            just about keywords, but about providing comprehensive, authoritative answers.
            
            ## Key Strategies for 2024
            
            1. Focus on user experience and Core Web Vitals
            2. Create topically comprehensive content
            3. Optimize for voice search and AI assistants
            4. Build strong content clusters around main topics
            
            This guide will help you implement these strategies effectively.
            """,
            content_type=ContentType.ARTICLE,
            status="draft",
            tenant_id="test-tenant",
            author_id="test-author",
            metadata=ContentMetadata(
                meta_description="Learn the latest SEO content optimization strategies for 2024. Complete guide with actionable tips and best practices.",
                meta_keywords=["seo content optimization", "content seo", "seo strategies 2024"]
            )
        )
        
        # Validate content
        validation_result = await agent.validate_content(
            content_item=content_item,
            validation_scope=["seo", "content", "brand"],
            strictness_level="standard"
        )
        
        print(f"Quality validation completed: {validation_result.qa_id}")
        print(f"Overall quality score: {validation_result.overall_quality_score}")
        print(f"Quality grade: {validation_result.quality_grade}")
        print(f"Readiness status: {validation_result.readiness_status}")
        print(f"Approval required: {validation_result.approval_required}")
        
        if validation_result.critical_issues:
            print(f"Critical issues: {validation_result.critical_issues}")
        
        if validation_result.improvement_priorities:
            print(f"Top improvements: {validation_result.improvement_priorities[:3]}")
        
        # Generate quality report
        report = await agent.generate_quality_report(
            content_ids=[content_item.id],
            report_type="summary"
        )
        
        print(f"Quality report generated: {report}")
        
        # Get agent stats
        stats = agent.get_agent_stats()
        print(f"Agent stats: {stats}")
    
    asyncio.run(main())