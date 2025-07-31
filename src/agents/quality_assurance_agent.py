"""
Quality Assurance Agent for SEO Content Knowledge Graph System.

This agent ensures content quality, brand voice compliance, SEO optimization,
and maintains editorial standards across all generated and ingested content.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pydantic import BaseModel, Field
from pydantic_ai import tool
import json
import re

from .base_agent import BaseAgent, AgentContext, AgentResult, agent_registry

logger = logging.getLogger(__name__)


class QualityAssuranceRequest(BaseModel):
    """Request model for quality assurance tasks."""
    content_id: Optional[str] = None
    content_text: Optional[str] = None
    content_title: Optional[str] = None
    assessment_type: str = Field(default="comprehensive")  # comprehensive, brand_voice, seo, readability, factual, plagiarism
    target_keywords: List[str] = Field(default_factory=list)
    content_type: str = Field(default="blog_post")
    target_audience: str = Field(default="general")
    quality_threshold: float = Field(default=0.8)
    include_suggestions: bool = True
    auto_fix_minor_issues: bool = False
    reference_guidelines: Optional[Dict[str, Any]] = None


class QualityAssuranceAgent(BaseAgent):
    """
    AI agent for comprehensive content quality assurance and compliance verification.
    
    Capabilities:
    - Brand voice compliance verification
    - SEO optimization assessment and validation
    - Content readability and user experience evaluation
    - Factual accuracy and citation verification
    - Plagiarism detection and originality assessment
    - Editorial standards and style guide compliance
    - Accessibility and inclusivity evaluation
    - Legal and regulatory compliance checking
    - Performance prediction and optimization recommendations
    """
    
    def __init__(self):
        super().__init__(
            name="quality_assurance",
            description="Ensures content quality, brand compliance, and editorial standards across all content"
        )
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Quality Assurance Agent."""
        return """You are an expert Quality Assurance Agent specializing in content quality, compliance, and editorial excellence.

Your role is to ensure all content meets the highest standards for:
- Brand voice consistency and authenticity
- SEO optimization and search engine compatibility
- Editorial quality and readability
- Factual accuracy and credibility
- Legal and regulatory compliance
- Accessibility and inclusivity
- User experience and engagement potential
- Business objective alignment

Always consider:
1. The organization's brand guidelines and voice requirements
2. SEO best practices and algorithm preferences
3. Editorial standards and style guide compliance
4. Target audience needs and preferences
5. Legal and regulatory requirements
6. Accessibility standards and inclusive language
7. Performance metrics and optimization opportunities
8. Competitive differentiation and unique value

Provide detailed, actionable feedback with specific improvement recommendations and quality scores."""
    
    def _register_tools(self) -> None:
        """Register tools specific to quality assurance."""
        
        @self._agent.tool
        async def assess_brand_voice_compliance(content: str, brand_guidelines: Dict) -> Dict[str, Any]:
            """Assess content compliance with brand voice guidelines."""
            return await self._assess_brand_voice_compliance(content, brand_guidelines)
        
        @self._agent.tool
        async def evaluate_seo_optimization(content: str, title: str, keywords: List[str]) -> Dict[str, Any]:
            """Evaluate SEO optimization and provide recommendations."""
            return await self._evaluate_seo_optimization(content, title, keywords)
        
        @self._agent.tool
        async def analyze_readability_and_ux(content: str, target_audience: str) -> Dict[str, Any]:
            """Analyze content readability and user experience factors."""
            return await self._analyze_readability_and_ux(content, target_audience)
        
        @self._agent.tool
        async def check_factual_accuracy(content: str, content_type: str) -> Dict[str, Any]:
            """Check factual accuracy and credibility of content."""
            return await self._check_factual_accuracy(content, content_type)
        
        @self._agent.tool
        async def detect_plagiarism_and_originality(content: str) -> Dict[str, Any]:
            """Detect potential plagiarism and assess content originality."""
            return await self._detect_plagiarism_and_originality(content)
        
        @self._agent.tool
        async def validate_editorial_standards(content: str, style_guide: Dict) -> Dict[str, Any]:
            """Validate content against editorial standards and style guide."""
            return await self._validate_editorial_standards(content, style_guide)
        
        @self._agent.tool
        async def assess_accessibility_compliance(content: str) -> Dict[str, Any]:
            """Assess content accessibility and inclusivity."""
            return await self._assess_accessibility_compliance(content)
        
        @self._agent.tool
        async def check_legal_compliance(content: str, industry: str) -> Dict[str, Any]:
            """Check content for legal and regulatory compliance."""
            return await self._check_legal_compliance(content, industry)
        
        @self._agent.tool
        async def predict_performance_metrics(content: str, content_type: str, keywords: List[str]) -> Dict[str, Any]:
            """Predict content performance metrics and optimization potential."""
            return await self._predict_performance_metrics(content, content_type, keywords)
    
    async def _execute_task(self, task_data: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        """Execute quality assurance task."""
        request = QualityAssuranceRequest(**task_data)
        
        # Get content text
        content_text = await self._get_content_text(request)
        if not content_text:
            raise ValueError("No content provided for quality assessment")
        
        # Get brand voice and compliance guidelines
        brand_voice = await self._get_brand_voice_config()
        seo_preferences = await self._get_seo_preferences()
        
        # Merge with provided guidelines
        guidelines = {
            "brand_voice": brand_voice,
            "seo_preferences": seo_preferences,
            "industry_context": context.industry_context
        }
        
        if request.reference_guidelines:
            guidelines.update(request.reference_guidelines)
        
        # Perform assessment based on type
        if request.assessment_type == "comprehensive":
            return await self._comprehensive_assessment(request, content_text, guidelines, context)
        elif request.assessment_type == "brand_voice":
            return await self._brand_voice_assessment(request, content_text, guidelines)
        elif request.assessment_type == "seo":
            return await self._seo_assessment(request, content_text, guidelines)
        elif request.assessment_type == "readability":
            return await self._readability_assessment(request, content_text, guidelines)
        elif request.assessment_type == "factual":
            return await self._factual_assessment(request, content_text, guidelines)
        elif request.assessment_type == "plagiarism":
            return await self._plagiarism_assessment(request, content_text, guidelines)
        else:
            raise ValueError(f"Unknown assessment type: {request.assessment_type}")
    
    async def _comprehensive_assessment(self, request: QualityAssuranceRequest, content: str,
                                      guidelines: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        """Perform comprehensive quality assessment."""
        
        # Run comprehensive assessment using Pydantic AI agent
        assessment_prompt = f"""
        Perform comprehensive quality assurance assessment for this content:

        CONTENT TITLE: {request.content_title or 'No title provided'}
        CONTENT TYPE: {request.content_type}
        TARGET AUDIENCE: {request.target_audience}
        TARGET KEYWORDS: {', '.join(request.target_keywords)}
        QUALITY THRESHOLD: {request.quality_threshold}

        CONTENT TO ASSESS:
        {content}

        BRAND GUIDELINES:
        {json.dumps(guidelines.get('brand_voice', {}), indent=2)}

        SEO PREFERENCES:
        {json.dumps(guidelines.get('seo_preferences', {}), indent=2)}

        INDUSTRY CONTEXT: {guidelines.get('industry_context', 'General')}

        Conduct thorough assessment covering:
        1. Brand voice compliance and consistency
        2. SEO optimization and keyword integration
        3. Content readability and user experience
        4. Editorial quality and standards
        5. Factual accuracy and credibility
        6. Accessibility and inclusivity
        7. Legal and regulatory compliance
        8. Performance potential and optimization opportunities

        Use available tools to gather detailed metrics and provide specific, actionable recommendations.
        """
        
        # Execute AI assessment
        ai_result = await self._agent.run(assessment_prompt)
        
        # Perform programmatic assessments
        brand_compliance = await self._assess_brand_voice_compliance(content, guidelines.get('brand_voice', {}))
        seo_evaluation = await self._evaluate_seo_optimization(content, request.content_title or "", request.target_keywords)
        readability_analysis = await self._analyze_readability_and_ux(content, request.target_audience)
        editorial_validation = await self._validate_editorial_standards(content, guidelines.get('style_guide', {}))
        
        # Calculate overall quality score
        quality_scores = {
            "brand_compliance": brand_compliance.get("compliance_score", 0) / 100,
            "seo_optimization": seo_evaluation.get("seo_score", 0) / 100,
            "readability": readability_analysis.get("readability_score", 0) / 100,
            "editorial_quality": editorial_validation.get("quality_score", 0) / 100
        }
        
        overall_quality_score = sum(quality_scores.values()) / len(quality_scores)
        
        # Generate recommendations and fixes
        recommendations = await self._generate_comprehensive_recommendations(
            brand_compliance, seo_evaluation, readability_analysis, editorial_validation
        )
        
        # Auto-fix minor issues if requested
        auto_fixes = []
        if request.auto_fix_minor_issues:
            auto_fixes = await self._apply_auto_fixes(content, recommendations)
        
        return {
            "assessment_type": "comprehensive",
            "content_id": request.content_id,
            "overall_quality_score": round(overall_quality_score, 3),
            "quality_grade": self._get_quality_grade(overall_quality_score),
            "meets_threshold": overall_quality_score >= request.quality_threshold,
            "ai_assessment": ai_result.data if hasattr(ai_result, 'data') else str(ai_result),
            "detailed_scores": quality_scores,
            "brand_compliance": brand_compliance,
            "seo_evaluation": seo_evaluation,
            "readability_analysis": readability_analysis,
            "editorial_validation": editorial_validation,
            "recommendations": recommendations,
            "auto_fixes_applied": auto_fixes,
            "assessment_timestamp": datetime.now().isoformat(),
            "confidence_score": 0.9
        }
    
    async def _brand_voice_assessment(self, request: QualityAssuranceRequest, content: str,
                                    guidelines: Dict[str, Any]) -> Dict[str, Any]:
        """Perform brand voice focused assessment."""
        brand_guidelines = guidelines.get('brand_voice', {})
        brand_compliance = await self._assess_brand_voice_compliance(content, brand_guidelines)
        
        return {
            "assessment_type": "brand_voice",
            "content_id": request.content_id,
            "brand_compliance": brand_compliance,
            "recommendations": await self._generate_brand_voice_recommendations(brand_compliance),
            "confidence_score": 0.85
        }
    
    async def _seo_assessment(self, request: QualityAssuranceRequest, content: str,
                            guidelines: Dict[str, Any]) -> Dict[str, Any]:
        """Perform SEO focused assessment."""
        seo_evaluation = await self._evaluate_seo_optimization(content, request.content_title or "", request.target_keywords)
        
        return {
            "assessment_type": "seo",
            "content_id": request.content_id,
            "seo_evaluation": seo_evaluation,
            "recommendations": await self._generate_seo_recommendations(seo_evaluation),
            "confidence_score": 0.9
        }
    
    async def _readability_assessment(self, request: QualityAssuranceRequest, content: str,
                                    guidelines: Dict[str, Any]) -> Dict[str, Any]:
        """Perform readability focused assessment."""
        readability_analysis = await self._analyze_readability_and_ux(content, request.target_audience)
        
        return {
            "assessment_type": "readability",
            "content_id": request.content_id,
            "readability_analysis": readability_analysis,
            "recommendations": await self._generate_readability_recommendations(readability_analysis),
            "confidence_score": 0.85
        }
    
    async def _factual_assessment(self, request: QualityAssuranceRequest, content: str,
                                guidelines: Dict[str, Any]) -> Dict[str, Any]:
        """Perform factual accuracy assessment."""
        factual_check = await self._check_factual_accuracy(content, request.content_type)
        
        return {
            "assessment_type": "factual",
            "content_id": request.content_id,
            "factual_check": factual_check,
            "recommendations": await self._generate_factual_recommendations(factual_check),
            "confidence_score": 0.75
        }
    
    async def _plagiarism_assessment(self, request: QualityAssuranceRequest, content: str,
                                   guidelines: Dict[str, Any]) -> Dict[str, Any]:
        """Perform plagiarism and originality assessment."""
        originality_check = await self._detect_plagiarism_and_originality(content)
        
        return {
            "assessment_type": "plagiarism",
            "content_id": request.content_id,
            "originality_check": originality_check,
            "recommendations": await self._generate_originality_recommendations(originality_check),
            "confidence_score": 0.8
        }
    
    # Tool implementation methods
    
    async def _assess_brand_voice_compliance(self, content: str, brand_guidelines: Dict) -> Dict[str, Any]:
        """Assess content compliance with brand voice guidelines."""
        if not brand_guidelines:
            return {
                "compliance_score": 50,
                "assessment": "No brand guidelines available for assessment",
                "violations": [],
                "strengths": []
            }
        
        compliance_score = 100
        violations = []
        strengths = []
        content_lower = content.lower()
        
        # Check tone compliance
        target_tone = brand_guidelines.get('tone', 'professional')
        tone_assessment = self._assess_tone_compliance(content, target_tone)
        if tone_assessment['compliant']:
            strengths.append(f"Content maintains {target_tone} tone effectively")
        else:
            compliance_score -= 15
            violations.append(f"Tone inconsistency: {tone_assessment['issue']}")
        
        # Check formality level
        target_formality = brand_guidelines.get('formality', 'semi-formal')
        formality_assessment = self._assess_formality_compliance(content, target_formality)
        if formality_assessment['compliant']:
            strengths.append(f"Appropriate {target_formality} formality level")
        else:
            compliance_score -= 10
            violations.append(f"Formality issue: {formality_assessment['issue']}")
        
        # Check prohibited terms
        prohibited_terms = brand_guidelines.get('prohibitedTerms', [])
        for term in prohibited_terms:
            if term.lower() in content_lower:
                compliance_score -= 20
                violations.append(f"Contains prohibited term: '{term}'")
        
        # Check preferred phrases usage
        preferred_phrases = brand_guidelines.get('preferredPhrases', [])
        used_preferred = 0
        for phrase in preferred_phrases:
            if phrase.lower() in content_lower:
                used_preferred += 1
                strengths.append(f"Uses preferred phrase: '{phrase}'")
        
        if used_preferred > 0:
            compliance_score += min(10, used_preferred * 2)
        
        # Industry context alignment
        industry_context = brand_guidelines.get('industryContext', '')
        if industry_context:
            context_alignment = self._assess_industry_context_alignment(content, industry_context)
            if context_alignment['aligned']:
                strengths.append("Content aligns well with industry context")
            else:
                compliance_score -= 10
                violations.append("Weak industry context alignment")
        
        compliance_score = max(0, min(100, compliance_score))
        
        return {
            "compliance_score": compliance_score,
            "grade": self._get_compliance_grade(compliance_score),
            "violations": violations,
            "strengths": strengths,
            "tone_assessment": tone_assessment,
            "formality_assessment": formality_assessment,
            "preferred_phrases_used": used_preferred,
            "improvement_areas": self._identify_brand_improvement_areas(violations)
        }
    
    async def _evaluate_seo_optimization(self, content: str, title: str, keywords: List[str]) -> Dict[str, Any]:
        """Evaluate SEO optimization and provide recommendations."""
        seo_score = 0
        seo_issues = []
        seo_strengths = []
        
        word_count = len(content.split())
        content_lower = content.lower()
        
        # Word count assessment
        if 800 <= word_count <= 2500:
            seo_score += 20
            seo_strengths.append(f"Optimal word count: {word_count} words")
        elif 500 <= word_count <= 3000:
            seo_score += 15
            seo_strengths.append(f"Acceptable word count: {word_count} words")
        else:
            seo_issues.append(f"Suboptimal word count: {word_count} words (aim for 800-2500)")
        
        # Title optimization
        if title:
            title_lower = title.lower()
            if 30 <= len(title) <= 60:
                seo_score += 15
                seo_strengths.append("Title length is SEO-optimized")
            else:
                seo_issues.append(f"Title length ({len(title)} chars) should be 30-60 characters")
            
            # Title keyword optimization
            if keywords and any(kw.lower() in title_lower for kw in keywords):
                seo_score += 15
                seo_strengths.append("Title contains target keywords")
            else:
                seo_issues.append("Title should contain primary target keyword")
        
        # Keyword optimization
        keyword_analysis = {}
        for keyword in keywords:
            keyword_lower = keyword.lower()
            occurrences = content_lower.count(keyword_lower)
            density = (occurrences / word_count) * 100 if word_count > 0 else 0
            
            keyword_analysis[keyword] = {
                "occurrences": occurrences,
                "density": round(density, 2),
                "optimal": 1 <= density <= 3
            }
            
            if 1 <= density <= 3:
                seo_score += 10
                seo_strengths.append(f"Optimal keyword density for '{keyword}': {density:.1f}%")
            elif density < 1:
                seo_issues.append(f"Low keyword density for '{keyword}': {density:.1f}% (increase usage)")
            else:
                seo_issues.append(f"High keyword density for '{keyword}': {density:.1f}% (may be keyword stuffing)")
        
        # Header structure
        headers = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        if len(headers) >= 3:
            seo_score += 15
            seo_strengths.append(f"Good header structure with {len(headers)} headers")
        elif len(headers) >= 1:
            seo_score += 10
            seo_strengths.append(f"Basic header structure with {len(headers)} headers")
        else:
            seo_issues.append("Missing headers - add H2 and H3 tags for better structure")
        
        # Internal linking potential
        internal_links = len(re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content))
        if internal_links >= 3:
            seo_score += 10
            seo_strengths.append(f"Good internal linking with {internal_links} links")
        elif internal_links >= 1:
            seo_score += 5
        else:
            seo_issues.append("Add internal links to improve SEO and user navigation")
        
        # Readability for SEO
        readability_score = self._calculate_readability_score(content)
        if readability_score >= 60:
            seo_score += 15
            seo_strengths.append(f"Good readability score: {readability_score:.1f}")
        else:
            seo_issues.append(f"Low readability score: {readability_score:.1f} (aim for 60+)")
        
        return {
            "seo_score": min(100, seo_score),
            "grade": self._get_seo_grade(seo_score),
            "word_count": word_count,
            "keyword_analysis": keyword_analysis,
            "header_count": len(headers),
            "internal_links": internal_links,
            "readability_score": readability_score,
            "seo_strengths": seo_strengths,
            "seo_issues": seo_issues,
            "optimization_opportunities": self._identify_seo_opportunities(seo_issues)
        }
    
    async def _analyze_readability_and_ux(self, content: str, target_audience: str) -> Dict[str, Any]:
        """Analyze content readability and user experience factors."""
        readability_score = self._calculate_readability_score(content)
        
        # Sentence analysis
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
        sentence_lengths = [len(sentence.split()) for sentence in sentences]
        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        
        # Paragraph analysis
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        paragraph_lengths = [len(p.split()) for p in paragraphs]
        avg_paragraph_length = sum(paragraph_lengths) / len(paragraph_lengths) if paragraph_lengths else 0
        
        # Vocabulary complexity
        words = content.lower().split()
        complex_words = [word for word in words if len(word) > 6]
        complexity_ratio = len(complex_words) / len(words) if words else 0
        
        # Structure analysis
        structure_score = 0
        structure_notes = []
        
        # Headers
        headers = re.findall(r'^#+\s+', content, re.MULTILINE)
        if len(headers) >= 3:
            structure_score += 25
            structure_notes.append("Good header structure for scannability")
        elif len(headers) >= 1:
            structure_score += 15
        else:
            structure_notes.append("Add headers to improve content structure")
        
        # Lists
        lists = len(re.findall(r'^\s*[-*+•]\s+|^\s*\d+\.\s+', content, re.MULTILINE))
        if lists >= 2:
            structure_score += 25
            structure_notes.append("Good use of lists for readability")
        elif lists >= 1:
            structure_score += 15
        else:
            structure_notes.append("Consider adding bullet points or numbered lists")
        
        # Paragraph breaks
        if len(paragraphs) >= 3:
            structure_score += 25
            structure_notes.append("Well-structured with multiple paragraphs")
        elif len(paragraphs) >= 2:
            structure_score += 15
        else:
            structure_notes.append("Break up content into smaller paragraphs")
        
        # Engagement elements
        engagement_score = 0
        questions = len(re.findall(r'\?', content))
        if questions >= 2:
            engagement_score += 20
            structure_notes.append("Good use of questions for engagement")
        
        # Audience-specific assessment
        audience_assessment = self._assess_audience_alignment(content, target_audience, readability_score)
        
        overall_ux_score = (readability_score + structure_score + engagement_score) / 3
        
        return {
            "readability_score": round(readability_score, 1),
            "ux_score": round(overall_ux_score, 1),
            "grade": self._get_readability_grade(readability_score),
            "sentence_analysis": {
                "total_sentences": len(sentences),
                "average_length": round(avg_sentence_length, 1),
                "optimal_range": "15-20 words",
                "assessment": "good" if 10 <= avg_sentence_length <= 25 else "needs_improvement"
            },
            "paragraph_analysis": {
                "total_paragraphs": len(paragraphs),
                "average_length": round(avg_paragraph_length, 1),
                "optimal_range": "50-100 words",
                "assessment": "good" if 30 <= avg_paragraph_length <= 120 else "needs_improvement"
            },
            "vocabulary_analysis": {
                "complexity_ratio": round(complexity_ratio, 3),
                "assessment": "appropriate" if complexity_ratio <= 0.3 else "too_complex"
            },
            "structure_score": structure_score,
            "structure_notes": structure_notes,
            "audience_assessment": audience_assessment,
            "improvement_suggestions": self._generate_readability_improvements(
                readability_score, avg_sentence_length, avg_paragraph_length, complexity_ratio
            )
        }
    
    async def _check_factual_accuracy(self, content: str, content_type: str) -> Dict[str, Any]:
        """Check factual accuracy and credibility of content."""
        # This would integrate with fact-checking APIs in production
        
        # Extract potential facts and claims
        factual_claims = self._extract_factual_claims(content)
        
        # Check for citations and sources
        citations = len(re.findall(r'\[([^\]]+)\]|\(([^)]+)\)', content))
        external_links = len(re.findall(r'https?://[^\s)]+', content))
        
        # Assess claim verifiability
        verifiable_claims = sum(1 for claim in factual_claims if self._is_verifiable_claim(claim))
        
        # Calculate credibility score
        credibility_score = 50  # Base score
        
        if citations >= 3:
            credibility_score += 20
        elif citations >= 1:
            credibility_score += 10
        
        if external_links >= 2:
            credibility_score += 15
        elif external_links >= 1:
            credibility_score += 8
        
        if verifiable_claims / len(factual_claims) >= 0.7 if factual_claims else True:
            credibility_score += 15
        
        credibility_score = min(100, credibility_score)
        
        return {
            "credibility_score": credibility_score,
            "factual_claims_found": len(factual_claims),
            "verifiable_claims": verifiable_claims,
            "citations_found": citations,
            "external_links": external_links,
            "fact_check_notes": [
                "Statistical claims should include sources",
                "Industry-specific claims need verification",
                "Historical facts should be cited"
            ],
            "source_recommendations": self._generate_source_recommendations(content_type),
            "verification_needed": [claim for claim in factual_claims if not self._is_verifiable_claim(claim)]
        }
    
    async def _detect_plagiarism_and_originality(self, content: str) -> Dict[str, Any]:
        """Detect potential plagiarism and assess content originality."""
        # This would integrate with plagiarism detection APIs in production
        
        # Analyze content uniqueness patterns
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
        
        # Simple originality assessment
        originality_score = 85  # Mock score
        
        # Check for common phrases that might indicate copied content
        common_phrases = [
            "according to studies",
            "research shows that",
            "it is important to note",
            "in today's digital world"
        ]
        
        phrase_matches = sum(1 for phrase in common_phrases if phrase.lower() in content.lower())
        if phrase_matches > 3:
            originality_score -= 10
        
        # Assess sentence structure diversity
        sentence_starters = [sentence.split()[0].lower() for sentence in sentences if sentence.split()]
        starter_diversity = len(set(sentence_starters)) / len(sentence_starters) if sentence_starters else 1
        
        if starter_diversity < 0.5:
            originality_score -= 15
        
        return {
            "originality_score": max(0, originality_score),
            "plagiarism_risk": "low" if originality_score >= 80 else "medium" if originality_score >= 60 else "high",
            "sentence_diversity": round(starter_diversity, 2),
            "common_phrase_usage": phrase_matches,
            "uniqueness_indicators": [
                "Unique perspective and insights",
                "Original examples and case studies",
                "Personal experience integration"
            ],
            "similarity_flags": [],
            "recommendations": [
                "Add more personal insights and examples",
                "Vary sentence structure and beginnings",
                "Include original research or data"
            ]
        }
    
    async def _validate_editorial_standards(self, content: str, style_guide: Dict) -> Dict[str, Any]:
        """Validate content against editorial standards and style guide."""
        quality_score = 80  # Base score
        issues = []
        strengths = []
        
        # Grammar and spelling (simplified check)
        grammar_issues = self._check_basic_grammar(content)
        if grammar_issues:
            quality_score -= len(grammar_issues) * 5
            issues.extend(grammar_issues)
        else:
            strengths.append("No obvious grammar issues detected")
        
        # Consistency checks
        consistency_issues = self._check_consistency(content)
        if consistency_issues:
            quality_score -= len(consistency_issues) * 3
            issues.extend(consistency_issues)
        
        # Style guide compliance
        if style_guide:
            style_compliance = self._check_style_compliance(content, style_guide)
            quality_score += style_compliance.get('bonus_points', 0)
            issues.extend(style_compliance.get('violations', []))
            strengths.extend(style_compliance.get('compliant_elements', []))
        
        # Structure and flow
        flow_assessment = self._assess_content_flow(content)
        if flow_assessment['good_flow']:
            strengths.append("Good content flow and logical structure")
        else:
            quality_score -= 10
            issues.append("Content flow could be improved")
        
        quality_score = max(0, min(100, quality_score))
        
        return {
            "quality_score": quality_score,
            "grade": self._get_quality_grade(quality_score / 100),
            "grammar_check": {
                "issues_found": len(grammar_issues),
                "details": grammar_issues[:5]  # Limit to top 5
            },
            "consistency_check": {
                "issues_found": len(consistency_issues),
                "details": consistency_issues
            },
            "style_compliance": style_guide and self._check_style_compliance(content, style_guide),
            "flow_assessment": flow_assessment,
            "editorial_issues": issues,
            "editorial_strengths": strengths,
            "improvement_priorities": self._prioritize_editorial_improvements(issues)
        }
    
    async def _assess_accessibility_compliance(self, content: str) -> Dict[str, Any]:
        """Assess content accessibility and inclusivity."""
        accessibility_score = 70  # Base score
        accessibility_notes = []
        improvements = []
        
        # Check for inclusive language
        inclusive_language = self._check_inclusive_language(content)
        if inclusive_language['score'] >= 80:
            accessibility_score += 15
            accessibility_notes.append("Uses inclusive language appropriately")
        else:
            improvements.extend(inclusive_language['suggestions'])
        
        # Check for alt text mentions (for images)
        image_references = len(re.findall(r'!\[([^\]]*)\]', content))
        if image_references > 0:
            accessibility_notes.append(f"Content references {image_references} images - ensure alt text is provided")
        
        # Check for clear headings and structure
        headers = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        if len(headers) >= 2:
            accessibility_score += 10
            accessibility_notes.append("Good heading structure for screen readers")
        else:
            improvements.append("Add more headings for better navigation")
        
        # Check for plain language
        readability = self._calculate_readability_score(content)
        if readability >= 70:
            accessibility_score += 10
            accessibility_notes.append("Good readability for diverse audiences")
        else:
            improvements.append("Simplify language for better accessibility")
        
        return {
            "accessibility_score": min(100, accessibility_score),
            "inclusive_language": inclusive_language,
            "readability_for_accessibility": readability,
            "structure_accessibility": {
                "header_count": len(headers),
                "image_references": image_references
            },
            "accessibility_notes": accessibility_notes,
            "improvement_suggestions": improvements,
            "compliance_level": "good" if accessibility_score >= 80 else "needs_improvement"
        }
    
    async def _check_legal_compliance(self, content: str, industry: str) -> Dict[str, Any]:
        """Check content for legal and regulatory compliance."""
        compliance_score = 90  # Base score (assume compliant unless issues found)
        legal_notes = []
        warnings = []
        
        # Check for medical claims (if health-related industry)
        if industry.lower() in ['healthcare', 'medical', 'fitness', 'nutrition']:
            medical_claims = self._check_medical_claims(content)
            if medical_claims['claims_found']:
                warnings.append("Medical/health claims detected - ensure FDA compliance and disclaimers")
                compliance_score -= 15
        
        # Check for financial advice (if finance-related industry)
        if industry.lower() in ['finance', 'investment', 'banking']:
            financial_advice = self._check_financial_advice(content)
            if financial_advice['advice_detected']:
                warnings.append("Financial advice detected - ensure SEC compliance and disclaimers")
                compliance_score -= 15
        
        # Check for copyright issues
        copyright_check = self._check_copyright_usage(content)
        if copyright_check['potential_issues']:
            warnings.extend(copyright_check['issues'])
            compliance_score -= 10
        
        # Check for required disclaimers
        disclaimer_check = self._check_required_disclaimers(content, industry)
        if not disclaimer_check['adequate']:
            warnings.append("Consider adding appropriate disclaimers for industry compliance")
            compliance_score -= 5
        
        return {
            "compliance_score": max(0, compliance_score),
            "compliance_level": "high" if compliance_score >= 85 else "medium" if compliance_score >= 70 else "low",
            "legal_warnings": warnings,
            "legal_notes": legal_notes,
            "industry_specific_checks": {
                "medical_claims": industry.lower() in ['healthcare', 'medical', 'fitness', 'nutrition'],
                "financial_advice": industry.lower() in ['finance', 'investment', 'banking']
            },
            "disclaimer_requirements": disclaimer_check,
            "recommendations": self._generate_legal_recommendations(warnings, industry)
        }
    
    async def _predict_performance_metrics(self, content: str, content_type: str, keywords: List[str]) -> Dict[str, Any]:
        """Predict content performance metrics and optimization potential."""
        # This would use ML models in production
        
        word_count = len(content.split())
        readability = self._calculate_readability_score(content)
        
        # Predict engagement metrics (simplified)
        engagement_prediction = {
            "estimated_time_on_page": max(1, word_count / 200),  # Assuming 200 WPM reading speed
            "predicted_bounce_rate": max(20, 80 - (readability - 50)),  # Better readability = lower bounce
            "social_share_potential": "high" if readability >= 70 and word_count >= 800 else "medium",
            "conversion_potential": self._assess_conversion_potential(content, content_type)
        }
        
        # SEO performance prediction
        seo_prediction = {
            "ranking_potential": self._predict_ranking_potential(content, keywords),
            "organic_traffic_potential": "medium",  # Would be based on keyword volumes
            "featured_snippet_chance": self._assess_snippet_potential(content, keywords)
        }
        
        # Overall performance score
        performance_factors = [
            min(100, word_count / 10),  # Word count factor
            readability,  # Readability factor
            len(keywords) * 10,  # Keyword optimization factor
            60  # Base content quality factor
        ]
        
        performance_score = sum(performance_factors) / len(performance_factors)
        
        return {
            "performance_score": min(100, performance_score),
            "engagement_prediction": engagement_prediction,
            "seo_prediction": seo_prediction,
            "optimization_opportunities": [
                "Improve readability for better engagement",
                "Add more internal links for SEO",
                "Include compelling CTAs for conversions"
            ],
            "success_probability": "high" if performance_score >= 80 else "medium" if performance_score >= 60 else "low"
        }
    
    # Helper methods (many simplified for demo purposes)
    
    async def _get_content_text(self, request: QualityAssuranceRequest) -> Optional[str]:
        """Get content text from various sources."""
        if request.content_text:
            return request.content_text
        
        if request.content_id:
            # Would fetch from database
            return "Sample content for assessment"
        
        return None
    
    def _get_quality_grade(self, score: float) -> str:
        """Get quality grade from score."""
        if score >= 0.9:
            return "A+"
        elif score >= 0.8:
            return "A"
        elif score >= 0.7:
            return "B+"
        elif score >= 0.6:
            return "B"
        elif score >= 0.5:
            return "C+"
        else:
            return "C"
    
    def _assess_tone_compliance(self, content: str, target_tone: str) -> Dict[str, Any]:
        """Assess if content matches target tone."""
        # Simplified tone analysis
        content_lower = content.lower()
        
        tone_indicators = {
            "professional": ["expertise", "implement", "strategy", "optimize"],
            "friendly": ["you", "we", "help", "easy", "simple"],
            "authoritative": ["research", "studies", "proven", "data"],
            "conversational": ["you know", "let's", "imagine", "think about"]
        }
        
        indicators = tone_indicators.get(target_tone, [])
        matches = sum(1 for indicator in indicators if indicator in content_lower)
        
        return {
            "compliant": matches >= 2,
            "confidence": min(1.0, matches / len(indicators)) if indicators else 0.5,
            "issue": f"Insufficient {target_tone} tone indicators" if matches < 2 else None
        }
    
    def _assess_formality_compliance(self, content: str, target_formality: str) -> Dict[str, Any]:
        """Assess if content matches target formality level."""
        # Simplified formality analysis
        contractions = len(re.findall(r"\w+'\w+", content))
        informal_words = len(re.findall(r'\b(gonna|wanna|kinda|sorta)\b', content, re.IGNORECASE))
        
        formality_scores = {
            "formal": contractions == 0 and informal_words == 0,
            "semi-formal": contractions <= 3 and informal_words == 0,
            "casual": True  # Always acceptable for casual
        }
        
        compliant = formality_scores.get(target_formality, True)
        
        return {
            "compliant": compliant,
            "contractions_found": contractions,
            "informal_words": informal_words,
            "issue": f"Too informal for {target_formality} content" if not compliant else None
        }
    
    def _assess_industry_context_alignment(self, content: str, industry_context: str) -> Dict[str, Any]:
        """Assess alignment with industry context."""
        context_words = industry_context.lower().split()
        content_lower = content.lower()
        
        matches = sum(1 for word in context_words if word in content_lower)
        alignment_score = matches / len(context_words) if context_words else 0
        
        return {
            "aligned": alignment_score >= 0.3,
            "alignment_score": alignment_score,
            "matches_found": matches
        }
    
    def _get_compliance_grade(self, score: int) -> str:
        """Get compliance grade from score."""
        if score >= 90:
            return "Excellent"
        elif score >= 80:
            return "Good"
        elif score >= 70:
            return "Acceptable"
        elif score >= 60:
            return "Needs Improvement"
        else:
            return "Poor"
    
    def _identify_brand_improvement_areas(self, violations: List[str]) -> List[str]:
        """Identify brand voice improvement areas."""
        areas = []
        
        for violation in violations:
            if "tone" in violation.lower():
                areas.append("Tone adjustment needed")
            elif "prohibited" in violation.lower():
                areas.append("Remove prohibited terms")
            elif "formality" in violation.lower():
                areas.append("Adjust formality level")
        
        return list(set(areas))
    
    def _get_seo_grade(self, score: int) -> str:
        """Get SEO grade from score."""
        return self._get_compliance_grade(score)
    
    def _identify_seo_opportunities(self, issues: List[str]) -> List[str]:
        """Identify SEO optimization opportunities."""
        opportunities = []
        
        for issue in issues:
            if "keyword" in issue.lower():
                opportunities.append("Keyword optimization")
            elif "header" in issue.lower():
                opportunities.append("Header structure improvement")
            elif "link" in issue.lower():
                opportunities.append("Internal linking enhancement")
        
        return list(set(opportunities))
    
    def _get_readability_grade(self, score: float) -> str:
        """Get readability grade from score."""
        if score >= 90:
            return "Very Easy"
        elif score >= 80:
            return "Easy"
        elif score >= 70:
            return "Fairly Easy"
        elif score >= 60:
            return "Standard"
        elif score >= 50:
            return "Fairly Difficult"
        else:
            return "Difficult"
    
    def _assess_audience_alignment(self, content: str, target_audience: str, readability_score: float) -> Dict[str, Any]:
        """Assess content alignment with target audience."""
        audience_requirements = {
            "general": {"min_readability": 60, "max_complexity": 0.3},
            "technical": {"min_readability": 50, "max_complexity": 0.5},
            "academic": {"min_readability": 40, "max_complexity": 0.6},
            "beginner": {"min_readability": 70, "max_complexity": 0.2}
        }
        
        requirements = audience_requirements.get(target_audience, audience_requirements["general"])
        
        return {
            "aligned": readability_score >= requirements["min_readability"],
            "readability_requirement": requirements["min_readability"],
            "current_readability": readability_score,
            "recommendations": [
                f"Adjust content for {target_audience} audience level",
                f"Target readability score: {requirements['min_readability']}+"
            ]
        }
    
    def _generate_readability_improvements(self, readability: float, avg_sentence: float, 
                                         avg_paragraph: float, complexity: float) -> List[str]:
        """Generate readability improvement suggestions."""
        improvements = []
        
        if readability < 60:
            improvements.append("Simplify vocabulary and sentence structure")
        
        if avg_sentence > 25:
            improvements.append("Break up long sentences (current avg: {:.1f} words)".format(avg_sentence))
        
        if avg_paragraph > 120:
            improvements.append("Create shorter paragraphs for better readability")
        
        if complexity > 0.3:
            improvements.append("Reduce complex vocabulary usage")
        
        return improvements
    
    # Additional helper methods for comprehensive functionality
    
    def _extract_factual_claims(self, content: str) -> List[str]:
        """Extract potential factual claims from content."""
        # Simplified extraction - would use NLP in production
        claims = []
        
        # Look for statistical claims
        stats = re.findall(r'\d+%|\d+\.\d+%|\d+ percent', content)
        claims.extend([f"Statistical claim: {stat}" for stat in stats])
        
        # Look for definitive statements
        definitive_patterns = [
            r'research shows that',
            r'studies indicate',
            r'according to',
            r'data reveals'
        ]
        
        for pattern in definitive_patterns:
            matches = re.findall(pattern + r'[^.!?]*[.!?]', content, re.IGNORECASE)
            claims.extend(matches)
        
        return claims[:10]  # Limit to top 10
    
    def _is_verifiable_claim(self, claim: str) -> bool:
        """Check if a claim is verifiable."""
        # Simplified verification check
        verifiable_indicators = ['study', 'research', 'data', 'survey', 'according to']
        return any(indicator in claim.lower() for indicator in verifiable_indicators)
    
    def _generate_source_recommendations(self, content_type: str) -> List[str]:
        """Generate source recommendations based on content type."""
        recommendations = {
            "blog_post": ["Industry reports", "Case studies", "Expert interviews"],
            "guide": ["Official documentation", "Research papers", "Best practice guides"],
            "article": ["Peer-reviewed studies", "Government sources", "Industry statistics"]
        }
        
        return recommendations.get(content_type, recommendations["blog_post"])
    
    def _check_basic_grammar(self, content: str) -> List[str]:
        """Check for basic grammar issues."""
        issues = []
        
        # Check for common issues (simplified)
        if re.search(r'\bi\b', content):  # Should be "I"
            issues.append("Use capital 'I' for first person")
        
        # Check for repeated words
        words = content.split()
        for i in range(len(words) - 1):
            if words[i].lower() == words[i + 1].lower():
                issues.append(f"Repeated word: '{words[i]}'")
                break
        
        return issues[:5]  # Limit to top 5
    
    def _check_consistency(self, content: str) -> List[str]:
        """Check for consistency issues."""
        issues = []
        
        # Check date format consistency
        date_formats = re.findall(r'\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2}', content)
        if len(set(date_formats)) > 1:
            issues.append("Inconsistent date formats found")
        
        # Check capitalization consistency for brand terms
        # This would be more sophisticated in production
        
        return issues
    
    def _check_style_compliance(self, content: str, style_guide: Dict) -> Dict[str, Any]:
        """Check compliance with style guide."""
        return {
            "bonus_points": 0,
            "violations": [],
            "compliant_elements": ["Consistent formatting"]
        }
    
    def _assess_content_flow(self, content: str) -> Dict[str, Any]:
        """Assess content flow and logical structure."""
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        # Simple flow assessment
        good_flow = len(paragraphs) >= 3 and len(content.split()) >= 300
        
        return {
            "good_flow": good_flow,
            "paragraph_count": len(paragraphs),
            "logical_progression": True,  # Would analyze actual progression
            "transition_quality": "good"
        }
    
    def _prioritize_editorial_improvements(self, issues: List[str]) -> List[str]:
        """Prioritize editorial improvements by impact."""
        priority_order = ["grammar", "structure", "consistency", "style"]
        
        prioritized = []
        for priority in priority_order:
            matching_issues = [issue for issue in issues if priority in issue.lower()]
            prioritized.extend(matching_issues)
        
        return prioritized
    
    def _check_inclusive_language(self, content: str) -> Dict[str, Any]:
        """Check for inclusive language usage."""
        # Simplified inclusive language check
        problematic_terms = ['guys', 'blacklist', 'whitelist', 'master/slave']
        content_lower = content.lower()
        
        issues = [term for term in problematic_terms if term in content_lower]
        
        return {
            "score": 100 - (len(issues) * 20),
            "issues_found": issues,
            "suggestions": [f"Consider alternatives to '{issue}'" for issue in issues]
        }
    
    def _check_medical_claims(self, content: str) -> Dict[str, Any]:
        """Check for medical/health claims."""
        medical_keywords = ['cure', 'treat', 'heal', 'diagnose', 'prevent disease']
        content_lower = content.lower()
        
        claims = [keyword for keyword in medical_keywords if keyword in content_lower]
        
        return {
            "claims_found": len(claims) > 0,
            "claims": claims
        }
    
    def _check_financial_advice(self, content: str) -> Dict[str, Any]:
        """Check for financial advice."""
        financial_keywords = ['invest in', 'buy stocks', 'guaranteed returns', 'financial advice']
        content_lower = content.lower()
        
        advice = [keyword for keyword in financial_keywords if keyword in content_lower]
        
        return {
            "advice_detected": len(advice) > 0,
            "advice_terms": advice
        }
    
    def _check_copyright_usage(self, content: str) -> Dict[str, Any]:
        """Check for potential copyright issues."""
        return {
            "potential_issues": False,
            "issues": []
        }
    
    def _check_required_disclaimers(self, content: str, industry: str) -> Dict[str, Any]:
        """Check for required disclaimers."""
        disclaimer_keywords = ['disclaimer', 'not financial advice', 'consult professional']
        content_lower = content.lower()
        
        has_disclaimers = any(keyword in content_lower for keyword in disclaimer_keywords)
        
        return {
            "adequate": has_disclaimers,
            "found_disclaimers": has_disclaimers,
            "industry_requirements": f"Industry '{industry}' may require specific disclaimers"
        }
    
    def _generate_legal_recommendations(self, warnings: List[str], industry: str) -> List[str]:
        """Generate legal compliance recommendations."""
        recommendations = []
        
        if warnings:
            recommendations.append("Review content with legal counsel")
            recommendations.append("Add appropriate disclaimers")
        
        recommendations.append(f"Ensure compliance with {industry} industry regulations")
        
        return recommendations
    
    def _assess_conversion_potential(self, content: str, content_type: str) -> str:
        """Assess conversion potential of content."""
        cta_count = len(re.findall(r'click here|learn more|get started|download|subscribe', content, re.IGNORECASE))
        
        if content_type in ['landing_page', 'product_description'] and cta_count >= 2:
            return "high"
        elif cta_count >= 1:
            return "medium"
        else:
            return "low"
    
    def _predict_ranking_potential(self, content: str, keywords: List[str]) -> str:
        """Predict ranking potential for keywords."""
        word_count = len(content.split())
        keyword_usage = sum(1 for kw in keywords if kw.lower() in content.lower())
        
        if word_count >= 800 and keyword_usage >= len(keywords) * 0.8:
            return "high"
        elif word_count >= 500 and keyword_usage >= len(keywords) * 0.5:
            return "medium"
        else:
            return "low"
    
    def _assess_snippet_potential(self, content: str, keywords: List[str]) -> float:
        """Assess featured snippet potential."""
        # Check for question-answer format
        questions = len(re.findall(r'\?', content))
        lists = len(re.findall(r'^\s*[-*+•]\s+|^\s*\d+\.\s+', content, re.MULTILINE))
        
        if questions >= 2 and lists >= 1:
            return 0.8
        elif questions >= 1 or lists >= 1:
            return 0.6
        else:
            return 0.3
    
    # Recommendation generation methods
    
    async def _generate_comprehensive_recommendations(self, brand_compliance: Dict, seo_evaluation: Dict,
                                                    readability_analysis: Dict, editorial_validation: Dict) -> List[Dict[str, Any]]:
        """Generate comprehensive improvement recommendations."""
        recommendations = []
        
        # Brand voice recommendations
        if brand_compliance.get("compliance_score", 0) < 80:
            recommendations.append({
                "category": "brand_voice",
                "priority": "high",
                "description": "Improve brand voice compliance",
                "specific_actions": brand_compliance.get("improvement_areas", [])
            })
        
        # SEO recommendations
        if seo_evaluation.get("seo_score", 0) < 70:
            recommendations.append({
                "category": "seo",
                "priority": "high", 
                "description": "Optimize for better SEO performance",
                "specific_actions": seo_evaluation.get("optimization_opportunities", [])
            })
        
        # Readability recommendations
        if readability_analysis.get("readability_score", 0) < 60:
            recommendations.append({
                "category": "readability",
                "priority": "medium",
                "description": "Improve content readability",
                "specific_actions": readability_analysis.get("improvement_suggestions", [])
            })
        
        return recommendations
    
    async def _generate_brand_voice_recommendations(self, brand_compliance: Dict) -> List[str]:
        """Generate brand voice specific recommendations."""
        return brand_compliance.get("improvement_areas", [])
    
    async def _generate_seo_recommendations(self, seo_evaluation: Dict) -> List[str]:
        """Generate SEO specific recommendations."""
        return seo_evaluation.get("optimization_opportunities", [])
    
    async def _generate_readability_recommendations(self, readability_analysis: Dict) -> List[str]:
        """Generate readability specific recommendations."""
        return readability_analysis.get("improvement_suggestions", [])
    
    async def _generate_factual_recommendations(self, factual_check: Dict) -> List[str]:
        """Generate factual accuracy recommendations."""
        recommendations = []
        
        if factual_check.get("citations_found", 0) < 2:
            recommendations.append("Add more citations and sources")
        
        if factual_check.get("verification_needed"):
            recommendations.append("Verify factual claims with authoritative sources")
        
        return recommendations
    
    async def _generate_originality_recommendations(self, originality_check: Dict) -> List[str]:
        """Generate originality improvement recommendations."""
        return originality_check.get("recommendations", [])
    
    async def _apply_auto_fixes(self, content: str, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply automatic fixes for minor issues."""
        auto_fixes = []
        
        # This would implement actual auto-fixing logic
        # For now, return simulated fixes
        
        for rec in recommendations:
            if rec.get("priority") == "low" and "spelling" in rec.get("description", "").lower():
                auto_fixes.append({
                    "type": "spelling_correction",
                    "description": "Fixed spelling errors",
                    "applied": True
                })
        
        return auto_fixes


# Register the agent
quality_assurance_agent = QualityAssuranceAgent()
agent_registry.register(quality_assurance_agent)