"""
Brand Voice Configuration for the SEO Content Knowledge Graph System.

This module provides structured brand voice validation, consistency scoring,
tone and style guidelines, and brand voice learning capabilities.
"""

import asyncio
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass
import uuid

import structlog
from pydantic import BaseModel, Field, validator
from cachetools import TTLCache
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from database.neo4j_client import Neo4jClient
from models.content_models import ContentItem, ContentType
from config.settings import get_settings

logger = structlog.get_logger(__name__)


class BrandVoiceError(Exception):
    """Raised when brand voice operations fail."""
    pass


class VoiceAnalysisError(BrandVoiceError):
    """Raised when voice analysis fails."""
    pass


class ToneType(str, Enum):
    """Types of brand voice tones."""
    
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    AUTHORITATIVE = "authoritative"
    CONVERSATIONAL = "conversational"
    FORMAL = "formal"
    HUMOROUS = "humorous"
    EMPATHETIC = "empathetic"
    ENTHUSIASTIC = "enthusiastic"
    SERIOUS = "serious"


class StyleAttribute(str, Enum):
    """Style attributes for brand voice."""
    
    TECHNICAL = "technical"
    SIMPLE = "simple"
    DETAILED = "detailed"
    CONCISE = "concise"
    DESCRIPTIVE = "descriptive"
    INSTRUCTIONAL = "instructional"
    NARRATIVE = "narrative"
    PERSUASIVE = "persuasive"
    INFORMATIVE = "informative"
    ENGAGING = "engaging"


class VoiceConsistencyLevel(str, Enum):
    """Levels of voice consistency."""
    
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    INCONSISTENT = "inconsistent"


class BrandVoiceGuidelines(BaseModel):
    """Brand voice guidelines configuration."""
    
    guideline_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Core voice characteristics
    primary_tone: ToneType = Field(..., description="Primary brand tone")
    secondary_tones: List[ToneType] = Field(default_factory=list, description="Secondary acceptable tones")
    style_attributes: List[StyleAttribute] = Field(default_factory=list, description="Style attributes")
    
    # Language preferences
    preferred_pronouns: List[str] = Field(default_factory=list, description="Preferred pronouns (we, you, I)")
    voice_person: str = Field("second", description="Voice person (first, second, third)")
    formality_level: float = Field(0.5, ge=0.0, le=1.0, description="Formality level (0=casual, 1=formal)")
    
    # Content guidelines
    vocabulary_preferences: Dict[str, List[str]] = Field(
        default_factory=dict, 
        description="Preferred vocabulary (industry_terms, avoid_words, etc.)"
    )
    sentence_structure: Dict[str, Any] = Field(
        default_factory=dict,
        description="Sentence structure preferences"
    )
    
    # Brand-specific elements
    brand_values: List[str] = Field(default_factory=list, description="Core brand values")
    brand_personality: List[str] = Field(default_factory=list, description="Brand personality traits")
    target_audience: str = Field(..., description="Target audience description")
    
    # Content type specific guidelines
    content_type_overrides: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Content type specific overrides"
    )
    
    # Validation rules
    mandatory_elements: List[str] = Field(default_factory=list, description="Mandatory elements")
    prohibited_elements: List[str] = Field(default_factory=list, description="Prohibited elements")
    
    # Metadata
    tenant_id: str = Field(..., description="Tenant identifier")
    created_by: str = Field(..., description="Guidelines creator")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    version: str = Field("1.0", description="Guidelines version")
    
    def get_guidelines_for_content_type(self, content_type: str) -> Dict[str, Any]:
        """Get guidelines specific to content type."""
        base_guidelines = {
            'primary_tone': self.primary_tone,
            'secondary_tones': self.secondary_tones,
            'style_attributes': self.style_attributes,
            'formality_level': self.formality_level,
            'vocabulary_preferences': self.vocabulary_preferences,
            'sentence_structure': self.sentence_structure
        }
        
        # Apply content type overrides
        if content_type in self.content_type_overrides:
            base_guidelines.update(self.content_type_overrides[content_type])
        
        return base_guidelines


class VoiceAnalysis(BaseModel):
    """Voice analysis results for content."""
    
    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content_id: str = Field(..., description="Content identifier")
    
    # Detected characteristics
    detected_tone: ToneType = Field(..., description="Detected primary tone")
    tone_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in tone detection")
    detected_style: List[StyleAttribute] = Field(default_factory=list, description="Detected style attributes")
    
    # Consistency scores
    overall_consistency: float = Field(0.0, ge=0.0, le=1.0, description="Overall voice consistency")
    tone_consistency: float = Field(0.0, ge=0.0, le=1.0, description="Tone consistency")
    style_consistency: float = Field(0.0, ge=0.0, le=1.0, description="Style consistency")
    vocabulary_consistency: float = Field(0.0, ge=0.0, le=1.0, description="Vocabulary consistency")
    
    # Detailed analysis
    formality_score: float = Field(0.0, ge=0.0, le=1.0, description="Formality level")
    readability_score: float = Field(0.0, ge=0.0, le=1.0, description="Readability score")
    engagement_score: float = Field(0.0, ge=0.0, le=1.0, description="Engagement potential")
    
    # Issues and recommendations
    consistency_issues: List[str] = Field(default_factory=list, description="Identified consistency issues")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")
    
    # Compliance
    guidelines_compliance: float = Field(0.0, ge=0.0, le=1.0, description="Guidelines compliance score")
    mandatory_elements_present: List[str] = Field(default_factory=list, description="Present mandatory elements")
    prohibited_elements_found: List[str] = Field(default_factory=list, description="Found prohibited elements")
    
    # Metadata
    tenant_id: str = Field(..., description="Tenant identifier")
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    analyzer_version: str = Field("1.0", description="Analyzer version")
    
    def get_consistency_level(self) -> VoiceConsistencyLevel:
        """Get consistency level based on overall score."""
        if self.overall_consistency >= 0.9:
            return VoiceConsistencyLevel.EXCELLENT
        elif self.overall_consistency >= 0.8:
            return VoiceConsistencyLevel.GOOD
        elif self.overall_consistency >= 0.7:
            return VoiceConsistencyLevel.FAIR
        elif self.overall_consistency >= 0.6:
            return VoiceConsistencyLevel.POOR
        else:
            return VoiceConsistencyLevel.INCONSISTENT


class BrandVoiceLearning(BaseModel):
    """Brand voice learning data and insights."""
    
    learning_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Learning data
    content_samples: List[str] = Field(default_factory=list, description="Content samples for learning")
    voice_patterns: Dict[str, float] = Field(default_factory=dict, description="Learned voice patterns")
    
    # Statistical insights
    common_phrases: List[Tuple[str, float]] = Field(default_factory=list, description="Common phrases and frequency")
    vocabulary_distribution: Dict[str, int] = Field(default_factory=dict, description="Vocabulary usage")
    tone_distribution: Dict[str, float] = Field(default_factory=dict, description="Tone usage distribution")
    
    # Temporal analysis
    voice_evolution: Dict[str, List[float]] = Field(default_factory=dict, description="Voice evolution over time")
    trend_analysis: Dict[str, str] = Field(default_factory=dict, description="Voice trend analysis")
    
    # Metadata
    tenant_id: str = Field(..., description="Tenant identifier")
    learning_period_start: datetime = Field(..., description="Learning period start")
    learning_period_end: datetime = Field(..., description="Learning period end")
    content_count: int = Field(0, ge=0, description="Number of content samples")
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BrandVoiceConfiguration:
    """
    Brand voice configuration and analysis service.
    
    Provides:
    - Brand voice guidelines management
    - Voice consistency scoring
    - Content analysis and recommendations
    - Brand voice learning and evolution
    """
    
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j_client = neo4j_client
        self.settings = get_settings()
        
        # Caches
        self.guidelines_cache = TTLCache(maxsize=100, ttl=3600)  # 1 hour
        self.analysis_cache = TTLCache(maxsize=1000, ttl=1800)  # 30 minutes
        self.learning_cache = TTLCache(maxsize=50, ttl=7200)    # 2 hours
        
        # Voice analysis tools
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        
        # Tone detection patterns
        self.tone_patterns = {
            ToneType.PROFESSIONAL: [
                r'\b(furthermore|however|therefore|consequently)\b',
                r'\b(recommend|suggest|advise|propose)\b',
                r'\b(analysis|evaluation|assessment|examination)\b'
            ],
            ToneType.CASUAL: [
                r'\b(hey|hi|yeah|cool|awesome)\b',
                r'\b(gonna|wanna|kinda|sorta)\b',
                r'[!]{2,}|[?]{2,}'
            ],
            ToneType.FRIENDLY: [
                r'\b(welcome|thanks|appreciate|glad)\b',
                r'\b(help|assist|support|guide)\b',
                r'[!](?![!])'
            ],
            ToneType.AUTHORITATIVE: [
                r'\b(must|should|will|shall)\b',
                r'\b(evidence|research|studies|data)\b',
                r'\b(proven|established|demonstrated)\b'
            ],
            ToneType.CONVERSATIONAL: [
                r'\b(you|your|we|our|us)\b',
                r'\?(?![?])',
                r'\b(think|feel|believe|wonder)\b'
            ]
        }
        
        logger.info("Brand voice configuration initialized")
    
    async def initialize(self) -> None:
        """Initialize the brand voice configuration service."""
        try:
            # Set up Neo4j constraints
            await self._setup_constraints()
            
            # Load default guidelines if needed
            await self._load_default_guidelines()
            
            logger.info("Brand voice configuration service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize brand voice configuration: {e}")
            raise BrandVoiceError(f"Initialization failed: {e}")
    
    async def _setup_constraints(self) -> None:
        """Set up Neo4j constraints for brand voice data."""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (g:BrandVoiceGuidelines) REQUIRE g.guideline_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:VoiceAnalysis) REQUIRE a.analysis_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (l:BrandVoiceLearning) REQUIRE l.learning_id IS UNIQUE"
        ]
        
        for constraint in constraints:
            await self.neo4j_client.execute_query(constraint)
    
    async def _load_default_guidelines(self) -> None:
        """Load default brand voice guidelines if none exist."""
        query = """
        MATCH (g:BrandVoiceGuidelines)
        RETURN count(g) as count
        """
        
        result = await self.neo4j_client.execute_query(query)
        if result[0]['count'] == 0:
            # Create default guidelines
            default_guidelines = BrandVoiceGuidelines(
                primary_tone=ToneType.PROFESSIONAL,
                secondary_tones=[ToneType.FRIENDLY, ToneType.CONVERSATIONAL],
                style_attributes=[StyleAttribute.INFORMATIVE, StyleAttribute.ENGAGING],
                preferred_pronouns=["you", "we"],
                voice_person="second",
                formality_level=0.7,
                vocabulary_preferences={
                    "industry_terms": ["SEO", "content marketing", "optimization"],
                    "avoid_words": ["spam", "cheap", "guaranteed"]
                },
                brand_values=["quality", "innovation", "reliability"],
                brand_personality=["expert", "helpful", "trustworthy"],
                target_audience="Content marketers and SEO professionals",
                tenant_id="default"
            )
            
            await self.save_guidelines(default_guidelines)
    
    async def save_guidelines(self, guidelines: BrandVoiceGuidelines) -> str:
        """Save brand voice guidelines."""
        try:
            # Update timestamp
            guidelines.updated_at = datetime.now(timezone.utc)
            
            # Save to Neo4j
            query = """
            MERGE (g:BrandVoiceGuidelines {guideline_id: $guideline_id})
            SET g += $properties
            RETURN g.guideline_id as guideline_id
            """
            
            properties = guidelines.dict()
            properties.pop('guideline_id', None)
            
            result = await self.neo4j_client.execute_query(
                query,
                guideline_id=guidelines.guideline_id,
                properties=properties
            )
            
            # Update cache
            cache_key = f"guidelines:{guidelines.tenant_id}"
            self.guidelines_cache[cache_key] = guidelines
            
            logger.info(f"Saved brand voice guidelines: {guidelines.guideline_id}")
            return guidelines.guideline_id
            
        except Exception as e:
            logger.error(f"Failed to save brand voice guidelines: {e}")
            raise BrandVoiceError(f"Failed to save guidelines: {e}")
    
    async def get_guidelines(self, tenant_id: str) -> Optional[BrandVoiceGuidelines]:
        """Get brand voice guidelines for tenant."""
        try:
            cache_key = f"guidelines:{tenant_id}"
            
            # Check cache first
            if cache_key in self.guidelines_cache:
                return self.guidelines_cache[cache_key]
            
            # Query Neo4j
            query = """
            MATCH (g:BrandVoiceGuidelines)
            WHERE g.tenant_id = $tenant_id
            RETURN g
            ORDER BY g.created_at DESC
            LIMIT 1
            """
            
            result = await self.neo4j_client.execute_query(query, tenant_id=tenant_id)
            
            if result:
                guidelines_data = result[0]['g']
                guidelines = BrandVoiceGuidelines(**guidelines_data)
                
                # Cache the result
                self.guidelines_cache[cache_key] = guidelines
                
                return guidelines
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get brand voice guidelines: {e}")
            raise BrandVoiceError(f"Failed to get guidelines: {e}")
    
    async def analyze_content_voice(self, 
                                  content: str, 
                                  content_id: str,
                                  content_type: str,
                                  tenant_id: str) -> VoiceAnalysis:
        """Analyze content voice consistency."""
        try:
            cache_key = f"analysis:{content_id}"
            
            # Check cache first
            if cache_key in self.analysis_cache:
                return self.analysis_cache[cache_key]
            
            # Get guidelines
            guidelines = await self.get_guidelines(tenant_id)
            if not guidelines:
                raise VoiceAnalysisError("No brand voice guidelines found")
            
            # Perform analysis
            analysis = VoiceAnalysis(
                content_id=content_id,
                tenant_id=tenant_id,
                detected_tone=await self._detect_tone(content),
                tone_confidence=0.0,
                detected_style=await self._detect_style(content)
            )
            
            # Calculate consistency scores
            analysis.tone_consistency = await self._calculate_tone_consistency(
                content, guidelines, analysis.detected_tone
            )
            analysis.style_consistency = await self._calculate_style_consistency(
                content, guidelines, analysis.detected_style
            )
            analysis.vocabulary_consistency = await self._calculate_vocabulary_consistency(
                content, guidelines
            )
            
            # Overall consistency
            analysis.overall_consistency = (
                analysis.tone_consistency * 0.4 +
                analysis.style_consistency * 0.3 +
                analysis.vocabulary_consistency * 0.3
            )
            
            # Additional metrics
            analysis.formality_score = await self._calculate_formality(content)
            analysis.readability_score = await self._calculate_readability(content)
            analysis.engagement_score = await self._calculate_engagement(content)
            
            # Guidelines compliance
            analysis.guidelines_compliance = await self._check_guidelines_compliance(
                content, guidelines, analysis
            )
            
            # Generate recommendations
            analysis.recommendations = await self._generate_recommendations(
                content, guidelines, analysis
            )
            
            # Save analysis
            await self._save_analysis(analysis)
            
            # Cache the result
            self.analysis_cache[cache_key] = analysis
            
            logger.info(f"Analyzed content voice: {content_id}")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze content voice: {e}")
            raise VoiceAnalysisError(f"Voice analysis failed: {e}")
    
    async def _detect_tone(self, content: str) -> ToneType:
        """Detect primary tone in content."""
        tone_scores = {}
        
        for tone, patterns in self.tone_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, content, re.IGNORECASE))
                score += matches
            
            # Normalize by content length
            tone_scores[tone] = score / len(content.split())
        
        # Return tone with highest score
        if tone_scores:
            return max(tone_scores, key=tone_scores.get)
        
        return ToneType.PROFESSIONAL  # Default
    
    async def _detect_style(self, content: str) -> List[StyleAttribute]:
        """Detect style attributes in content."""
        style_patterns = {
            StyleAttribute.TECHNICAL: [
                r'\b(API|SDK|algorithm|framework|implementation)\b',
                r'\b(configure|initialize|optimize|integrate)\b'
            ],
            StyleAttribute.SIMPLE: [
                r'\b(easy|simple|straightforward|basic)\b',
                r'\b(just|only|simply|merely)\b'
            ],
            StyleAttribute.DETAILED: [
                r'\b(specifically|particularly|precisely|exactly)\b',
                r'\b(furthermore|additionally|moreover|consequently)\b'
            ],
            StyleAttribute.INSTRUCTIONAL: [
                r'\b(step|follow|guide|tutorial|how to)\b',
                r'\b(first|second|third|next|then|finally)\b'
            ]
        }
        
        detected_styles = []
        
        for style, patterns in style_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, content, re.IGNORECASE))
                score += matches
            
            # Threshold for style detection
            if score / len(content.split()) > 0.01:  # 1% threshold
                detected_styles.append(style)
        
        return detected_styles
    
    async def _calculate_tone_consistency(self, 
                                        content: str, 
                                        guidelines: BrandVoiceGuidelines,
                                        detected_tone: ToneType) -> float:
        """Calculate tone consistency score."""
        if detected_tone == guidelines.primary_tone:
            return 1.0
        elif detected_tone in guidelines.secondary_tones:
            return 0.8
        else:
            return 0.4
    
    async def _calculate_style_consistency(self, 
                                         content: str, 
                                         guidelines: BrandVoiceGuidelines,
                                         detected_style: List[StyleAttribute]) -> float:
        """Calculate style consistency score."""
        if not guidelines.style_attributes:
            return 1.0
        
        # Calculate overlap
        overlap = set(detected_style).intersection(set(guidelines.style_attributes))
        if guidelines.style_attributes:
            return len(overlap) / len(guidelines.style_attributes)
        
        return 1.0
    
    async def _calculate_vocabulary_consistency(self, 
                                              content: str, 
                                              guidelines: BrandVoiceGuidelines) -> float:
        """Calculate vocabulary consistency score."""
        content_lower = content.lower()
        score = 1.0
        
        # Check for preferred vocabulary
        vocab_prefs = guidelines.vocabulary_preferences
        
        if 'industry_terms' in vocab_prefs:
            preferred_count = sum(1 for term in vocab_prefs['industry_terms'] 
                                if term.lower() in content_lower)
            if vocab_prefs['industry_terms']:
                score *= 0.8 + (0.2 * preferred_count / len(vocab_prefs['industry_terms']))
        
        # Check for avoided words
        if 'avoid_words' in vocab_prefs:
            avoided_count = sum(1 for word in vocab_prefs['avoid_words'] 
                              if word.lower() in content_lower)
            if avoided_count > 0:
                score *= 0.5  # Significant penalty for avoided words
        
        return score
    
    async def _calculate_formality(self, content: str) -> float:
        """Calculate formality score."""
        formal_indicators = [
            r'\b(furthermore|however|therefore|consequently|nevertheless)\b',
            r'\b(recommend|suggest|advise|propose|conclude)\b',
            r'\b(analysis|evaluation|assessment|examination)\b'
        ]
        
        informal_indicators = [
            r'\b(yeah|cool|awesome|great|nice)\b',
            r'\b(gonna|wanna|kinda|sorta)\b',
            r'[!]{2,}'
        ]
        
        formal_score = sum(len(re.findall(pattern, content, re.IGNORECASE)) 
                          for pattern in formal_indicators)
        informal_score = sum(len(re.findall(pattern, content, re.IGNORECASE)) 
                           for pattern in informal_indicators)
        
        total_score = formal_score + informal_score
        if total_score == 0:
            return 0.5  # Neutral
        
        return formal_score / total_score
    
    async def _calculate_readability(self, content: str) -> float:
        """Calculate readability score (simplified)."""
        words = content.split()
        sentences = re.split(r'[.!?]+', content)
        
        if not sentences or not words:
            return 0.0
        
        avg_words_per_sentence = len(words) / len(sentences)
        
        # Simple readability score (lower is better)
        if avg_words_per_sentence <= 15:
            return 1.0
        elif avg_words_per_sentence <= 25:
            return 0.8
        elif avg_words_per_sentence <= 35:
            return 0.6
        else:
            return 0.4
    
    async def _calculate_engagement(self, content: str) -> float:
        """Calculate engagement potential score."""
        engagement_indicators = [
            r'\?',  # Questions
            r'\b(you|your)\b',  # Direct address
            r'\b(discover|learn|find out|explore)\b',  # Action words
            r'[!](?![!])',  # Exclamations (single)
            r'\b(tip|secret|trick|hack)\b'  # Engaging words
        ]
        
        total_score = 0
        for pattern in engagement_indicators:
            matches = len(re.findall(pattern, content, re.IGNORECASE))
            total_score += matches
        
        # Normalize by content length
        words = len(content.split())
        if words == 0:
            return 0.0
        
        engagement_ratio = total_score / words
        return min(engagement_ratio * 10, 1.0)  # Cap at 1.0
    
    async def _check_guidelines_compliance(self, 
                                         content: str, 
                                         guidelines: BrandVoiceGuidelines,
                                         analysis: VoiceAnalysis) -> float:
        """Check compliance with brand voice guidelines."""
        compliance_score = 1.0
        
        # Check mandatory elements
        content_lower = content.lower()
        for element in guidelines.mandatory_elements:
            if element.lower() not in content_lower:
                analysis.consistency_issues.append(f"Missing mandatory element: {element}")
                compliance_score -= 0.1
        
        # Check prohibited elements
        for element in guidelines.prohibited_elements:
            if element.lower() in content_lower:
                analysis.consistency_issues.append(f"Contains prohibited element: {element}")
                analysis.prohibited_elements_found.append(element)
                compliance_score -= 0.2
        
        # Check pronoun usage
        if guidelines.preferred_pronouns:
            pronoun_found = any(pronoun.lower() in content_lower 
                              for pronoun in guidelines.preferred_pronouns)
            if not pronoun_found:
                analysis.consistency_issues.append("Preferred pronouns not used")
                compliance_score -= 0.05
        
        return max(compliance_score, 0.0)
    
    async def _generate_recommendations(self, 
                                      content: str, 
                                      guidelines: BrandVoiceGuidelines,
                                      analysis: VoiceAnalysis) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Tone recommendations
        if analysis.tone_consistency < 0.8:
            recommendations.append(
                f"Adjust tone to be more {guidelines.primary_tone.value}"
            )
        
        # Style recommendations
        if analysis.style_consistency < 0.7:
            missing_styles = set(guidelines.style_attributes) - set(analysis.detected_style)
            if missing_styles:
                recommendations.append(
                    f"Incorporate more {', '.join(s.value for s in missing_styles)} elements"
                )
        
        # Vocabulary recommendations
        if analysis.vocabulary_consistency < 0.8:
            vocab_prefs = guidelines.vocabulary_preferences
            if 'industry_terms' in vocab_prefs:
                recommendations.append(
                    f"Use more industry-specific terms: {', '.join(vocab_prefs['industry_terms'][:3])}"
                )
        
        # Readability recommendations
        if analysis.readability_score < 0.7:
            recommendations.append("Improve readability by using shorter sentences")
        
        # Engagement recommendations
        if analysis.engagement_score < 0.6:
            recommendations.append("Increase engagement with questions and direct address")
        
        return recommendations
    
    async def _save_analysis(self, analysis: VoiceAnalysis) -> None:
        """Save voice analysis to Neo4j."""
        query = """
        MERGE (a:VoiceAnalysis {analysis_id: $analysis_id})
        SET a += $properties
        RETURN a.analysis_id as analysis_id
        """
        
        properties = analysis.dict()
        properties.pop('analysis_id', None)
        
        await self.neo4j_client.execute_query(
            query,
            analysis_id=analysis.analysis_id,
            properties=properties
        )
    
    async def learn_from_content(self, 
                                content_items: List[ContentItem],
                                tenant_id: str) -> BrandVoiceLearning:
        """Learn brand voice patterns from content samples."""
        try:
            # Extract content samples
            content_samples = [item.content for item in content_items if item.content]
            
            if not content_samples:
                raise BrandVoiceError("No content samples provided for learning")
            
            # Create learning data
            learning = BrandVoiceLearning(
                tenant_id=tenant_id,
                learning_period_start=min(item.created_at for item in content_items),
                learning_period_end=max(item.created_at for item in content_items),
                content_count=len(content_samples),
                content_samples=content_samples[:100]  # Limit samples
            )
            
            # Analyze common phrases
            all_text = ' '.join(content_samples)
            learning.common_phrases = await self._extract_common_phrases(all_text)
            
            # Vocabulary distribution
            learning.vocabulary_distribution = await self._analyze_vocabulary_distribution(all_text)
            
            # Tone distribution
            learning.tone_distribution = await self._analyze_tone_distribution(content_samples)
            
            # Voice patterns
            learning.voice_patterns = await self._extract_voice_patterns(content_samples)
            
            # Save learning data
            await self._save_learning_data(learning)
            
            logger.info(f"Learned brand voice patterns from {len(content_samples)} samples")
            return learning
            
        except Exception as e:
            logger.error(f"Failed to learn brand voice patterns: {e}")
            raise BrandVoiceError(f"Learning failed: {e}")
    
    async def _extract_common_phrases(self, text: str) -> List[Tuple[str, float]]:
        """Extract common phrases from text."""
        # Simple n-gram extraction
        words = text.lower().split()
        phrases = {}
        
        # Extract 2-4 word phrases
        for n in range(2, 5):
            for i in range(len(words) - n + 1):
                phrase = ' '.join(words[i:i+n])
                if len(phrase) > 10:  # Minimum phrase length
                    phrases[phrase] = phrases.get(phrase, 0) + 1
        
        # Sort by frequency and return top phrases
        sorted_phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)
        
        # Convert to relative frequencies
        total_phrases = sum(phrases.values())
        return [(phrase, count / total_phrases) for phrase, count in sorted_phrases[:50]]
    
    async def _analyze_vocabulary_distribution(self, text: str) -> Dict[str, int]:
        """Analyze vocabulary distribution."""
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Count word frequencies
        word_counts = {}
        for word in words:
            if len(word) > 3:  # Skip very short words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Return top words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_words[:100])
    
    async def _analyze_tone_distribution(self, content_samples: List[str]) -> Dict[str, float]:
        """Analyze tone distribution across content samples."""
        tone_counts = {}
        
        for content in content_samples:
            detected_tone = await self._detect_tone(content)
            tone_counts[detected_tone.value] = tone_counts.get(detected_tone.value, 0) + 1
        
        # Convert to percentages
        total_samples = len(content_samples)
        return {tone: count / total_samples for tone, count in tone_counts.items()}
    
    async def _extract_voice_patterns(self, content_samples: List[str]) -> Dict[str, float]:
        """Extract voice patterns from content samples."""
        patterns = {}
        
        for content in content_samples:
            # Sentence length patterns
            sentences = re.split(r'[.!?]+', content)
            avg_sentence_length = sum(len(s.split()) for s in sentences if s.strip()) / len(sentences)
            patterns['avg_sentence_length'] = patterns.get('avg_sentence_length', 0) + avg_sentence_length
            
            # Punctuation patterns
            exclamations = content.count('!')
            questions = content.count('?')
            patterns['exclamation_rate'] = patterns.get('exclamation_rate', 0) + exclamations / len(content)
            patterns['question_rate'] = patterns.get('question_rate', 0) + questions / len(content)
            
            # Formality patterns
            formality = await self._calculate_formality(content)
            patterns['formality'] = patterns.get('formality', 0) + formality
        
        # Average across samples
        sample_count = len(content_samples)
        return {pattern: value / sample_count for pattern, value in patterns.items()}
    
    async def _save_learning_data(self, learning: BrandVoiceLearning) -> None:
        """Save learning data to Neo4j."""
        query = """
        MERGE (l:BrandVoiceLearning {learning_id: $learning_id})
        SET l += $properties
        RETURN l.learning_id as learning_id
        """
        
        properties = learning.dict()
        properties.pop('learning_id', None)
        
        await self.neo4j_client.execute_query(
            query,
            learning_id=learning.learning_id,
            properties=properties
        )
    
    async def get_voice_insights(self, tenant_id: str) -> Dict[str, Any]:
        """Get voice insights and analytics."""
        try:
            # Get recent analyses
            query = """
            MATCH (a:VoiceAnalysis)
            WHERE a.tenant_id = $tenant_id
            AND a.analyzed_at > datetime() - duration('P30D')
            RETURN a
            ORDER BY a.analyzed_at DESC
            LIMIT 100
            """
            
            analyses = await self.neo4j_client.execute_query(query, tenant_id=tenant_id)
            
            if not analyses:
                return {"message": "No voice analyses found"}
            
            # Calculate insights
            consistency_scores = [a['a']['overall_consistency'] for a in analyses]
            tone_distribution = {}
            
            for analysis in analyses:
                tone = analysis['a']['detected_tone']
                tone_distribution[tone] = tone_distribution.get(tone, 0) + 1
            
            insights = {
                'total_analyses': len(analyses),
                'average_consistency': sum(consistency_scores) / len(consistency_scores),
                'consistency_trend': self._calculate_trend(consistency_scores),
                'tone_distribution': tone_distribution,
                'top_issues': self._get_top_issues(analyses),
                'recommendations': self._get_global_recommendations(analyses)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to get voice insights: {e}")
            raise BrandVoiceError(f"Failed to get insights: {e}")
    
    def _calculate_trend(self, scores: List[float]) -> str:
        """Calculate trend direction from scores."""
        if len(scores) < 2:
            return "stable"
        
        recent_avg = sum(scores[:10]) / min(10, len(scores))
        older_avg = sum(scores[-10:]) / min(10, len(scores))
        
        if recent_avg > older_avg * 1.05:
            return "improving"
        elif recent_avg < older_avg * 0.95:
            return "declining"
        else:
            return "stable"
    
    def _get_top_issues(self, analyses: List[Dict]) -> List[str]:
        """Get top consistency issues."""
        issue_counts = {}
        
        for analysis in analyses:
            issues = analysis['a'].get('consistency_issues', [])
            for issue in issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        # Return top 5 issues
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        return [issue for issue, count in sorted_issues[:5]]
    
    def _get_global_recommendations(self, analyses: List[Dict]) -> List[str]:
        """Get global recommendations based on all analyses."""
        recommendations = []
        
        # Analyze consistency patterns
        consistency_scores = [a['a']['overall_consistency'] for a in analyses]
        avg_consistency = sum(consistency_scores) / len(consistency_scores)
        
        if avg_consistency < 0.8:
            recommendations.append("Focus on improving overall brand voice consistency")
        
        # Analyze tone consistency
        tone_consistency_scores = [a['a']['tone_consistency'] for a in analyses]
        avg_tone_consistency = sum(tone_consistency_scores) / len(tone_consistency_scores)
        
        if avg_tone_consistency < 0.7:
            recommendations.append("Work on maintaining consistent tone across content")
        
        # Analyze vocabulary consistency
        vocab_consistency_scores = [a['a']['vocabulary_consistency'] for a in analyses]
        avg_vocab_consistency = sum(vocab_consistency_scores) / len(vocab_consistency_scores)
        
        if avg_vocab_consistency < 0.7:
            recommendations.append("Improve vocabulary consistency and use more brand-specific terms")
        
        return recommendations


# =============================================================================
# Request/Response Models
# =============================================================================

class GuidelinesRequest(BaseModel):
    """Request model for brand voice guidelines."""
    
    primary_tone: ToneType = Field(..., description="Primary brand tone")
    secondary_tones: List[ToneType] = Field(default_factory=list, description="Secondary tones")
    style_attributes: List[StyleAttribute] = Field(default_factory=list, description="Style attributes")
    
    # Language preferences
    preferred_pronouns: List[str] = Field(default_factory=list, description="Preferred pronouns")
    voice_person: str = Field("second", description="Voice person")
    formality_level: float = Field(0.5, ge=0.0, le=1.0, description="Formality level")
    
    # Brand elements
    brand_values: List[str] = Field(default_factory=list, description="Brand values")
    brand_personality: List[str] = Field(default_factory=list, description="Brand personality")
    target_audience: str = Field(..., description="Target audience")
    
    # Vocabulary
    vocabulary_preferences: Dict[str, List[str]] = Field(default_factory=dict, description="Vocabulary preferences")
    mandatory_elements: List[str] = Field(default_factory=list, description="Mandatory elements")
    prohibited_elements: List[str] = Field(default_factory=list, description="Prohibited elements")


class VoiceAnalysisRequest(BaseModel):
    """Request model for voice analysis."""
    
    content: str = Field(..., description="Content to analyze")
    content_type: str = Field("article", description="Content type")
    
    # Optional context
    target_keywords: Optional[List[str]] = Field(None, description="Target keywords")
    target_audience: Optional[str] = Field(None, description="Target audience")


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Mock Neo4j client for testing
        from unittest.mock import Mock
        neo4j_client = Mock()
        
        # Create brand voice configuration
        brand_voice = BrandVoiceConfiguration(neo4j_client)
        
        # Create sample guidelines
        guidelines = BrandVoiceGuidelines(
            primary_tone=ToneType.PROFESSIONAL,
            secondary_tones=[ToneType.FRIENDLY],
            style_attributes=[StyleAttribute.INFORMATIVE, StyleAttribute.ENGAGING],
            preferred_pronouns=["you", "we"],
            target_audience="Content marketers",
            tenant_id="test-tenant"
        )
        
        print(f"Created guidelines: {guidelines.primary_tone}")
        print(f"Style attributes: {guidelines.style_attributes}")
        
        # Test voice analysis
        sample_content = """
        Welcome to our comprehensive guide on SEO content optimization. 
        In this article, we'll explore advanced techniques that will help you 
        improve your search rankings and drive more organic traffic to your website.
        """
        
        # This would normally require actual Neo4j connection
        # analysis = await brand_voice.analyze_content_voice(
        #     sample_content, "test-content", "article", "test-tenant"
        # )
        
        print("Brand voice configuration example completed")
    
    # Run example
    # asyncio.run(main())
    print("Brand voice configuration module loaded successfully")