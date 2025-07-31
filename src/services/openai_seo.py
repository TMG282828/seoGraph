"""
OpenAI-powered SEO analysis and suggestions service.

This service provides AI-driven SEO recommendations, content analysis,
and smart tag suggestions using OpenAI's API.
"""

import os
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import openai
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)


class OpenAISEOService:
    """Service for OpenAI-powered SEO analysis and recommendations."""
    
    def __init__(self):
        """Initialize OpenAI service with API credentials."""
        self.client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.model = os.getenv("OPENAI_MODEL", "gpt-4")
        
    def analyze_content_seo(
        self, 
        content: str, 
        keywords: List[str] = None, 
        page_type: str = "blog_post",
        current_tags: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze content for SEO optimization and provide suggestions.
        
        Args:
            content (str): Content to analyze
            keywords (List[str], optional): Target keywords
            page_type (str): Type of page (blog_post, product, landing_page, etc.)
            current_tags (List[str], optional): Currently applied tags
            
        Returns:
            Dict[str, Any]: Comprehensive SEO analysis and suggestions
        """
        try:
            if not content or len(content.strip()) < 50:
                return {
                    "success": False,
                    "error": "Content too short for meaningful analysis (minimum 50 characters)"
                }
            
            # Analyze content structure and metrics
            analysis = self._analyze_content_structure(content, keywords or [])
            
            # Get AI-powered suggestions
            suggestions = self._get_ai_seo_suggestions(
                content, keywords or [], page_type, analysis
            )
            
            # Generate smart tags
            recommended_tags = self._generate_smart_tags(
                content, keywords or [], page_type, current_tags or []
            )
            
            # Get technical recommendations
            technical_recommendations = self._get_technical_recommendations(
                content, page_type, analysis
            )
            
            return {
                "success": True,
                "analysis": analysis,
                "suggestions": suggestions,
                "recommended_tags": recommended_tags,
                "technical_recommendations": technical_recommendations,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze content SEO: {e}")
            return {
                "success": False,
                "error": f"SEO analysis failed: {str(e)}"
            }
    
    def _analyze_content_structure(self, content: str, keywords: List[str]) -> Dict[str, Any]:
        """
        Analyze content structure and calculate SEO metrics.
        
        Args:
            content (str): Content to analyze
            keywords (List[str]): Target keywords
            
        Returns:
            Dict[str, Any]: Content analysis metrics
        """
        try:
            # Basic content metrics
            word_count = len(content.split())
            char_count = len(content)
            paragraph_count = len([p for p in content.split('\n\n') if p.strip()])
            
            # Readability score (simplified Flesch Reading Ease approximation)
            sentences = len(re.findall(r'[.!?]+', content))
            avg_sentence_length = word_count / max(sentences, 1)
            syllables = self._count_syllables(content)
            avg_syllables_per_word = syllables / max(word_count, 1)
            
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            readability_score = max(0, min(100, flesch_score))
            
            # Keyword analysis
            primary_keyword = keywords[0] if keywords else ""
            keyword_density = {}
            
            if keywords:
                content_lower = content.lower()
                for keyword in keywords:
                    keyword_lower = keyword.lower()
                    occurrences = content_lower.count(keyword_lower)
                    density = (occurrences / max(word_count, 1)) * 100
                    keyword_density[keyword] = {
                        "occurrences": occurrences,
                        "density": round(density, 2)
                    }
            
            # Content quality score (0-100)
            content_score = self._calculate_content_score(
                word_count, readability_score, keyword_density, paragraph_count
            )
            
            return {
                "content_length": word_count,
                "character_count": char_count,
                "paragraph_count": paragraph_count,
                "sentence_count": sentences,
                "readability_score": round(readability_score, 1),
                "content_score": content_score,
                "keyword_density": {
                    "primary_keyword": keyword_density.get(primary_keyword, {}).get("density", 0),
                    "all_keywords": keyword_density
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze content structure: {e}")
            return {"content_length": 0, "content_score": 0, "readability_score": 0}
    
    def _get_ai_seo_suggestions(
        self, 
        content: str, 
        keywords: List[str], 
        page_type: str, 
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Get AI-powered SEO suggestions for content optimization.
        
        Args:
            content (str): Content to analyze
            keywords (List[str]): Target keywords
            page_type (str): Type of page
            analysis (Dict[str, Any]): Content analysis results
            
        Returns:
            List[Dict[str, Any]]: SEO optimization suggestions
        """
        try:
            # Prepare context for AI
            content_preview = content[:1000] + "..." if len(content) > 1000 else content
            keywords_str = ", ".join(keywords) if keywords else "No specific keywords provided"
            
            prompt = f"""
            As an SEO expert, analyze this {page_type} content and provide specific optimization suggestions.
            
            Content Preview: {content_preview}
            Target Keywords: {keywords_str}
            Current Metrics:
            - Word count: {analysis.get('content_length', 0)}
            - Readability score: {analysis.get('readability_score', 0)}/100
            - Content score: {analysis.get('content_score', 0)}/100
            
            Provide 3-5 specific, actionable SEO suggestions. For each suggestion, include:
            1. Type (title_optimization, keyword_usage, content_structure, meta_description, etc.)
            2. Priority (high, medium, low)
            3. Specific suggestion text
            4. Expected impact
            5. Example implementation (if applicable)
            
            Focus on practical improvements that will enhance search rankings and user experience.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert SEO analyst providing actionable optimization recommendations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            suggestions_text = response.choices[0].message.content
            
            # Parse AI response into structured suggestions
            suggestions = self._parse_ai_suggestions(suggestions_text)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to get AI SEO suggestions: {e}")
            return self._get_fallback_suggestions(analysis, keywords)
    
    def _generate_smart_tags(
        self, 
        content: str, 
        keywords: List[str], 
        page_type: str, 
        current_tags: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Generate smart tags for content categorization and SEO.
        
        Args:
            content (str): Content to analyze
            keywords (List[str]): Target keywords
            page_type (str): Type of page
            current_tags (List[str]): Currently applied tags
            
        Returns:
            List[Dict[str, Any]]: Recommended tags with confidence scores
        """
        try:
            content_preview = content[:800] + "..." if len(content) > 800 else content
            keywords_str = ", ".join(keywords) if keywords else "No specific keywords"
            current_tags_str = ", ".join(current_tags) if current_tags else "None"
            
            prompt = f"""
            Analyze this {page_type} content and suggest relevant tags for SEO and content categorization.
            
            Content: {content_preview}
            Keywords: {keywords_str}
            Current Tags: {current_tags_str}
            
            Suggest 8-12 tags across these categories:
            - topic: Main subject areas and themes
            - keyword: SEO-focused keyword tags
            - audience: Target audience segments
            - intent: User search intent (informational, commercial, transactional)
            - technical: Technical aspects or features
            
            For each tag, provide:
            - tag: The tag text
            - category: One of the categories above
            - confidence: Float between 0.0-1.0 indicating relevance
            - reason: Brief explanation why this tag is relevant
            
            Avoid suggesting tags that are already in the current tags list.
            Focus on tags that will improve content discoverability and SEO performance.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert content strategist specializing in SEO tagging and content categorization."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.4
            )
            
            tags_text = response.choices[0].message.content
            
            # Parse AI response into structured tags
            tags = self._parse_ai_tags(tags_text, current_tags)
            
            return tags
            
        except Exception as e:
            logger.error(f"Failed to generate smart tags: {e}")
            return self._get_fallback_tags(keywords, page_type)
    
    def _get_technical_recommendations(
        self, 
        content: str, 
        page_type: str, 
        analysis: Dict[str, Any]
    ) -> List[str]:
        """
        Get technical SEO recommendations based on content analysis.
        
        Args:
            content (str): Content to analyze
            page_type (str): Type of page
            analysis (Dict[str, Any]): Content analysis results
            
        Returns:
            List[str]: Technical SEO recommendations
        """
        recommendations = []
        
        try:
            word_count = analysis.get('content_length', 0)
            readability = analysis.get('readability_score', 0)
            
            # Word count recommendations
            if page_type == "blog_post" and word_count < 300:
                recommendations.append("Increase content length to at least 300 words for better SEO performance")
            elif page_type == "product" and word_count < 150:
                recommendations.append("Add more product description content (minimum 150 words)")
            elif word_count > 3000:
                recommendations.append("Consider breaking long content into multiple pages or sections")
            
            # Readability recommendations
            if readability < 30:
                recommendations.append("Improve readability by using shorter sentences and simpler words")
            elif readability > 90:
                recommendations.append("Content may be too simple - consider adding more detailed explanations")
            
            # Structure recommendations
            if analysis.get('paragraph_count', 0) < 3 and word_count > 200:
                recommendations.append("Break content into more paragraphs for better readability")
            
            # Add page-type specific recommendations
            if page_type == "blog_post":
                recommendations.extend([
                    "Add internal links to related content",
                    "Include relevant images with alt text",
                    "Consider adding a table of contents for long articles"
                ])
            elif page_type == "product":
                recommendations.extend([
                    "Include product specifications and features",
                    "Add customer reviews and ratings",
                    "Optimize for local SEO if applicable"
                ])
            
            return recommendations[:6]  # Limit to 6 recommendations
            
        except Exception as e:
            logger.error(f"Failed to get technical recommendations: {e}")
            return ["Optimize page loading speed", "Ensure mobile responsiveness"]
    
    def _count_syllables(self, text: str) -> int:
        """Count syllables in text for readability calculation."""
        text = text.lower()
        vowels = "aeiouy"
        syllable_count = 0
        previous_was_vowel = False
        
        for char in text:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Adjust for silent e
        if text.endswith('e'):
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _calculate_content_score(
        self, 
        word_count: int, 
        readability: float, 
        keyword_density: Dict[str, Dict[str, Any]], 
        paragraph_count: int
    ) -> int:
        """Calculate overall content quality score (0-100)."""
        score = 0
        
        # Word count score (0-30 points)
        if word_count >= 300:
            score += min(30, word_count // 20)
        
        # Readability score (0-25 points)
        if 30 <= readability <= 70:
            score += 25
        elif readability > 70:
            score += 20
        else:
            score += 10
        
        # Keyword optimization (0-25 points)
        if keyword_density:
            primary_density = list(keyword_density.values())[0].get('density', 0)
            if 1 <= primary_density <= 3:
                score += 25
            elif 0.5 <= primary_density < 1 or 3 < primary_density <= 5:
                score += 15
            else:
                score += 5
        
        # Structure score (0-20 points)
        if paragraph_count >= 3:
            score += 20
        elif paragraph_count >= 2:
            score += 15
        else:
            score += 5
        
        return min(100, max(0, score))
    
    def _parse_ai_suggestions(self, suggestions_text: str) -> List[Dict[str, Any]]:
        """Parse AI response into structured suggestions."""
        suggestions = []
        
        try:
            # This is a simplified parser - in production, you might want more robust parsing
            lines = suggestions_text.split('\n')
            current_suggestion = {}
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Look for structured patterns in AI response
                if 'type:' in line.lower() or 'category:' in line.lower():
                    if current_suggestion:
                        suggestions.append(current_suggestion)
                    current_suggestion = {
                        'type': self._extract_value(line, ['type:', 'category:']),
                        'priority': 'medium',
                        'suggestion': '',
                        'impact': '',
                        'example': ''
                    }
                elif 'priority:' in line.lower():
                    current_suggestion['priority'] = self._extract_value(line, ['priority:']).lower()
                elif 'suggestion:' in line.lower() or 'recommendation:' in line.lower():
                    current_suggestion['suggestion'] = self._extract_value(line, ['suggestion:', 'recommendation:'])
                elif 'impact:' in line.lower():
                    current_suggestion['impact'] = self._extract_value(line, ['impact:'])
                elif 'example:' in line.lower():
                    current_suggestion['example'] = self._extract_value(line, ['example:'])
            
            if current_suggestion:
                suggestions.append(current_suggestion)
            
            # If parsing failed, create fallback suggestions
            if not suggestions:
                suggestions = [
                    {
                        "type": "content_optimization",
                        "priority": "high",
                        "suggestion": "Optimize content structure with clear headings and subheadings",
                        "impact": "Improved readability and SEO rankings",
                        "example": "Use H1 for main title, H2 for sections, H3 for subsections"
                    }
                ]
            
            return suggestions[:5]  # Limit to 5 suggestions
            
        except Exception as e:
            logger.error(f"Failed to parse AI suggestions: {e}")
            return self._get_fallback_suggestions({}, [])
    
    def _parse_ai_tags(self, tags_text: str, current_tags: List[str]) -> List[Dict[str, Any]]:
        """Parse AI response into structured tags."""
        tags = []
        
        try:
            lines = tags_text.split('\n')
            current_tag = {}
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if 'tag:' in line.lower():
                    if current_tag and current_tag.get('tag') not in current_tags:
                        tags.append(current_tag)
                    current_tag = {
                        'tag': self._extract_value(line, ['tag:']),
                        'category': 'topic',
                        'confidence': 0.7,
                        'reason': ''
                    }
                elif 'category:' in line.lower():
                    current_tag['category'] = self._extract_value(line, ['category:']).lower()
                elif 'confidence:' in line.lower():
                    try:
                        confidence_str = self._extract_value(line, ['confidence:'])
                        current_tag['confidence'] = float(confidence_str)
                    except:
                        current_tag['confidence'] = 0.7
                elif 'reason:' in line.lower():
                    current_tag['reason'] = self._extract_value(line, ['reason:'])
            
            if current_tag and current_tag.get('tag') not in current_tags:
                tags.append(current_tag)
            
            # If parsing failed, create fallback tags
            if not tags:
                tags = self._get_fallback_tags([], "blog_post")
            
            return tags[:12]  # Limit to 12 tags
            
        except Exception as e:
            logger.error(f"Failed to parse AI tags: {e}")
            return self._get_fallback_tags([], "blog_post")
    
    def _extract_value(self, line: str, prefixes: List[str]) -> str:
        """Extract value after specified prefixes."""
        for prefix in prefixes:
            if prefix in line.lower():
                return line.split(prefix, 1)[1].strip()
        return line.strip()
    
    def _get_fallback_suggestions(self, analysis: Dict[str, Any], keywords: List[str]) -> List[Dict[str, Any]]:
        """Get fallback suggestions when AI parsing fails."""
        return [
            {
                "type": "keyword_optimization",
                "priority": "high",
                "suggestion": "Optimize keyword density and placement throughout the content",
                "impact": "Better search engine rankings for target keywords",
                "example": "Include primary keyword in title, first paragraph, and naturally throughout content"
            },
            {
                "type": "content_structure",
                "priority": "medium",
                "suggestion": "Improve content structure with clear headings and bullet points",
                "impact": "Enhanced readability and user engagement",
                "example": "Use H2 tags for main sections and bullet points for lists"
            }
        ]
    
    def _get_fallback_tags(self, keywords: List[str], page_type: str) -> List[Dict[str, Any]]:
        """Get fallback tags when AI parsing fails."""
        base_tags = [
            {"tag": "seo-optimization", "category": "technical", "confidence": 0.8, "reason": "Content focused on SEO"},
            {"tag": "content-marketing", "category": "topic", "confidence": 0.7, "reason": "Marketing-related content"},
            {"tag": "informational", "category": "intent", "confidence": 0.6, "reason": "Educational content"},
        ]
        
        # Add keyword-based tags
        for keyword in keywords[:3]:
            base_tags.append({
                "tag": keyword.replace(" ", "-").lower(),
                "category": "keyword",
                "confidence": 0.9,
                "reason": f"Target keyword: {keyword}"
            })
        
        return base_tags

    async def generate_seo_analysis(self, prompt: str, context: str = "general") -> Dict[str, Any]:
        """Generate comprehensive SEO analysis using OpenAI."""
        try:
            # Check if OpenAI is configured
            if not self.client:
                logger.warning("OpenAI client not configured, using fallback analysis")
                return await self._generate_fallback_analysis(prompt, context)
            
            # Make OpenAI API call
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert SEO strategist and content analyst. Provide comprehensive, actionable SEO insights based on research data. Always return professional HTML content with proper structure."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=4000,
                temperature=0.7
            )
            
            analysis_content = response.choices[0].message.content.strip()
            
            return {
                "success": True,
                "analysis": analysis_content,
                "context": context,
                "generated_at": datetime.now().isoformat(),
                "model": "gpt-4o-mini"
            }
            
        except Exception as e:
            logger.error(f"OpenAI SEO analysis failed: {e}")
            return await self._generate_fallback_analysis(prompt, context)
    
    async def _generate_fallback_analysis(self, prompt: str, context: str) -> Dict[str, Any]:
        """Generate fallback analysis when OpenAI is unavailable."""
        
        # Extract key information from prompt
        keywords_match = re.search(r'Top Keywords: ([^-\n]*)', prompt)
        market_match = re.search(r'Market: ([^-\n]*)', prompt)
        
        keywords = keywords_match.group(1).strip() if keywords_match else "Not specified"
        market = market_match.group(1).strip() if market_match else "US-en"
        
        fallback_html = f"""
        <h1>SEO Research Analysis Report</h1>
        
        <div class="metric">
            <h2>Executive Summary</h2>
            <p>This analysis covers the {market} market with focus on keywords: {keywords}</p>
            <p>The research indicates several opportunities for content optimization and strategic positioning.</p>
        </div>
        
        <h2>Keyword Strategy Analysis</h2>
        <div class="opportunity">
            <p><strong>Primary Focus Keywords:</strong> {keywords}</p>
            <p><strong>Recommendation:</strong> Develop content clusters around these core terms to establish topical authority.</p>
            <p><strong>Priority Level:</strong> High - Begin implementation within 2 weeks</p>
        </div>
        
        <h2>Content Gap Opportunities</h2>
        <div class="opportunity">
            <p><strong>Identified Gaps:</strong> Comprehensive guides and tutorials in your target niche</p>
            <p><strong>Strategic Approach:</strong> Create in-depth, authoritative content that addresses user intent</p>
            <p><strong>Expected Impact:</strong> 25-40% improvement in organic visibility</p>
        </div>
        
        <h2>Action Plan</h2>
        <div class="metric">
            <h3>Phase 1 (Weeks 1-2):</h3>
            <ul>
                <li>Content audit of existing materials</li>
                <li>Keyword mapping and clustering</li>
                <li>Competitor content gap analysis</li>
            </ul>
            
            <h3>Phase 2 (Weeks 3-6):</h3>
            <ul>
                <li>Create priority content pieces</li>
                <li>Optimize existing high-potential pages</li>
                <li>Implement technical SEO improvements</li>
            </ul>
            
            <h3>Phase 3 (Weeks 7-12):</h3>
            <ul>
                <li>Scale content production</li>
                <li>Build topical authority</li>
                <li>Monitor and adjust strategy</li>
            </ul>
        </div>
        
        <h2>Success Metrics</h2>
        <div class="metric">
            <ul>
                <li><strong>Organic Traffic:</strong> Target 30% increase in 6 months</li>
                <li><strong>Keyword Rankings:</strong> Top 10 positions for primary keywords</li>
                <li><strong>Content Performance:</strong> 25% improvement in engagement metrics</li>
                <li><strong>Domain Authority:</strong> Steady monthly growth in authority scores</li>
            </ul>
        </div>
        """
        
        return {
            "success": True,
            "analysis": fallback_html,
            "context": context,
            "generated_at": datetime.now().isoformat(),
            "model": "fallback_template",
            "note": "Generated using fallback template - OpenAI service unavailable"
        }


# Initialize global service instance
openai_seo_service = OpenAISEOService()