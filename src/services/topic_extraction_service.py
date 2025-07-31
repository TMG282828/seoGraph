"""
AI-Powered Topic Extraction Service.

Uses JsonAnalysisAgent to dynamically extract meaningful topics from brief content,
user messages, and content context. Eliminates reliance on generic metadata titles
through intelligent AI-driven topic analysis.
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from ..agents.json_analysis.agent import JsonAnalysisAgent
from .brief_structure_parser import brief_structure_parser

# Try to import BeautifulSoup for HTML parsing
try:
    from bs4 import BeautifulSoup
    HTML_PARSING_AVAILABLE = True
except ImportError:
    HTML_PARSING_AVAILABLE = False

logger = logging.getLogger(__name__)


class TopicExtractionService:
    """
    AI-powered service for extracting meaningful content topics.
    
    Uses JsonAnalysisAgent to analyze brief content, user messages, and context
    to determine the most relevant topic for content generation. All topic
    decisions are made through AI analysis, not programmatic logic.
    """
    
    def __init__(self):
        """Initialize the topic extraction service."""
        self.json_agent = JsonAnalysisAgent()
        logger.info(f"TopicExtractionService initialized with JsonAnalysisAgent (HTML parsing: {HTML_PARSING_AVAILABLE})")
    
    async def extract_topic(self, 
                          brief_content: Optional[str] = None,
                          user_message: Optional[str] = None,
                          brief_summary: Optional[Dict[str, Any]] = None,
                          content_goals: Optional[Dict[str, Any]] = None) -> str:
        """
        Extract the most relevant topic using AI analysis.
        
        Args:
            brief_content: Raw brief content to analyze
            user_message: User's original message/request
            brief_summary: Brief metadata (may contain generic titles)
            content_goals: Content objectives and context
            
        Returns:
            str: AI-extracted topic optimized for content generation
        """
        try:
            logger.info("🔍 Starting AI-powered topic extraction")
            
            # DEBUG: Log all input data to understand what AI receives
            logger.info(f"📋 DEBUG - Brief content length: {len(brief_content) if brief_content else 0}")
            logger.info(f"📋 DEBUG - Brief content preview: {brief_content[:200] if brief_content else 'None'}...")
            logger.info(f"💬 DEBUG - User message: {user_message}")
            logger.info(f"📊 DEBUG - Brief summary: {brief_summary}")
            logger.info(f"🎯 DEBUG - Content goals: {content_goals}")
            
            # First try structured brief parsing for titles
            if brief_content:
                parsed_brief = brief_structure_parser.parse_brief(brief_content)
                structured_title = brief_structure_parser.get_content_title(parsed_brief)
                if structured_title:
                    logger.info(f"🏗️ Found structured title: '{structured_title}'")
                    return structured_title
            
            # Fallback to manual title extraction
            extracted_title = self._extract_title_from_content(brief_content)
            if extracted_title:
                logger.info(f"🏷️ Found explicit title: '{extracted_title}'")
                return extracted_title
            
            # Prepare content for AI analysis
            analysis_content = self._prepare_analysis_content(
                brief_content, user_message, brief_summary, content_goals
            )
            
            logger.info(f"🔄 DEBUG - Formatted analysis content length: {len(analysis_content)}")
            logger.info(f"🔄 DEBUG - Formatted analysis content preview: {analysis_content[:300]}...")
            
            if not analysis_content.strip():
                error_msg = "No content available for topic extraction - brief_content, user_message, and content_goals all empty"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            # Use AI to extract topic from content
            topic = await self._extract_topic_with_ai(analysis_content, {
                "brief_content": brief_content,
                "user_message": user_message,
                "brief_summary": brief_summary,
                "content_goals": content_goals
            })
            
            # AI quality validation
            validated_topic = await self._validate_topic_quality(topic, analysis_content)
            
            logger.info(f"✅ AI extracted topic: '{validated_topic}'")
            return validated_topic
            
        except Exception as e:
            logger.error(f"Topic extraction failed: {e}")
            # Re-raise the error for transparency instead of masking it
            raise Exception(f"AI topic extraction failed: {str(e)}. Please check brief content format or try again.")
    
    def _extract_title_from_content(self, brief_content: Optional[str]) -> Optional[str]:
        """Extract explicit titles from content using HTML parsing and text patterns."""
        if not brief_content or not brief_content.strip():
            return None
        
        try:
            # Try HTML parsing first if available
            if HTML_PARSING_AVAILABLE and ('<' in brief_content and '>' in brief_content):
                logger.info("🔍 Attempting HTML title extraction")
                soup = BeautifulSoup(brief_content, 'html.parser')
                
                # Look for title tag
                title_tag = soup.find('title')
                if title_tag and title_tag.string and title_tag.string.strip():
                    title = title_tag.string.strip()
                    logger.info(f"📄 Found HTML title tag: '{title}'")
                    if not self._is_generic_title(title):
                        return title
                
                # Look for h1 tag
                h1_tag = soup.find('h1')
                if h1_tag and h1_tag.get_text() and h1_tag.get_text().strip():
                    title = h1_tag.get_text().strip()
                    logger.info(f"📄 Found H1 heading: '{title}'")
                    if not self._is_generic_title(title):
                        return title
                        
                # Look for h2 tag as fallback
                h2_tag = soup.find('h2')
                if h2_tag and h2_tag.get_text() and h2_tag.get_text().strip():
                    title = h2_tag.get_text().strip()
                    logger.info(f"📄 Found H2 heading: '{title}'")
                    if not self._is_generic_title(title):
                        return title
            
            # Try text-based title patterns
            logger.info("🔍 Attempting text pattern title extraction")
            lines = brief_content.strip().split('\n')
            
            for line in lines[:10]:  # Check first 10 lines
                line = line.strip()
                if not line:
                    continue
                    
                # Look for "Title:" pattern and specific brief title patterns
                title_patterns = [
                    r'^(?:title|subject|topic):\s*(.+)$',
                    r'^title\s*/\s*h1:\s*(.+)$',  # "Title / H1: content"
                    r'^title\s*/\s*h1\s*(.+)$',   # "Title / H1 content" 
                    r'^h1:\s*(.+)$',              # "H1: content"
                ]
                
                for pattern in title_patterns:
                    title_match = re.match(pattern, line, re.IGNORECASE)
                    if title_match:
                        title = title_match.group(1).strip()
                        logger.info(f"📄 Found title pattern '{pattern}': '{title}'")
                        if not self._is_generic_title(title):
                            return title
                
                # Look for lines that could be titles (not too long, not questions)
                if (20 <= len(line) <= 100 and 
                    not line.endswith('?') and 
                    not line.startswith('#') and
                    ':' not in line):
                    logger.info(f"📄 Potential title line: '{line}'")
                    if not self._is_generic_title(line):
                        return line
                        
        except Exception as e:
            logger.warning(f"Title extraction failed: {e}")
        
        return None
    
    def _is_generic_title(self, title: str) -> bool:
        """Check if a title is generic and should be ignored."""
        if not title or not title.strip():
            return True
            
        title_lower = title.lower().strip()
        
        # Generic patterns to ignore
        generic_patterns = [
            r'manual brief.*\d{1,2}/\d{1,2}/\d{4}',
            r'content.*analysis',
            r'brief.*\d{4}',
            r'document.*\d',
            r'untitled.*',
            r'new.*document',
            r'draft.*',
            r'promote.*and.*upsell',  # Ignore objective-style titles
            r'objectives?\s*$',       # Ignore "Objectives" section headers
            r'^call to action\s*$',   # Ignore "Call to Action" headers
            r'^target audience\s*$',  # Ignore "Target Audience" headers
        ]
        
        for pattern in generic_patterns:
            if re.search(pattern, title_lower):
                logger.info(f"🚫 Ignoring generic title pattern: '{title}'")
                return True
        
        return False
    
    def _prepare_analysis_content(self, 
                                brief_content: Optional[str],
                                user_message: Optional[str],
                                brief_summary: Optional[Dict[str, Any]],
                                content_goals: Optional[Dict[str, Any]]) -> str:
        """Prepare content for AI analysis by combining available sources."""
        content_parts = []
        
        if brief_content and brief_content.strip():
            content_parts.append(f"BRIEF CONTENT:\n{brief_content.strip()}")
        
        if user_message and user_message.strip():
            content_parts.append(f"USER REQUEST:\n{user_message.strip()}")
        
        if content_goals and isinstance(content_goals, dict):
            goals_text = self._format_content_goals(content_goals)
            if goals_text:
                content_parts.append(f"CONTENT OBJECTIVES:\n{goals_text}")
        
        if brief_summary and isinstance(brief_summary, dict):
            summary_text = self._format_brief_summary(brief_summary)
            if summary_text:
                content_parts.append(f"BRIEF SUMMARY:\n{summary_text}")
        
        return "\n\n".join(content_parts)
    
    def _format_content_goals(self, content_goals: Dict[str, Any]) -> str:
        """Format content goals for AI analysis."""
        formatted_parts = []
        
        for key, value in content_goals.items():
            if value and str(value).strip():
                formatted_parts.append(f"- {key.replace('_', ' ').title()}: {value}")
        
        return "\n".join(formatted_parts)
    
    def _format_brief_summary(self, brief_summary: Dict[str, Any]) -> str:
        """Format brief summary for AI analysis, excluding generic titles."""
        formatted_parts = []
        
        for key, value in brief_summary.items():
            if key == "title":
                # Let AI decide if title is meaningful
                continue
            elif value and str(value).strip():
                formatted_parts.append(f"- {key.replace('_', ' ').title()}: {value}")
        
        return "\n".join(formatted_parts)
    
    async def _extract_topic_with_ai(self, content: str, context: Dict[str, Any]) -> str:
        """Use JsonAnalysisAgent to extract topic from content."""
        try:
            system_prompt = """You are a title and topic extraction specialist. Your job is to find the ACTUAL TITLE or TOPIC of the content for an article or blog post.

PRIORITY ORDER:
1. Look for "Title / H1:", "H1:", or "Title:" fields - these are the actual article titles
2. Find content titles that would be used as article headlines  
3. Identify main themes about the subject matter
4. NEVER use business objectives, goals, or promotional descriptions as titles

CRITICAL DISTINCTIONS:
- ARTICLE TITLE: "July 2025 Health Trends: What's Shaping Wellness This Summer" ✅ 
- BUSINESS OBJECTIVE: "Promote trending health tips and upsell products" ❌
- SECTION HEADER: "Objectives", "Call to Action", "Target Audience" ❌

Focus on CONTENT TITLES that would appear as H1 headings, not business goals or objectives.

Return your extraction as JSON:
{
    "extracted_topic": "the actual article title or content topic",
    "confidence_score": 0.95,
    "topic_source": "title_field|heading|content_theme|inferred",
    "reasoning": "why this is the content title (not business objective)"
}"""

            user_prompt = f"""Extract the article title from this content brief:

{content}

Look for:
1. "Title / H1:" field with the actual article title
2. Content headlines that would be H1 headings
3. Main topic themes (NOT business objectives)

IGNORE: Objectives, goals, promotional descriptions, section headers like "Call to Action"
FIND: The actual article title or content topic that readers would see as the headline."""

            # DEBUG: Log what we're sending to AI
            logger.info(f"🤖 DEBUG - Sending to AI - Content length: {len(content)}")
            logger.info(f"🤖 DEBUG - AI prompt preview: {user_prompt[:150]}...")
            
            result = await self.json_agent.analyze({
                "analysis_type": "topic_extraction",
                "content": content,
                "schema": {
                    "extracted_topic": "string",
                    "confidence_score": "number",
                    "topic_source": "string", 
                    "reasoning": "string"
                },
                "prompt": f"{system_prompt}\n\n{user_prompt}",
                "max_tokens": 1000
            })
            
            # DEBUG: Log AI response
            logger.info(f"🤖 DEBUG - AI result success: {result.success}")
            logger.info(f"🤖 DEBUG - AI result data: {result.data}")
            logger.info(f"🤖 DEBUG - AI error: {result.error_message if not result.success else 'None'}")
            
            if result.success and result.data:
                topic = result.data.get("extracted_topic", "").strip()
                confidence = result.data.get("confidence_score", 0)
                source = result.data.get("topic_source", "unknown")
                reasoning = result.data.get("reasoning", "")
                
                logger.info(f"🎯 AI topic extraction: '{topic}' (confidence: {confidence:.2f}, source: {source})")
                logger.info(f"🧠 AI reasoning: {reasoning}")
                
                if topic and confidence > 0.5:
                    return topic
                else:
                    raise Exception(f"AI extracted low confidence topic: '{topic}' (confidence: {confidence:.2f})")
                    
            else:
                raise Exception(f"AI topic extraction failed: {result.error_message or 'No data returned'}")
            
        except Exception as e:
            logger.error(f"AI topic extraction failed: {e}")
            # Try fallback AI content analysis instead of hardcoded response
            return await self._analyze_content_for_topic(content)
    
    async def _analyze_content_for_topic(self, content: str) -> str:
        """Fallback AI analysis to infer topic from content."""
        try:
            system_prompt = """You are a topic inference specialist. Read the content and determine what it is about - the main subject, theme, or topic.

Your job is to identify WHAT this content is about, not describe the process of analysis.

Focus on:
- The main subject matter
- Key themes and topics discussed
- What this content would be titled
- The primary focus area

Return only the topic as a JSON object:
{
    "topic": "the main topic/subject this content is about"
}"""

            user_prompt = f"""What is this content about? What is the main topic/subject?

{content[:1000]}...

Identify the primary topic or theme - what would you title this content?"""

            result = await self.json_agent.analyze({
                "analysis_type": "content_topic_inference",
                "content": content,
                "schema": {
                    "topic": "string"
                },
                "prompt": f"{system_prompt}\n\n{user_prompt}",
                "max_tokens": 500
            })
            
            if result.success and result.data:
                topic = result.data.get("topic", "").strip()
                if topic:
                    logger.info(f"🔍 AI inferred topic: '{topic}'")
                    return topic
                else:
                    raise Exception("AI returned empty topic from content analysis")
                    
        except Exception as e:
            logger.error(f"AI content analysis failed: {e}")
            raise Exception(f"AI content analysis failed: {str(e)}. Unable to extract topic from content.")
    
    async def _validate_topic_quality(self, topic: str, content: str) -> str:
        """Use AI to validate and improve topic quality."""
        try:
            system_prompt = """You are a topic quality validator. Evaluate the provided topic and determine if it's suitable for content generation.

Check if the topic is:
1. Specific and clear
2. Relevant to the content
3. Not generic or vague
4. Suitable for SEO and content creation

If the topic needs improvement, suggest a better alternative based on the content.

Return your evaluation as JSON:
{
    "is_quality_topic": true|false,
    "improved_topic": "better topic if needed",
    "quality_score": 0.85,
    "improvement_reason": "why improvement was needed"
}"""

            user_prompt = f"""Evaluate this topic for quality and relevance:

TOPIC: "{topic}"

CONTENT CONTEXT:
{content[:500]}...

Is this a good topic for content generation? If not, suggest an improvement."""

            result = await self.json_agent.analyze({
                "analysis_type": "topic_quality_validation",
                "content": f"Topic: {topic}\n\nContent: {content}",
                "schema": {
                    "is_quality_topic": "boolean",
                    "improved_topic": "string",
                    "quality_score": "number",
                    "improvement_reason": "string"
                },
                "prompt": f"{system_prompt}\n\n{user_prompt}",
                "max_tokens": 800
            })
            
            if result.success and result.data:
                is_quality = result.data.get("is_quality_topic", True)
                improved_topic = result.data.get("improved_topic", "").strip()
                quality_score = result.data.get("quality_score", 0.8)
                reason = result.data.get("improvement_reason", "")
                
                if not is_quality and improved_topic:
                    logger.info(f"🔧 AI improved topic: '{topic}' → '{improved_topic}' (reason: {reason})")
                    return improved_topic
                elif quality_score < 0.6 and improved_topic:
                    logger.info(f"🔧 AI enhanced topic: '{topic}' → '{improved_topic}' (score: {quality_score:.2f})")
                    return improved_topic
                else:
                    logger.info(f"✅ Topic validated: '{topic}' (score: {quality_score:.2f})")
                    return topic
                    
        except Exception as e:
            logger.error(f"Topic quality validation failed: {e}")
        
        return topic
    
    async def _generate_emergency_topic(self, user_message: Optional[str], brief_content: Optional[str]) -> str:
        """Generate topic using AI when extraction fails."""
        try:
            available_content = ""
            if user_message:
                available_content += f"User Request: {user_message}\n"
            if brief_content:
                available_content += f"Brief Content: {brief_content[:200]}...\n"
            
            if not available_content.strip():
                raise Exception("No content available for emergency topic generation - both user_message and brief_content are empty")
            
            system_prompt = """You are an emergency topic generator. Create a meaningful topic from whatever content is available.

Generate a clear, specific topic that would be suitable for content creation, even from limited information.

Return as JSON:
{
    "emergency_topic": "generated topic"
}"""

            result = await self.json_agent.analyze({
                "analysis_type": "emergency_topic_generation",
                "content": available_content,
                "schema": {
                    "emergency_topic": "string"
                },
                "prompt": f"{system_prompt}\n\nAvailable content:\n{available_content}",
                "max_tokens": 300
            })
            
            if result.success and result.data:
                emergency_topic = result.data.get("emergency_topic", "").strip()
                if emergency_topic:
                    logger.info(f"🚨 AI generated emergency topic: '{emergency_topic}'")
                    return emergency_topic
                else:
                    raise Exception("AI emergency topic generation returned empty result")
                    
        except Exception as e:
            logger.error(f"Emergency topic generation failed: {e}")
            raise Exception(f"All AI topic extraction methods failed: {str(e)}. Cannot generate topic without valid content.")


# Global instance for use across the application
topic_extraction_service = TopicExtractionService()