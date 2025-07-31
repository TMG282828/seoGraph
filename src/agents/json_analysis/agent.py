"""
JSON Analysis Agent for PRP Workflow.

Specialized agent designed specifically for structured JSON responses 
in workflow analysis phases. This is a lightweight agent focused solely
on JSON data extraction, not inheriting from BaseAgent.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
import openai
from openai import AsyncOpenAI

from config.settings import get_settings

logger = logging.getLogger(__name__)


class JsonAnalysisRequest(BaseModel):
    """Request model for JSON analysis tasks."""
    analysis_type: str = Field(description="Type of analysis (brief_analysis, content_planning, etc.)")
    content: str = Field(description="Content to analyze")
    json_schema: Dict[str, Any] = Field(description="Expected JSON schema structure", alias="schema")
    prompt: str = Field(description="Analysis prompt/instructions")
    max_tokens: Optional[int] = Field(default=2000, description="Maximum tokens in response")


class JsonAnalysisResult(BaseModel):
    """Result model for JSON analysis."""
    success: bool
    data: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time_ms: float = 0
    tokens_used: int = 0


class JsonAnalysisAgent:
    """
    Specialized agent for structured JSON responses in PRP workflow analysis.
    
    This agent is optimized for returning valid JSON structures rather than
    readable content, making it ideal for workflow coordination tasks.
    """
    
    def __init__(self):
        """Initialize the JSON Analysis Agent."""
        self.agent_name = "json_analysis"
        self.agent_version = "1.0.0"
        
        # Initialize OpenAI client directly for better JSON control
        settings = get_settings()
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        
        logger.info("JSON Analysis Agent initialized for structured responses")
    
    async def execute(self, task_data: Dict[str, Any], context: Optional[Any] = None) -> JsonAnalysisResult:
        """
        Execute JSON analysis task and return structured data.
        
        Args:
            task_data: Task configuration including analysis type and content
            context: Optional execution context (unused in this lightweight agent)
            
        Returns:
            JsonAnalysisResult with parsed JSON data
        """
        start_time = datetime.now()
        
        try:
            # Validate required fields
            required_fields = ['analysis_type', 'content', 'prompt']
            for field in required_fields:
                if field not in task_data:
                    raise ValueError(f"Missing required field: {field}")
            
            analysis_type = task_data['analysis_type']
            content = task_data['content']
            prompt = task_data['prompt']
            schema = task_data.get('schema', {})
            max_tokens = task_data.get('max_tokens', 2000)
            
            logger.info(f"🔍 Starting {analysis_type} analysis for {len(content)} characters")
            
            # Create system prompt optimized for JSON responses
            system_prompt = self._create_json_system_prompt(analysis_type)
            
            # Create user prompt with explicit JSON-only instructions
            user_prompt = f"""{prompt}

CRITICAL REQUIREMENTS:
1. Respond ONLY with valid JSON - no markdown, no explanations, no other text
2. The JSON must be properly formatted and parseable
3. Include all required fields from the expected schema
4. Do not wrap JSON in code blocks or add any prefixes/suffixes

Expected JSON structure reference:
{json.dumps(schema, indent=2) if schema else "Follow the structure specified in the prompt above"}

Content to analyze:
{content}

JSON Response:"""
            
            # Make OpenAI API call with JSON mode enforcement
            response = await self._make_json_api_call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens
            )
            
            # Parse and validate JSON response
            json_result = await self._parse_and_validate_json(response, schema)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.info(f"✅ {analysis_type} analysis completed in {execution_time:.0f}ms")
            
            return JsonAnalysisResult(
                success=True,
                data=json_result,
                execution_time_ms=execution_time,
                tokens_used=max_tokens
            )
            
        except Exception as e:
            logger.error(f"❌ JSON analysis failed for {task_data.get('analysis_type', 'unknown')}: {e}")
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return JsonAnalysisResult(
                success=False,
                data={},
                execution_time_ms=execution_time,
                error_message=f"JSON analysis failed: {str(e)}",
                tokens_used=0
            )
    
    def _create_json_system_prompt(self, analysis_type: str) -> str:
        """Create system prompt optimized for JSON responses."""
        return f"""You are a specialized JSON API that analyzes content and returns structured data.

ROLE: Content Analysis API for {analysis_type}
OUTPUT FORMAT: Valid JSON only - no markdown, no explanations, no other text
RESPONSE STYLE: Structured data extraction, not content generation

CRITICAL RULES:
1. ONLY return valid JSON - nothing else
2. JSON must be properly formatted and parseable  
3. Include all required fields specified in the user prompt
4. Use appropriate data types (strings, numbers, arrays, objects)
5. Ensure all string values are properly escaped
6. Do not add markdown formatting, code blocks, or explanations

You are NOT a content generation AI. You are a data extraction and analysis API that returns structured JSON responses for workflow coordination."""
    
    async def _make_json_api_call(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """Make OpenAI API call optimized for JSON responses."""
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Use GPT-4o-mini which supports JSON mode
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.1,  # Low temperature for consistent structured output
                response_format={"type": "json_object"},  # Force JSON mode
                timeout=60.0
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise Exception(f"Failed to get JSON response from AI: {str(e)}")
    
    async def _parse_and_validate_json(self, response: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate JSON response."""
        try:
            # Parse JSON
            json_data = json.loads(response)
            
            # Basic validation - ensure it's a dictionary
            if not isinstance(json_data, dict):
                raise ValueError("Response is not a JSON object")
            
            # Schema validation if provided
            if schema:
                # Basic schema validation - check required fields
                required_fields = schema.get('required', [])
                for field in required_fields:
                    if field not in json_data:
                        logger.warning(f"Missing required field in JSON response: {field}")
            
            logger.info(f"✅ JSON parsed successfully with {len(json_data)} fields")
            return json_data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.error(f"Raw response: {response[:500]}...")
            raise Exception(f"Invalid JSON format in AI response: {str(e)}")
        except Exception as e:
            logger.error(f"JSON validation failed: {e}")
            raise Exception(f"JSON validation error: {str(e)}")
    
    async def analyze_brief(self, brief_content: str, topic: str) -> Dict[str, Any]:
        """
        Analyze brief content and return structured JSON.
        
        Convenience method for brief analysis specifically.
        """
        schema = {
            "required": ["main_topic", "key_themes", "target_audience", "content_type", "estimated_complexity"],
            "properties": {
                "main_topic": {"type": "string"},
                "key_themes": {"type": "array", "items": {"type": "string"}},
                "target_audience": {"type": "string"},
                "content_type": {"type": "string"},
                "estimated_complexity": {"type": "string", "enum": ["low", "medium", "high"]},
                "word_count_brief": {"type": "number"},
                "analysis_summary": {"type": "string"}
            }
        }
        
        prompt = f"""Analyze this content brief and return structured JSON data:

BRIEF TOPIC: {topic}
BRIEF CONTENT: {brief_content}

Return JSON with this exact structure:
{{
    "main_topic": "the primary topic/subject",
    "key_themes": ["theme1", "theme2", "theme3"],
    "target_audience": "who this content is for",
    "content_type": "article/guide/blog_post/landing_page",
    "estimated_complexity": "low/medium/high",
    "word_count_brief": {len(brief_content.split())},
    "analysis_summary": "brief summary of what this content should accomplish"
}}"""
        
        task_data = {
            "analysis_type": "brief_analysis",
            "content": brief_content,
            "prompt": prompt,
            "schema": schema,
            "max_tokens": 1000
        }
        
        result = await self.execute(task_data)
        
        if result.success:
            return result.data
        else:
            raise Exception(result.error_message or "Brief analysis failed")