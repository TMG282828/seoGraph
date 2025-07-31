"""
PRP Workflow Content Generator.

Handles the actual content generation phase of the PRP workflow,
using the ContentGenerationAgent to create the final content.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from .models import PRPWorkflowState
from ..brief_structure_parser import brief_structure_parser

logger = logging.getLogger(__name__)


class ContentGenerator:
    """Handles content generation for PRP workflow."""
    
    async def generate_content(self, workflow: PRPWorkflowState) -> Dict[str, Any]:
        """Generate the actual content using ContentGenerationAgent with structured brief data."""
        try:
            logger.info(f"🤖 Starting content generation for: {workflow.topic}")
            
            # Parse the brief content for structured data
            parsed_brief = None
            if workflow.brief_content:
                parsed_brief = brief_structure_parser.parse_brief(workflow.brief_content)
                logger.info(f"📋 Parsed brief: Title='{parsed_brief.title}', Keywords={len(parsed_brief.keywords)}, Headings={len(parsed_brief.heading_structure)}")
            
            # Import the content generation agent
            from ...agents.content_generation.agent import ContentGenerationAgent
            from ...agents.base_agent import AgentContext
            
            # Get content generation agent
            agent = ContentGenerationAgent()
            
            # Create agent context with enhanced brand voice
            enhanced_brand_voice = workflow.brand_voice.copy()
            if parsed_brief and parsed_brief.tone_of_voice:
                enhanced_brand_voice["tone"] = parsed_brief.tone_of_voice
            
            agent_context = AgentContext(
                organization_id="00000000-0000-0000-0000-000000000001",
                user_id="anonymous",
                brand_voice_config=enhanced_brand_voice,
                industry_context=workflow.content_goals.get('industry', 'health and wellness'),
                session_id=f"prp_{workflow.workflow_id}"
            )
            
            # Determine content specifications from parsed brief
            word_count_range = parsed_brief.word_count_range if parsed_brief else None
            content_length = self._determine_content_length(word_count_range)
            target_keywords = parsed_brief.keywords if parsed_brief and parsed_brief.keywords else [workflow.topic]
            target_audience = parsed_brief.target_audience if parsed_brief else workflow.content_goals.get("audience", "general")
            
            # Prepare enhanced content generation request with structured data
            task_data = {
                "content_type": "blog_post",
                "topic": workflow.topic,
                "target_keywords": target_keywords,
                "content_length": content_length,
                "writing_style": enhanced_brand_voice.get("tone", "helpful"),
                "target_audience": target_audience,
                "outline_only": False,
                "include_meta_tags": True,
                "include_internal_links": True,
                "reference_content": [workflow.brief_content] if workflow.brief_content else [],
                "use_knowledge_graph": True,
                "use_vector_search": True,
                "similarity_threshold": 0.7,
                "max_related_content": 5,
                "type": "content_generation",
                # Enhanced fields from structured brief
                "parsed_brief": self._convert_parsed_brief_to_dict(parsed_brief) if parsed_brief else {},
                "heading_structure": parsed_brief.heading_structure if parsed_brief else [],
                "meta_description": parsed_brief.meta_description if parsed_brief else None,
                "title_tag": parsed_brief.title_tag if parsed_brief else None,
                "competitor_articles": parsed_brief.competitor_articles if parsed_brief else [],
                "objectives": parsed_brief.objectives if parsed_brief else None,
                "call_to_action": parsed_brief.call_to_action if parsed_brief else None,
            }
            
            # Execute content generation
            logger.info(f"🔄 Sending content generation request to AI...")
            result = await agent.execute(task_data, agent_context)
            
            if result.success and result.result_data:
                generated_data = result.result_data
                logger.info(f"✅ Content generation completed: {generated_data.get('word_count', 0)} words")
                
                return {
                    "title": generated_data.get("title", f"Complete Guide to {workflow.topic}"),
                    "content": generated_data.get("content", "Generated content not available"),
                    "word_count": generated_data.get("word_count", 0),
                    "seo_score": generated_data.get("seo_score", 75),
                    "readability_score": generated_data.get("readability_score", 80),
                    "brand_voice_compliance": generated_data.get("brand_voice_compliance", 85),
                    "sections_generated": len(generated_data.get("content_outline", {}).get("sections", [])),
                    "internal_links": len(generated_data.get("internal_links", [])),
                    "meta_tags": generated_data.get("meta_tags", {}),
                    "knowledge_sources": generated_data.get("knowledge_sources", []),
                    "related_topics": generated_data.get("related_topics", []),
                    "improvement_suggestions": generated_data.get("improvement_suggestions", []),
                    "ai_powered": True,
                    "generation_method": "content_generation_agent",
                    "workflow_id": workflow.workflow_id
                }
            else:
                error_msg = result.error_message if result else "No result returned from content generation"
                logger.error(f"❌ Content generation failed: {error_msg}")
                raise Exception(f"Content generation failed: {error_msg}")
                
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            raise Exception(f"Content generation failed: {str(e)}. Cannot proceed without generated content.")
    
    def _determine_content_length(self, word_count_range: Optional[str]) -> str:
        """Determine content length specification from word count range."""
        if not word_count_range:
            return "medium"
        
        # Extract numbers from word count range
        import re
        numbers = re.findall(r'\d+', word_count_range)
        if not numbers:
            return "medium"
        
        # Take the first number as the target
        target_words = int(numbers[0])
        
        if target_words < 500:
            return "short"
        elif target_words > 1500:
            return "long" 
        else:
            return "medium"
    
    def _convert_parsed_brief_to_dict(self, parsed_brief) -> dict:
        """Convert ParsedBrief dataclass to dictionary."""
        from dataclasses import asdict
        return asdict(parsed_brief)