"""
PRP Workflow Phase Analyzers.

Contains AI-powered analysis methods for each phase of the PRP workflow:
- Brief Analysis
- Content Planning  
- Requirements Definition
- Process Definition
- Final Review
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

from .models import PRPWorkflowState, WorkflowPhase
from src.agents.json_analysis.agent import JsonAnalysisAgent
from src.services.langfuse_service import monitor_ai_agent

logger = logging.getLogger(__name__)


class BriefAnalyzer:
    """AI-powered brief analysis for PRP workflow."""
    
    def __init__(self):
        """Initialize BriefAnalyzer with JsonAnalysisAgent."""
        self.json_agent = JsonAnalysisAgent()
    
    @monitor_ai_agent("brief_analyzer", "brief_analysis")
    async def analyze_brief(self, brief_content: str, topic: str) -> Dict[str, Any]:
        """Analyze the brief using AI - NO hardcoded responses."""
        if not brief_content or brief_content.strip() == "":
            raise ValueError("Brief content is required for AI analysis. Cannot proceed without content.")
        
        try:
            logger.info(f"🤖 Starting AI-powered brief analysis for {len(brief_content.split())} words...")
            
            # Use JsonAnalysisAgent for structured response
            try:
                analysis_data = await self.json_agent.analyze_brief(brief_content, topic)
                
                # Add PRP-specific metadata
                analysis_data["brief_preview"] = brief_content[:200] + "..." if len(brief_content) > 200 else brief_content
                analysis_data["ai_powered"] = True
                analysis_data["analysis_method"] = "json_analysis_agent"
                
                logger.info(f"📊 AI Brief analysis complete - Topic: {analysis_data.get('main_topic')}, Themes: {analysis_data.get('key_themes')}")
                return analysis_data
                
            except Exception as json_error:
                # Log JSON parsing error to Langfuse
                from src.services.langfuse_service import langfuse_service
                langfuse_service.log_json_parsing_error(
                    agent_name="brief_analyzer",
                    raw_response=str(json_error),
                    error=f"JsonAnalysisAgent failed: {str(json_error)}",
                    organization_id="00000000-0000-0000-0000-000000000001"
                )
                raise Exception(f"AI returned invalid JSON format. Cannot proceed with analysis.")
                
        except asyncio.TimeoutError:
            logger.error("AI brief analysis timed out after 60 seconds")
            raise Exception("AI analysis timed out. Please try again.")
        except Exception as e:
            logger.error(f"AI brief analysis failed: {e}")
            raise Exception(f"AI brief analysis failed: {str(e)}. Cannot proceed without AI analysis.")


class ContentPlanner:
    """AI-powered content planning for PRP workflow."""
    
    def __init__(self):
        """Initialize ContentPlanner with JsonAnalysisAgent."""
        self.json_agent = JsonAnalysisAgent()
    
    @monitor_ai_agent("content_planner", "content_planning")
    async def create_content_plan(self, workflow: PRPWorkflowState) -> Dict[str, Any]:
        """Create comprehensive content planning strategy using AI - NO hardcoded responses."""
        try:
            logger.info(f"🤖 Starting AI-powered content planning for: {workflow.topic}")
            
            # Build context from workflow state
            context_info = f"""
TOPIC: {workflow.topic}
BRIEF CONTENT: {workflow.brief_content[:500]}...
CONTENT GOALS: {workflow.content_goals}
BRAND VOICE: {workflow.brand_voice}
CHECKIN FREQUENCY: {workflow.checkin_frequency}
            """.strip()
            
            # Prepare planning request
            planning_prompt = f"""Create a comprehensive content planning strategy based on this brief and requirements:

{context_info}

Provide a structured JSON response with exactly this format:
{{
    "content_strategy": {{
        "primary_goal": "the main objective for this content",
        "approach": "the strategic approach (e.g., comprehensive guide, how-to, analysis)",
        "tone": "the recommended tone based on brand voice",
        "target_seo_focus": "primary SEO strategy"
    }},
    "sections": [
        {{"title": "Section Title", "estimated_words": 300, "purpose": "what this section accomplishes"}},
        {{"title": "Another Section", "estimated_words": 400, "purpose": "purpose description"}}
    ],
    "seo_strategy": {{
        "primary_keywords": ["keyword1", "keyword2"],
        "secondary_keywords": ["keyword3", "keyword4"],
        "header_optimization": true,
        "internal_linking": true,
        "meta_focus": "meta description focus"
    }},
    "estimated_total_words": 2000,
    "complexity_assessment": "low/medium/high"
}}

Respond only with the JSON structure, no other text."""
            
            # Use JsonAnalysisAgent for structured response
            schema = {
                "required": ["content_strategy", "sections", "seo_strategy"],
                "properties": {
                    "content_strategy": {"type": "object"},
                    "sections": {"type": "array"},
                    "seo_strategy": {"type": "object"},
                    "estimated_total_words": {"type": "number"},
                    "complexity_assessment": {"type": "string"}
                }
            }
            
            task_data = {
                "analysis_type": "content_planning",
                "content": context_info,
                "prompt": planning_prompt,
                "schema": schema,
                "max_tokens": 2000
            }
            
            from src.agents.base_agent import AgentContext
            agent_context = AgentContext(
                organization_id="00000000-0000-0000-0000-000000000001",
                user_id="anonymous",
                brand_voice_config=workflow.brand_voice,
                session_id=f"content_plan_{workflow.workflow_id}"
            )
            
            # Execute AI planning with timeout
            logger.info(f"🔄 Sending content planning request to AI...")
            result = await asyncio.wait_for(
                self.json_agent.execute(task_data, agent_context),
                timeout=60.0
            )
            
            if not result.success:
                raise Exception(f"AI content planning failed: {result.error_message}")
            
            planning_data = result.data
            
            # Add metadata
            planning_data["ai_powered"] = True
            planning_data["planning_method"] = "json_analysis_agent"
            planning_data["workflow_id"] = workflow.workflow_id
            
            logger.info(f"📊 AI Content planning complete - Strategy: {planning_data.get('content_strategy', {}).get('approach')}, Sections: {len(planning_data.get('sections', []))}")
            return planning_data
                
        except asyncio.TimeoutError:
            logger.error("AI content planning timed out after 60 seconds")
            raise Exception("AI content planning timed out. Please try again.")
        except Exception as e:
            logger.error(f"AI content planning failed: {e}")
            raise Exception(f"AI content planning failed: {str(e)}. Cannot proceed without AI planning.")


class RequirementsDefiner:
    """AI-powered requirements definition for PRP workflow."""
    
    def __init__(self):
        """Initialize RequirementsDefiner with JsonAnalysisAgent."""
        self.json_agent = JsonAnalysisAgent()
    
    @monitor_ai_agent("requirements_definer", "requirements_definition")
    async def define_content_requirements(self, workflow: PRPWorkflowState) -> Dict[str, Any]:
        """Define specific content requirements using AI - NO hardcoded responses."""
        try:
            logger.info(f"🤖 Starting AI-powered requirements definition for: {workflow.topic}")
            
            # Build context with planning results
            planning_context = ""
            if workflow.planning_result:
                planning_context = f"""
PLANNING RESULTS:
Content Strategy: {workflow.planning_result.get('content_strategy', {})}
Sections Planned: {len(workflow.planning_result.get('sections', []))} sections
SEO Strategy: {workflow.planning_result.get('seo_strategy', {})}
                """.strip()
            
            context_info = f"""
TOPIC: {workflow.topic}
BRIEF CONTENT: {workflow.brief_content[:300]}...
CONTENT GOALS: {workflow.content_goals}
BRAND VOICE: {workflow.brand_voice}
{planning_context}
            """.strip()
            
            # Prepare requirements request
            requirements_prompt = f"""Based on this content brief and planning, define specific content requirements:

{context_info}

Provide a structured JSON response with exactly this format:
{{
    "length_requirements": {{
        "min_words": 1000,
        "max_words": 3000,
        "target_words": 2000,
        "rationale": "why this length is appropriate"
    }},
    "seo_requirements": {{
        "primary_keywords": ["keyword1", "keyword2"],
        "keyword_density": "1-3%",
        "header_structure": "H1, H2, H3, H4",
        "meta_tags": true,
        "internal_links_target": 3,
        "seo_focus": "specific SEO strategy"
    }},
    "brand_requirements": {{
        "tone": "specific tone from brand voice",
        "voice": "formal/semi-formal/casual",
        "key_messages": ["message1", "message2"],
        "style_guidelines": "specific writing style requirements"
    }},
    "quality_standards": {{
        "readability_target": "Flesch-Kincaid grade level",
        "expertise_level": "beginner/intermediate/advanced",
        "fact_checking": true,
        "source_requirements": "citation and source standards"
    }}
}}

Respond only with the JSON structure, no other text."""
            
            # Use JsonAnalysisAgent for structured response
            schema = {
                "required": ["length_requirements", "seo_requirements", "brand_requirements", "quality_standards"],
                "properties": {
                    "length_requirements": {"type": "object"},
                    "seo_requirements": {"type": "object"},
                    "brand_requirements": {"type": "object"},
                    "quality_standards": {"type": "object"}
                }
            }
            
            task_data = {
                "analysis_type": "requirements_definition",
                "content": context_info,
                "prompt": requirements_prompt,
                "schema": schema,
                "max_tokens": 2000
            }
            
            from src.agents.base_agent import AgentContext
            agent_context = AgentContext(
                organization_id="00000000-0000-0000-0000-000000000001",
                user_id="anonymous",
                brand_voice_config=workflow.brand_voice,
                session_id=f"requirements_{workflow.workflow_id}"
            )
            
            # Execute AI requirements definition with timeout
            logger.info(f"🔄 Sending requirements definition request to AI...")
            result = await asyncio.wait_for(
                self.json_agent.execute(task_data, agent_context),
                timeout=60.0
            )
            
            if not result.success:
                raise Exception(f"AI requirements definition failed: {result.error_message}")
            
            requirements_data = result.data
            
            # Add metadata
            requirements_data["ai_powered"] = True
            requirements_data["requirements_method"] = "json_analysis_agent"
            requirements_data["workflow_id"] = workflow.workflow_id
            
            logger.info(f"📊 AI Requirements definition complete - Target words: {requirements_data.get('length_requirements', {}).get('target_words')}, SEO focus: {requirements_data.get('seo_requirements', {}).get('seo_focus')}")
            return requirements_data
                
        except asyncio.TimeoutError:
            logger.error("AI requirements definition timed out after 60 seconds")
            raise Exception("AI requirements definition timed out. Please try again.")
        except Exception as e:
            logger.error(f"AI requirements definition failed: {e}")
            raise Exception(f"AI requirements definition failed: {str(e)}. Cannot proceed without AI requirements.")


class ProcessDefiner:
    """AI-powered process definition for PRP workflow."""
    
    def __init__(self):
        """Initialize ProcessDefiner with JsonAnalysisAgent."""
        self.json_agent = JsonAnalysisAgent()
    
    @monitor_ai_agent("process_definer", "process_definition")
    async def define_content_process(self, workflow: PRPWorkflowState) -> Dict[str, Any]:
        """Define the content creation process using AI - NO hardcoded responses."""
        try:
            logger.info(f"🤖 Starting AI-powered process definition for: {workflow.topic}")
            
            # Use JsonAnalysisAgent for structured JSON response
            # (initialized in __init__ as self.json_agent)
            
            # Build context with previous results
            previous_context = ""
            if workflow.planning_result:
                previous_context += f"PLANNING: {workflow.planning_result.get('content_strategy', {})}\n"
            if workflow.requirements_result:
                previous_context += f"REQUIREMENTS: Target words: {workflow.requirements_result.get('length_requirements', {}).get('target_words', 'Unknown')}, SEO focus: {workflow.requirements_result.get('seo_requirements', {}).get('seo_focus', 'Unknown')}\n"
            
            context_info = f"""
TOPIC: {workflow.topic}
BRIEF CONTENT: {workflow.brief_content[:300]}...
CONTENT GOALS: {workflow.content_goals}
BRAND VOICE: {workflow.brand_voice}
CHECKIN FREQUENCY: {workflow.checkin_frequency}
{previous_context}
            """.strip()
            
            # Prepare process definition request
            process_prompt = f"""Based on this content brief and previous planning/requirements, define the optimal content creation process:

{context_info}

Provide a structured JSON response with exactly this format:
{{
    "generation_approach": "how to approach content creation (e.g., section_by_section, outline_first, research_heavy)",
    "research_process": {{
        "primary_sources": ["source_type1", "source_type2"],
        "fact_verification": "strict/moderate/basic",
        "competitor_analysis": true,
        "expert_quotes": true
    }},
    "quality_checks": [
        "specific_check1",
        "specific_check2",
        "specific_check3"
    ],
    "review_steps": [
        {{"step": "step_name", "focus": "what to focus on", "duration": "estimated_time"}},
        {{"step": "another_step", "focus": "focus_area", "duration": "time_estimate"}}
    ],
    "tools_to_use": [
        "specific_tool1",
        "specific_tool2",
        "specific_tool3"
    ],
    "optimization_sequence": [
        "first_optimization",
        "second_optimization",
        "final_optimization"
    ],
    "timeline_estimate": {{
        "research_hours": 2,
        "writing_hours": 4,
        "review_hours": 1,
        "total_hours": 7
    }}
}}

Respond only with the JSON structure, no other text."""

            # Use JsonAnalysisAgent for structured response
            schema = {
                "required": ["generation_approach", "quality_checks", "review_steps", "tools_to_use"],
                "properties": {
                    "generation_approach": {"type": "string"},
                    "research_process": {"type": "object"},
                    "quality_checks": {"type": "array"},
                    "review_steps": {"type": "array"},
                    "tools_to_use": {"type": "array"},
                    "optimization_sequence": {"type": "array"},
                    "timeline_estimate": {"type": "object"}
                }
            }
            
            task_data = {
                "analysis_type": "process_definition",
                "content": context_info,
                "prompt": process_prompt,
                "schema": schema,
                "max_tokens": 2000
            }
            
            from src.agents.base_agent import AgentContext
            agent_context = AgentContext(
                organization_id="00000000-0000-0000-0000-000000000001",
                user_id="anonymous",
                brand_voice_config=workflow.brand_voice,
                session_id=f"process_{workflow.workflow_id}"
            )
            
            # Execute AI process definition with timeout
            logger.info(f"🔄 Sending process definition request to AI...")
            result = await asyncio.wait_for(
                self.json_agent.execute(task_data, agent_context),
                timeout=60.0
            )
            
            if not result.success:
                raise Exception(f"AI process definition failed: {result.error_message}")
                
            process_data = result.data
            
            # Add metadata
            process_data["ai_powered"] = True
            process_data["process_method"] = "json_analysis_agent"
            process_data["workflow_id"] = workflow.workflow_id
            
            logger.info(f"📊 AI Process definition complete - Approach: {process_data.get('generation_approach')}, Timeline: {process_data.get('timeline_estimate', {}).get('total_hours', 'Unknown')} hours")
            return process_data
                
        except asyncio.TimeoutError:
            logger.error("AI process definition timed out after 60 seconds")
            raise Exception("AI process definition timed out. Please try again.")
        except Exception as e:
            logger.error(f"AI process definition failed: {e}")
            raise Exception(f"AI process definition failed: {str(e)}. Cannot proceed without AI process definition.")


class FinalReviewer:
    """AI-powered final review for PRP workflow."""
    
    def __init__(self):
        """Initialize FinalReviewer with JsonAnalysisAgent."""
        self.json_agent = JsonAnalysisAgent()
    
    @monitor_ai_agent("final_reviewer", "final_review")
    async def perform_final_review(self, workflow: PRPWorkflowState) -> Dict[str, Any]:
        """Perform final review and optimization using AI - NO hardcoded responses."""
        try:
            logger.info(f"🤖 Starting AI-powered final review for: {workflow.topic}")
            
            # Use JsonAnalysisAgent for structured JSON response
            # (initialized in __init__ as self.json_agent)
            
            # Get generated content for review
            content_to_review = ""
            if workflow.generation_result:
                content_to_review = workflow.generation_result.get("content", "")[:2000]  # First 2000 chars for review
            
            # Build comprehensive context
            review_context = f"""
TOPIC: {workflow.topic}
ORIGINAL BRIEF: {workflow.brief_content[:300]}...
CONTENT GOALS: {workflow.content_goals}
BRAND VOICE: {workflow.brand_voice}
GENERATED CONTENT (preview): {content_to_review}...
REQUIREMENTS: {workflow.requirements_result.get('length_requirements', {}) if workflow.requirements_result else 'None'}
PROCESS USED: {workflow.process_result.get('generation_approach', 'Unknown') if workflow.process_result else 'Unknown'}
            """.strip()
            
            # Prepare final review request
            review_prompt = f"""Perform a comprehensive final review and quality assessment of this content based on the brief, requirements, and brand voice:

{review_context}

Provide a structured JSON response with exactly this format:
{{
    "seo_analysis": {{
        "seo_score": 85,
        "keyword_optimization": "excellent/good/fair/poor",
        "header_structure": "excellent/good/fair/poor",
        "meta_optimization": "excellent/good/fair/poor",
        "internal_linking": "excellent/good/fair/poor"
    }},
    "brand_compliance": {{
        "compliance_score": 90,
        "tone_alignment": "excellent/good/fair/poor",
        "voice_consistency": "excellent/good/fair/poor",
        "key_message_delivery": "excellent/good/fair/poor"
    }},
    "content_quality": {{
        "readability_score": 88,
        "engagement_level": "high/medium/low",
        "information_accuracy": "verified/likely/unverified",
        "completeness": "complete/mostly_complete/incomplete"
    }},
    "optimization_applied": [
        "specific_optimization1",
        "specific_optimization2",
        "specific_optimization3"
    ],
    "identified_issues": [
        "issue1_description",
        "issue2_description"
    ],
    "recommendations": [
        "specific_actionable_recommendation1",
        "specific_actionable_recommendation2",
        "specific_actionable_recommendation3"
    ],
    "overall_assessment": {{
        "final_score": 87,
        "readiness": "ready/needs_minor_revisions/needs_major_revisions",
        "estimated_revision_time": "30_minutes/2_hours/4_hours"
    }}
}}

Respond only with the JSON structure, no other text."""

            # Use JsonAnalysisAgent for structured response
            schema = {
                "required": ["seo_analysis", "brand_compliance", "content_quality", "recommendations"],
                "properties": {
                    "seo_analysis": {"type": "object"},
                    "brand_compliance": {"type": "object"},
                    "content_quality": {"type": "object"},
                    "optimization_applied": {"type": "array"},
                    "identified_issues": {"type": "array"},
                    "recommendations": {"type": "array"},
                    "overall_assessment": {"type": "object"}
                }
            }
            
            task_data = {
                "analysis_type": "final_review",
                "content": review_context,
                "prompt": review_prompt,
                "schema": schema,
                "max_tokens": 2000
            }
            
            from src.agents.base_agent import AgentContext
            agent_context = AgentContext(
                organization_id="00000000-0000-0000-0000-000000000001",
                user_id="anonymous",
                brand_voice_config=workflow.brand_voice,
                session_id=f"final_review_{workflow.workflow_id}"
            )
            
            # Execute AI final review with timeout
            logger.info(f"🔄 Sending final review request to AI...")
            result = await asyncio.wait_for(
                self.json_agent.execute(task_data, agent_context),
                timeout=60.0
            )
            
            if not result.success:
                raise Exception(f"AI final review failed: {result.error_message}")
                
            review_data = result.data
            
            # Add metadata
            review_data["ai_powered"] = True
            review_data["review_method"] = "json_analysis_agent"
            review_data["workflow_id"] = workflow.workflow_id
            review_data["review_timestamp"] = datetime.now().isoformat()
            
            logger.info(f"📊 AI Final review complete - Overall score: {review_data.get('overall_assessment', {}).get('final_score', 'Unknown')}, Readiness: {review_data.get('overall_assessment', {}).get('readiness', 'Unknown')}")
            return review_data
                
        except asyncio.TimeoutError:
            logger.error("AI final review timed out after 60 seconds")
            raise Exception("AI final review timed out. Please try again.")
        except Exception as e:
            logger.error(f"AI final review failed: {e}")
            raise Exception(f"AI final review failed: {str(e)}. Cannot proceed without AI review.")