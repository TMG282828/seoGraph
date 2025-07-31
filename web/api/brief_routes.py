"""
Brief management API routes using SQLAlchemy database with AI integration.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from typing import Optional, List, Dict, Any
import logging
import asyncio
from datetime import datetime

# Import auth dependencies with fallback
def get_current_user():
    """Simple mock auth for testing - returns anonymous user."""
    return {"id": "anonymous", "email": "user@localhost"}

# Import database and models
from src.database.database import get_db_session
from src.database.models import ContentBrief, SavedContent, ContentItem
from sqlalchemy.orm import Session
from sqlalchemy import desc

# Import AI agent and related dependencies
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.agents.content_generation import (
        ContentGenerationAgent, 
        create_content_generation_agent, 
        ContentGenerationRequest as AIContentBrief
    )
    from src.agents.base_agent import AgentContext
    from src.services.prp_workflow import (
        PRPWorkflowService, 
        WorkflowPhase, 
        CheckpointStatus,
        ContentWorkflow
    )
    from src.services.topic_extraction_service import topic_extraction_service
    from models.content_models import ContentType
    from models.seo_models import SearchIntent
    AI_AVAILABLE = True
    logging.info("AI content generation agent and PRP workflow service loaded successfully")
    
    # Create shared instances to avoid repeated initialization
    _shared_agent = None
    _shared_workflow_service = None
    
    def get_shared_agent():
        """Get a shared agent instance to avoid repeated initialization."""
        global _shared_agent
        if _shared_agent is None:
            _shared_agent = ContentGenerationAgent()
            logging.info("Shared ContentGenerationAgent instance created")
        return _shared_agent
    
    def get_shared_workflow_service():
        """Get a shared PRP workflow service instance."""
        global _shared_workflow_service
        if _shared_workflow_service is None:
            _shared_workflow_service = PRPWorkflowService()
            logging.info("Shared PRPWorkflowService instance created")
        return _shared_workflow_service
        
except ImportError as e:
    logging.warning(f"AI agent not available: {e}")
    AI_AVAILABLE = False
    def get_shared_agent():
        return None
    def get_shared_workflow_service():
        return None

logger = logging.getLogger(__name__)
router = APIRouter()


# Helper functions for content generation workflows
async def handle_prp_workflow(message: str, request: dict, tenant_id: str, context: str, start_time: datetime):
    """Handle PRP workflow with human-in-loop checkpoints."""
    try:
        logger.info(f"🔄 PRP Workflow starting for message: {message[:50]}...")
        
        workflow_service = get_shared_workflow_service()
        if workflow_service is None:
            logger.error("❌ PRP Workflow service not available")
            return {
                "success": False,
                "response": "PRP Workflow service not available",
                "context": context,
                "error": "workflow_service_unavailable"
            }
        
        # Extract request data
        brief_content = request.get("brief_content")
        brief_summary = request.get("brief_summary", {})
        human_in_loop = request.get("human_in_loop", {})
        content_goals = request.get("content_goals", {})
        brand_voice = request.get("brand_voice", {})
        
        logger.info(f"📋 PRP Request data - Brief content: {bool(brief_content)}, Brief summary: {bool(brief_summary)}")
        logger.info(f"📄 Brief content preview: {brief_content[:200] if brief_content else 'None'}...")
        logger.info(f"📊 Brief summary content: {brief_summary}")
        logger.info(f"👤 Human-in-loop settings: {human_in_loop}")
        logger.info(f"🎯 Content goals: {content_goals}")
        logger.info(f"🗣️ Brand voice: {brand_voice}")
        
        # Create workflow for this content generation request
        workflow_config = {
            "user_message": message,
            "brief_content": brief_content,
            "brief_summary": brief_summary,
            "brand_voice": brand_voice,
            "content_goals": content_goals,
            "human_in_loop": human_in_loop
        }
        
        # Start PRP workflow
        logger.info(f"🚀 Creating PRP workflow with config: {list(workflow_config.keys())}")
        workflow = await workflow_service.create_workflow(
            tenant_id=tenant_id,
            workflow_type="content_generation",
            config=workflow_config,
            user_id="anonymous"
        )
        
        logger.info(f"✅ Workflow created with ID: {workflow.id}")
        
        # Execute first phase (planning) and return checkpoint
        logger.info(f"🔄 Executing first phase for workflow: {workflow.id}")
        checkpoint = await workflow_service.execute_next_phase(workflow.id)
        
        if not checkpoint:
            logger.error(f"❌ No checkpoint returned from execute_next_phase")
            return {
                "success": False,
                "response": "Failed to create initial checkpoint",
                "context": context,
                "error": "checkpoint_creation_failed"
            }
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"✅ PRP workflow checkpoint created in {duration:.2f}s: phase={checkpoint.phase}, title={checkpoint.title}")
        
        # Handle checkpoint phase value safely
        try:
            phase_value = checkpoint.phase.value if hasattr(checkpoint.phase, 'value') else str(checkpoint.phase)
        except:
            phase_value = "brief_analysis"
        
        response_data = {
            "success": True,
            "response": checkpoint.content.get("message", "Planning phase completed. Please review and approve to continue."),
            "context": context,
            "prpWorkflow": {
                "workflowId": workflow.id,
                "phase": phase_value,
                "progress": workflow_service.get_progress_percentage(workflow.id),
                "checkpoint_id": checkpoint.id
            },
            "checkpointActions": {
                "workflowId": workflow.id,
                "checkpointId": checkpoint.id,
                "phase": phase_value,
                "title": checkpoint.title,
                "description": checkpoint.description
            },
            "knowledge_graph_used": True,
            "generation_time": f"{duration:.2f}s",
            "recommendations": [
                "PRP workflow initiated with structured checkpoints",
                "Review and approve to continue to next phase"
            ]
        }
        
        logger.info(f"📤 PRP Response structure: {list(response_data.keys())}")
        logger.info(f"📤 PRP Checkpoint actions: {response_data.get('checkpointActions', {})}")
        
        return response_data
        
    except Exception as e:
        logger.error(f"PRP workflow failed: {e}")
        return {
            "success": False,
            "response": f"PRP workflow encountered an error: {str(e)}",
            "context": context,
            "error": "prp_workflow_error"
        }


async def handle_direct_generation(message: str, request: dict, tenant_id: str, context: str, start_time: datetime):
    """Handle direct content generation without PRP workflow."""
    try:
        logger.info(f"⚡ Direct generation starting for message: {message[:50]}...")
        
        agent = get_shared_agent()
        if agent is None:
            logger.error("❌ Content generation agent not available")
            return {
                "success": False,
                "response": "Content generation agent not available",
                "context": context,
                "error": "agent_unavailable"
            }
        
        # Extract request data
        brief_content = request.get("brief_content")
        brief_summary = request.get("brief_summary", {})
        human_in_loop = request.get("human_in_loop", {})
        content_goals = request.get("content_goals", {})
        brand_voice = request.get("brand_voice", {})
        
        logger.info(f"📋 Direct Request data - Brief content: {bool(brief_content)}, Brief summary: {bool(brief_summary)}")
        logger.info(f"📋 Brief content preview: {brief_content[:100] if brief_content else 'None'}...")
        logger.info(f"🎯 Content goals: {content_goals}")
        logger.info(f"🗣️ Brand voice: {brand_voice}")
        
        # Prepare content generation based on whether brief is provided
        if brief_content and brief_summary:
            # With brief content - use AI topic extraction instead of generic title
            logger.info("🤖 Using AI topic extraction for direct generation")
            title = await topic_extraction_service.extract_topic(
                brief_content=brief_content,
                user_message=message,
                brief_summary=brief_summary,
                content_goals=content_goals
            )
            logger.info(f"✅ AI extracted topic for direct generation: {title}")
            
            keywords = brief_summary.get('keywords', [])
            word_count = brief_summary.get('word_count', 500)
            
            # Determine content length category
            if word_count < 500:
                content_length = "short"
            elif word_count > 1500:
                content_length = "long"
            else:
                content_length = "medium"
            
            content_request = AIContentBrief(
                content_type="blog_post",
                topic=title,
                target_keywords=keywords if isinstance(keywords, list) else [keywords] if keywords else [],
                content_length=content_length,
                writing_style=brand_voice.get("tone", "informational"),
                target_audience=content_goals.get("audience", "general"),
                outline_only=False,
                include_meta_tags=True,
                include_internal_links=True,
                reference_content=[brief_content],
                use_knowledge_graph=True,
                use_vector_search=True,
                similarity_threshold=0.7,
                max_related_content=5
            )
        else:
            # Without brief - simple chat response
            content_request = AIContentBrief(
                content_type="response",
                topic=message,
                target_keywords=[],
                content_length="short",
                writing_style="conversational",
                target_audience="general",
                outline_only=False,
                use_knowledge_graph=True,
                use_vector_search=True,
                similarity_threshold=0.8,
                max_related_content=2
            )
        
        # Create agent context
        agent_context = AgentContext(
            organization_id=tenant_id,
            user_id="anonymous",
            brand_voice_config=brand_voice,
            industry_context=content_goals.get('industry', ''),
            session_id=f"chat_{int(datetime.now().timestamp())}"
        )
        
        # Convert to task data
        task_data = content_request.dict()
        task_data['type'] = 'content_generation'
        task_data['user_message'] = message
        
        # Execute with timeout
        logger.info(f"About to execute direct generation with task_data: {list(task_data.keys())}")
        try:
            result = await asyncio.wait_for(
                agent.execute(task_data, agent_context),
                timeout=30.0
            )
            logger.info(f"Direct generation completed, success: {result.success}")
        except asyncio.TimeoutError:
            return {
                "success": False,
                "response": "Content generation timed out. Please try again with a simpler request.",
                "context": context,
                "error": "timeout_error"
            }
        
        if result.success and result.result_data:
            generated_content = result.result_data.get("content", "")
            knowledge_sources = result.result_data.get("knowledge_sources", [])
            related_topics = result.result_data.get("related_topics", [])
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"✅ Direct generation completed in {duration:.2f}s: {len(generated_content)} chars, {len(knowledge_sources)} sources, {len(related_topics)} topics")
            
            response_data = {
                "success": True,
                "response": generated_content,
                "context": context,
                "word_count": len(generated_content.split()),
                "seo_score": result.result_data.get("seo_score", 75),
                "readability_score": result.result_data.get("readability_score", 80),
                "quality_score": result.result_data.get("brand_voice_compliance", 85),
                "knowledge_graph_used": True,
                "knowledge_sources": knowledge_sources,
                "related_topics": related_topics,
                "generation_time": f"{duration:.2f}s",
                "recommendations": result.result_data.get("improvement_suggestions", [
                    "Content enhanced with knowledge graph context",
                    "Enable PRP workflow for structured content creation with checkpoints"
                ])
            }
            
            logger.info(f"📤 Direct Response structure: {list(response_data.keys())}")
            logger.info(f"📤 Content preview: {generated_content[:100] if generated_content else 'No content'}...")
            
            return response_data
        else:
            return {
                "success": False,
                "response": f"Direct generation failed: {result.error_message}",
                "context": context,
                "error": "direct_generation_failed"
            }
            
    except Exception as e:
        logger.error(f"Direct generation failed: {e}")
        return {
            "success": False,
            "response": f"Direct generation encountered an error: {str(e)}",
            "context": context,
            "error": "direct_generation_error"
        }

@router.post("/briefs")
async def save_brief(request: dict):
    """Save content brief to database."""
    try:
        title = request.get("title", "")
        content = request.get("content", "")
        word_count = request.get("word_count", 0)
        keywords = request.get("keywords", [])
        summary = request.get("summary", "")
        source_type = request.get("source_type", "manual")  # manual, file, url
        
        if not title.strip():
            raise HTTPException(status_code=400, detail="Brief title is required")
        if not content.strip():
            raise HTTPException(status_code=400, detail="Brief content is required")
        
        # Get database session
        db = get_db_session()
        current_user = get_current_user()
        
        try:
            # Create new brief
            brief = ContentBrief(
                user_id=current_user.get('id', 'anonymous'),
                title=title,
                content=content,
                summary=summary,
                word_count=word_count,
                keywords=keywords[:10] if keywords else [],
                source_type=source_type,
                original_filename=request.get("original_filename"),
                source_url=request.get("source_url")
            )
            
            db.add(brief)
            db.commit()
            db.refresh(brief)
            
            return {
                "success": True,
                "id": brief.id,
                "message": "Brief saved successfully"
            }
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Brief save failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save brief: {str(e)}")

@router.get("/briefs")
async def list_briefs(limit: int = 20, offset: int = 0):
    """List user's saved briefs."""
    try:
        current_user = get_current_user()
        user_id = current_user.get('id', 'anonymous')
        db = get_db_session()
        
        try:
            briefs = db.query(ContentBrief).filter(
                ContentBrief.user_id == user_id
            ).order_by(desc(ContentBrief.created_at)).limit(limit).offset(offset).all()
            
            return [brief.to_dict() for brief in briefs]
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Failed to list briefs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list briefs: {str(e)}")

@router.get("/briefs/{brief_id}")
async def get_brief(brief_id: int):
    """Get a specific brief by ID."""
    try:
        current_user = get_current_user()
        user_id = current_user.get('id', 'anonymous')
        db = get_db_session()
        
        try:
            brief = db.query(ContentBrief).filter(
                ContentBrief.id == brief_id,
                ContentBrief.user_id == user_id
            ).first()
            
            if not brief:
                raise HTTPException(status_code=404, detail="Brief not found")
                
            return brief.to_dict()
            
        finally:
            db.close()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get brief {brief_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get brief: {str(e)}")

@router.delete("/briefs/{brief_id}")
async def delete_brief(brief_id: int):
    """Delete a brief by ID."""
    try:
        current_user = get_current_user()
        user_id = current_user.get('id', 'anonymous')
        db = get_db_session()
        
        try:
            brief = db.query(ContentBrief).filter(
                ContentBrief.id == brief_id,
                ContentBrief.user_id == user_id
            ).first()
            
            if not brief:
                raise HTTPException(status_code=404, detail="Brief not found")
            
            db.delete(brief)
            db.commit()
            
            return {"success": True, "message": "Brief deleted successfully"}
            
        finally:
            db.close()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete brief {brief_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete brief: {str(e)}")

@router.post("/save")
async def save_content(request: dict):
    """Manually save generated content."""
    try:
        title = request.get("title", "")
        content = request.get("content", "")
        content_type = request.get("content_type", "manual_save")
        
        if not title.strip():
            raise HTTPException(status_code=400, detail="Content title is required")
        if not content.strip():
            raise HTTPException(status_code=400, detail="Content cannot be empty")
        
        # Get database session
        db = get_db_session()
        current_user = get_current_user()
        
        try:
            # Create new saved content
            saved_content = SavedContent(
                user_id=current_user.get('id', 'anonymous'),
                title=title,
                content=content,
                content_type=content_type,
                word_count=request.get("word_count", 0),
                seo_score=request.get("seo_score"),
                readability_score=request.get("readability_score"),
                keywords=request.get("keywords", []),
                brief_used=request.get("brief_used"),
                brief_id=request.get("brief_id"),
                created_via=request.get("created_via", "manual"),
                auto_saved=request.get("auto_saved", False)
            )
            
            db.add(saved_content)
            db.commit()
            db.refresh(saved_content)
            
            return {
                "success": True,
                "id": saved_content.id,
                "message": "Content saved successfully"
            }
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Content save failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save content: {str(e)}")

@router.post("/auto-save")
async def auto_save_content(request: dict):
    """Auto-save generated content."""
    try:
        # Validate required fields for auto-save
        if not request.get("title"):
            # Generate a default title if missing
            request["title"] = f"Auto-saved Content {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        if not request.get("content"):
            # For auto-save, allow empty content but log it
            logger.warning("Auto-save attempted with empty content")
            return {
                "success": False,
                "message": "Cannot auto-save empty content",
                "auto_saved": False
            }
        
        # Ensure auto-save specific fields are set
        request["auto_saved"] = True
        request["created_via"] = request.get("created_via", "auto_save")
        
        # Add basic metrics if missing
        if "word_count" not in request:
            content = request.get("content", "")
            request["word_count"] = len(content.split()) if content else 0
        
        return await save_content(request)
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Auto-save failed: {e}")
        # Return graceful error response instead of raising exception
        return {
            "success": False,
            "message": "Auto-save temporarily unavailable",
            "error": str(e),
            "auto_saved": False
        }

@router.get("/list")
async def list_recent_content(limit: int = 10, offset: int = 0):
    """List recent saved content from both SavedContent and ContentItem tables."""
    try:
        current_user = get_current_user()
        user_id = current_user.get('id', 'anonymous')
        db = get_db_session()
        
        try:
            # Get content from both tables and combine them
            all_content = []
            
            # Get from SavedContent table (manual saves, older content)
            saved_content_items = db.query(SavedContent).filter(
                SavedContent.user_id == user_id
            ).order_by(desc(SavedContent.created_at)).all()
            
            for item in saved_content_items:
                content_dict = item.to_dict()
                content_dict['source_table'] = 'saved_content'
                all_content.append(content_dict)
            
            # Get from ContentItem table (PRP workflow content, Knowledge Base)
            # Use organization context - map user to org (for demo, use default org)
            organization_id = "demo-org"  # TODO: Use real org ID from auth token when authentication is implemented
            
            content_items = db.query(ContentItem).filter(
                ContentItem.organization_id == organization_id
            ).order_by(desc(ContentItem.created_at)).all()
            
            for item in content_items:
                content_dict = item.to_dict()
                # Map ContentItem fields to match SavedContent format for frontend
                mapped_content = {
                    'id': content_dict['id'],
                    'title': content_dict['title'],
                    'content': content_dict['content'],
                    'content_type': content_dict.get('content_type', 'prp_workflow'),
                    'word_count': content_dict.get('word_count', 0),
                    'seo_score': content_dict.get('seo_score'),
                    'readability_score': content_dict.get('readability_score'),
                    'keywords': content_dict.get('keywords', []),
                    'created_at': content_dict['created_at'],
                    'source_table': 'content_item',
                    'brief_used': None,
                    'brief_id': None,
                    'created_via': 'prp_workflow' if 'prp_' in str(content_dict['id']) else 'knowledge_base',
                    'auto_saved': True
                }
                all_content.append(mapped_content)
            
            # Sort all content by created_at descending
            all_content.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            
            # Apply pagination
            paginated_content = all_content[offset:offset + limit]
            
            logger.info(f"✅ Recent content API: Found {len(all_content)} total items ({len(saved_content_items)} SavedContent, {len(content_items)} ContentItem)")
            
            return {
                "content": paginated_content,
                "total": len(all_content),
                "limit": limit,
                "offset": offset,
                "sources": {
                    "saved_content": len(saved_content_items),
                    "content_item": len(content_items)
                }
            }
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Failed to list recent content: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list recent content: {str(e)}")

@router.post("/chat")
async def content_chat(request: dict):
    """Chat interface for AI-powered content generation."""
    try:
        message = request.get("message", "")
        context = request.get("context", "content_studio")
        brief_content = request.get("brief_content")
        brief_summary = request.get("brief_summary", {})
        
        if not message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Check if AI is available
        logger.info(f"🤖 AI_AVAILABLE status: {AI_AVAILABLE}")
        if not AI_AVAILABLE:
            logger.info(f"❌ AI not available - returning fallback response")
            # Provide a simple fallback response
            return {
                "success": True,
                "response": f"I received your message: '{message}'\n\nThe AI content generation service is currently being configured. In the meantime, I can acknowledge your requests and provide basic assistance. Please try again shortly for full AI-powered content generation.",
                "context": context,
                "word_count": 25,
                "seo_score": 60,
                "readability_score": 85,
                "knowledge_graph_used": False,
                "generation_time": "fallback_mode",
                "recommendations": [
                    "AI service is being configured",
                    "Basic response provided as fallback"
                ]
            }
        
        logger.info(f"✅ AI is available - proceeding with content generation")
        
        start_time = datetime.now()
        
        # Extract workflow mode from request
        prp_workflow_enabled = request.get("prp_workflow", False)
        human_in_loop = request.get("human_in_loop", {})
        require_approval = human_in_loop.get("requireApproval", False)
        
        # Debug logging for workflow routing
        logger.info(f"🔍 ROUTING DEBUG - prp_workflow: {prp_workflow_enabled} (type: {type(prp_workflow_enabled)})")
        logger.info(f"🔍 ROUTING DEBUG - requireApproval: {require_approval} (type: {type(require_approval)})")
        logger.info(f"🔍 ROUTING DEBUG - Human in loop settings: {human_in_loop}")
        logger.info(f"🔍 ROUTING DEBUG - Request keys: {list(request.keys())}")
        logger.info(f"🔍 ROUTING DEBUG - Raw prp_workflow value: {repr(request.get('prp_workflow'))}")
        
        # Create user context
        current_user = get_current_user()
        tenant_id = current_user.get('org_id', 'demo-org')
        
        # Route to appropriate workflow based on settings
        # If PRP workflow is enabled, use it (regardless of requireApproval setting)
        if prp_workflow_enabled:
            logger.info(f"🔄 ROUTING DECISION: PRP workflow selected - prp_workflow_enabled={prp_workflow_enabled}")
            logger.info(f"🔄 Starting PRP workflow with checkpoints for: {message[:100]}")
            return await handle_prp_workflow(message, request, tenant_id, context, start_time)
        else:
            logger.info(f"⚡ ROUTING DECISION: Direct generation selected - prp_workflow_enabled={prp_workflow_enabled}")
            logger.info(f"⚡ Starting direct content generation for: {message[:100]}")
            return await handle_direct_generation(message, request, tenant_id, context, start_time)
            
    except Exception as e:
        logger.error(f"Content generation failed: {e}")
        return {
            "success": False,
            "response": f"Content generation encountered an error: {str(e)}",
            "context": context,
            "error": "generation_error"
        }

@router.post("/checkpoint-response")
async def handle_checkpoint_response(request: dict):
    """Handle user response to PRP workflow checkpoint."""
    if not AI_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI service not available")
    
    try:
        workflow_id = request.get("workflow_id")
        checkpoint_id = request.get("checkpoint_id") 
        response = request.get("response")  # 'approved', 'rejected', 'modified'
        feedback = request.get("feedback", "")
        
        logger.info(f"🔄 CHECKPOINT RESPONSE: workflow_id={workflow_id}, checkpoint_id={checkpoint_id}, response={response}")
        
        if not workflow_id or not checkpoint_id or not response:
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        workflow_service = get_shared_workflow_service()
        if workflow_service is None:
            raise HTTPException(status_code=503, detail="Workflow service not available")
        
        # Process checkpoint response
        logger.info(f"📋 Processing checkpoint response for {workflow_id}...")
        result = await workflow_service.handle_checkpoint_response(
            workflow_id=workflow_id,
            checkpoint_id=checkpoint_id,
            response=response,
            feedback=feedback
        )
        logger.info(f"✅ Checkpoint response processed: {result.get('success', False)}")
        
        if response == "approved":
            # Continue to next phase
            logger.info(f"🚀 Approval received - executing next phase for {workflow_id}")
            next_checkpoint = await workflow_service.execute_next_phase(workflow_id)
            logger.info(f"📤 Next checkpoint result: {next_checkpoint.id if next_checkpoint else 'None'}")
            
            if next_checkpoint:
                # Handle checkpoint phase value safely
                try:
                    phase_value = next_checkpoint.phase.value if hasattr(next_checkpoint.phase, 'value') else str(next_checkpoint.phase)
                except:
                    phase_value = "unknown"
                
                # Prepare base response
                response_data = {
                    "success": True,
                    "message": "Checkpoint approved, proceeding to next phase",
                    "prpWorkflow": {
                        "workflowId": workflow_id,
                        "phase": phase_value,
                        "progress": workflow_service.get_progress_percentage(workflow_id),
                        "checkpoint_id": next_checkpoint.id
                    },
                    "checkpointActions": {
                        "workflowId": workflow_id,
                        "checkpointId": next_checkpoint.id,
                        "phase": phase_value,
                        "title": next_checkpoint.title,
                        "description": next_checkpoint.description
                    },
                    "response": next_checkpoint.content.get("message", "Next phase ready for review")
                }
                
                # Include generated content if available (generation phase or later)
                if phase_value in ["generation", "review", "complete"]:
                    try:
                        generated_content = await workflow_service.get_final_content(workflow_id)
                        if generated_content and generated_content.strip():
                            response_data["generatedContent"] = generated_content
                            logger.info(f"✅ Including generated content ({len(generated_content)} chars) in checkpoint response")
                    except Exception as e:
                        logger.warning(f"Could not retrieve generated content for workflow {workflow_id}: {e}")
                
                return response_data
            else:
                # CRITICAL FIX: Workflow complete - execute_next_phase returned None
                logger.info(f"🏁 Workflow {workflow_id} completed - execute_next_phase returned None")
                workflow = workflow_service.active_workflows.get(workflow_id)
                
                final_content = await workflow_service.get_final_content(workflow_id)
                success_message = "🎉 Content Successfully Saved to Knowledge Base!"
                
                # Look for COMPLETE checkpoint with storage success message
                if workflow and workflow.checkpoints:
                    complete_checkpoint = None
                    for cp in reversed(workflow.checkpoints):  # Check latest first
                        if hasattr(cp.phase, 'value'):
                            phase_value = cp.phase.value
                        else:
                            phase_value = str(cp.phase)
                        
                        if phase_value == "complete":
                            complete_checkpoint = cp
                            break
                    
                    if complete_checkpoint:
                        logger.info(f"📋 Found COMPLETE checkpoint: {complete_checkpoint.title}")
                        success_message = complete_checkpoint.content.get("message", complete_checkpoint.title)
                        if not success_message:
                            success_message = complete_checkpoint.title
                
                logger.info(f"🎯 WORKFLOW COMPLETION: Returning final success response for {workflow_id}")
                return {
                    "success": True,
                    "message": success_message,
                    "response": final_content,
                    "generatedContent": final_content,  # Ensure content is included
                    "prpWorkflow": {
                        "workflowId": workflow_id,
                        "phase": "complete",
                        "progress": 100,
                        "completed": True  # Flag for frontend
                    },
                    "knowledge_graph_used": True,
                    "workflowCompleted": True,  # Clear flag for UI
                    "recommendations": [
                        "PRP workflow completed successfully",
                        "Content saved to Knowledge Base with full RAG integration",
                        "Content is now available in Recent Content section"
                    ]
                }
        else:
            # Rejected or modification requested
            return {
                "success": True,
                "message": f"Checkpoint {response}. Please provide feedback for improvements.",
                "response": f"I understand you want changes to the {result.get('phase', 'current')} phase. Please let me know what specific modifications you'd like me to make.",
                "prpWorkflow": {
                    "workflowId": workflow_id,
                    "phase": result.get("phase", "unknown"),
                    "progress": workflow_service.get_progress_percentage(workflow_id)
                }
            }
            
    except Exception as e:
        logger.error(f"Checkpoint response handling failed: {e}")
        raise HTTPException(status_code=500, detail=f"Checkpoint response failed: {str(e)}")
