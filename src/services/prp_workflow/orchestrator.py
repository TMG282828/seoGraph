"""
PRP Workflow Orchestrator.

Main orchestration service that coordinates the PRP workflow phases
and manages the overall workflow state and checkpoints.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

from .models import (
    PRPWorkflowState, WorkflowCheckpoint, ContentWorkflow, WorkflowConfig,
    WorkflowPhase, CheckpointStatus, PhaseResult
)
from .phase_analyzers import BriefAnalyzer, ContentPlanner, RequirementsDefiner, ProcessDefiner, FinalReviewer
from .content_generator import ContentGenerator
from ..topic_extraction_service import topic_extraction_service

logger = logging.getLogger(__name__)


class PRPWorkflowOrchestrator:
    """
    Main orchestration service for PRP workflows.
    
    Coordinates the execution of workflow phases:
    1. Brief Analysis - Understanding the user's request
    2. Planning - Creating content strategy and approach
    3. Requirements - Defining specific content requirements
    4. Process - Outlining the content creation process
    5. Generation - Creating the actual content
    6. Review - Quality assurance and optimization
    """
    
    def __init__(self):
        """Initialize the PRP workflow orchestrator."""
        self.active_workflows: Dict[str, PRPWorkflowState] = {}
        
        # Initialize phase analyzers
        self.brief_analyzer = BriefAnalyzer()
        self.content_planner = ContentPlanner()
        self.requirements_definer = RequirementsDefiner()
        self.process_definer = ProcessDefiner()
        self.content_generator = ContentGenerator()
        self.final_reviewer = FinalReviewer()
        
        logger.info("PRP Workflow Orchestrator initialized with AI-powered phase analyzers")
    
    async def create_workflow(self, tenant_id: str, workflow_type: str, 
                            config: Dict[str, Any], user_id: str) -> ContentWorkflow:
        """Create a new workflow."""
        workflow_id = f"prp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(config)) % 10000}"
        
        # Extract config values
        user_message = config.get("user_message", "")
        brief_content = config.get("brief_content", "")
        brief_summary = config.get("brief_summary", {})
        human_in_loop = config.get("human_in_loop", {})
        content_goals = config.get("content_goals", {})
        brand_voice = config.get("brand_voice", {})
        
        # Use AI-powered topic extraction instead of generic metadata titles
        logger.info("🤖 Starting AI-powered topic extraction from content analysis")
        topic = await topic_extraction_service.extract_topic(
            brief_content=brief_content,
            user_message=user_message,
            brief_summary=brief_summary,
            content_goals=content_goals
        )
        
        logger.info(f"✅ AI extracted topic: {topic}")
        
        # Map frontend camelCase to backend snake_case for human-in-loop settings
        checkin_frequency = human_in_loop.get("checkin_frequency") or human_in_loop.get("checkinFrequency", "medium")
        agent_aggressiveness = human_in_loop.get("agent_aggressiveness") or human_in_loop.get("agentAggressiveness", 5)
        require_approval = human_in_loop.get("require_approval") or human_in_loop.get("requireApproval", True)
        
        logger.info(f"🔧 Field mapping - checkin: {checkin_frequency}, aggressiveness: {agent_aggressiveness}, approval: {require_approval}")
        
        # Create workflow state
        workflow_state = PRPWorkflowState(
            workflow_id=workflow_id,
            brief_content=brief_content or user_message,
            topic=topic,
            checkin_frequency=checkin_frequency,
            agent_aggressiveness=agent_aggressiveness,
            require_approval=require_approval,
            content_goals=content_goals or {},
            brand_voice=brand_voice or {}
        )
        
        self.active_workflows[workflow_id] = workflow_state
        logger.info(f"Created PRP workflow {workflow_id} for topic: {topic}")
        
        # Begin with brief analysis
        await self._execute_brief_analysis(workflow_id)
        
        # Return a simple workflow object
        return ContentWorkflow(id=workflow_id)
    
    async def execute_next_phase(self, workflow_id: str) -> Optional[WorkflowCheckpoint]:
        """Execute the next phase and return checkpoint if needed."""
        if workflow_id not in self.active_workflows:
            return None
        
        workflow = self.active_workflows[workflow_id]
        
        # CRITICAL FIX: If workflow is already complete, don't execute any more phases
        if workflow.current_phase == WorkflowPhase.COMPLETE:
            logger.info(f"🏁 Workflow {workflow_id} is already COMPLETE - no further phases to execute")
            return None
        
        # If there's already a pending checkpoint, return it
        for checkpoint in workflow.checkpoints:
            if checkpoint.status == CheckpointStatus.PENDING:
                return checkpoint
        
        # If no pending checkpoints and workflow just started, the brief analysis should have created one
        if not workflow.checkpoints:
            logger.warning(f"No checkpoints found for workflow {workflow_id}, re-executing brief analysis")
            await self._execute_brief_analysis(workflow_id)
        
        # Return the latest checkpoint
        if workflow.checkpoints:
            return workflow.checkpoints[-1]
        
        return None
    
    async def handle_checkpoint_response(self, workflow_id: str, checkpoint_id: str, 
                                       response: str, feedback: str = "") -> Dict[str, Any]:
        """Handle user response to checkpoint."""
        if response == "approved":
            status = CheckpointStatus.APPROVED
        elif response == "rejected":
            status = CheckpointStatus.REJECTED
        else:
            status = CheckpointStatus.MODIFIED
        
        return await self._process_checkpoint_response(workflow_id, checkpoint_id, status, feedback)
    
    async def get_final_content(self, workflow_id: str) -> str:
        """Get the final generated content."""
        if workflow_id not in self.active_workflows:
            return "Workflow not found"
        
        workflow = self.active_workflows[workflow_id]
        
        if workflow.generation_result:
            return workflow.generation_result.get("content", "Content generation completed but no content available")
        
        return "Content generation not yet completed"
    
    def get_progress_percentage(self, workflow_id: str) -> int:
        """Get workflow progress percentage."""
        if workflow_id not in self.active_workflows:
            return 0
        
        workflow = self.active_workflows[workflow_id]
        return self._calculate_progress_percentage(workflow)
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a workflow."""
        if workflow_id not in self.active_workflows:
            return None
        
        workflow = self.active_workflows[workflow_id]
        
        return {
            "workflow_id": workflow_id,
            "current_phase": workflow.current_phase,
            "topic": workflow.topic,
            "progress_percentage": self._calculate_progress_percentage(workflow),
            "pending_checkpoints": [
                {
                    "id": cp.id,
                    "phase": cp.phase,
                    "title": cp.title,
                    "description": cp.description,
                    "status": cp.status
                }
                for cp in workflow.checkpoints 
                if cp.status == CheckpointStatus.PENDING
            ],
            "last_checkpoint": {
                "id": workflow.checkpoints[-1].id,
                "phase": workflow.checkpoints[-1].phase,
                "title": workflow.checkpoints[-1].title,
                "description": workflow.checkpoints[-1].description,
                "status": workflow.checkpoints[-1].status,
                "created_at": workflow.checkpoints[-1].created_at
            } if workflow.checkpoints else None,
            "created_at": workflow.created_at,
            "updated_at": workflow.updated_at
        }
    
    # Private methods for phase execution
    
    async def _execute_brief_analysis(self, workflow_id: str) -> None:
        """Execute brief analysis phase."""
        try:
            workflow = self.active_workflows[workflow_id]
            logger.info(f"🔄 Executing brief analysis for workflow: {workflow_id}")
            
            # Analyze brief using AI
            analysis_result = await self.brief_analyzer.analyze_brief(
                workflow.brief_content, workflow.topic
            )
            
            # Create checkpoint for approval
            checkpoint = WorkflowCheckpoint(
                id=f"checkpoint_{uuid.uuid4().hex[:8]}",
                phase=WorkflowPhase.BRIEF_ANALYSIS,
                title="Brief Analysis Complete",
                description="AI has analyzed your brief. Please review the understanding before proceeding.",
                content={
                    "message": f"I've analyzed your brief about '{workflow.topic}'. Here's my understanding:\n\n" +
                              f"**Main Topic**: {analysis_result.get('main_topic', 'Unknown')}\n" +
                              f"**Key Themes**: {', '.join(analysis_result.get('key_themes', []))}\n" +
                              f"**Target Audience**: {analysis_result.get('target_audience', 'General')}\n" +
                              f"**Content Type**: {analysis_result.get('content_type', 'Article')}\n" +
                              f"**Complexity**: {analysis_result.get('estimated_complexity', 'Medium')}\n\n" +
                              "Would you like me to proceed with creating a content plan based on this analysis?",
                    "analysis_data": analysis_result
                }
            )
            
            workflow.checkpoints.append(checkpoint)
            workflow.current_phase = WorkflowPhase.BRIEF_ANALYSIS
            workflow.updated_at = datetime.now()
            
            logger.info(f"✅ Brief analysis checkpoint created for workflow: {workflow_id}")
            
        except Exception as e:
            logger.error(f"Brief analysis failed for workflow {workflow_id}: {e}")
            raise
    
    async def _execute_planning_phase(self, workflow_id: str) -> None:
        """Execute content planning phase."""
        try:
            workflow = self.active_workflows[workflow_id]
            logger.info(f"🔄 Executing planning phase for workflow: {workflow_id}")
            
            # Create content plan using AI
            planning_result = await self.content_planner.create_content_plan(workflow)
            workflow.planning_result = planning_result
            
            # Create checkpoint for approval
            strategy = planning_result.get('content_strategy', {})
            sections = planning_result.get('sections', [])
            seo_strategy = planning_result.get('seo_strategy', {})
            
            checkpoint = WorkflowCheckpoint(
                id=f"checkpoint_{uuid.uuid4().hex[:8]}",
                phase=WorkflowPhase.PLANNING,
                title="Content Strategy Ready",
                description="AI has created a comprehensive content strategy. Please review before proceeding.",
                content={
                    "message": f"I've created a content strategy for '{workflow.topic}':\n\n" +
                              f"**Primary Goal**: {strategy.get('primary_goal', 'Not specified')}\n" +
                              f"**Approach**: {strategy.get('approach', 'Standard approach')}\n" +
                              f"**Tone**: {strategy.get('tone', 'Professional')}\n" +
                              f"**Sections Planned**: {len(sections)} sections\n" +
                              f"**Estimated Words**: {planning_result.get('estimated_total_words', 'Unknown')}\n" +
                              f"**SEO Focus**: {seo_strategy.get('primary_keywords', [])}\n\n" +
                              "Does this strategy align with your vision? Approve to continue to requirements definition.",
                    "planning_data": planning_result
                }
            )
            
            workflow.checkpoints.append(checkpoint)
            workflow.current_phase = WorkflowPhase.PLANNING
            workflow.updated_at = datetime.now()
            
            logger.info(f"✅ Planning checkpoint created for workflow: {workflow_id}")
            
        except Exception as e:
            logger.error(f"Planning phase failed for workflow {workflow_id}: {e}")
            raise
    
    async def _execute_requirements_phase(self, workflow_id: str) -> None:
        """Execute requirements definition phase."""
        try:
            workflow = self.active_workflows[workflow_id]
            logger.info(f"🔄 Executing requirements phase for workflow: {workflow_id}")
            
            # Define requirements using AI
            requirements_result = await self.requirements_definer.define_content_requirements(workflow)
            workflow.requirements_result = requirements_result
            
            # Create checkpoint for approval
            length_req = requirements_result.get('length_requirements', {})
            seo_req = requirements_result.get('seo_requirements', {})
            brand_req = requirements_result.get('brand_requirements', {})
            quality_req = requirements_result.get('quality_standards', {})
            
            checkpoint = WorkflowCheckpoint(
                id=f"checkpoint_{uuid.uuid4().hex[:8]}",
                phase=WorkflowPhase.REQUIREMENTS,
                title="Content Requirements Defined",
                description="AI has defined specific content requirements. Please review before proceeding.",
                content={
                    "message": f"I've defined specific requirements for '{workflow.topic}':\n\n" +
                              f"**Target Length**: {length_req.get('target_words', 'Unknown')} words\n" +
                              f"**SEO Focus**: {', '.join(seo_req.get('primary_keywords', []))}\n" +
                              f"**Brand Tone**: {brand_req.get('tone', 'Professional')}\n" +
                              f"**Quality Level**: {quality_req.get('expertise_level', 'Intermediate')}\n" +
                              f"**Readability Target**: {quality_req.get('readability_target', 'Grade 8-10')}\n\n" +
                              "These requirements will guide the content creation process. Approve to continue to process definition.",
                    "requirements_data": requirements_result
                }
            )
            
            workflow.checkpoints.append(checkpoint)
            workflow.current_phase = WorkflowPhase.REQUIREMENTS
            workflow.updated_at = datetime.now()
            
            logger.info(f"✅ Requirements checkpoint created for workflow: {workflow_id}")
            
        except Exception as e:
            logger.error(f"Requirements phase failed for workflow {workflow_id}: {e}")
            raise
    
    async def _execute_process_phase(self, workflow_id: str) -> None:
        """Execute process definition phase."""
        try:
            workflow = self.active_workflows[workflow_id]
            logger.info(f"🔄 Executing process phase for workflow: {workflow_id}")
            
            # Define process using AI
            process_result = await self.process_definer.define_content_process(workflow)
            workflow.process_result = process_result
            
            # Create checkpoint for approval
            timeline = process_result.get('timeline_estimate', {})
            approach = process_result.get('generation_approach', 'Standard approach')
            quality_checks = process_result.get('quality_checks', [])
            
            checkpoint = WorkflowCheckpoint(
                id=f"checkpoint_{uuid.uuid4().hex[:8]}",
                phase=WorkflowPhase.PROCESS,
                title="Content Process Defined",
                description="AI has defined the optimal content creation process. Review before content generation.",
                content={
                    "message": f"I've defined the optimal process for creating '{workflow.topic}':\n\n" +
                              f"**Generation Approach**: {approach}\n" +
                              f"**Estimated Timeline**: {timeline.get('total_hours', 'Unknown')} hours\n" +
                              f"**Quality Checks**: {len(quality_checks)} checks planned\n" +
                              f"**Research Level**: {process_result.get('research_process', {}).get('fact_verification', 'Standard')}\n\n" +
                              "This process will ensure high-quality content creation. Approve to begin content generation.",
                    "process_data": process_result
                }
            )
            
            workflow.checkpoints.append(checkpoint)
            workflow.current_phase = WorkflowPhase.PROCESS
            workflow.updated_at = datetime.now()
            
            logger.info(f"✅ Process checkpoint created for workflow: {workflow_id}")
            
        except Exception as e:
            logger.error(f"Process phase failed for workflow {workflow_id}: {e}")
            raise
    
    async def _execute_generation_phase(self, workflow_id: str) -> None:
        """Execute content generation phase."""
        try:
            workflow = self.active_workflows[workflow_id]
            logger.info(f"🔄 Executing generation phase for workflow: {workflow_id}")
            
            # Generate content using AI
            generation_result = await self.content_generator.generate_content(workflow)
            workflow.generation_result = generation_result
            
            # Create checkpoint for approval
            word_count = generation_result.get('word_count', 0)
            seo_score = generation_result.get('seo_score', 0)
            readability_score = generation_result.get('readability_score', 0)
            
            checkpoint = WorkflowCheckpoint(
                id=f"checkpoint_{uuid.uuid4().hex[:8]}",
                phase=WorkflowPhase.GENERATION,
                title="Content Generated",
                description="AI has generated your content. Please review the quality before final review.",
                content={
                    "message": f"I've generated content for '{workflow.topic}':\n\n" +
                              f"**Word Count**: {word_count} words\n" +
                              f"**SEO Score**: {seo_score}/100\n" +
                              f"**Readability Score**: {readability_score}/100\n" +
                              f"**Internal Links**: {generation_result.get('internal_links', 0)}\n" +
                              f"**Knowledge Sources**: {len(generation_result.get('knowledge_sources', []))}\n\n" +
                              "The content is ready for your review. Approve to proceed to final quality assessment.",
                    "generation_data": generation_result,
                    "content_preview": generation_result.get('content', '')[:500] + "..." if generation_result.get('content', '') else "No content generated"
                }
            )
            
            workflow.checkpoints.append(checkpoint)
            workflow.current_phase = WorkflowPhase.GENERATION
            workflow.updated_at = datetime.now()
            
            logger.info(f"✅ Generation checkpoint created for workflow: {workflow_id}")
            
        except Exception as e:
            logger.error(f"Generation phase failed for workflow {workflow_id}: {e}")
            raise
    
    async def _execute_review_phase(self, workflow_id: str) -> None:
        """Execute final review phase."""
        try:
            workflow = self.active_workflows[workflow_id]
            logger.info(f"🔄 Executing review phase for workflow: {workflow_id}")
            
            # Perform final review using AI
            review_result = await self.final_reviewer.perform_final_review(workflow)
            
            # Mark workflow as complete
            workflow.current_phase = WorkflowPhase.COMPLETE
            workflow.completed_at = datetime.now()
            workflow.updated_at = datetime.now()
            
            # Create final checkpoint
            overall_assessment = review_result.get('overall_assessment', {})
            final_score = overall_assessment.get('final_score', 0)
            readiness = overall_assessment.get('readiness', 'unknown')
            
            checkpoint = WorkflowCheckpoint(
                id=f"checkpoint_{uuid.uuid4().hex[:8]}",
                phase=WorkflowPhase.REVIEW,
                title="Final Review Complete",
                description="AI has completed comprehensive quality assessment. Workflow finished.",
                content={
                    "message": f"Final review completed for '{workflow.topic}':\n\n" +
                              f"**Overall Score**: {final_score}/100\n" +
                              f"**Readiness**: {readiness.replace('_', ' ').title()}\n" +
                              f"**SEO Analysis**: {review_result.get('seo_analysis', {}).get('seo_score', 'Unknown')}/100\n" +
                              f"**Brand Compliance**: {review_result.get('brand_compliance', {}).get('compliance_score', 'Unknown')}/100\n" +
                              f"**Content Quality**: {review_result.get('content_quality', {}).get('readability_score', 'Unknown')}/100\n\n" +
                              "Your content is ready! All phases of the PRP workflow have been completed successfully.",
                    "review_data": review_result
                },
                status=CheckpointStatus.APPROVED  # Auto-approve final review
            )
            
            workflow.checkpoints.append(checkpoint)
            
            logger.info(f"✅ Review phase completed for workflow: {workflow_id}")
            
        except Exception as e:
            logger.error(f"Review phase failed for workflow {workflow_id}: {e}")
            raise
    
    async def _execute_storage_phase(self, workflow_id: str):
        """Phase 7: Store generated content to Knowledge Base."""
        workflow = self.active_workflows[workflow_id]
        workflow.current_phase = WorkflowPhase.STORAGE
        
        try:
            logger.info(f"🗄️ CRITICAL: Starting content storage for workflow {workflow_id}")
            logger.info(f"📊 Current workflow state: phase={workflow.current_phase}, checkpoints={len(workflow.checkpoints)}")
            logger.info(f"📄 Generation result available: {bool(workflow.generation_result)}")
            
            # Store the generated content using the content storage service
            storage_result = await self._store_generated_content(workflow)
            
            if storage_result.get("success"):
                # Mark workflow as complete
                workflow.current_phase = WorkflowPhase.COMPLETE
                workflow.completed_at = datetime.now()
                
                # Create final success checkpoint - AUTO-APPROVED
                checkpoint = WorkflowCheckpoint(
                    id=f"{workflow_id}_complete",
                    phase=WorkflowPhase.COMPLETE,
                    title="Content Saved Successfully!",
                    description="Your content has been saved to the Knowledge Base and is ready for use.",
                    content={
                        "message": "🎉 **Content Successfully Saved!**\n\nYour generated content has been stored in the Knowledge Base and is now available for:\n\n✅ **Knowledge Graph Integration** - Connected to related topics and concepts\n✅ **Vector Search** - Findable through semantic similarity\n✅ **RAG Enhancement** - Will improve future content generation\n✅ **Content Library** - Accessible in Recent Content and Knowledge Base\n\nThe content is immediately searchable and will enhance all future AI responses.",
                        "storage_result": storage_result,
                        "final_content": workflow.generation_result,
                        "content_location": f"/knowledge-base/content/{storage_result.get('content_id')}",
                        "metadata": {
                            "word_count": len(workflow.generation_result.get("content", "").split()) if workflow.generation_result else 0,
                            "content_id": storage_result.get("content_id"),
                            "stored_at": datetime.now().isoformat(),
                            "searchable": True,
                            "available_in_rag": True
                        },
                        "next_steps": [
                            "Content is now searchable in your Knowledge Base",
                            "Available for RAG-powered content generation",
                            "Can be referenced in future content creation"
                        ]
                    },
                    status=CheckpointStatus.APPROVED  # Auto-approve COMPLETE checkpoint
                )
                
                workflow.checkpoints.append(checkpoint)
                logger.info(f"✅ Workflow {workflow_id} completed successfully - Content stored with ID: {storage_result.get('content_id')}")
                logger.info(f"🎯 CRITICAL: Created COMPLETE checkpoint {checkpoint.id} - should be returned to frontend")
                
            else:
                # Storage failed - create error checkpoint
                checkpoint = WorkflowCheckpoint(
                    id=f"{workflow_id}_storage_error",
                    phase=WorkflowPhase.STORAGE,
                    title="Content Storage Failed",
                    description="There was an issue saving your content. The content is still available in this workflow.",
                    content={
                        "error": storage_result.get("error", "Unknown storage error"),
                        "final_content": workflow.generation_result,
                        "retry_available": True,
                        "manual_copy_available": True
                    }
                )
                
                workflow.checkpoints.append(checkpoint)
                logger.error(f"❌ Content storage failed for workflow {workflow_id}: {storage_result.get('error')}")
                
        except Exception as e:
            logger.error(f"❌ Exception during content storage for workflow {workflow_id}: {e}")
            
            # Create error checkpoint
            checkpoint = WorkflowCheckpoint(
                id=f"{workflow_id}_storage_exception",
                phase=WorkflowPhase.STORAGE,
                title="Content Storage Error",
                description="An unexpected error occurred while saving your content.",
                content={
                    "error": str(e),
                    "final_content": workflow.generation_result,
                    "retry_available": True
                }
            )
            
            workflow.checkpoints.append(checkpoint)
    
    async def _store_generated_content(self, workflow: 'PRPWorkflowState') -> Dict[str, Any]:
        """Store the generated content to the Knowledge Base using the content storage service."""
        try:
            logger.info(f"📚 Starting content storage for workflow {workflow.workflow_id}")
            
            if not workflow.generation_result:
                return {
                    "success": False,
                    "error": "No generated content available for storage"
                }
            
            generated_content = workflow.generation_result
            content_text = generated_content.get("content", "")
            
            if not content_text or len(content_text.strip()) == 0:
                return {
                    "success": False,
                    "error": "Generated content is empty"
                }
            
            # Import the content storage function
            from web.api.content_storage import store_content
            
            # Prepare content data for storage
            filename = f"{workflow.topic}.md"
            content_type = "text/markdown"
            
            # Create analysis result from workflow data
            analysis_result = {
                "word_count": len(content_text.split()),
                "analysis": {
                    "seo_metrics": {
                        "overall_seo_score": generated_content.get("seo_score", 75),
                        "readability_score": generated_content.get("readability_score", 80)
                    }
                },
                "extracted_topics": generated_content.get("related_topics", []),
                "recommendations": generated_content.get("improvement_suggestions", []),
                "summary": f"AI-generated content about {workflow.topic} created via PRP workflow",
                "keywords": [workflow.topic] + generated_content.get("related_topics", [])[:5],
                "workflow_metadata": {
                    "workflow_id": workflow.workflow_id,
                    "created_via": "prp_workflow",
                    "checkin_frequency": workflow.checkin_frequency,
                    "brand_voice": workflow.brand_voice,
                    "content_goals": workflow.content_goals,
                    "planning_result": getattr(workflow, 'planning_result', None),
                    "requirements_result": getattr(workflow, 'requirements_result', None),
                    "process_result": getattr(workflow, 'process_result', None)
                }
            }
            
            # Use user organization context from authentication
            current_user = {
                "org_id": "demo-org",  # TODO: Use real organization ID from authenticated user
                "user_id": "demo_user",
                "email": "user@demo.com"
            }
            
            # Store the content
            storage_success = await store_content(
                filename=filename,
                content_text=content_text,
                content_type=content_type,
                analysis_result=analysis_result,
                current_user=current_user
            )
            
            if storage_success:
                logger.info(f"✅ Content successfully stored for workflow {workflow.workflow_id}")
                return {
                    "success": True,
                    "content_id": f"prp_{workflow.workflow_id}",
                    "message": "Content stored successfully in Knowledge Base"
                }
            else:
                logger.error(f"❌ Content storage failed for workflow {workflow.workflow_id}")
                return {
                    "success": False,
                    "error": "Failed to store content in Knowledge Base"
                }
                
        except Exception as e:
            logger.error(f"Exception in content storage for workflow {workflow.workflow_id}: {e}")
            return {
                "success": False,
                "error": f"Storage exception: {str(e)}"
            }
    
    async def _process_checkpoint_response(self, workflow_id: str, checkpoint_id: str, 
                                         status: CheckpointStatus, feedback: str = None) -> Dict[str, Any]:
        """Process user response to a checkpoint."""
        if workflow_id not in self.active_workflows:
            return {"success": False, "error": "Workflow not found"}
        
        workflow = self.active_workflows[workflow_id]
        
        # Find and update checkpoint
        checkpoint = None
        for cp in workflow.checkpoints:
            if cp.id == checkpoint_id:
                checkpoint = cp
                break
        
        if not checkpoint:
            return {"success": False, "error": "Checkpoint not found"}
        
        checkpoint.status = status
        checkpoint.feedback = feedback
        checkpoint.reviewed_at = datetime.now()
        workflow.updated_at = datetime.now()
        
        logger.info(f"Checkpoint {checkpoint_id} in workflow {workflow_id} marked as {status}")
        
        # Continue workflow based on response
        if status == CheckpointStatus.APPROVED:
            await self._continue_workflow(workflow_id, checkpoint.phase)
        
        return {
            "success": True,
            "phase": checkpoint.phase,
            "status": status,
            "message": f"Checkpoint {status.value} successfully"
        }
    
    async def _continue_workflow(self, workflow_id: str, current_phase: WorkflowPhase) -> None:
        """Continue workflow to next phase."""
        try:
            # Determine next phase based on current phase
            if current_phase == WorkflowPhase.BRIEF_ANALYSIS:
                await self._execute_planning_phase(workflow_id)
            elif current_phase == WorkflowPhase.PLANNING:
                await self._execute_requirements_phase(workflow_id)
            elif current_phase == WorkflowPhase.REQUIREMENTS:
                await self._execute_process_phase(workflow_id)
            elif current_phase == WorkflowPhase.PROCESS:
                await self._execute_generation_phase(workflow_id)
            elif current_phase == WorkflowPhase.GENERATION:
                await self._execute_review_phase(workflow_id)
            elif current_phase == WorkflowPhase.REVIEW:
                logger.info(f"▶️ Transitioning: REVIEW → STORAGE (CRITICAL FIX)")
                await self._execute_storage_phase(workflow_id)
            elif current_phase == WorkflowPhase.STORAGE:
                logger.info(f"Workflow {workflow_id} completed - no further phases to execute")
            
        except Exception as e:
            logger.error(f"Failed to continue workflow {workflow_id} from phase {current_phase}: {e}")
            raise
    
    def _calculate_progress_percentage(self, workflow: PRPWorkflowState) -> int:
        """Calculate workflow progress percentage."""
        phase_weights = {
            WorkflowPhase.BRIEF_ANALYSIS: 15,
            WorkflowPhase.PLANNING: 35,
            WorkflowPhase.REQUIREMENTS: 50,
            WorkflowPhase.PROCESS: 65,
            WorkflowPhase.GENERATION: 85,
            WorkflowPhase.REVIEW: 90,
            WorkflowPhase.STORAGE: 95,
            WorkflowPhase.COMPLETE: 100
        }
        
        return phase_weights.get(workflow.current_phase, 0)