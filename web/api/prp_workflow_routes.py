"""
API routes for PRP-style content workflow management.

This module provides endpoints for managing multi-phase content creation
workflows with human-in-loop checkpoints.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

try:
    from src.services.prp_workflow_service import prp_workflow_service, CheckpointStatus
    PRP_WORKFLOW_AVAILABLE = True
except ImportError as e:
    PRP_WORKFLOW_AVAILABLE = False
    # Create a mock service for basic functionality
    class MockPRPService:
        def __init__(self):
            self.active_workflows = set()
        async def get_workflow_status(self, workflow_id):
            return None
    prp_workflow_service = MockPRPService()

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/prp-workflow", tags=["PRP Workflow"])


# Request/Response Models

class StartWorkflowRequest(BaseModel):
    """Request to start a new PRP workflow."""
    brief_content: str = Field(..., description="Content brief or requirements")
    topic: str = Field(..., description="Main topic for content creation")
    human_in_loop: Dict[str, Any] = Field(default_factory=dict, description="Human-in-loop settings")
    content_goals: Dict[str, Any] = Field(default_factory=dict, description="Content creation goals")
    brand_voice: Dict[str, Any] = Field(default_factory=dict, description="Brand voice configuration")


class CheckpointResponseRequest(BaseModel):
    """Request to respond to a workflow checkpoint."""
    workflow_id: str = Field(..., description="Workflow identifier")
    checkpoint_id: str = Field(..., description="Checkpoint identifier")
    status: str = Field(..., description="Response status: approved, rejected, modified")
    feedback: Optional[str] = Field(None, description="User feedback or modifications")


class WorkflowStatusResponse(BaseModel):
    """Response with workflow status."""
    success: bool
    workflow_id: Optional[str] = None
    current_phase: Optional[str] = None
    progress_percentage: Optional[int] = None
    pending_checkpoints: list = Field(default_factory=list)
    last_checkpoint: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@router.post("/start", response_model=WorkflowStatusResponse)
async def start_prp_workflow(request: StartWorkflowRequest, background_tasks: BackgroundTasks):
    """Start a new PRP-style content creation workflow."""
    
    if not PRP_WORKFLOW_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="PRP Workflow service not available"
        )
    
    try:
        logger.info(f"Starting PRP workflow for topic: {request.topic}")
        
        # Start the workflow
        workflow_id = await prp_workflow_service.start_workflow(
            brief_content=request.brief_content,
            topic=request.topic,
            human_in_loop_settings=request.human_in_loop,
            content_goals=request.content_goals,
            brand_voice=request.brand_voice
        )
        
        # Get initial status
        status = await prp_workflow_service.get_workflow_status(workflow_id)
        
        return WorkflowStatusResponse(
            success=True,
            workflow_id=workflow_id,
            current_phase=status.get("current_phase"),
            progress_percentage=status.get("progress_percentage"),
            pending_checkpoints=status.get("pending_checkpoints", []),
            last_checkpoint=status.get("last_checkpoint")
        )
        
    except Exception as e:
        logger.error(f"Error starting PRP workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start workflow: {str(e)}")


@router.get("/status/{workflow_id}", response_model=WorkflowStatusResponse)
async def get_workflow_status(workflow_id: str):
    """Get current status of a PRP workflow."""
    
    if not PRP_WORKFLOW_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="PRP Workflow service not available"
        )
    
    try:
        status = await prp_workflow_service.get_workflow_status(workflow_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return WorkflowStatusResponse(
            success=True,
            workflow_id=workflow_id,
            current_phase=status.get("current_phase"),
            progress_percentage=status.get("progress_percentage"),
            pending_checkpoints=status.get("pending_checkpoints", []),
            last_checkpoint=status.get("last_checkpoint")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/checkpoint/{workflow_id}/next")
async def get_next_checkpoint(workflow_id: str):
    """Get the next pending checkpoint for user review."""
    
    if not PRP_WORKFLOW_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="PRP Workflow service not available"
        )
    
    try:
        checkpoint = await prp_workflow_service.get_next_checkpoint(workflow_id)
        
        if not checkpoint:
            return {
                "success": True,
                "checkpoint": None,
                "message": "No pending checkpoints"
            }
        
        return {
            "success": True,
            "checkpoint": checkpoint
        }
        
    except Exception as e:
        logger.error(f"Error getting next checkpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get checkpoint: {str(e)}")


@router.post("/checkpoint/respond", response_model=WorkflowStatusResponse)
async def respond_to_checkpoint(request: CheckpointResponseRequest, background_tasks: BackgroundTasks):
    """Respond to a workflow checkpoint."""
    
    if not PRP_WORKFLOW_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="PRP Workflow service not available"
        )
    
    try:
        # Validate status
        valid_statuses = ["approved", "rejected", "modified"]
        if request.status not in valid_statuses:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid status. Must be one of: {valid_statuses}"
            )
        
        # Convert string status to enum
        status_mapping = {
            "approved": CheckpointStatus.APPROVED,
            "rejected": CheckpointStatus.REJECTED,
            "modified": CheckpointStatus.MODIFIED
        }
        
        checkpoint_status = status_mapping[request.status]
        
        logger.info(f"Processing checkpoint response: {request.workflow_id} - {request.checkpoint_id} - {request.status}")
        
        # Process the checkpoint response
        result = await prp_workflow_service.process_checkpoint_response(
            workflow_id=request.workflow_id,
            checkpoint_id=request.checkpoint_id,
            status=checkpoint_status,
            feedback=request.feedback
        )
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Checkpoint processing failed"))
        
        workflow_status = result.get("workflow_status", {})
        
        return WorkflowStatusResponse(
            success=True,
            workflow_id=request.workflow_id,
            current_phase=workflow_status.get("current_phase"),
            progress_percentage=workflow_status.get("progress_percentage"),
            pending_checkpoints=workflow_status.get("pending_checkpoints", []),
            last_checkpoint=workflow_status.get("last_checkpoint")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing checkpoint response: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process response: {str(e)}")


@router.get("/workflows/active")
async def get_active_workflows():
    """Get list of active workflows."""
    
    try:
        # Get active workflows from service
        active_workflows = []
        
        if PRP_WORKFLOW_AVAILABLE and hasattr(prp_workflow_service, 'active_workflows'):
            for workflow_id in prp_workflow_service.active_workflows:
                status = await prp_workflow_service.get_workflow_status(workflow_id)
                if status:
                    active_workflows.append({
                        "workflow_id": workflow_id,
                        "topic": status.get("topic"),
                        "current_phase": status.get("current_phase"),
                        "progress_percentage": status.get("progress_percentage"),
                        "created_at": status.get("created_at"),
                        "has_pending_checkpoints": len(status.get("pending_checkpoints", [])) > 0
                    })
        
        return {
            "success": True,
            "workflows": active_workflows,
            "count": len(active_workflows),
            "service_available": PRP_WORKFLOW_AVAILABLE
        }
        
    except Exception as e:
        logger.error(f"Error getting active workflows: {e}")
        # Return empty result instead of error for better UX
        return {
            "success": True,
            "workflows": [],
            "count": 0,
            "service_available": False,
            "error": str(e)
        }


@router.delete("/workflow/{workflow_id}")
async def cancel_workflow(workflow_id: str):
    """Cancel/delete a workflow."""
    
    if not PRP_WORKFLOW_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="PRP Workflow service not available"
        )
    
    try:
        if workflow_id in prp_workflow_service.active_workflows:
            del prp_workflow_service.active_workflows[workflow_id]
            logger.info(f"Cancelled workflow: {workflow_id}")
            
            return {
                "success": True,
                "message": f"Workflow {workflow_id} cancelled successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Workflow not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel workflow: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint for PRP workflow service."""
    return {
        "status": "healthy",
        "service": "PRP Workflow",
        "available": PRP_WORKFLOW_AVAILABLE,
        "active_workflows": len(prp_workflow_service.active_workflows) if PRP_WORKFLOW_AVAILABLE else 0
    }