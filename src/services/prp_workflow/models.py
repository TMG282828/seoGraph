"""
PRP Workflow Models and Data Structures.

Defines the data models used throughout the PRP workflow system,
including workflow states, checkpoints, and phase definitions.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class WorkflowPhase(str, Enum):
    """Workflow phases for content creation."""
    BRIEF_ANALYSIS = "brief_analysis"
    PLANNING = "planning"
    REQUIREMENTS = "requirements" 
    PROCESS = "process"
    GENERATION = "generation"
    REVIEW = "review"
    STORAGE = "storage"
    OPTIMIZATION = "optimization"
    COMPLETE = "complete"


class CheckpointStatus(str, Enum):
    """Status of workflow checkpoints."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"


class WorkflowCheckpoint(BaseModel):
    """Represents a human-in-loop checkpoint."""
    id: str
    phase: WorkflowPhase
    title: str
    description: str
    content: Dict[str, Any]
    status: CheckpointStatus = CheckpointStatus.PENDING
    feedback: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    reviewed_at: Optional[datetime] = None


class PRPWorkflowState(BaseModel):
    """Complete state of a PRP workflow instance."""
    workflow_id: str
    current_phase: WorkflowPhase = WorkflowPhase.BRIEF_ANALYSIS
    brief_content: str
    topic: str
    
    # Human-in-Loop Settings
    checkin_frequency: str = "medium"  # high, medium, low
    agent_aggressiveness: int = 5
    require_approval: bool = True
    
    # Content Goals and Brand Voice
    content_goals: Dict[str, Any] = Field(default_factory=dict)
    brand_voice: Dict[str, Any] = Field(default_factory=dict)
    
    # Phase Results
    planning_result: Optional[Dict[str, Any]] = None
    requirements_result: Optional[Dict[str, Any]] = None
    process_result: Optional[Dict[str, Any]] = None
    generation_result: Optional[Dict[str, Any]] = None
    
    # Checkpoints
    checkpoints: List[WorkflowCheckpoint] = Field(default_factory=list)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class ContentWorkflow(BaseModel):
    """Simple workflow model for API compatibility."""
    id: str
    status: str = "active"
    created_at: datetime = Field(default_factory=datetime.now)


class WorkflowConfig(BaseModel):
    """Configuration for workflow creation."""
    user_message: str
    brief_content: Optional[str] = None
    brief_summary: Dict[str, Any] = Field(default_factory=dict)
    brand_voice: Dict[str, Any] = Field(default_factory=dict)
    content_goals: Dict[str, Any] = Field(default_factory=dict)
    human_in_loop: Dict[str, Any] = Field(default_factory=dict)


class PhaseResult(BaseModel):
    """Result from executing a workflow phase."""
    phase: WorkflowPhase
    success: bool
    content: Dict[str, Any]
    analysis: Optional[Dict[str, Any]] = None
    recommendations: List[str] = Field(default_factory=list)
    processing_time: Optional[float] = None
    error_message: Optional[str] = None