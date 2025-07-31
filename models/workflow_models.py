"""
Workflow-related Pydantic models for the SEO Content Knowledge Graph System.

This module defines data models for content workflows, briefs, and 
human-in-the-loop processes.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator
import structlog

logger = structlog.get_logger(__name__)


class WorkflowStatus(str, Enum):
    """Content workflow status."""
    
    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    APPROVED = "approved"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowStepType(str, Enum):
    """Types of workflow steps."""
    
    AGENT_EXECUTION = "agent_execution"
    HUMAN_REVIEW = "human_review"
    VALIDATION = "validation"
    PUBLICATION = "publication"
    NOTIFICATION = "notification"
    INTEGRATION = "integration"


class ContentBriefType(str, Enum):
    """Types of content briefs."""
    
    BLOG_POST = "blog_post"
    ARTICLE = "article"
    LANDING_PAGE = "landing_page"
    PRODUCT_PAGE = "product_page"
    WHITEPAPER = "whitepaper"
    CASE_STUDY = "case_study"
    GUIDE = "guide"
    TUTORIAL = "tutorial"
    NEWS = "news"
    PRESS_RELEASE = "press_release"


class Priority(str, Enum):
    """Priority levels."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class ContentBrief(BaseModel):
    """Content brief with detailed requirements and specifications."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique brief identifier")
    title: str = Field(..., min_length=1, max_length=200, description="Brief title")
    
    # Content specifications
    content_type: ContentBriefType = Field(..., description="Type of content to create")
    target_keywords: List[str] = Field(..., min_items=1, description="Target keywords")
    target_audience: str = Field(..., description="Target audience description")
    tone: str = Field(..., description="Content tone and style")
    
    # Content requirements
    word_count: int = Field(..., ge=100, le=10000, description="Target word count")
    outline_requirements: List[str] = Field(default_factory=list, description="Outline requirements")
    key_points: List[str] = Field(default_factory=list, description="Key points to cover")
    
    # SEO requirements
    meta_description: Optional[str] = Field(None, max_length=160, description="Meta description")
    meta_keywords: List[str] = Field(default_factory=list, description="Meta keywords")
    internal_links: List[str] = Field(default_factory=list, description="Required internal links")
    
    # Resources and references
    reference_urls: List[str] = Field(default_factory=list, description="Reference URLs")
    competitor_analysis: Optional[str] = Field(None, description="Competitor analysis notes")
    brand_guidelines: Optional[str] = Field(None, description="Brand guideline notes")
    
    # Project management
    priority: Priority = Field(Priority.MEDIUM, description="Priority level")
    deadline: Optional[datetime] = Field(None, description="Content deadline")
    estimated_hours: Optional[float] = Field(None, ge=0.0, description="Estimated hours")
    
    # Google Drive integration
    google_drive_id: Optional[str] = Field(None, description="Google Drive file ID")
    google_drive_url: Optional[str] = Field(None, description="Google Drive URL")
    last_synced: Optional[datetime] = Field(None, description="Last sync timestamp")
    
    # Ownership and assignment
    tenant_id: str = Field(..., description="Tenant identifier")
    created_by: str = Field(..., description="User who created brief")
    assigned_to: Optional[str] = Field(None, description="Assigned user ID")
    
    # Status and tracking
    status: WorkflowStatus = Field(WorkflowStatus.DRAFT, description="Brief status")
    completion_percentage: float = Field(0.0, ge=0.0, le=100.0, description="Completion percentage")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    tags: List[str] = Field(default_factory=list, description="Content tags")
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    @validator('target_keywords')
    def validate_keywords(cls, v):
        """Validate and clean keywords."""
        return [kw.strip().lower() for kw in v if kw.strip()]
    
    @validator('word_count')
    def validate_word_count(cls, v, values):
        """Validate word count based on content type."""
        content_type = values.get('content_type')
        
        if content_type == ContentBriefType.BLOG_POST and v < 300:
            raise ValueError("Blog posts should be at least 300 words")
        elif content_type == ContentBriefType.WHITEPAPER and v < 1000:
            raise ValueError("Whitepapers should be at least 1000 words")
        elif content_type == ContentBriefType.LANDING_PAGE and v < 200:
            raise ValueError("Landing pages should be at least 200 words")
        
        return v
    
    def update_status(self, new_status: WorkflowStatus) -> None:
        """Update status and timestamp."""
        self.status = new_status
        self.updated_at = datetime.now(timezone.utc)
    
    def calculate_completion_percentage(self) -> float:
        """Calculate completion percentage based on filled fields."""
        total_fields = 0
        completed_fields = 0
        
        # Required fields
        required_fields = [
            'title', 'content_type', 'target_keywords', 'target_audience', 
            'tone', 'word_count'
        ]
        
        for field in required_fields:
            total_fields += 1
            if getattr(self, field, None):
                completed_fields += 1
        
        # Optional but important fields
        optional_fields = [
            'outline_requirements', 'key_points', 'meta_description',
            'reference_urls', 'deadline'
        ]
        
        for field in optional_fields:
            total_fields += 1
            value = getattr(self, field, None)
            if value and (isinstance(value, list) and len(value) > 0 or value):
                completed_fields += 1
        
        self.completion_percentage = (completed_fields / total_fields) * 100
        return self.completion_percentage


class WorkflowStep(BaseModel):
    """Individual workflow step."""
    
    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Step name")
    description: Optional[str] = Field(None, description="Step description")
    
    step_type: WorkflowStepType = Field(..., description="Type of step")
    order: int = Field(..., ge=0, description="Step order")
    
    # Configuration
    config: Dict[str, Any] = Field(default_factory=dict, description="Step configuration")
    dependencies: List[str] = Field(default_factory=list, description="Dependent step IDs")
    
    # Execution
    status: WorkflowStatus = Field(WorkflowStatus.DRAFT, description="Step status")
    assigned_to: Optional[str] = Field(None, description="Assigned user/agent")
    
    # Results
    result: Optional[Dict[str, Any]] = Field(None, description="Step execution result")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    
    # Timing
    estimated_duration: Optional[float] = Field(None, ge=0.0, description="Estimated duration in hours")
    actual_duration: Optional[float] = Field(None, ge=0.0, description="Actual duration in hours")
    
    started_at: Optional[datetime] = Field(None, description="Step start time")
    completed_at: Optional[datetime] = Field(None, description="Step completion time")
    
    def start_step(self) -> None:
        """Mark step as started."""
        self.status = WorkflowStatus.IN_PROGRESS
        self.started_at = datetime.now(timezone.utc)
    
    def complete_step(self, result: Optional[Dict[str, Any]] = None) -> None:
        """Mark step as completed."""
        self.status = WorkflowStatus.APPROVED
        self.completed_at = datetime.now(timezone.utc)
        
        if result:
            self.result = result
        
        # Calculate actual duration
        if self.started_at:
            duration = (self.completed_at - self.started_at).total_seconds() / 3600
            self.actual_duration = duration
    
    def fail_step(self, error_message: str) -> None:
        """Mark step as failed."""
        self.status = WorkflowStatus.FAILED
        self.error_message = error_message
        self.completed_at = datetime.now(timezone.utc)


class ContentWorkflow(BaseModel):
    """Content workflow with multiple steps and state management."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique workflow identifier")
    name: str = Field(..., description="Workflow name")
    description: Optional[str] = Field(None, description="Workflow description")
    
    # Associated content
    brief_id: str = Field(..., description="Associated content brief ID")
    content_id: Optional[str] = Field(None, description="Generated content ID")
    
    # Workflow configuration
    workflow_type: str = Field(..., description="Type of workflow")
    template_id: Optional[str] = Field(None, description="Workflow template ID")
    
    # Steps and execution
    steps: List[WorkflowStep] = Field(default_factory=list, description="Workflow steps")
    current_step: int = Field(0, ge=0, description="Current step index")
    
    # Assignment and review
    assigned_agents: List[str] = Field(default_factory=list, description="Assigned AI agents")
    human_reviewers: List[str] = Field(default_factory=list, description="Human reviewers")
    
    # Status and progress
    status: WorkflowStatus = Field(WorkflowStatus.DRAFT, description="Workflow status")
    progress_percentage: float = Field(0.0, ge=0.0, le=100.0, description="Progress percentage")
    
    # Timing
    estimated_duration: Optional[float] = Field(None, ge=0.0, description="Estimated total duration")
    actual_duration: Optional[float] = Field(None, ge=0.0, description="Actual duration")
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    started_at: Optional[datetime] = Field(None, description="Workflow start time")
    completed_at: Optional[datetime] = Field(None, description="Workflow completion time")
    
    # Metadata
    tenant_id: str = Field(..., description="Tenant identifier")
    created_by: str = Field(..., description="User who created workflow")
    
    def add_step(self, step: WorkflowStep) -> None:
        """Add a step to the workflow."""
        step.order = len(self.steps)
        self.steps.append(step)
        self.updated_at = datetime.now(timezone.utc)
    
    def get_current_step(self) -> Optional[WorkflowStep]:
        """Get the current step."""
        if 0 <= self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return None
    
    def advance_step(self) -> bool:
        """Advance to the next step."""
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            self.updated_at = datetime.now(timezone.utc)
            self.calculate_progress()
            return True
        return False
    
    def calculate_progress(self) -> float:
        """Calculate workflow progress percentage."""
        if not self.steps:
            self.progress_percentage = 0.0
            return 0.0
        
        completed_steps = sum(1 for step in self.steps if step.status == WorkflowStatus.APPROVED)
        self.progress_percentage = (completed_steps / len(self.steps)) * 100
        return self.progress_percentage
    
    def start_workflow(self) -> None:
        """Start the workflow."""
        self.status = WorkflowStatus.IN_PROGRESS
        self.started_at = datetime.now(timezone.utc)
        self.updated_at = self.started_at
        
        # Start first step if available
        if self.steps:
            self.steps[0].start_step()
    
    def complete_workflow(self) -> None:
        """Complete the workflow."""
        self.status = WorkflowStatus.APPROVED
        self.completed_at = datetime.now(timezone.utc)
        self.updated_at = self.completed_at
        self.progress_percentage = 100.0
        
        # Calculate total duration
        if self.started_at:
            duration = (self.completed_at - self.started_at).total_seconds() / 3600
            self.actual_duration = duration
    
    def fail_workflow(self, error_message: str) -> None:
        """Fail the workflow."""
        self.status = WorkflowStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
        self.updated_at = self.completed_at
        
        # Mark current step as failed
        current_step = self.get_current_step()
        if current_step:
            current_step.fail_step(error_message)
    
    def get_next_pending_step(self) -> Optional[WorkflowStep]:
        """Get the next pending step."""
        for step in self.steps:
            if step.status == WorkflowStatus.DRAFT:
                return step
        return None
    
    def get_failed_steps(self) -> List[WorkflowStep]:
        """Get all failed steps."""
        return [step for step in self.steps if step.status == WorkflowStatus.FAILED]
    
    def is_complete(self) -> bool:
        """Check if workflow is complete."""
        return self.status == WorkflowStatus.APPROVED and all(
            step.status == WorkflowStatus.APPROVED for step in self.steps
        )
    
    def can_advance(self) -> bool:
        """Check if workflow can advance to next step."""
        current_step = self.get_current_step()
        if not current_step:
            return False
        
        # Check if current step is complete
        if current_step.status != WorkflowStatus.APPROVED:
            return False
        
        # Check dependencies for next step
        if self.current_step + 1 < len(self.steps):
            next_step = self.steps[self.current_step + 1]
            
            # Check if all dependencies are satisfied
            for dep_id in next_step.dependencies:
                dep_step = next((s for s in self.steps if s.step_id == dep_id), None)
                if not dep_step or dep_step.status != WorkflowStatus.APPROVED:
                    return False
        
        return True


class WorkflowTemplate(BaseModel):
    """Template for creating standardized workflows."""
    
    template_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Template name")
    description: Optional[str] = Field(None, description="Template description")
    
    # Template configuration
    category: str = Field(..., description="Template category")
    content_types: List[ContentBriefType] = Field(default_factory=list, description="Applicable content types")
    
    # Template steps
    step_templates: List[Dict[str, Any]] = Field(default_factory=list, description="Step templates")
    
    # Settings
    default_reviewers: List[str] = Field(default_factory=list, description="Default reviewers")
    default_agents: List[str] = Field(default_factory=list, description="Default agents")
    estimated_duration: Optional[float] = Field(None, ge=0.0, description="Estimated duration")
    
    # Metadata
    tenant_id: str = Field(..., description="Tenant identifier")
    created_by: str = Field(..., description="Creator user ID")
    is_active: bool = Field(True, description="Template is active")
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    def create_workflow(self, brief_id: str, name: str, created_by: str) -> ContentWorkflow:
        """Create a workflow from this template."""
        workflow = ContentWorkflow(
            name=name,
            brief_id=brief_id,
            workflow_type=self.category,
            template_id=self.template_id,
            tenant_id=self.tenant_id,
            created_by=created_by,
            assigned_agents=self.default_agents.copy(),
            human_reviewers=self.default_reviewers.copy(),
            estimated_duration=self.estimated_duration
        )
        
        # Create steps from templates
        for i, step_template in enumerate(self.step_templates):
            step = WorkflowStep(
                name=step_template.get('name', f'Step {i+1}'),
                description=step_template.get('description'),
                step_type=WorkflowStepType(step_template.get('step_type', 'agent_execution')),
                order=i,
                config=step_template.get('config', {}),
                dependencies=step_template.get('dependencies', []),
                assigned_to=step_template.get('assigned_to'),
                estimated_duration=step_template.get('estimated_duration')
            )
            workflow.add_step(step)
        
        return workflow


class WorkflowNotification(BaseModel):
    """Workflow notification for users."""
    
    notification_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = Field(..., description="Associated workflow ID")
    
    # Notification details
    recipient: str = Field(..., description="Recipient user ID")
    notification_type: str = Field(..., description="Type of notification")
    title: str = Field(..., description="Notification title")
    message: str = Field(..., description="Notification message")
    
    # Status
    is_read: bool = Field(False, description="Has been read")
    is_sent: bool = Field(False, description="Has been sent")
    
    # Metadata
    tenant_id: str = Field(..., description="Tenant identifier")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    read_at: Optional[datetime] = Field(None, description="Read timestamp")
    sent_at: Optional[datetime] = Field(None, description="Sent timestamp")
    
    def mark_as_read(self) -> None:
        """Mark notification as read."""
        self.is_read = True
        self.read_at = datetime.now(timezone.utc)
    
    def mark_as_sent(self) -> None:
        """Mark notification as sent."""
        self.is_sent = True
        self.sent_at = datetime.now(timezone.utc)


# =============================================================================
# Request/Response Models
# =============================================================================

class CreateContentBriefRequest(BaseModel):
    """Request to create a new content brief."""
    
    title: str = Field(..., min_length=1, max_length=200)
    content_type: ContentBriefType = Field(...)
    target_keywords: List[str] = Field(..., min_items=1)
    target_audience: str = Field(...)
    tone: str = Field(...)
    word_count: int = Field(..., ge=100, le=10000)
    
    # Optional fields
    outline_requirements: Optional[List[str]] = Field(None)
    key_points: Optional[List[str]] = Field(None)
    meta_description: Optional[str] = Field(None, max_length=160)
    reference_urls: Optional[List[str]] = Field(None)
    deadline: Optional[datetime] = Field(None)
    priority: Optional[Priority] = Field(Priority.MEDIUM)
    assigned_to: Optional[str] = Field(None)
    tags: Optional[List[str]] = Field(None)


class UpdateContentBriefRequest(BaseModel):
    """Request to update a content brief."""
    
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    content_type: Optional[ContentBriefType] = Field(None)
    target_keywords: Optional[List[str]] = Field(None)
    target_audience: Optional[str] = Field(None)
    tone: Optional[str] = Field(None)
    word_count: Optional[int] = Field(None, ge=100, le=10000)
    
    outline_requirements: Optional[List[str]] = Field(None)
    key_points: Optional[List[str]] = Field(None)
    meta_description: Optional[str] = Field(None, max_length=160)
    reference_urls: Optional[List[str]] = Field(None)
    deadline: Optional[datetime] = Field(None)
    priority: Optional[Priority] = Field(None)
    assigned_to: Optional[str] = Field(None)
    status: Optional[WorkflowStatus] = Field(None)
    tags: Optional[List[str]] = Field(None)


class CreateWorkflowRequest(BaseModel):
    """Request to create a new workflow."""
    
    name: str = Field(..., description="Workflow name")
    brief_id: str = Field(..., description="Associated brief ID")
    workflow_type: str = Field(..., description="Workflow type")
    template_id: Optional[str] = Field(None, description="Template ID")
    
    assigned_agents: Optional[List[str]] = Field(None)
    human_reviewers: Optional[List[str]] = Field(None)
    custom_steps: Optional[List[Dict[str, Any]]] = Field(None)


class WorkflowExecutionRequest(BaseModel):
    """Request to execute a workflow step."""
    
    workflow_id: str = Field(..., description="Workflow ID")
    step_id: Optional[str] = Field(None, description="Specific step ID")
    action: str = Field(..., description="Action to perform")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Action parameters")


if __name__ == "__main__":
    # Example usage
    brief = ContentBrief(
        title="Complete Guide to Content Marketing",
        content_type=ContentBriefType.GUIDE,
        target_keywords=["content marketing", "digital marketing", "content strategy"],
        target_audience="Marketing professionals and business owners",
        tone="Professional and informative",
        word_count=2000,
        priority=Priority.HIGH,
        tenant_id="example-tenant",
        created_by="user-123"
    )
    
    print(f"Created brief: {brief.title}")
    print(f"Completion: {brief.calculate_completion_percentage():.1f}%")
    
    # Create workflow
    workflow = ContentWorkflow(
        name="Content Creation Workflow",
        brief_id=brief.id,
        workflow_type="content_creation",
        tenant_id="example-tenant",
        created_by="user-123"
    )
    
    # Add steps
    research_step = WorkflowStep(
        name="Research and Analysis",
        step_type=WorkflowStepType.AGENT_EXECUTION,
        order=0,
        assigned_to="research_agent"
    )
    
    writing_step = WorkflowStep(
        name="Content Writing",
        step_type=WorkflowStepType.AGENT_EXECUTION,
        order=1,
        assigned_to="writing_agent",
        dependencies=[research_step.step_id]
    )
    
    review_step = WorkflowStep(
        name="Human Review",
        step_type=WorkflowStepType.HUMAN_REVIEW,
        order=2,
        assigned_to="reviewer-123",
        dependencies=[writing_step.step_id]
    )
    
    workflow.add_step(research_step)
    workflow.add_step(writing_step)
    workflow.add_step(review_step)
    
    print(f"Created workflow: {workflow.name}")
    print(f"Steps: {len(workflow.steps)}")
    print(f"Progress: {workflow.calculate_progress():.1f}%")