"""
Tests for the Workflow Models module.
"""

import pytest
from datetime import datetime, timezone
from enum import Enum

from models.workflow_models import (
    WorkflowStatus, 
    ContentBrief, 
    ContentWorkflow,
    WorkflowStep
)


class TestWorkflowStatus:
    """Test suite for WorkflowStatus enum."""

    def test_workflow_status_values(self):
        """Test that WorkflowStatus has expected values."""
        assert WorkflowStatus.PENDING == "pending"
        assert WorkflowStatus.IN_PROGRESS == "in_progress"
        assert WorkflowStatus.COMPLETED == "completed"
        assert WorkflowStatus.FAILED == "failed"
        assert WorkflowStatus.CANCELLED == "cancelled"

    def test_workflow_status_is_enum(self):
        """Test that WorkflowStatus is an enum."""
        assert issubclass(WorkflowStatus, Enum)


class TestContentBrief:
    """Test suite for ContentBrief model."""

    def test_content_brief_creation(self):
        """Test creation of ContentBrief with valid data."""
        brief = ContentBrief(
            title="Test Article",
            description="Test description",
            keywords=["test", "article", "seo"],
            target_audience="developers",
            content_type="article",
            word_count=1000
        )
        
        assert brief.title == "Test Article"
        assert brief.description == "Test description"
        assert brief.keywords == ["test", "article", "seo"]
        assert brief.target_audience == "developers"
        assert brief.content_type == "article"
        assert brief.word_count == 1000

    def test_content_brief_defaults(self):
        """Test ContentBrief with default values."""
        brief = ContentBrief(
            title="Test Article",
            description="Test description",
            keywords=["test"],
            target_audience="general",
            content_type="article"
        )
        
        assert brief.word_count == 500  # Default value
        assert brief.tone == "professional"  # Default value
        assert brief.seo_requirements == {}  # Default value

    def test_content_brief_validation(self):
        """Test ContentBrief validation."""
        # Test minimum word count
        with pytest.raises(ValueError, match="Word count must be at least 100"):
            ContentBrief(
                title="Test",
                description="Test",
                keywords=["test"],
                target_audience="general",
                content_type="article",
                word_count=50
            )

    def test_content_brief_keyword_validation(self):
        """Test ContentBrief keyword validation."""
        # Test empty keywords
        with pytest.raises(ValueError, match="At least one keyword is required"):
            ContentBrief(
                title="Test",
                description="Test",
                keywords=[],
                target_audience="general",
                content_type="article"
            )


class TestContentWorkflow:
    """Test suite for ContentWorkflow model."""

    def test_content_workflow_creation(self):
        """Test creation of ContentWorkflow with valid data."""
        brief = ContentBrief(
            title="Test Article",
            description="Test description",
            keywords=["test"],
            target_audience="developers",
            content_type="article"
        )
        
        workflow = ContentWorkflow(
            brief=brief,
            workflow_type="content_creation",
            tenant_id="test_tenant"
        )
        
        assert workflow.brief == brief
        assert workflow.workflow_type == "content_creation"
        assert workflow.tenant_id == "test_tenant"
        assert workflow.status == WorkflowStatus.PENDING
        assert isinstance(workflow.created_at, datetime)

    def test_content_workflow_step_progression(self):
        """Test workflow step progression."""
        brief = ContentBrief(
            title="Test Article",
            description="Test description",
            keywords=["test"],
            target_audience="developers",
            content_type="article"
        )
        
        workflow = ContentWorkflow(
            brief=brief,
            workflow_type="content_creation",
            tenant_id="test_tenant"
        )
        
        # Test initial state
        assert workflow.current_step == 0
        assert len(workflow.steps) == 0
        
        # Add a step
        step = WorkflowStep(
            name="research",
            description="Research phase",
            estimated_duration=30
        )
        workflow.steps.append(step)
        
        assert len(workflow.steps) == 1
        assert workflow.steps[0].name == "research"

    def test_content_workflow_validation(self):
        """Test ContentWorkflow validation."""
        brief = ContentBrief(
            title="Test Article",
            description="Test description",
            keywords=["test"],
            target_audience="developers",
            content_type="article"
        )
        
        workflow = ContentWorkflow(
            brief=brief,
            workflow_type="content_creation",
            tenant_id="test_tenant"
        )
        
        # Test validation passes for valid workflow
        assert workflow.validate() is True

    def test_content_workflow_progress_calculation(self):
        """Test workflow progress calculation."""
        brief = ContentBrief(
            title="Test Article",
            description="Test description",
            keywords=["test"],
            target_audience="developers",
            content_type="article"
        )
        
        workflow = ContentWorkflow(
            brief=brief,
            workflow_type="content_creation",
            tenant_id="test_tenant"
        )
        
        # Add steps
        step1 = WorkflowStep(name="research", description="Research", estimated_duration=30)
        step2 = WorkflowStep(name="writing", description="Writing", estimated_duration=60)
        workflow.steps = [step1, step2]
        
        # Test progress calculation
        progress = workflow.calculate_progress()
        assert progress >= 0.0
        assert progress <= 1.0


class TestWorkflowStep:
    """Test suite for WorkflowStep model."""

    def test_workflow_step_creation(self):
        """Test creation of WorkflowStep with valid data."""
        step = WorkflowStep(
            name="research",
            description="Research phase",
            estimated_duration=30
        )
        
        assert step.name == "research"
        assert step.description == "Research phase"
        assert step.estimated_duration == 30
        assert step.status == "pending"
        assert isinstance(step.created_at, datetime)

    def test_workflow_step_defaults(self):
        """Test WorkflowStep with default values."""
        step = WorkflowStep(
            name="research",
            description="Research phase"
        )
        
        assert step.estimated_duration == 15  # Default value
        assert step.status == "pending"  # Default value
        assert step.output is None  # Default value

    def test_workflow_step_completion(self):
        """Test WorkflowStep completion."""
        step = WorkflowStep(
            name="research",
            description="Research phase",
            estimated_duration=30
        )
        
        # Complete the step
        step.complete_step("Research completed successfully")
        
        assert step.status == "completed"
        assert step.output == "Research completed successfully"
        assert step.completed_at is not None
        assert isinstance(step.completed_at, datetime)

    def test_workflow_step_failure(self):
        """Test WorkflowStep failure."""
        step = WorkflowStep(
            name="research",
            description="Research phase",
            estimated_duration=30
        )
        
        # Fail the step
        step.fail_step("Research failed due to API error")
        
        assert step.status == "failed"
        assert step.error_message == "Research failed due to API error"
        assert step.completed_at is not None