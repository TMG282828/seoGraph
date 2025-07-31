"""
PRP Workflow Module - Backward Compatibility Layer.

This module maintains 100% backward compatibility with the original
prp_workflow_service.py while organizing code into modular components.

All existing imports and usage patterns continue to work unchanged.
"""

from .models import (
    WorkflowPhase,
    CheckpointStatus, 
    WorkflowCheckpoint,
    PRPWorkflowState,
    ContentWorkflow,
    WorkflowConfig,
    PhaseResult
)

from .orchestrator import PRPWorkflowOrchestrator
from .phase_analyzers import BriefAnalyzer, ContentPlanner, RequirementsDefiner, ProcessDefiner, FinalReviewer
from .content_generator import ContentGenerator

# Backward compatibility alias - existing code expects PRPWorkflowService
PRPWorkflowService = PRPWorkflowOrchestrator

# Create global service instance (same as original)
prp_workflow_service = PRPWorkflowOrchestrator()

# Export all the classes and functions that the original module exported
__all__ = [
    'WorkflowPhase',
    'CheckpointStatus',
    'WorkflowCheckpoint', 
    'PRPWorkflowState',
    'ContentWorkflow',
    'WorkflowConfig',
    'PhaseResult',
    'PRPWorkflowService',
    'PRPWorkflowOrchestrator',
    'BriefAnalyzer',
    'ContentPlanner', 
    'RequirementsDefiner',
    'ProcessDefiner',
    'FinalReviewer',
    'ContentGenerator',
    'prp_workflow_service'
]