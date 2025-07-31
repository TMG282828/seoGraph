"""
Workflow Orchestrator for the SEO Content Knowledge Graph System.

This module provides comprehensive workflow orchestration with state machine management,
human-in-the-loop processes, automated task execution, and workflow analytics.
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum
import uuid

import structlog
from cachetools import TTLCache
from pydantic_ai import Agent
from tenacity import retry, stop_after_attempt, wait_exponential

from database.neo4j_client import Neo4jClient
from models.workflow_models import (
    ContentBrief,
    ContentWorkflow,
    WorkflowStep,
    WorkflowTemplate,
    WorkflowNotification,
    WorkflowStatus,
    WorkflowStepType,
    CreateWorkflowRequest,
    WorkflowExecutionRequest
)
from models.content_models import ContentItem
from .analytics_service import AnalyticsService
from src.agents.content_analysis_agent import ContentAnalysisAgent
from src.agents.content_generation import ContentGenerationAgent  
from src.agents.quality_assurance_agent import QualityAssuranceAgent
from config.settings import get_settings

logger = structlog.get_logger(__name__)


class WorkflowOrchestrationError(Exception):
    """Raised when workflow orchestration fails."""
    pass


class WorkflowExecutionError(WorkflowOrchestrationError):
    """Raised when workflow execution fails."""
    pass


class WorkflowStateError(WorkflowOrchestrationError):
    """Raised when workflow state is invalid."""
    pass


class NotificationService:
    """Service for sending workflow notifications."""
    
    def __init__(self):
        self.notification_channels = {
            'email': self._send_email_notification,
            'slack': self._send_slack_notification,
            'webhook': self._send_webhook_notification,
            'in_app': self._send_in_app_notification
        }
    
    async def send_notification(self, 
                              notification: WorkflowNotification,
                              channels: List[str]) -> bool:
        """Send notification to specified channels."""
        try:
            success_count = 0
            
            for channel in channels:
                if channel in self.notification_channels:
                    try:
                        await self.notification_channels[channel](notification)
                        success_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to send notification to {channel}: {e}")
                else:
                    logger.warning(f"Unknown notification channel: {channel}")
            
            # Mark as sent if at least one channel succeeded
            if success_count > 0:
                notification.mark_as_sent()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return False
    
    async def _send_email_notification(self, notification: WorkflowNotification) -> None:
        """Send email notification (placeholder implementation)."""
        logger.info(f"EMAIL NOTIFICATION: {notification.title} to {notification.recipient}")
    
    async def _send_slack_notification(self, notification: WorkflowNotification) -> None:
        """Send Slack notification (placeholder implementation)."""
        logger.info(f"SLACK NOTIFICATION: {notification.title} to {notification.recipient}")
    
    async def _send_webhook_notification(self, notification: WorkflowNotification) -> None:
        """Send webhook notification (placeholder implementation)."""
        logger.info(f"WEBHOOK NOTIFICATION: {notification.title} to {notification.recipient}")
    
    async def _send_in_app_notification(self, notification: WorkflowNotification) -> None:
        """Send in-app notification (placeholder implementation)."""
        logger.info(f"IN-APP NOTIFICATION: {notification.title} to {notification.recipient}")


class WorkflowExecutor:
    """Executes individual workflow steps."""
    
    def __init__(self, 
                 neo4j_client: Neo4jClient,
                 analytics_service: AnalyticsService):
        self.neo4j_client = neo4j_client
        self.analytics_service = analytics_service
        
        # Step executors
        self.step_executors = {
            WorkflowStepType.AGENT_EXECUTION: self._execute_agent_step,
            WorkflowStepType.HUMAN_REVIEW: self._execute_human_review_step,
            WorkflowStepType.VALIDATION: self._execute_validation_step,
            WorkflowStepType.PUBLICATION: self._execute_publication_step,
            WorkflowStepType.NOTIFICATION: self._execute_notification_step,
            WorkflowStepType.INTEGRATION: self._execute_integration_step
        }
        
        # Agent registry
        self.agents = {}
        
        logger.info("Workflow executor initialized")
    
    def register_agent(self, agent_name: str, agent_instance: Any) -> None:
        """Register an agent for workflow execution."""
        self.agents[agent_name] = agent_instance
        logger.info(f"Registered agent: {agent_name}")
    
    async def execute_step(self, 
                         workflow: ContentWorkflow, 
                         step: WorkflowStep) -> Dict[str, Any]:
        """Execute a workflow step."""
        try:
            logger.info(f"Executing step: {step.name}", 
                       workflow_id=workflow.id,
                       step_id=step.step_id,
                       step_type=step.step_type)
            
            # Mark step as started
            step.start_step()
            
            # Get executor for step type
            executor = self.step_executors.get(step.step_type)
            if not executor:
                raise WorkflowExecutionError(f"No executor for step type: {step.step_type}")
            
            # Execute step
            result = await executor(workflow, step)
            
            # Mark step as completed
            step.complete_step(result)
            
            logger.info(f"Step completed successfully: {step.name}",
                       step_id=step.step_id,
                       duration=step.actual_duration)
            
            return result
            
        except Exception as e:
            logger.error(f"Step execution failed: {step.name}: {e}")
            step.fail_step(str(e))
            raise WorkflowExecutionError(f"Step execution failed: {e}")
    
    async def _execute_agent_step(self, 
                                workflow: ContentWorkflow, 
                                step: WorkflowStep) -> Dict[str, Any]:
        """Execute an agent-based workflow step."""
        try:
            agent_name = step.assigned_to
            if not agent_name or agent_name not in self.agents:
                raise WorkflowExecutionError(f"Agent not found: {agent_name}")
            
            agent = self.agents[agent_name]
            config = step.config
            
            # Get brief for context
            brief = await self._get_workflow_brief(workflow)
            
            # Prepare agent prompt based on step configuration
            prompt = self._build_agent_prompt(step, brief, config)
            
            # Execute agent
            result = await agent.run(prompt)
            
            return {
                'agent_result': result.data if hasattr(result, 'data') else str(result),
                'agent_usage': result.usage if hasattr(result, 'usage') else None,
                'execution_time': step.actual_duration
            }
            
        except Exception as e:
            logger.error(f"Agent step execution failed: {e}")
            raise WorkflowExecutionError(f"Agent execution failed: {e}")
    
    async def _execute_human_review_step(self, 
                                       workflow: ContentWorkflow, 
                                       step: WorkflowStep) -> Dict[str, Any]:
        """Execute a human review step."""
        try:
            # Create review notification
            reviewer_id = step.assigned_to
            if not reviewer_id:
                raise WorkflowExecutionError("No reviewer assigned for human review step")
            
            # Get content for review
            content = await self._get_step_content_for_review(workflow, step)
            
            # Create review request
            review_request = {
                'workflow_id': workflow.id,
                'step_id': step.step_id,
                'content': content,
                'instructions': step.config.get('review_instructions', 'Please review this content'),
                'deadline': step.config.get('review_deadline'),
                'reviewer_id': reviewer_id
            }
            
            # Store review request in database
            await self._store_review_request(review_request)
            
            # For now, mark as pending human input
            # In a real system, this would wait for human review
            return {
                'status': 'pending_human_review',
                'review_request_id': review_request.get('id'),
                'reviewer_id': reviewer_id,
                'content_preview': content[:200] if isinstance(content, str) else str(content)[:200]
            }
            
        except Exception as e:
            logger.error(f"Human review step execution failed: {e}")
            raise WorkflowExecutionError(f"Human review failed: {e}")
    
    async def _execute_validation_step(self, 
                                     workflow: ContentWorkflow, 
                                     step: WorkflowStep) -> Dict[str, Any]:
        """Execute a validation step."""
        try:
            validation_type = step.config.get('validation_type', 'content_quality')
            content = await self._get_step_content_for_validation(workflow, step)
            
            validation_results = {}
            
            if validation_type == 'content_quality':
                validation_results = await self._validate_content_quality(content, step.config)
            elif validation_type == 'seo_compliance':
                validation_results = await self._validate_seo_compliance(content, step.config)
            elif validation_type == 'brand_voice':
                validation_results = await self._validate_brand_voice(content, step.config)
            else:
                raise WorkflowExecutionError(f"Unknown validation type: {validation_type}")
            
            # Check if validation passed
            passed = validation_results.get('passed', False)
            if not passed and step.config.get('fail_on_validation_error', True):
                raise WorkflowExecutionError(f"Validation failed: {validation_results.get('errors', [])}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Validation step execution failed: {e}")
            raise WorkflowExecutionError(f"Validation failed: {e}")
    
    async def _execute_publication_step(self, 
                                      workflow: ContentWorkflow, 
                                      step: WorkflowStep) -> Dict[str, Any]:
        """Execute a publication step."""
        try:
            publication_config = step.config
            content = await self._get_finalized_content(workflow)
            
            # Get publication target
            target = publication_config.get('target', 'draft')
            
            if target == 'cms':
                result = await self._publish_to_cms(content, publication_config)
            elif target == 'social':
                result = await self._publish_to_social(content, publication_config)
            elif target == 'email':
                result = await self._publish_to_email(content, publication_config)
            else:
                # Default to saving as draft
                result = await self._save_as_draft(content, publication_config)
            
            return result
            
        except Exception as e:
            logger.error(f"Publication step execution failed: {e}")
            raise WorkflowExecutionError(f"Publication failed: {e}")
    
    async def _execute_notification_step(self, 
                                       workflow: ContentWorkflow, 
                                       step: WorkflowStep) -> Dict[str, Any]:
        """Execute a notification step."""
        try:
            notification_config = step.config
            
            # Create notification
            notification = WorkflowNotification(
                workflow_id=workflow.id,
                recipient=notification_config.get('recipient'),
                notification_type=notification_config.get('type', 'workflow_update'),
                title=notification_config.get('title', f'Workflow Update: {workflow.name}'),
                message=notification_config.get('message', 'Workflow step completed'),
                tenant_id=workflow.tenant_id
            )
            
            # Send notification
            channels = notification_config.get('channels', ['in_app'])
            notification_service = NotificationService()
            success = await notification_service.send_notification(notification, channels)
            
            return {
                'notification_sent': success,
                'notification_id': notification.notification_id,
                'channels': channels
            }
            
        except Exception as e:
            logger.error(f"Notification step execution failed: {e}")
            raise WorkflowExecutionError(f"Notification failed: {e}")
    
    async def _execute_integration_step(self, 
                                      workflow: ContentWorkflow, 
                                      step: WorkflowStep) -> Dict[str, Any]:
        """Execute an integration step."""
        try:
            integration_config = step.config
            integration_type = integration_config.get('type')
            
            if integration_type == 'google_drive':
                result = await self._integrate_google_drive(workflow, integration_config)
            elif integration_type == 'analytics':
                result = await self._integrate_analytics(workflow, integration_config)
            elif integration_type == 'webhook':
                result = await self._integrate_webhook(workflow, integration_config)
            else:
                raise WorkflowExecutionError(f"Unknown integration type: {integration_type}")
            
            return result
            
        except Exception as e:
            logger.error(f"Integration step execution failed: {e}")
            raise WorkflowExecutionError(f"Integration failed: {e}")
    
    # Helper methods for step execution
    
    async def _get_workflow_brief(self, workflow: ContentWorkflow) -> Optional[ContentBrief]:
        """Get content brief for workflow."""
        try:
            query = """
            MATCH (b:ContentBrief {id: $brief_id})
            RETURN b
            """
            
            result = await self.neo4j_client.run_query(query, {"brief_id": workflow.brief_id})
            
            if result:
                brief_data = result[0]['b']
                return ContentBrief(**brief_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get workflow brief: {e}")
            return None
    
    def _build_agent_prompt(self, 
                          step: WorkflowStep, 
                          brief: Optional[ContentBrief], 
                          config: Dict[str, Any]) -> str:
        """Build agent prompt based on step configuration."""
        try:
            base_prompt = config.get('prompt', '')
            
            if brief:
                # Add brief context to prompt
                brief_context = f"""
                Content Brief Context:
                - Title: {brief.title}
                - Content Type: {brief.content_type}
                - Target Keywords: {', '.join(brief.target_keywords)}
                - Target Audience: {brief.target_audience}
                - Tone: {brief.tone}
                - Word Count: {brief.word_count}
                """
                
                if brief.key_points:
                    brief_context += f"- Key Points: {', '.join(brief.key_points)}"
                
                base_prompt = f"{brief_context}\n\n{base_prompt}"
            
            return base_prompt
            
        except Exception as e:
            logger.error(f"Failed to build agent prompt: {e}")
            return config.get('prompt', 'Please process this request.')
    
    async def _get_step_content_for_review(self, workflow: ContentWorkflow, step: WorkflowStep) -> Any:
        """Get content for human review."""
        # Get content from previous steps or workflow
        # This is a placeholder implementation
        return f"Content for review from workflow {workflow.id}, step {step.step_id}"
    
    async def _get_step_content_for_validation(self, workflow: ContentWorkflow, step: WorkflowStep) -> Any:
        """Get content for validation."""
        # Get content from previous steps
        # This is a placeholder implementation
        return f"Content for validation from workflow {workflow.id}, step {step.step_id}"
    
    async def _get_finalized_content(self, workflow: ContentWorkflow) -> Any:
        """Get finalized content for publication."""
        # Get final content from workflow
        # This is a placeholder implementation
        return f"Finalized content from workflow {workflow.id}"
    
    async def _validate_content_quality(self, content: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate content quality."""
        # Placeholder implementation
        return {
            'passed': True,
            'score': 0.85,
            'checks': ['readability', 'grammar', 'structure'],
            'errors': [],
            'warnings': []
        }
    
    async def _validate_seo_compliance(self, content: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate SEO compliance."""
        # Placeholder implementation
        return {
            'passed': True,
            'score': 0.90,
            'checks': ['keywords', 'meta_description', 'headings'],
            'errors': [],
            'warnings': []
        }
    
    async def _validate_brand_voice(self, content: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate brand voice consistency."""
        # Placeholder implementation
        return {
            'passed': True,
            'score': 0.88,
            'checks': ['tone', 'style', 'terminology'],
            'errors': [],
            'warnings': []
        }
    
    async def _store_review_request(self, review_request: Dict[str, Any]) -> None:
        """Store review request in database."""
        # Placeholder implementation
        review_request['id'] = str(uuid.uuid4())
        logger.info(f"Stored review request: {review_request['id']}")
    
    async def _publish_to_cms(self, content: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Publish content to CMS."""
        # Placeholder implementation
        return {
            'published': True,
            'url': 'https://example.com/published-content',
            'cms': config.get('cms_type', 'wordpress')
        }
    
    async def _publish_to_social(self, content: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Publish content to social media."""
        # Placeholder implementation
        return {
            'published': True,
            'platforms': config.get('platforms', ['twitter', 'linkedin']),
            'post_ids': ['tweet_123', 'linkedin_456']
        }
    
    async def _publish_to_email(self, content: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Publish content via email."""
        # Placeholder implementation
        return {
            'sent': True,
            'recipients': config.get('recipients', []),
            'email_id': 'email_789'
        }
    
    async def _save_as_draft(self, content: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Save content as draft."""
        # Placeholder implementation
        return {
            'saved': True,
            'draft_id': str(uuid.uuid4()),
            'location': config.get('draft_location', 'database')
        }
    
    async def _integrate_google_drive(self, workflow: ContentWorkflow, config: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate with Google Drive."""
        # Placeholder implementation
        return {
            'integrated': True,
            'action': config.get('action', 'sync'),
            'files_processed': 1
        }
    
    async def _integrate_analytics(self, workflow: ContentWorkflow, config: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate with analytics service."""
        # Placeholder implementation
        return {
            'integrated': True,
            'metrics_tracked': config.get('metrics', ['workflow_completion']),
            'tracking_id': str(uuid.uuid4())
        }
    
    async def _integrate_webhook(self, workflow: ContentWorkflow, config: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate with external webhook."""
        # Placeholder implementation
        return {
            'webhook_called': True,
            'url': config.get('webhook_url'),
            'response_status': 200
        }


class WorkflowOrchestrator:
    """
    Comprehensive workflow orchestrator with state machine management.
    
    Provides:
    - Workflow creation and execution
    - State machine management
    - Human-in-the-loop processes
    - Agent orchestration
    - Workflow templates
    - Analytics and monitoring
    """
    
    def __init__(self, 
                 neo4j_client: Neo4jClient,
                 analytics_service: AnalyticsService):
        self.neo4j_client = neo4j_client
        self.analytics_service = analytics_service
        self.settings = get_settings()
        
        # Core components
        self.workflow_executor = WorkflowExecutor(neo4j_client, analytics_service)
        self.notification_service = NotificationService()
        
        # Active workflows
        self.active_workflows: Dict[str, ContentWorkflow] = {}
        
        # Workflow templates
        self.workflow_templates: Dict[str, WorkflowTemplate] = {}
        
        # Execution queue
        self.execution_queue: asyncio.Queue = asyncio.Queue()
        self.execution_workers: List[asyncio.Task] = []
        self.is_running = False
        
        # Caching
        self.workflow_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour
        
        # Statistics
        self.stats = {
            'workflows_created': 0,
            'workflows_completed': 0,
            'workflows_failed': 0,
            'steps_executed': 0,
            'avg_execution_time': 0.0
        }
        
        logger.info("Workflow orchestrator initialized")
    
    async def initialize(self) -> None:
        """Initialize the workflow orchestrator."""
        try:
            # Load workflow templates
            await self._load_workflow_templates()
            
            # Load active workflows
            await self._load_active_workflows()
            
            # Register default agents
            await self._register_default_agents()
            
            # Start execution workers
            await self._start_execution_workers()
            
            logger.info("Workflow orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize workflow orchestrator: {e}")
            raise WorkflowOrchestrationError(f"Initialization failed: {e}")
    
    async def close(self) -> None:
        """Close the workflow orchestrator."""
        try:
            self.is_running = False
            
            # Cancel execution workers
            for worker in self.execution_workers:
                worker.cancel()
                try:
                    await worker
                except asyncio.CancelledError:
                    pass
            
            logger.info("Workflow orchestrator closed")
            
        except Exception as e:
            logger.error(f"Error closing workflow orchestrator: {e}")
    
    # =============================================================================
    # Workflow Management
    # =============================================================================
    
    async def create_workflow(self, request: CreateWorkflowRequest, created_by: str) -> ContentWorkflow:
        """Create a new content workflow."""
        try:
            # Get or create template
            template = None
            if request.template_id:
                template = await self.get_workflow_template(request.template_id)
                if not template:
                    raise WorkflowOrchestrationError(f"Template not found: {request.template_id}")
            
            # Create workflow
            if template:
                workflow = template.create_workflow(
                    brief_id=request.brief_id,
                    name=request.name,
                    created_by=created_by
                )
            else:
                workflow = ContentWorkflow(
                    name=request.name,
                    brief_id=request.brief_id,
                    workflow_type=request.workflow_type,
                    tenant_id="default",  # TODO: Get from context
                    created_by=created_by,
                    assigned_agents=request.assigned_agents or [],
                    human_reviewers=request.human_reviewers or []
                )
                
                # Add custom steps if provided
                if request.custom_steps:
                    for i, step_config in enumerate(request.custom_steps):
                        step = WorkflowStep(
                            name=step_config.get('name', f'Step {i+1}'),
                            description=step_config.get('description'),
                            step_type=WorkflowStepType(step_config.get('step_type', 'agent_execution')),
                            order=i,
                            config=step_config.get('config', {}),
                            assigned_to=step_config.get('assigned_to')
                        )
                        workflow.add_step(step)
            
            # Store workflow
            await self._store_workflow(workflow)
            
            # Add to active workflows
            self.active_workflows[workflow.id] = workflow
            
            # Update statistics
            self.stats['workflows_created'] += 1
            
            logger.info(f"Created workflow: {workflow.name}", workflow_id=workflow.id)
            return workflow
            
        except Exception as e:
            logger.error(f"Failed to create workflow: {e}")
            raise WorkflowOrchestrationError(f"Workflow creation failed: {e}")
    
    async def get_workflow(self, workflow_id: str) -> Optional[ContentWorkflow]:
        """Get workflow by ID."""
        try:
            # Check active workflows first
            if workflow_id in self.active_workflows:
                return self.active_workflows[workflow_id]
            
            # Check cache
            if workflow_id in self.workflow_cache:
                return self.workflow_cache[workflow_id]
            
            # Load from database
            workflow = await self._load_workflow(workflow_id)
            if workflow:
                self.workflow_cache[workflow_id] = workflow
                
                # Add to active if not completed
                if not workflow.is_complete():
                    self.active_workflows[workflow_id] = workflow
            
            return workflow
            
        except Exception as e:
            logger.error(f"Failed to get workflow {workflow_id}: {e}")
            return None
    
    async def start_workflow(self, workflow_id: str) -> bool:
        """Start workflow execution."""
        try:
            workflow = await self.get_workflow(workflow_id)
            if not workflow:
                raise WorkflowOrchestrationError(f"Workflow not found: {workflow_id}")
            
            if workflow.status != WorkflowStatus.DRAFT:
                raise WorkflowStateError(f"Workflow not in draft state: {workflow.status}")
            
            # Start workflow
            workflow.start_workflow()
            
            # Queue for execution
            await self.execution_queue.put(workflow)
            
            # Update in database
            await self._update_workflow_status(workflow)
            
            logger.info(f"Started workflow: {workflow.name}", workflow_id=workflow_id)
            return True
            
        except Exception as e:
            logger.error(f"Failed to start workflow {workflow_id}: {e}")
            return False
    
    async def execute_workflow_step(self, request: WorkflowExecutionRequest) -> Dict[str, Any]:
        """Execute a specific workflow step."""
        try:
            workflow = await self.get_workflow(request.workflow_id)
            if not workflow:
                raise WorkflowOrchestrationError(f"Workflow not found: {request.workflow_id}")
            
            # Get step to execute
            if request.step_id:
                step = next((s for s in workflow.steps if s.step_id == request.step_id), None)
                if not step:
                    raise WorkflowOrchestrationError(f"Step not found: {request.step_id}")
            else:
                step = workflow.get_current_step()
                if not step:
                    raise WorkflowOrchestrationError("No current step to execute")
            
            # Execute step
            result = await self.workflow_executor.execute_step(workflow, step)
            
            # Update workflow progress
            workflow.calculate_progress()
            
            # Check if workflow can advance
            if workflow.can_advance():
                workflow.advance_step()
                
                # Queue next step for execution if automated
                next_step = workflow.get_current_step()
                if next_step and next_step.step_type != WorkflowStepType.HUMAN_REVIEW:
                    await self.execution_queue.put(workflow)
            
            # Check if workflow is complete
            if workflow.is_complete():
                workflow.complete_workflow()
                self._remove_from_active_workflows(workflow.id)
                self.stats['workflows_completed'] += 1
            
            # Update in database
            await self._update_workflow_status(workflow)
            
            # Update statistics
            self.stats['steps_executed'] += 1
            
            return {
                'step_result': result,
                'workflow_status': workflow.status,
                'progress': workflow.progress_percentage,
                'next_step': workflow.get_current_step().name if workflow.get_current_step() else None
            }
            
        except Exception as e:
            logger.error(f"Failed to execute workflow step: {e}")
            
            # Mark workflow as failed if critical error
            if request.workflow_id in self.active_workflows:
                workflow = self.active_workflows[request.workflow_id]
                workflow.fail_workflow(str(e))
                await self._update_workflow_status(workflow)
                self.stats['workflows_failed'] += 1
            
            raise WorkflowExecutionError(f"Step execution failed: {e}")
    
    # =============================================================================
    # Workflow Templates
    # =============================================================================
    
    async def create_workflow_template(self, template: WorkflowTemplate) -> str:
        """Create a new workflow template."""
        try:
            # Store template
            await self._store_workflow_template(template)
            
            # Cache template
            self.workflow_templates[template.template_id] = template
            
            logger.info(f"Created workflow template: {template.name}")
            return template.template_id
            
        except Exception as e:
            logger.error(f"Failed to create workflow template: {e}")
            raise WorkflowOrchestrationError(f"Template creation failed: {e}")
    
    async def get_workflow_template(self, template_id: str) -> Optional[WorkflowTemplate]:
        """Get workflow template by ID."""
        try:
            # Check cache first
            if template_id in self.workflow_templates:
                return self.workflow_templates[template_id]
            
            # Load from database
            template = await self._load_workflow_template(template_id)
            if template:
                self.workflow_templates[template_id] = template
            
            return template
            
        except Exception as e:
            logger.error(f"Failed to get workflow template {template_id}: {e}")
            return None
    
    async def list_workflow_templates(self, category: Optional[str] = None) -> List[WorkflowTemplate]:
        """List available workflow templates."""
        try:
            templates = list(self.workflow_templates.values())
            
            if category:
                templates = [t for t in templates if t.category == category]
            
            return templates
            
        except Exception as e:
            logger.error(f"Failed to list workflow templates: {e}")
            return []
    
    # =============================================================================
    # Execution Workers
    # =============================================================================
    
    async def _start_execution_workers(self, num_workers: int = 3) -> None:
        """Start workflow execution workers."""
        try:
            self.is_running = True
            
            for i in range(num_workers):
                worker = asyncio.create_task(self._execution_worker(f"worker-{i}"))
                self.execution_workers.append(worker)
            
            logger.info(f"Started {num_workers} workflow execution workers")
            
        except Exception as e:
            logger.error(f"Failed to start execution workers: {e}")
            raise WorkflowOrchestrationError(f"Worker startup failed: {e}")
    
    async def _execution_worker(self, worker_name: str) -> None:
        """Workflow execution worker."""
        logger.info(f"Started execution worker: {worker_name}")
        
        try:
            while self.is_running:
                try:
                    # Get workflow from queue (with timeout)
                    workflow = await asyncio.wait_for(
                        self.execution_queue.get(), 
                        timeout=5.0
                    )
                    
                    # Execute next step
                    await self._execute_next_workflow_step(workflow, worker_name)
                    
                    # Mark queue task as done
                    self.execution_queue.task_done()
                    
                except asyncio.TimeoutError:
                    # No work available, continue
                    continue
                except Exception as e:
                    logger.error(f"Worker {worker_name} execution error: {e}")
                    continue
                    
        except asyncio.CancelledError:
            logger.info(f"Execution worker {worker_name} cancelled")
        except Exception as e:
            logger.error(f"Execution worker {worker_name} failed: {e}")
    
    async def _execute_next_workflow_step(self, workflow: ContentWorkflow, worker_name: str) -> None:
        """Execute the next step in a workflow."""
        try:
            current_step = workflow.get_current_step()
            if not current_step:
                logger.warning(f"No current step for workflow {workflow.id}")
                return
            
            # Skip human review steps in automated execution
            if current_step.step_type == WorkflowStepType.HUMAN_REVIEW:
                logger.info(f"Skipping human review step in automated execution: {current_step.name}")
                return
            
            logger.info(f"Worker {worker_name} executing step: {current_step.name}",
                       workflow_id=workflow.id,
                       step_id=current_step.step_id)
            
            # Execute step
            result = await self.workflow_executor.execute_step(workflow, current_step)
            
            # Update workflow
            workflow.calculate_progress()
            
            # Advance if possible
            if workflow.can_advance():
                workflow.advance_step()
                
                # Queue next step if not human review
                next_step = workflow.get_current_step()
                if next_step and next_step.step_type != WorkflowStepType.HUMAN_REVIEW:
                    await self.execution_queue.put(workflow)
            
            # Check completion
            if workflow.is_complete():
                workflow.complete_workflow()
                self._remove_from_active_workflows(workflow.id)
                self.stats['workflows_completed'] += 1
                
                logger.info(f"Workflow completed: {workflow.name}", workflow_id=workflow.id)
            
            # Update database
            await self._update_workflow_status(workflow)
            
        except Exception as e:
            logger.error(f"Failed to execute workflow step: {e}")
            workflow.fail_workflow(str(e))
            await self._update_workflow_status(workflow)
            self.stats['workflows_failed'] += 1
    
    # =============================================================================
    # Database Operations
    # =============================================================================
    
    async def _store_workflow(self, workflow: ContentWorkflow) -> None:
        """Store workflow in database."""
        try:
            query = """
            CREATE (w:ContentWorkflow {
                id: $id,
                name: $name,
                description: $description,
                brief_id: $brief_id,
                content_id: $content_id,
                workflow_type: $workflow_type,
                template_id: $template_id,
                current_step: $current_step,
                assigned_agents: $assigned_agents,
                human_reviewers: $human_reviewers,
                status: $status,
                progress_percentage: $progress_percentage,
                estimated_duration: $estimated_duration,
                actual_duration: $actual_duration,
                tenant_id: $tenant_id,
                created_by: $created_by,
                created_at: datetime(),
                updated_at: datetime(),
                started_at: $started_at,
                completed_at: $completed_at
            })
            """
            
            await self.neo4j_client.run_query(query, {
                **workflow.dict(exclude={'steps'}),
                'started_at': workflow.started_at,
                'completed_at': workflow.completed_at
            })
            
            # Store workflow steps
            for step in workflow.steps:
                await self._store_workflow_step(workflow.id, step)
            
        except Exception as e:
            logger.error(f"Failed to store workflow: {e}")
            raise WorkflowOrchestrationError(f"Workflow storage failed: {e}")
    
    async def _store_workflow_step(self, workflow_id: str, step: WorkflowStep) -> None:
        """Store workflow step in database."""
        try:
            query = """
            MATCH (w:ContentWorkflow {id: $workflow_id})
            CREATE (s:WorkflowStep {
                step_id: $step_id,
                name: $name,
                description: $description,
                step_type: $step_type,
                order: $order,
                config: $config,
                dependencies: $dependencies,
                status: $status,
                assigned_to: $assigned_to,
                result: $result,
                error_message: $error_message,
                estimated_duration: $estimated_duration,
                actual_duration: $actual_duration,
                started_at: $started_at,
                completed_at: $completed_at
            })
            CREATE (w)-[:HAS_STEP]->(s)
            """
            
            await self.neo4j_client.run_query(query, {
                'workflow_id': workflow_id,
                **step.dict()
            })
            
        except Exception as e:
            logger.error(f"Failed to store workflow step: {e}")
    
    async def _load_workflow(self, workflow_id: str) -> Optional[ContentWorkflow]:
        """Load workflow from database."""
        try:
            # Load workflow
            query = """
            MATCH (w:ContentWorkflow {id: $workflow_id})
            RETURN w
            """
            
            result = await self.neo4j_client.run_query(query, {"workflow_id": workflow_id})
            
            if not result:
                return None
            
            workflow_data = result[0]['w']
            workflow = ContentWorkflow(**workflow_data)
            
            # Load workflow steps
            steps_query = """
            MATCH (w:ContentWorkflow {id: $workflow_id})-[:HAS_STEP]->(s:WorkflowStep)
            RETURN s
            ORDER BY s.order
            """
            
            steps_result = await self.neo4j_client.run_query(steps_query, {"workflow_id": workflow_id})
            
            for step_record in steps_result:
                step_data = step_record['s']
                step = WorkflowStep(**step_data)
                workflow.steps.append(step)
            
            return workflow
            
        except Exception as e:
            logger.error(f"Failed to load workflow {workflow_id}: {e}")
            return None
    
    async def _update_workflow_status(self, workflow: ContentWorkflow) -> None:
        """Update workflow status in database."""
        try:
            query = """
            MATCH (w:ContentWorkflow {id: $workflow_id})
            SET w.status = $status,
                w.progress_percentage = $progress_percentage,
                w.current_step = $current_step,
                w.actual_duration = $actual_duration,
                w.updated_at = datetime(),
                w.started_at = $started_at,
                w.completed_at = $completed_at
            """
            
            await self.neo4j_client.run_query(query, {
                'workflow_id': workflow.id,
                'status': workflow.status,
                'progress_percentage': workflow.progress_percentage,
                'current_step': workflow.current_step,
                'actual_duration': workflow.actual_duration,
                'started_at': workflow.started_at,
                'completed_at': workflow.completed_at
            })
            
            # Update step statuses
            for step in workflow.steps:
                await self._update_workflow_step_status(step)
            
        except Exception as e:
            logger.error(f"Failed to update workflow status: {e}")
    
    async def _update_workflow_step_status(self, step: WorkflowStep) -> None:
        """Update workflow step status in database."""
        try:
            query = """
            MATCH (s:WorkflowStep {step_id: $step_id})
            SET s.status = $status,
                s.result = $result,
                s.error_message = $error_message,
                s.actual_duration = $actual_duration,
                s.started_at = $started_at,
                s.completed_at = $completed_at
            """
            
            await self.neo4j_client.run_query(query, {
                'step_id': step.step_id,
                'status': step.status,
                'result': step.result,
                'error_message': step.error_message,
                'actual_duration': step.actual_duration,
                'started_at': step.started_at,
                'completed_at': step.completed_at
            })
            
        except Exception as e:
            logger.error(f"Failed to update workflow step status: {e}")
    
    # =============================================================================
    # Utility Methods
    # =============================================================================
    
    async def _load_workflow_templates(self) -> None:
        """Load workflow templates from database."""
        try:
            query = """
            MATCH (t:WorkflowTemplate {is_active: true})
            RETURN t
            """
            
            result = await self.neo4j_client.run_query(query)
            
            for record in result:
                template_data = record['t']
                template = WorkflowTemplate(**template_data)
                self.workflow_templates[template.template_id] = template
            
            logger.info(f"Loaded {len(self.workflow_templates)} workflow templates")
            
        except Exception as e:
            logger.error(f"Failed to load workflow templates: {e}")
    
    async def _load_active_workflows(self) -> None:
        """Load active workflows from database."""
        try:
            query = """
            MATCH (w:ContentWorkflow)
            WHERE w.status IN ['in_progress', 'review']
            RETURN w.id as workflow_id
            """
            
            result = await self.neo4j_client.run_query(query)
            
            for record in result:
                workflow_id = record['workflow_id']
                workflow = await self._load_workflow(workflow_id)
                if workflow:
                    self.active_workflows[workflow_id] = workflow
            
            logger.info(f"Loaded {len(self.active_workflows)} active workflows")
            
        except Exception as e:
            logger.error(f"Failed to load active workflows: {e}")
    
    async def _register_default_agents(self) -> None:
        """Register default agents with the workflow executor."""
        try:
            # Register agents (these would be imported and instantiated)
            # For now, using placeholder agents
            
            # Content analysis agent
            self.workflow_executor.register_agent('content_analysis', 'ContentAnalysisAgent')
            
            # Content generation agent
            self.workflow_executor.register_agent('content_generation', 'ContentGenerationAgent')
            
            # Quality assurance agent
            self.workflow_executor.register_agent('quality_assurance', 'QualityAssuranceAgent')
            
            logger.info("Registered default agents")
            
        except Exception as e:
            logger.error(f"Failed to register default agents: {e}")
    
    def _remove_from_active_workflows(self, workflow_id: str) -> None:
        """Remove workflow from active workflows."""
        if workflow_id in self.active_workflows:
            del self.active_workflows[workflow_id]
    
    async def _store_workflow_template(self, template: WorkflowTemplate) -> None:
        """Store workflow template in database."""
        # Placeholder implementation
        logger.info(f"Stored workflow template: {template.template_id}")
    
    async def _load_workflow_template(self, template_id: str) -> Optional[WorkflowTemplate]:
        """Load workflow template from database."""
        # Placeholder implementation
        return None
    
    async def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get workflow orchestrator statistics."""
        try:
            stats = {
                **self.stats,
                'active_workflows': len(self.active_workflows),
                'workflow_templates': len(self.workflow_templates),
                'execution_queue_size': self.execution_queue.qsize(),
                'execution_workers': len(self.execution_workers),
                'is_running': self.is_running
            }
            
            # Add detailed workflow stats
            if self.active_workflows:
                statuses = {}
                for workflow in self.active_workflows.values():
                    status = workflow.status
                    statuses[status] = statuses.get(status, 0) + 1
                
                stats['workflow_statuses'] = statuses
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get orchestrator stats: {e}")
            return {'error': str(e)}


# =============================================================================
# Utility Functions
# =============================================================================

async def create_simple_workflow(brief_id: str, 
                                workflow_type: str = "content_creation",
                                tenant_id: str = "default") -> Optional[ContentWorkflow]:
    """
    Simple function to create a basic content workflow.
    
    Args:
        brief_id: Content brief ID
        workflow_type: Type of workflow
        tenant_id: Tenant identifier
        
    Returns:
        Created workflow if successful
    """
    # Initialize required services
    settings = get_settings()
    
    neo4j_client = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_username,
        password=settings.neo4j_password
    )
    
    analytics_service = AnalyticsService(neo4j_client, None)
    
    # Create orchestrator
    orchestrator = WorkflowOrchestrator(neo4j_client, analytics_service)
    
    try:
        await orchestrator.initialize()
        
        # Create workflow request
        request = CreateWorkflowRequest(
            name=f"Content Creation Workflow",
            brief_id=brief_id,
            workflow_type=workflow_type,
            assigned_agents=["content_analysis", "content_generation", "quality_assurance"],
            human_reviewers=["reviewer-1"]
        )
        
        # Create workflow
        workflow = await orchestrator.create_workflow(request, "system")
        
        return workflow
        
    finally:
        await orchestrator.close()


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        # Test workflow creation
        workflow = await create_simple_workflow(
            brief_id="brief-123",
            workflow_type="content_creation"
        )
        
        if workflow:
            print(f"Created workflow: {workflow.name}")
            print(f"Steps: {len(workflow.steps)}")
            print(f"Status: {workflow.status}")
        else:
            print("Failed to create workflow")

    asyncio.run(main())