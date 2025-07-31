"""
Agent Coordinator for SEO Content Knowledge Graph System.

This service orchestrates the execution of multiple AI agents, manages workflows,
and provides a unified interface for complex multi-agent operations.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pydantic import BaseModel, Field
import json

from .base_agent import BaseAgent, AgentContext, AgentResult, agent_registry
from .content_analysis_agent import content_analysis_agent
from .seo_research_agent import seo_research_agent
from .content_generation_agent import content_generation_agent
from .graph_management_agent import graph_management_agent
from .quality_assurance_agent import quality_assurance_agent

logger = logging.getLogger(__name__)


class WorkflowRequest(BaseModel):
    """Request model for multi-agent workflows."""
    workflow_type: str = Field(default="content_creation")  # content_creation, content_optimization, research_analysis, graph_maintenance
    input_data: Dict[str, Any] = Field(default_factory=dict)
    agent_sequence: Optional[List[str]] = None
    parallel_execution: bool = False
    quality_threshold: float = 0.8
    auto_retry_on_failure: bool = True
    max_retries: int = 2
    include_analytics: bool = True


class AgentCoordinator:
    """
    Coordinator service for managing multi-agent workflows and operations.
    
    Capabilities:
    - Multi-agent workflow orchestration
    - Sequential and parallel agent execution
    - Result aggregation and analysis
    - Quality assurance integration
    - Error handling and retry logic
    - Performance monitoring and optimization
    - Workflow analytics and reporting
    """
    
    def __init__(self):
        """Initialize the agent coordinator."""
        self.agents = agent_registry.get_all_agents()
        self.workflow_templates = self._initialize_workflow_templates()
        
        logger.info(f"Agent Coordinator initialized with {len(self.agents)} agents")
    
    def _initialize_workflow_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize predefined workflow templates."""
        return {
            "content_creation": {
                "description": "Complete content creation workflow from research to publication",
                "agents": ["seo_research", "content_generation", "quality_assurance"],
                "parallel_steps": [],
                "dependencies": {
                    "content_generation": ["seo_research"],
                    "quality_assurance": ["content_generation"]
                }
            },
            "content_optimization": {
                "description": "Optimize existing content for better performance",
                "agents": ["content_analysis", "seo_research", "quality_assurance"],
                "parallel_steps": [["content_analysis", "seo_research"]],
                "dependencies": {
                    "quality_assurance": ["content_analysis", "seo_research"]
                }
            },
            "research_analysis": {
                "description": "Comprehensive research and competitive analysis",
                "agents": ["seo_research", "content_analysis"],
                "parallel_steps": [["seo_research", "content_analysis"]],
                "dependencies": {}
            },
            "graph_maintenance": {
                "description": "Knowledge graph maintenance and optimization",
                "agents": ["graph_management", "content_analysis"],
                "parallel_steps": [],
                "dependencies": {
                    "content_analysis": ["graph_management"]
                }
            },
            "quality_audit": {
                "description": "Comprehensive quality audit of content",
                "agents": ["content_analysis", "quality_assurance"],
                "parallel_steps": [],
                "dependencies": {
                    "quality_assurance": ["content_analysis"]
                }
            }
        }
    
    async def execute_workflow(self, request: WorkflowRequest, context: AgentContext) -> Dict[str, Any]:
        """Execute a multi-agent workflow."""
        workflow_start = datetime.now()
        
        try:
            # Get workflow template or use custom sequence
            if request.agent_sequence:
                workflow = self._create_custom_workflow(request.agent_sequence)
            else:
                workflow = self.workflow_templates.get(request.workflow_type)
                if not workflow:
                    raise ValueError(f"Unknown workflow type: {request.workflow_type}")
            
            logger.info(f"Starting workflow: {request.workflow_type}")
            
            # Execute workflow
            if request.parallel_execution and workflow.get("parallel_steps"):
                results = await self._execute_parallel_workflow(workflow, request, context)
            else:
                results = await self._execute_sequential_workflow(workflow, request, context)
            
            # Aggregate and analyze results
            aggregated_results = await self._aggregate_workflow_results(results, request, context)
            
            # Calculate execution time
            execution_time = (datetime.now() - workflow_start).total_seconds()
            
            return {
                "workflow_type": request.workflow_type,
                "success": True,
                "execution_time_seconds": execution_time,
                "agent_results": results,
                "aggregated_results": aggregated_results,
                "workflow_analytics": await self._generate_workflow_analytics(results, execution_time),
                "quality_check": await self._perform_workflow_quality_check(aggregated_results, request.quality_threshold),
                "completed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            execution_time = (datetime.now() - workflow_start).total_seconds()
            logger.error(f"Workflow execution failed: {e}")
            
            return {
                "workflow_type": request.workflow_type,
                "success": False,
                "error": str(e),
                "execution_time_seconds": execution_time,
                "failed_at": datetime.now().isoformat()
            }
    
    async def execute_single_agent(self, agent_name: str, task_data: Dict[str, Any], 
                                 context: AgentContext) -> AgentResult:
        """Execute a single agent with retry logic."""
        agent = self.agents.get(agent_name)
        if not agent:
            raise ValueError(f"Agent not found: {agent_name}")
        
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Executing agent {agent_name} (attempt {attempt + 1})")
                result = await agent.execute(task_data, context)
                
                if result.success:
                    return result
                else:
                    last_error = result.error_message
                    logger.warning(f"Agent {agent_name} failed: {last_error}")
                    
            except Exception as e:
                last_error = str(e)
                logger.error(f"Agent {agent_name} execution error: {e}")
            
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # Create failed result
        return AgentResult(
            success=False,
            agent_name=agent_name,
            task_type="single_execution",
            error_message=f"Failed after {max_retries} attempts: {last_error}"
        )
    
    async def _execute_sequential_workflow(self, workflow: Dict[str, Any], request: WorkflowRequest,
                                         context: AgentContext) -> Dict[str, AgentResult]:
        """Execute workflow agents sequentially."""
        results = {}
        agent_sequence = workflow["agents"]
        dependencies = workflow.get("dependencies", {})
        
        for agent_name in agent_sequence:
            # Check dependencies
            if agent_name in dependencies:
                deps = dependencies[agent_name]
                for dep in deps:
                    if dep not in results or not results[dep].success:
                        raise Exception(f"Dependency {dep} failed for agent {agent_name}")
            
            # Prepare task data (may include results from previous agents)
            task_data = self._prepare_agent_task_data(agent_name, request.input_data, results)
            
            # Execute agent
            result = await self.execute_single_agent(agent_name, task_data, context)
            results[agent_name] = result
            
            if not result.success and not request.auto_retry_on_failure:
                logger.error(f"Agent {agent_name} failed, stopping workflow")
                break
        
        return results
    
    async def _execute_parallel_workflow(self, workflow: Dict[str, Any], request: WorkflowRequest,
                                       context: AgentContext) -> Dict[str, AgentResult]:
        """Execute workflow agents in parallel where possible."""
        results = {}
        parallel_steps = workflow.get("parallel_steps", [])
        remaining_agents = set(workflow["agents"])
        
        # Execute parallel steps
        for step in parallel_steps:
            tasks = []
            for agent_name in step:
                if agent_name in remaining_agents:
                    task_data = self._prepare_agent_task_data(agent_name, request.input_data, results)
                    tasks.append(self._execute_agent_task(agent_name, task_data, context))
                    remaining_agents.remove(agent_name)
            
            if tasks:
                step_results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, result in enumerate(step_results):
                    agent_name = step[i]
                    if isinstance(result, Exception):
                        results[agent_name] = AgentResult(
                            success=False,
                            agent_name=agent_name,
                            task_type="parallel_execution",
                            error_message=str(result)
                        )
                    else:
                        results[agent_name] = result
        
        # Execute remaining agents sequentially
        for agent_name in remaining_agents:
            task_data = self._prepare_agent_task_data(agent_name, request.input_data, results)
            result = await self.execute_single_agent(agent_name, task_data, context)
            results[agent_name] = result
        
        return results
    
    async def _execute_agent_task(self, agent_name: str, task_data: Dict[str, Any],
                                context: AgentContext) -> AgentResult:
        """Execute a single agent task (helper for parallel execution)."""
        return await self.execute_single_agent(agent_name, task_data, context)
    
    def _prepare_agent_task_data(self, agent_name: str, input_data: Dict[str, Any],
                                previous_results: Dict[str, AgentResult]) -> Dict[str, Any]:
        """Prepare task data for agent execution, including previous results."""
        task_data = input_data.copy()
        
        # Add relevant data from previous agent results
        if agent_name == "content_generation" and "seo_research" in previous_results:
            seo_result = previous_results["seo_research"]
            if seo_result.success:
                research_data = seo_result.result_data
                task_data.update({
                    "target_keywords": research_data.get("priority_keywords", []),
                    "competitor_analysis_data": research_data.get("competitor_analysis", {}),
                    "seo_requirements": research_data.get("seo_recommendations", {})
                })
        
        elif agent_name == "quality_assurance" and "content_generation" in previous_results:
            content_result = previous_results["content_generation"]
            if content_result.success:
                content_data = content_result.result_data
                task_data.update({
                    "content_text": content_data.get("content", ""),
                    "content_title": content_data.get("title", ""),
                    "target_keywords": content_data.get("target_keywords", [])
                })
        
        elif agent_name == "content_analysis" and previous_results:
            # Content analysis can benefit from any existing content
            for result in previous_results.values():
                if result.success and "content" in result.result_data:
                    task_data["content_text"] = result.result_data["content"]
                    break
        
        return task_data
    
    async def _aggregate_workflow_results(self, results: Dict[str, AgentResult], request: WorkflowRequest,
                                        context: AgentContext) -> Dict[str, Any]:
        """Aggregate results from multiple agents into unified insights."""
        aggregated = {
            "successful_agents": [],
            "failed_agents": [],
            "key_insights": [],
            "recommendations": [],
            "metrics": {}
        }
        
        # Categorize results
        for agent_name, result in results.items():
            if result.success:
                aggregated["successful_agents"].append(agent_name)
                
                # Extract key insights
                insights = self._extract_agent_insights(agent_name, result)
                aggregated["key_insights"].extend(insights)
                
                # Extract recommendations
                recommendations = self._extract_agent_recommendations(agent_name, result)
                aggregated["recommendations"].extend(recommendations)
                
                # Extract metrics
                metrics = self._extract_agent_metrics(agent_name, result)
                aggregated["metrics"].update(metrics)
                
            else:
                aggregated["failed_agents"].append({
                    "agent": agent_name,
                    "error": result.error_message
                })
        
        # Generate workflow-level insights
        workflow_insights = await self._generate_workflow_insights(results, request.workflow_type)
        aggregated["workflow_insights"] = workflow_insights
        
        # Generate consolidated recommendations
        consolidated_recommendations = self._consolidate_recommendations(aggregated["recommendations"])
        aggregated["consolidated_recommendations"] = consolidated_recommendations
        
        return aggregated
    
    def _extract_agent_insights(self, agent_name: str, result: AgentResult) -> List[str]:
        """Extract key insights from agent results."""
        insights = []
        
        if agent_name == "seo_research":
            research_data = result.result_data
            if "priority_keywords" in research_data:
                insights.append(f"Identified {len(research_data['priority_keywords'])} priority keywords")
            if "competitor_analysis" in research_data:
                insights.append("Completed competitive landscape analysis")
        
        elif agent_name == "content_generation":
            content_data = result.result_data
            if "word_count" in content_data:
                insights.append(f"Generated {content_data['word_count']} words of content")
            if "seo_score" in content_data:
                insights.append(f"Content SEO score: {content_data['seo_score']}")
        
        elif agent_name == "quality_assurance":
            qa_data = result.result_data
            if "overall_quality_score" in qa_data:
                score = qa_data["overall_quality_score"]
                insights.append(f"Content quality score: {score:.1%}")
        
        elif agent_name == "content_analysis":
            analysis_data = result.result_data
            if "topic_analysis" in analysis_data:
                insights.append("Completed content topic analysis")
        
        elif agent_name == "graph_management":
            graph_data = result.result_data
            if "health_score" in graph_data:
                insights.append(f"Graph health score: {graph_data['health_score']}")
        
        return insights
    
    def _extract_agent_recommendations(self, agent_name: str, result: AgentResult) -> List[Dict[str, Any]]:
        """Extract recommendations from agent results."""
        recommendations = []
        
        if "recommendations" in result.result_data:
            agent_recommendations = result.result_data["recommendations"]
            if isinstance(agent_recommendations, list):
                for rec in agent_recommendations:
                    if isinstance(rec, str):
                        recommendations.append({
                            "source": agent_name,
                            "type": "general",
                            "description": rec,
                            "priority": "medium"
                        })
                    elif isinstance(rec, dict):
                        rec["source"] = agent_name
                        recommendations.append(rec)
        
        return recommendations
    
    def _extract_agent_metrics(self, agent_name: str, result: AgentResult) -> Dict[str, Any]:
        """Extract metrics from agent results."""
        metrics = {}
        
        # Common metrics
        metrics[f"{agent_name}_execution_time"] = result.execution_time_ms
        metrics[f"{agent_name}_confidence"] = result.confidence_score
        
        # Agent-specific metrics
        if agent_name == "seo_research" and "keyword_opportunities" in result.result_data:
            metrics["keywords_discovered"] = len(result.result_data["keyword_opportunities"].get("all_opportunities", []))
        
        elif agent_name == "content_generation" and "word_count" in result.result_data:
            metrics["content_word_count"] = result.result_data["word_count"]
            
        elif agent_name == "quality_assurance" and "overall_quality_score" in result.result_data:
            metrics["quality_score"] = result.result_data["overall_quality_score"]
        
        return metrics
    
    async def _generate_workflow_insights(self, results: Dict[str, AgentResult], workflow_type: str) -> List[str]:
        """Generate workflow-level insights."""
        insights = []
        
        successful_agents = [name for name, result in results.items() if result.success]
        failed_agents = [name for name, result in results.items() if not result.success]
        
        insights.append(f"Workflow completion: {len(successful_agents)}/{len(results)} agents successful")
        
        if workflow_type == "content_creation":
            if "seo_research" in successful_agents and "content_generation" in successful_agents:
                insights.append("Successfully integrated SEO research into content generation")
            if "quality_assurance" in successful_agents:
                insights.append("Content passed quality assurance checks")
        
        elif workflow_type == "content_optimization":
            if "content_analysis" in successful_agents:
                insights.append("Baseline content analysis completed")
            if "seo_research" in successful_agents:
                insights.append("Optimization opportunities identified")
        
        if failed_agents:
            insights.append(f"Failed agents requiring attention: {', '.join(failed_agents)}")
        
        return insights
    
    def _consolidate_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Consolidate and prioritize recommendations from multiple agents."""
        # Group recommendations by type
        grouped = {}
        for rec in recommendations:
            rec_type = rec.get("type", "general")
            if rec_type not in grouped:
                grouped[rec_type] = []
            grouped[rec_type].append(rec)
        
        # Prioritize and deduplicate
        consolidated = []
        priority_order = ["high", "medium", "low"]
        
        for rec_type, type_recommendations in grouped.items():
            # Sort by priority
            sorted_recs = sorted(type_recommendations, 
                               key=lambda x: priority_order.index(x.get("priority", "medium")))
            
            # Take top recommendations per type
            consolidated.extend(sorted_recs[:3])
        
        return consolidated
    
    async def _generate_workflow_analytics(self, results: Dict[str, AgentResult], execution_time: float) -> Dict[str, Any]:
        """Generate analytics for workflow execution."""
        total_agents = len(results)
        successful_agents = sum(1 for result in results.values() if result.success)
        
        # Calculate average confidence
        confidences = [result.confidence_score for result in results.values() if result.success]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Calculate total tokens used (if available)
        total_tokens = sum(
            sum(result.tokens_used.values()) for result in results.values() 
            if result.tokens_used
        )
        
        return {
            "success_rate": successful_agents / total_agents if total_agents > 0 else 0,
            "average_confidence": avg_confidence,
            "total_execution_time": execution_time,
            "total_tokens_used": total_tokens,
            "agent_performance": {
                name: {
                    "success": result.success,
                    "execution_time": result.execution_time_ms,
                    "confidence": result.confidence_score
                }
                for name, result in results.items()
            }
        }
    
    async def _perform_workflow_quality_check(self, aggregated_results: Dict[str, Any], 
                                            quality_threshold: float) -> Dict[str, Any]:
        """Perform quality check on workflow results."""
        quality_metrics = aggregated_results.get("metrics", {})
        
        # Extract quality-related metrics
        quality_score = quality_metrics.get("quality_score", 0)
        confidence_scores = [
            metrics["confidence"] for agent, metrics in quality_metrics.get("agent_performance", {}).items()
            if "confidence" in metrics
        ]
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # Overall quality assessment
        overall_quality = (quality_score + avg_confidence) / 2 if quality_score > 0 else avg_confidence
        
        meets_threshold = overall_quality >= quality_threshold
        
        return {
            "overall_quality_score": overall_quality,
            "meets_threshold": meets_threshold,
            "quality_threshold": quality_threshold,
            "individual_scores": {
                "content_quality": quality_score,
                "execution_confidence": avg_confidence
            },
            "quality_recommendations": [
                "Review failed agents and retry if necessary",
                "Consider adjusting quality threshold based on results",
                "Monitor agent performance trends"
            ] if not meets_threshold else []
        }
    
    def _create_custom_workflow(self, agent_sequence: List[str]) -> Dict[str, Any]:
        """Create a custom workflow from agent sequence."""
        return {
            "description": "Custom workflow",
            "agents": agent_sequence,
            "parallel_steps": [],
            "dependencies": {}
        }
    
    # Additional workflow templates and utilities
    
    async def get_workflow_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get available workflow templates."""
        return self.workflow_templates
    
    async def get_agent_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Get capabilities of all registered agents."""
        capabilities = {}
        
        for agent_name, agent in self.agents.items():
            capabilities[agent_name] = {
                "name": agent.name,
                "description": agent.description,
                "available": True
            }
        
        return capabilities
    
    async def validate_workflow_request(self, request: WorkflowRequest) -> Dict[str, Any]:
        """Validate workflow request and check agent availability."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check workflow type
        if request.agent_sequence:
            agents_to_check = request.agent_sequence
        else:
            workflow = self.workflow_templates.get(request.workflow_type)
            if not workflow:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Unknown workflow type: {request.workflow_type}")
                return validation_result
            agents_to_check = workflow["agents"]
        
        # Check agent availability
        for agent_name in agents_to_check:
            if agent_name not in self.agents:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Agent not available: {agent_name}")
        
        # Check input data requirements
        if not request.input_data:
            validation_result["warnings"].append("No input data provided - some agents may not function optimally")
        
        return validation_result


# Global agent coordinator instance
agent_coordinator = AgentCoordinator()