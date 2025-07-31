"""
Base Agent class for SEO Content Knowledge Graph System.

This module provides the foundation for all AI agents with standardized
multi-tenant operations, logging, and integration with graph-vector services.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
import json

from ..database.neo4j_client import neo4j_client
from ..database.qdrant_client import qdrant_client
from ..database.supabase_client import supabase_client
from ..services.graph_vector_service import graph_vector_service

logger = logging.getLogger(__name__)


class AgentContext(BaseModel):
    """Context information passed to all agents."""
    organization_id: str
    user_id: Optional[str] = None
    brand_voice_config: Dict[str, Any] = Field(default_factory=dict)
    seo_preferences: Dict[str, Any] = Field(default_factory=dict)
    industry_context: str = ""
    session_id: Optional[str] = None
    task_id: Optional[str] = None


class AgentResult(BaseModel):
    """Standardized result format for all agents."""
    success: bool
    agent_name: str
    task_type: str
    result_data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float = 0
    tokens_used: Dict[str, int] = Field(default_factory=dict)
    error_message: Optional[str] = None
    recommendations: List[str] = Field(default_factory=list)
    confidence_score: float = 0.0
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class BaseAgent(ABC):
    """
    Abstract base class for all SEO Content Knowledge Graph agents.
    
    Provides standardized functionality for:
    - Multi-tenant context management
    - Graph and vector database integration
    - Logging and observability
    - Error handling and recovery
    - Result standardization
    """
    
    def __init__(self, name: str, description: str):
        """Initialize the base agent."""
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"agents.{name}")
        self._current_context: Optional[AgentContext] = None
        
        # Initialize Pydantic AI agent
        self._agent = Agent(
            model='openai:gpt-4o-mini',
            system_prompt=self._get_system_prompt(),
            retries=2
        )
        
        # Register tools with the agent
        self._register_tools()
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        pass
    
    @abstractmethod
    def _register_tools(self) -> None:
        """Register agent-specific tools."""
        pass
    
    async def execute(self, task_data: Dict[str, Any], context: AgentContext) -> AgentResult:
        """
        Main execution method for the agent.
        
        Args:
            task_data: Task-specific input data
            context: Agent execution context with organization info
            
        Returns:
            AgentResult: Standardized result with success status and data
        """
        start_time = datetime.now()
        self._current_context = context
        
        try:
            # Set organization context for all services
            self._set_organization_context(context.organization_id)
            
            # Log task start
            self.logger.info(f"Starting {self.name} task", extra={
                'organization_id': context.organization_id,
                'task_type': task_data.get('type', 'unknown'),
                'session_id': context.session_id
            })
            
            # Execute the specific agent task
            result_data = await self._execute_task(task_data, context)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create success result
            result = AgentResult(
                success=True,
                agent_name=self.name,
                task_type=task_data.get('type', 'unknown'),
                result_data=result_data,
                execution_time_ms=execution_time,
                metadata={
                    'organization_id': context.organization_id,
                    'user_id': context.user_id,
                    'session_id': context.session_id,
                    'task_id': context.task_id
                }
            )
            
            # Log successful completion
            self.logger.info(f"Completed {self.name} task successfully", extra={
                'execution_time_ms': execution_time,
                'organization_id': context.organization_id
            })
            
            return result
            
        except Exception as e:
            # Calculate execution time even for failures
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Log error
            self.logger.error(f"Failed to execute {self.name} task: {e}", extra={
                'organization_id': context.organization_id,
                'error': str(e),
                'execution_time_ms': execution_time
            })
            
            # Create error result
            return AgentResult(
                success=False,
                agent_name=self.name,
                task_type=task_data.get('type', 'unknown'),
                execution_time_ms=execution_time,
                error_message=str(e),
                metadata={
                    'organization_id': context.organization_id,
                    'user_id': context.user_id,
                    'session_id': context.session_id,
                    'task_id': context.task_id
                }
            )
    
    @abstractmethod
    async def _execute_task(self, task_data: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        """
        Execute the specific agent task. Must be implemented by subclasses.
        
        Args:
            task_data: Task-specific input data
            context: Agent execution context
            
        Returns:
            Dict[str, Any]: Task result data
        """
        pass
    
    def _set_organization_context(self, organization_id: str) -> None:
        """Set organization context for all database services."""
        try:
            neo4j_client.set_organization_context(organization_id)
            qdrant_client.set_organization_context(organization_id)
            graph_vector_service.set_organization_context(organization_id)
        except Exception as e:
            self.logger.warning(f"Failed to set organization context: {e}")
    
    async def _get_brand_voice_config(self) -> Dict[str, Any]:
        """Get brand voice configuration for current organization."""
        if not self._current_context:
            return {}
        
        if self._current_context.brand_voice_config:
            return self._current_context.brand_voice_config
        
        # Fallback to fetching from database
        try:
            org_data = await supabase_client.get_organization(self._current_context.organization_id)
            if org_data and 'brand_voice_config' in org_data:
                return org_data['brand_voice_config']
        except Exception as e:
            self.logger.warning(f"Failed to fetch brand voice config: {e}")
        
        return {}
    
    async def _get_seo_preferences(self) -> Dict[str, Any]:
        """Get SEO preferences for current organization."""
        if not self._current_context:
            return {}
        
        if self._current_context.seo_preferences:
            return self._current_context.seo_preferences
        
        # Fallback to fetching from database
        try:
            org_data = await supabase_client.get_organization(self._current_context.organization_id)
            if org_data and 'seo_preferences' in org_data:
                return org_data['seo_preferences']
        except Exception as e:
            self.logger.warning(f"Failed to fetch SEO preferences: {e}")
        
        return {
            'target_keyword_density': 1.5,
            'content_length_preference': 'medium',
            'internal_linking_style': 'contextual'
        }
    
    async def _store_result(self, result: AgentResult) -> bool:
        """Store agent result for future reference and analytics."""
        try:
            # Store in Supabase for analytics
            await supabase_client.create_content_task({
                'task_type': f"{self.name}:{result.task_type}",
                'status': 'completed' if result.success else 'failed',
                'result_data': result.result_data,
                'metadata': result.metadata,
                'execution_time_ms': result.execution_time_ms,
                'error_message': result.error_message,
                'confidence_score': result.confidence_score
            })
            return True
        except Exception as e:
            self.logger.warning(f"Failed to store agent result: {e}")
            return False
    
    # Common utility methods for all agents
    
    def _extract_keywords_from_text(self, text: str, max_keywords: int = 20) -> List[str]:
        """Extract potential keywords from text using simple heuristics."""
        import re
        from collections import Counter
        
        # Simple keyword extraction (in production, use proper NLP)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 
            'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these', 
            'those', 'will', 'would', 'could', 'should', 'can', 'may', 'might'
        }
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Get most common words
        word_counts = Counter(filtered_words)
        return [word for word, count in word_counts.most_common(max_keywords)]
    
    def _calculate_readability_score(self, text: str) -> float:
        """Calculate simple readability score (Flesch-Kincaid approximation)."""
        import re
        
        sentences = len(re.findall(r'[.!?]+', text))
        words = len(text.split())
        syllables = sum([self._count_syllables(word) for word in text.split()])
        
        if sentences == 0 or words == 0:
            return 0.0
        
        # Simplified Flesch Reading Ease
        score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
        return max(0, min(100, score))
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (approximation)."""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _validate_content_length(self, content: str, target_length: str = 'medium') -> Dict[str, Any]:
        """Validate content length against SEO preferences."""
        word_count = len(content.split())
        
        length_ranges = {
            'short': (300, 800),
            'medium': (800, 1500),
            'long': (1500, 3000)
        }
        
        min_words, max_words = length_ranges.get(target_length, (800, 1500))
        
        return {
            'word_count': word_count,
            'target_range': f"{min_words}-{max_words}",
            'meets_target': min_words <= word_count <= max_words,
            'recommendation': self._get_length_recommendation(word_count, min_words, max_words)
        }
    
    def _get_length_recommendation(self, word_count: int, min_words: int, max_words: int) -> str:
        """Get content length recommendation."""
        if word_count < min_words:
            needed = min_words - word_count
            return f"Add approximately {needed} words to meet minimum length requirement"
        elif word_count > max_words:
            excess = word_count - max_words
            return f"Consider reducing content by approximately {excess} words"
        else:
            return "Content length is within optimal range"


class AgentRegistry:
    """Registry for managing and discovering available agents."""
    
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
    
    def register(self, agent: BaseAgent) -> None:
        """Register an agent in the registry."""
        self._agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name."""
        return self._agents.get(name)
    
    def list_agents(self) -> List[str]:
        """List all registered agent names."""
        return list(self._agents.keys())
    
    def get_all_agents(self) -> Dict[str, BaseAgent]:
        """Get all registered agents."""
        return self._agents.copy()


# Global agent registry
agent_registry = AgentRegistry()