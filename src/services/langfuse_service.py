"""
Langfuse Monitoring Service.

Provides comprehensive AI call monitoring, error tracking, and performance
analytics for all agents in the TMG_conGen system.
"""

import asyncio
import logging
import functools
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import json
import os

try:
    from langfuse import Langfuse
    from langfuse.decorators import observe
    LANGFUSE_AVAILABLE = True
except ImportError:
    # Graceful degradation if Langfuse is not installed
    LANGFUSE_AVAILABLE = False
    logging.warning("Langfuse not available - monitoring will be disabled")

from config.settings import get_settings

logger = logging.getLogger(__name__)


class LangfuseService:
    """
    Langfuse monitoring service for AI agent observability.
    
    Provides monitoring decorators, error tracking, and performance analytics
    for all AI agents with multi-tenant support.
    """
    
    def __init__(self):
        """Initialize Langfuse monitoring service."""
        self.enabled = False
        self.client = None
        
        if LANGFUSE_AVAILABLE:
            try:
                settings = get_settings()
                
                # Check if Langfuse keys are configured
                if settings.langfuse_public_key and settings.langfuse_secret_key:
                    self.client = Langfuse(
                        public_key=settings.langfuse_public_key,
                        secret_key=settings.langfuse_secret_key,
                        host=settings.langfuse_host
                    )
                    self.enabled = True
                    logger.info("✅ Langfuse monitoring service initialized")
                else:
                    logger.warning("Langfuse keys not configured - monitoring disabled")
            except Exception as e:
                logger.error(f"Failed to initialize Langfuse: {e}")
                self.enabled = False
        else:
            logger.info("Langfuse not available - monitoring disabled")
    
    def monitor_agent_call(self, agent_name: str, task_type: str):
        """
        Decorator to monitor AI agent calls with Langfuse.
        
        Args:
            agent_name: Name of the agent being monitored
            task_type: Type of task being executed
            
        Returns:
            Decorated function with monitoring
        """
        def decorator(func: Callable) -> Callable:
            if not self.enabled:
                # Return original function if monitoring disabled
                return func
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                """Async wrapper for monitoring."""
                start_time = datetime.now()
                
                # Extract context information
                context_info = self._extract_context_info(args, kwargs)
                organization_id = context_info.get('organization_id', 'unknown')
                
                try:
                    # Create Langfuse trace
                    trace = self.client.trace(
                        name=f"{agent_name}_{task_type}",
                        user_id=context_info.get('user_id', 'anonymous'),
                        session_id=context_info.get('session_id'),
                        metadata={
                            "agent_name": agent_name,
                            "task_type": task_type,
                            "organization_id": organization_id,
                            "timestamp": start_time.isoformat()
                        }
                    )
                    
                    # Execute the original function
                    result = await func(*args, **kwargs)
                    
                    # Calculate execution time
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    # Log successful execution
                    trace.update(
                        output=self._sanitize_output(result),
                        metadata={
                            "execution_time_seconds": execution_time,
                            "success": True,
                            "organization_id": organization_id
                        }
                    )
                    
                    # Create generation span for AI calls
                    if hasattr(result, 'tokens_used'):
                        self._log_token_usage(trace, result, agent_name, task_type)
                    
                    logger.debug(f"📊 Langfuse: {agent_name}.{task_type} completed in {execution_time:.2f}s")
                    
                    return result
                    
                except Exception as e:
                    # Log error execution
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    trace.update(
                        output={"error": str(e)},
                        metadata={
                            "execution_time_seconds": execution_time,
                            "success": False,
                            "error_type": type(e).__name__,
                            "organization_id": organization_id
                        }
                    )
                    
                    logger.error(f"📊 Langfuse: {agent_name}.{task_type} failed after {execution_time:.2f}s: {e}")
                    
                    # Re-raise the original exception
                    raise
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                """Sync wrapper for monitoring."""
                if asyncio.iscoroutinefunction(func):
                    return async_wrapper(*args, **kwargs)
                
                start_time = datetime.now()
                context_info = self._extract_context_info(args, kwargs)
                organization_id = context_info.get('organization_id', 'unknown')
                
                try:
                    trace = self.client.trace(
                        name=f"{agent_name}_{task_type}",
                        user_id=context_info.get('user_id', 'anonymous'),
                        session_id=context_info.get('session_id'),
                        metadata={
                            "agent_name": agent_name,
                            "task_type": task_type,
                            "organization_id": organization_id,
                            "timestamp": start_time.isoformat()
                        }
                    )
                    
                    result = func(*args, **kwargs)
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    trace.update(
                        output=self._sanitize_output(result),
                        metadata={
                            "execution_time_seconds": execution_time,
                            "success": True,
                            "organization_id": organization_id
                        }
                    )
                    
                    return result
                    
                except Exception as e:
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    trace.update(
                        output={"error": str(e)},
                        metadata={
                            "execution_time_seconds": execution_time,
                            "success": False,
                            "error_type": type(e).__name__,
                            "organization_id": organization_id
                        }
                    )
                    
                    raise
            
            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def log_json_parsing_error(self, agent_name: str, raw_response: str, 
                             error: str, organization_id: str = "unknown"):
        """Log specific JSON parsing errors for debugging."""
        if not self.enabled:
            return
        
        try:
            self.client.trace(
                name=f"{agent_name}_json_parsing_error",
                metadata={
                    "error_type": "json_parsing_error",
                    "agent_name": agent_name,
                    "organization_id": organization_id,
                    "error_message": error,
                    "raw_response_preview": raw_response[:500],
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            logger.info(f"📊 Langfuse: JSON parsing error logged for {agent_name}")
            
        except Exception as e:
            logger.error(f"Failed to log JSON parsing error to Langfuse: {e}")
    
    def log_ai_api_call(self, agent_name: str, model: str, prompt: str, 
                       response: str, tokens_used: int, organization_id: str = "unknown"):
        """Log individual AI API calls for detailed monitoring."""
        if not self.enabled:
            return
        
        try:
            generation = self.client.generation(
                name=f"{agent_name}_ai_call",
                model=model,
                input=prompt[:1000],  # Truncate long prompts
                output=response[:1000],  # Truncate long responses
                usage={
                    "total_tokens": tokens_used,
                    "prompt_tokens": int(tokens_used * 0.7),  # Estimate
                    "completion_tokens": int(tokens_used * 0.3)  # Estimate
                },
                metadata={
                    "agent_name": agent_name,
                    "organization_id": organization_id,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            logger.debug(f"📊 Langfuse: AI API call logged for {agent_name}")
            
        except Exception as e:
            logger.error(f"Failed to log AI API call to Langfuse: {e}")
    
    def _extract_context_info(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Extract context information from function arguments."""
        context_info = {}
        
        # Look for AgentContext in arguments
        for arg in args:
            if hasattr(arg, 'organization_id'):
                context_info['organization_id'] = getattr(arg, 'organization_id', 'unknown')
                context_info['user_id'] = getattr(arg, 'user_id', 'anonymous')
                context_info['session_id'] = getattr(arg, 'session_id', None)
                break
        
        # Also check kwargs
        if 'context' in kwargs and hasattr(kwargs['context'], 'organization_id'):
            context = kwargs['context']
            context_info['organization_id'] = getattr(context, 'organization_id', 'unknown')
            context_info['user_id'] = getattr(context, 'user_id', 'anonymous')
            context_info['session_id'] = getattr(context, 'session_id', None)
        
        return context_info
    
    def _sanitize_output(self, result: Any) -> Dict[str, Any]:
        """Sanitize result for logging (remove sensitive data)."""
        try:
            if hasattr(result, 'result_data'):
                # For AgentResult objects
                return {
                    "success": getattr(result, 'success', False),
                    "agent_name": getattr(result, 'agent_name', 'unknown'),
                    "task_type": getattr(result, 'task_type', 'unknown'),
                    "execution_time_ms": getattr(result, 'execution_time_ms', 0),
                    "confidence_score": getattr(result, 'confidence_score', 0),
                    "data_keys": list(getattr(result, 'result_data', {}).keys()) if hasattr(result, 'result_data') else [],
                    "error_message": getattr(result, 'error_message', None)
                }
            elif isinstance(result, dict):
                # For dictionary results, log keys but not full content
                return {
                    "type": "dict",
                    "keys": list(result.keys()),
                    "size": len(result)
                }
            else:
                # For other types, log basic info
                return {
                    "type": type(result).__name__,
                    "value_preview": str(result)[:100] if result else None
                }
        except Exception as e:
            logger.warning(f"Failed to sanitize output for Langfuse: {e}")
            return {"error": "Failed to sanitize output"}
    
    def _log_token_usage(self, trace, result, agent_name: str, task_type: str):
        """Log token usage information."""
        try:
            if hasattr(result, 'tokens_used') and result.tokens_used:
                tokens = result.tokens_used
                
                trace.generation(
                    name=f"{agent_name}_{task_type}_tokens",
                    model="gpt-4",  # Default model
                    usage={
                        "total_tokens": tokens.get('total', 0),
                        "prompt_tokens": tokens.get('prompt', 0),
                        "completion_tokens": tokens.get('completion', 0)
                    },
                    metadata={
                        "agent_name": agent_name,
                        "task_type": task_type
                    }
                )
        except Exception as e:
            logger.warning(f"Failed to log token usage: {e}")
    
    def flush(self):
        """Flush any pending Langfuse events."""
        if self.enabled and self.client:
            try:
                self.client.flush()
            except Exception as e:
                logger.error(f"Failed to flush Langfuse events: {e}")


# Global service instance
langfuse_service = LangfuseService()


def monitor_ai_agent(agent_name: str, task_type: str):
    """
    Convenience decorator for monitoring AI agents.
    
    Usage:
        @monitor_ai_agent("content_generation", "brief_analysis")
        async def analyze_brief(self, content, context):
            # Agent logic here
            pass
    """
    return langfuse_service.monitor_agent_call(agent_name, task_type)