"""
Comprehensive logging configuration for SEO Content Knowledge Graph System.

Provides structured logging with file rotation, different log levels, and
component-specific log files for debugging and monitoring.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_dir: str = "logs"):
    """
    Setup comprehensive logging system with file rotation and structured output.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers to avoid duplication
    root_logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-25s | %(funcName)-20s:%(lineno)-4d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 1. Console handler for immediate feedback
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # 2. Main application log with rotation
    main_handler = logging.handlers.RotatingFileHandler(
        log_path / "application.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    main_handler.setLevel(logging.DEBUG)
    main_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(main_handler)
    
    # 3. Error-only log for critical issues
    error_handler = logging.handlers.RotatingFileHandler(
        log_path / "errors.log",
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)
    
    # 4. Component-specific loggers
    setup_component_loggers(log_path, detailed_formatter)
    
    # Log the logging setup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging system initialized - Level: {log_level}, Directory: {log_path}")
    logger.info(f"Log files: application.log, errors.log, api.log, database.log, agents.log")


def setup_component_loggers(log_path: Path, formatter: logging.Formatter):
    """Setup specialized loggers for different system components."""
    
    # API requests and responses
    api_logger = logging.getLogger('api')
    api_handler = logging.handlers.RotatingFileHandler(
        log_path / "api.log",
        maxBytes=10*1024*1024,
        backupCount=3
    )
    api_handler.setFormatter(formatter)
    api_logger.addHandler(api_handler)
    api_logger.setLevel(logging.DEBUG)
    
    # Database operations
    db_logger = logging.getLogger('database')
    db_handler = logging.handlers.RotatingFileHandler(
        log_path / "database.log",
        maxBytes=5*1024*1024,
        backupCount=3
    )
    db_handler.setFormatter(formatter)
    db_logger.addHandler(db_handler)
    db_logger.setLevel(logging.DEBUG)
    
    # AI agents and processing
    agents_logger = logging.getLogger('agents')
    agents_handler = logging.handlers.RotatingFileHandler(
        log_path / "agents.log",
        maxBytes=10*1024*1024,
        backupCount=3
    )
    agents_handler.setFormatter(formatter)
    agents_logger.addHandler(agents_handler)
    agents_logger.setLevel(logging.DEBUG)
    
    # Knowledge Graph operations
    graph_logger = logging.getLogger('graph')
    graph_handler = logging.handlers.RotatingFileHandler(
        log_path / "graph.log",
        maxBytes=5*1024*1024,
        backupCount=3
    )
    graph_handler.setFormatter(formatter)
    graph_logger.addHandler(graph_handler)
    graph_logger.setLevel(logging.DEBUG)
    
    # Performance monitoring
    perf_logger = logging.getLogger('performance')
    perf_handler = logging.handlers.RotatingFileHandler(
        log_path / "performance.log",
        maxBytes=5*1024*1024,
        backupCount=3
    )
    perf_handler.setFormatter(formatter)
    perf_logger.addHandler(perf_handler)
    perf_logger.setLevel(logging.DEBUG)


def get_component_logger(component: str) -> logging.Logger:
    """
    Get a logger for a specific component.
    
    Args:
        component: Component name (api, database, agents, graph, performance)
        
    Returns:
        Configured logger for the component
    """
    return logging.getLogger(component)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")


def log_api_request(endpoint: str, method: str, status_code: int, response_time: float, **kwargs):
    """Log API request with structured data."""
    api_logger = get_component_logger('api')
    api_logger.info(
        f"API Request: {method} {endpoint} -> {status_code} ({response_time:.3f}s)",
        extra={
            'endpoint': endpoint,
            'method': method,
            'status_code': status_code,
            'response_time': response_time,
            **kwargs
        }
    )


def log_database_operation(operation: str, table: str, duration: float, success: bool = True, **kwargs):
    """Log database operations with performance metrics."""
    db_logger = get_component_logger('database')
    level = logging.INFO if success else logging.ERROR
    status = "SUCCESS" if success else "FAILED"
    
    db_logger.log(
        level,
        f"DB {operation}: {table} -> {status} ({duration:.3f}s)",
        extra={
            'operation': operation,
            'table': table,
            'duration': duration,
            'success': success,
            **kwargs
        }
    )


def log_agent_execution(agent_name: str, task_type: str, duration: float, success: bool = True, **kwargs):
    """Log AI agent execution with performance data."""
    agents_logger = get_component_logger('agents')
    level = logging.INFO if success else logging.ERROR
    status = "SUCCESS" if success else "FAILED"
    
    agents_logger.log(
        level,
        f"Agent {agent_name}: {task_type} -> {status} ({duration:.3f}s)",
        extra={
            'agent_name': agent_name,
            'task_type': task_type,
            'duration': duration,
            'success': success,
            **kwargs
        }
    )


def log_performance_metric(metric_name: str, value: float, unit: str = "", **kwargs):
    """Log performance metrics for monitoring."""
    perf_logger = get_component_logger('performance')
    perf_logger.info(
        f"METRIC {metric_name}: {value}{unit}",
        extra={
            'metric_name': metric_name,
            'value': value,
            'unit': unit,
            'timestamp': datetime.utcnow().isoformat(),
            **kwargs
        }
    )


# Debugging utilities
def dump_logs_summary(hours: int = 24):
    """Generate a summary of recent log activity."""
    summary = {
        'api_requests': 0,
        'database_operations': 0,
        'agent_executions': 0,
        'error_count': 0,
        'performance_metrics': 0
    }
    
    # This would analyze log files for the last N hours
    # Implementation would parse log files and extract metrics
    logger = logging.getLogger(__name__)
    logger.info(f"Log summary for last {hours} hours: {summary}")
    return summary


if __name__ == "__main__":
    # Test the logging setup
    setup_logging("DEBUG")
    
    # Test different loggers
    main_logger = logging.getLogger(__name__)
    main_logger.info("Testing main application logger")
    
    log_api_request("/api/test", "GET", 200, 0.123, user_id="test-user")
    log_database_operation("SELECT", "content_items", 0.045, True, rows_returned=5)
    log_agent_execution("content_analysis", "analyze_text", 2.341, True, word_count=500)
    log_performance_metric("response_time", 0.123, "s", endpoint="/api/test")
    
    main_logger.error("Testing error logging")
    
    print("Logging test complete. Check the logs/ directory for output files.")