"""
OpenTelemetry Performance Monitoring Integration.

Provides comprehensive performance monitoring, tracing, and metrics collection
for the SEO Content Knowledge Graph System using OpenTelemetry standards.
"""

import os
import time
import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from functools import wraps
import structlog

from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
from opentelemetry.instrumentation.neo4j import Neo4jInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.util.http import get_excluded_urls

logger = structlog.get_logger(__name__)


class OTelConfig:
    """OpenTelemetry configuration."""
    
    def __init__(self):
        # Service identification
        self.service_name = os.getenv("OTEL_SERVICE_NAME", "seo-content-system")
        self.service_version = os.getenv("OTEL_SERVICE_VERSION", "1.0.0")
        self.environment = os.getenv("OTEL_ENVIRONMENT", "development")
        
        # Export endpoints
        self.otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        self.jaeger_endpoint = os.getenv("OTEL_EXPORTER_JAEGER_ENDPOINT")
        self.prometheus_port = int(os.getenv("OTEL_PROMETHEUS_PORT", "9090"))
        
        # Export configuration
        self.enable_console_export = os.getenv("OTEL_ENABLE_CONSOLE", "false").lower() == "true"
        self.enable_otlp_export = bool(self.otlp_endpoint)
        self.enable_jaeger_export = bool(self.jaeger_endpoint)
        self.enable_prometheus_export = os.getenv("OTEL_ENABLE_PROMETHEUS", "true").lower() == "true"
        
        # Sampling configuration
        self.trace_sample_rate = float(os.getenv("OTEL_TRACE_SAMPLE_RATE", "1.0"))
        
        # Resource attributes
        self.resource = Resource.create({
            SERVICE_NAME: self.service_name,
            SERVICE_VERSION: self.service_version,
            "deployment.environment": self.environment,
            "service.namespace": "seo-content",
            "service.instance.id": os.getenv("HOSTNAME", "localhost")
        })


class OTelMonitor:
    """
    OpenTelemetry Performance Monitor.
    
    Provides comprehensive performance monitoring with:
    - Distributed tracing for request flows
    - Custom metrics for business logic
    - Automatic instrumentation for frameworks
    - Performance dashboards and alerting
    """
    
    def __init__(self, config: Optional[OTelConfig] = None):
        self.config = config or OTelConfig()
        self.logger = structlog.get_logger(__name__)
        
        # OpenTelemetry components
        self.tracer_provider = None
        self.meter_provider = None
        self.tracer = None
        self.meter = None
        
        # Custom metrics
        self.metrics = {}
        
        # Performance tracking
        self.performance_cache = {}
        
        # Agent performance tracking
        self.agent_metrics = {
            "request_count": None,
            "request_duration": None,
            "error_count": None,
            "token_usage": None
        }
        
        # Database performance tracking
        self.db_metrics = {
            "query_count": None,
            "query_duration": None,
            "connection_count": None
        }
        
        # System metrics
        self.system_metrics = {
            "cpu_usage": None,
            "memory_usage": None,
            "disk_usage": None,
            "active_connections": None
        }
    
    async def initialize(self):
        """Initialize OpenTelemetry monitoring."""
        try:
            # Initialize tracing
            await self._setup_tracing()
            
            # Initialize metrics
            await self._setup_metrics()
            
            # Setup automatic instrumentation
            await self._setup_instrumentation()
            
            # Initialize custom metrics
            await self._initialize_custom_metrics()
            
            self.logger.info(
                "OpenTelemetry monitoring initialized",
                service_name=self.config.service_name,
                otlp_enabled=self.config.enable_otlp_export,
                jaeger_enabled=self.config.enable_jaeger_export,
                prometheus_enabled=self.config.enable_prometheus_export
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenTelemetry: {e}")
            raise
    
    async def _setup_tracing(self):
        """Setup distributed tracing."""
        # Create tracer provider
        self.tracer_provider = TracerProvider(resource=self.config.resource)
        
        # Add span processors/exporters
        exporters = []
        
        # Console exporter for development
        if self.config.enable_console_export:
            console_exporter = ConsoleSpanExporter()
            self.tracer_provider.add_span_processor(
                BatchSpanProcessor(console_exporter)
            )
            exporters.append("console")
        
        # OTLP exporter for observability platforms
        if self.config.enable_otlp_export:
            otlp_exporter = OTLPSpanExporter(endpoint=self.config.otlp_endpoint)
            self.tracer_provider.add_span_processor(
                BatchSpanProcessor(otlp_exporter)
            )
            exporters.append("otlp")
        
        # Jaeger exporter for distributed tracing
        if self.config.enable_jaeger_export:
            jaeger_exporter = JaegerExporter(endpoint=self.config.jaeger_endpoint)
            self.tracer_provider.add_span_processor(
                BatchSpanProcessor(jaeger_exporter)
            )
            exporters.append("jaeger")
        
        # Set global tracer provider
        trace.set_tracer_provider(self.tracer_provider)
        self.tracer = trace.get_tracer(__name__)
        
        self.logger.info(f"Tracing initialized with exporters: {exporters}")
    
    async def _setup_metrics(self):
        """Setup metrics collection."""
        readers = []
        
        # Prometheus metrics reader
        if self.config.enable_prometheus_export:
            prometheus_reader = PrometheusMetricReader()
            readers.append(prometheus_reader)
        
        # OTLP metrics reader
        if self.config.enable_otlp_export:
            otlp_metric_exporter = OTLPMetricExporter(endpoint=self.config.otlp_endpoint)
            otlp_reader = PeriodicExportingMetricReader(
                exporter=otlp_metric_exporter,
                export_interval_millis=10000  # 10 seconds
            )
            readers.append(otlp_reader)
        
        # Create meter provider
        self.meter_provider = MeterProvider(
            resource=self.config.resource,
            metric_readers=readers
        )
        
        # Set global meter provider
        metrics.set_meter_provider(self.meter_provider)
        self.meter = metrics.get_meter(__name__)
        
        self.logger.info(f"Metrics initialized with {len(readers)} readers")
    
    async def _setup_instrumentation(self):
        """Setup automatic instrumentation."""
        try:
            # HTTP client instrumentation
            HTTPXClientInstrumentor().instrument()
            RequestsInstrumentor().instrument()
            
            # Database instrumentation
            try:
                Psycopg2Instrumentor().instrument()
            except Exception as e:
                self.logger.warning(f"Could not instrument psycopg2: {e}")
            
            try:
                Neo4jInstrumentor().instrument()
            except Exception as e:
                self.logger.warning(f"Could not instrument Neo4j: {e}")
            
            self.logger.info("Automatic instrumentation setup complete")
            
        except Exception as e:
            self.logger.error(f"Failed to setup instrumentation: {e}")
    
    async def _initialize_custom_metrics(self):
        """Initialize custom application metrics."""
        try:
            # Agent performance metrics
            self.agent_metrics["request_count"] = self.meter.create_counter(
                name="agent_requests_total",
                description="Total number of agent requests",
                unit="1"
            )
            
            self.agent_metrics["request_duration"] = self.meter.create_histogram(
                name="agent_request_duration_seconds",
                description="Agent request duration in seconds",
                unit="s"
            )
            
            self.agent_metrics["error_count"] = self.meter.create_counter(
                name="agent_errors_total",
                description="Total number of agent errors",
                unit="1"
            )
            
            self.agent_metrics["token_usage"] = self.meter.create_counter(
                name="agent_tokens_total",
                description="Total number of tokens used by agents",
                unit="1"
            )
            
            # Database performance metrics
            self.db_metrics["query_count"] = self.meter.create_counter(
                name="database_queries_total",
                description="Total number of database queries",
                unit="1"
            )
            
            self.db_metrics["query_duration"] = self.meter.create_histogram(
                name="database_query_duration_seconds",
                description="Database query duration in seconds",
                unit="s"
            )
            
            self.db_metrics["connection_count"] = self.meter.create_up_down_counter(
                name="database_connections_active",
                description="Number of active database connections",
                unit="1"
            )
            
            # System performance metrics
            self.system_metrics["cpu_usage"] = self.meter.create_gauge(
                name="system_cpu_usage_percent",
                description="System CPU usage percentage",
                unit="%"
            )
            
            self.system_metrics["memory_usage"] = self.meter.create_gauge(
                name="system_memory_usage_percent",
                description="System memory usage percentage",
                unit="%"
            )
            
            self.system_metrics["disk_usage"] = self.meter.create_gauge(
                name="system_disk_usage_percent",
                description="System disk usage percentage",
                unit="%"
            )
            
            self.system_metrics["active_connections"] = self.meter.create_gauge(
                name="system_active_connections",
                description="Number of active system connections",
                unit="1"
            )
            
            self.logger.info("Custom metrics initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize custom metrics: {e}")
    
    def trace_agent_request(self, agent_type: str, operation: str):
        """Decorator for tracing agent requests."""
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(
                    f"agent.{agent_type}.{operation}",
                    attributes={
                        "agent.type": agent_type,
                        "agent.operation": operation
                    }
                ) as span:
                    start_time = time.time()
                    try:
                        # Record request
                        if self.agent_metrics["request_count"]:
                            self.agent_metrics["request_count"].add(
                                1, {"agent_type": agent_type, "operation": operation}
                            )
                        
                        # Execute function
                        result = await func(*args, **kwargs)
                        
                        # Record success
                        span.set_status(Status(StatusCode.OK))
                        span.set_attribute("success", True)
                        
                        # Record token usage if available
                        if hasattr(result, 'usage') and result.usage:
                            tokens = getattr(result.usage, 'total_tokens', 0)
                            if tokens and self.agent_metrics["token_usage"]:
                                self.agent_metrics["token_usage"].add(
                                    tokens, {"agent_type": agent_type, "operation": operation}
                                )
                            span.set_attribute("tokens.total", tokens)
                        
                        return result
                        
                    except Exception as e:
                        # Record error
                        if self.agent_metrics["error_count"]:
                            self.agent_metrics["error_count"].add(
                                1, {"agent_type": agent_type, "operation": operation, "error": str(type(e).__name__)}
                            )
                        
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.set_attribute("success", False)
                        span.set_attribute("error", str(e))
                        raise
                        
                    finally:
                        # Record duration
                        duration = time.time() - start_time
                        if self.agent_metrics["request_duration"]:
                            self.agent_metrics["request_duration"].record(
                                duration, {"agent_type": agent_type, "operation": operation}
                            )
                        span.set_attribute("duration", duration)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(
                    f"agent.{agent_type}.{operation}",
                    attributes={
                        "agent.type": agent_type,
                        "agent.operation": operation
                    }
                ) as span:
                    start_time = time.time()
                    try:
                        # Record request
                        if self.agent_metrics["request_count"]:
                            self.agent_metrics["request_count"].add(
                                1, {"agent_type": agent_type, "operation": operation}
                            )
                        
                        # Execute function
                        result = func(*args, **kwargs)
                        
                        # Record success
                        span.set_status(Status(StatusCode.OK))
                        span.set_attribute("success", True)
                        
                        return result
                        
                    except Exception as e:
                        # Record error
                        if self.agent_metrics["error_count"]:
                            self.agent_metrics["error_count"].add(
                                1, {"agent_type": agent_type, "operation": operation, "error": str(type(e).__name__)}
                            )
                        
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.set_attribute("success", False)
                        span.set_attribute("error", str(e))
                        raise
                        
                    finally:
                        # Record duration
                        duration = time.time() - start_time
                        if self.agent_metrics["request_duration"]:
                            self.agent_metrics["request_duration"].record(
                                duration, {"agent_type": agent_type, "operation": operation}
                            )
                        span.set_attribute("duration", duration)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def trace_database_query(self, database: str, operation: str):
        """Decorator for tracing database queries."""
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(
                    f"db.{database}.{operation}",
                    attributes={
                        "db.system": database,
                        "db.operation": operation
                    }
                ) as span:
                    start_time = time.time()
                    try:
                        # Record query
                        if self.db_metrics["query_count"]:
                            self.db_metrics["query_count"].add(
                                1, {"database": database, "operation": operation}
                            )
                        
                        # Execute function
                        result = await func(*args, **kwargs)
                        
                        span.set_status(Status(StatusCode.OK))
                        span.set_attribute("success", True)
                        
                        return result
                        
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.set_attribute("success", False)
                        raise
                        
                    finally:
                        # Record duration
                        duration = time.time() - start_time
                        if self.db_metrics["query_duration"]:
                            self.db_metrics["query_duration"].record(
                                duration, {"database": database, "operation": operation}
                            )
                        span.set_attribute("duration", duration)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(
                    f"db.{database}.{operation}",
                    attributes={
                        "db.system": database,
                        "db.operation": operation
                    }
                ) as span:
                    start_time = time.time()
                    try:
                        # Record query
                        if self.db_metrics["query_count"]:
                            self.db_metrics["query_count"].add(
                                1, {"database": database, "operation": operation}
                            )
                        
                        # Execute function
                        result = func(*args, **kwargs)
                        
                        span.set_status(Status(StatusCode.OK))
                        span.set_attribute("success", True)
                        
                        return result
                        
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.set_attribute("success", False)
                        raise
                        
                    finally:
                        # Record duration
                        duration = time.time() - start_time
                        if self.db_metrics["query_duration"]:
                            self.db_metrics["query_duration"].record(
                                duration, {"database": database, "operation": operation}
                            )
                        span.set_attribute("duration", duration)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    @asynccontextmanager
    async def trace_operation(self, operation_name: str, attributes: Optional[Dict[str, Any]] = None):
        """Context manager for tracing custom operations."""
        with self.tracer.start_as_current_span(operation_name, attributes=attributes or {}) as span:
            start_time = time.time()
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            finally:
                duration = time.time() - start_time
                span.set_attribute("duration", duration)
    
    async def record_system_metrics(self, cpu_usage: float, memory_usage: float, disk_usage: float, connections: int):
        """Record system performance metrics."""
        try:
            if self.system_metrics["cpu_usage"]:
                self.system_metrics["cpu_usage"].set(cpu_usage)
            
            if self.system_metrics["memory_usage"]:
                self.system_metrics["memory_usage"].set(memory_usage)
            
            if self.system_metrics["disk_usage"]:
                self.system_metrics["disk_usage"].set(disk_usage)
            
            if self.system_metrics["active_connections"]:
                self.system_metrics["active_connections"].set(connections)
                
        except Exception as e:
            self.logger.error(f"Failed to record system metrics: {e}")
    
    async def record_custom_metric(self, name: str, value: float, attributes: Optional[Dict[str, str]] = None):
        """Record a custom metric value."""
        try:
            # Create metric if it doesn't exist
            if name not in self.metrics:
                self.metrics[name] = self.meter.create_gauge(
                    name=name,
                    description=f"Custom metric: {name}"
                )
            
            # Record value
            self.metrics[name].set(value, attributes or {})
            
        except Exception as e:
            self.logger.error(f"Failed to record custom metric {name}: {e}")
    
    def instrument_fastapi(self, app):
        """Instrument FastAPI application."""
        try:
            FastAPIInstrumentor.instrument_app(
                app,
                tracer_provider=self.tracer_provider,
                meter_provider=self.meter_provider,
                excluded_urls=get_excluded_urls()
            )
            self.logger.info("FastAPI instrumentation enabled")
        except Exception as e:
            self.logger.error(f"Failed to instrument FastAPI: {e}")
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance monitoring summary."""
        return {
            "service": {
                "name": self.config.service_name,
                "version": self.config.service_version,
                "environment": self.config.environment
            },
            "monitoring": {
                "tracing_enabled": bool(self.tracer_provider),
                "metrics_enabled": bool(self.meter_provider),
                "exporters": {
                    "otlp": self.config.enable_otlp_export,
                    "jaeger": self.config.enable_jaeger_export,
                    "prometheus": self.config.enable_prometheus_export,
                    "console": self.config.enable_console_export
                }
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def shutdown(self):
        """Shutdown OpenTelemetry monitoring."""
        try:
            if self.tracer_provider:
                self.tracer_provider.shutdown()
            
            if self.meter_provider:
                self.meter_provider.shutdown()
            
            self.logger.info("OpenTelemetry monitoring shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during OpenTelemetry shutdown: {e}")


# Global instance
otel_monitor: Optional[OTelMonitor] = None


async def initialize_otel_monitoring(config: Optional[OTelConfig] = None) -> OTelMonitor:
    """Initialize global OpenTelemetry monitoring."""
    global otel_monitor
    
    if otel_monitor is None:
        otel_monitor = OTelMonitor(config)
        await otel_monitor.initialize()
    
    return otel_monitor


def get_otel_monitor() -> Optional[OTelMonitor]:
    """Get the global OpenTelemetry monitor instance."""
    return otel_monitor


# Convenience decorators
def trace_agent(agent_type: str, operation: str):
    """Convenience decorator for agent tracing."""
    monitor = get_otel_monitor()
    if monitor:
        return monitor.trace_agent_request(agent_type, operation)
    else:
        # No-op decorator if monitoring not initialized
        def decorator(func):
            return func
        return decorator


def trace_db(database: str, operation: str):
    """Convenience decorator for database tracing."""
    monitor = get_otel_monitor()
    if monitor:
        return monitor.trace_database_query(database, operation)
    else:
        # No-op decorator if monitoring not initialized
        def decorator(func):
            return func
        return decorator


# Export public interface
__all__ = [
    "OTelConfig",
    "OTelMonitor", 
    "initialize_otel_monitoring",
    "get_otel_monitor",
    "trace_agent",
    "trace_db"
]