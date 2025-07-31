"""
Health Check and Monitoring API Routes for Production.

Provides comprehensive health monitoring for all system components including:
- SEO Dashboard API health
- SerpBear integration status
- Database connections (Neo4j, Supabase, Qdrant)
- External API status
- System resource usage
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import psutil
import time
import logging
from datetime import datetime

# Import our service clients
from src.database.neo4j_client import neo4j_client
from src.database.supabase_client import supabase_client  
from src.database.qdrant_client import qdrant_client
from src.services.serpbear_client import serpbear_client

# Import OpenTelemetry monitoring
try:
    from monitoring.otel_monitor import get_otel_monitor
    OTEL_MONITORING_AVAILABLE = True
except ImportError:
    OTEL_MONITORING_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["health"])


class HealthStatus(BaseModel):
    """Health status response model."""
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: str
    uptime_seconds: float
    version: str = "1.0.0"
    components: Dict[str, Dict[str, Any]]


class ComponentHealth(BaseModel):
    """Individual component health status."""
    status: str  # "up", "down", "degraded"
    response_time_ms: Optional[float] = None
    message: Optional[str] = None
    last_check: str


# Store startup time for uptime calculation
startup_time = time.time()


@router.get("/health")
async def health_check() -> JSONResponse:
    """
    Main health check endpoint for load balancers and monitoring.
    
    Returns:
        Simple JSON response indicating overall system health
    """
    try:
        # Quick health checks for critical components
        health_status = await _check_critical_components()
        
        if health_status["status"] == "healthy":
            return JSONResponse(
                status_code=200,
                content={
                    "status": "healthy",
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": "All systems operational"
                }
            )
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "degraded",
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": "Some components experiencing issues"
                }
            )
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "Health check system failure"
            }
        )


@router.get("/health/detailed", response_model=HealthStatus)
async def detailed_health_check() -> HealthStatus:
    """
    Comprehensive health check with component details.
    
    Returns:
        Detailed health status of all system components
    """
    current_time = datetime.utcnow().isoformat()
    uptime = time.time() - startup_time
    
    # Check all components
    components = {
        "database": await _check_databases(),
        "serpbear": await _check_serpbear(),
        "system": await _check_system_resources(),
        "external_apis": await _check_external_apis(),
        "monitoring": await _check_monitoring()
    }
    
    # Determine overall status
    component_statuses = [comp["status"] for comp in components.values()]
    
    if all(status == "up" for status in component_statuses):
        overall_status = "healthy"
    elif any(status == "down" for status in component_statuses):
        overall_status = "unhealthy"
    else:
        overall_status = "degraded"
    
    return HealthStatus(
        status=overall_status,
        timestamp=current_time,
        uptime_seconds=uptime,
        components=components
    )


async def _check_critical_components() -> Dict[str, Any]:
    """Quick check of critical components for fast health endpoint."""
    
    # Check if we can connect to primary database
    try:
        # Quick Neo4j ping
        result = await neo4j_client.execute_query("RETURN 1 as ping")
        neo4j_ok = bool(result)
    except:
        neo4j_ok = False
    
    # Check if API is responding
    api_ok = True  # If we got this far, API is responding
    
    # Quick SerpBear check
    try:
        status = serpbear_client.get_connection_status()
        serpbear_ok = status.get("api_key_configured", False)
    except:
        serpbear_ok = False
    
    critical_checks = [neo4j_ok, api_ok]
    optional_checks = [serpbear_ok]
    
    if all(critical_checks):
        return {"status": "healthy"}
    elif any(critical_checks):
        return {"status": "degraded"}
    else:
        return {"status": "unhealthy"}


async def _check_databases() -> Dict[str, Any]:
    """Check all database connections."""
    
    checks = {}
    overall_status = "up"
    
    # Neo4j Check
    try:
        start_time = time.time()
        result = await neo4j_client.execute_query("RETURN 1 as test")
        response_time = (time.time() - start_time) * 1000
        
        checks["neo4j"] = {
            "status": "up",
            "response_time_ms": round(response_time, 2),
            "message": "Connection successful"
        }
    except Exception as e:
        checks["neo4j"] = {
            "status": "down", 
            "message": f"Connection failed: {str(e)}"
        }
        overall_status = "down"
    
    # Supabase Check
    try:
        start_time = time.time()
        # Simple query to check connection
        result = supabase_client.table("organizations").select("id").limit(1).execute()
        response_time = (time.time() - start_time) * 1000
        
        checks["supabase"] = {
            "status": "up",
            "response_time_ms": round(response_time, 2),
            "message": "Connection successful"
        }
    except Exception as e:
        checks["supabase"] = {
            "status": "down",
            "message": f"Connection failed: {str(e)}"
        }
        overall_status = "down"
    
    # Qdrant Check
    try:
        start_time = time.time()
        collections = await qdrant_client.get_collections()
        response_time = (time.time() - start_time) * 1000
        
        checks["qdrant"] = {
            "status": "up",
            "response_time_ms": round(response_time, 2),
            "message": f"Found {len(collections.collections)} collections"
        }
    except Exception as e:
        checks["qdrant"] = {
            "status": "degraded",  # Qdrant is optional for basic functionality
            "message": f"Connection failed: {str(e)}"
        }
        if overall_status == "up":
            overall_status = "degraded"
    
    return {
        "status": overall_status,
        "last_check": datetime.utcnow().isoformat(),
        "details": checks
    }


async def _check_serpbear() -> Dict[str, Any]:
    """Check SerpBear integration status."""
    
    try:
        start_time = time.time()
        
        # Check connection configuration
        status = serpbear_client.get_connection_status()
        response_time = (time.time() - start_time) * 1000
        
        if not status.get("api_key_configured"):
            return {
                "status": "degraded",
                "response_time_ms": round(response_time, 2),
                "message": "API key not configured",
                "last_check": datetime.utcnow().isoformat()
            }
        
        # Test actual connection
        async with serpbear_client as client:
            domains = await client.get_domains()
            
        return {
            "status": "up",
            "response_time_ms": round(response_time, 2),
            "message": f"Connected - tracking {len(domains)} domains",
            "last_check": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "down",
            "message": f"Connection failed: {str(e)}",
            "last_check": datetime.utcnow().isoformat()
        }


async def _check_system_resources() -> Dict[str, Any]:
    """Check system resource usage."""
    
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        # Determine status based on resource usage
        if cpu_percent > 90 or memory_percent > 90 or disk_percent > 90:
            status = "degraded"
            message = "High resource usage detected"
        elif cpu_percent > 95 or memory_percent > 95 or disk_percent > 95:
            status = "down"
            message = "Critical resource usage"
        else:
            status = "up"
            message = "Resource usage normal"
        
        return {
            "status": status,
            "message": message,
            "last_check": datetime.utcnow().isoformat(),
            "details": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_percent": disk_percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_free_gb": round(disk.free / (1024**3), 2)
            }
        }
        
    except Exception as e:
        return {
            "status": "degraded",
            "message": f"Resource check failed: {str(e)}",
            "last_check": datetime.utcnow().isoformat()
        }


async def _check_external_apis() -> Dict[str, Any]:
    """Check external API connections (OpenAI, Google, etc.)."""
    
    checks = {}
    overall_status = "up"
    
    # OpenAI API Check (basic test)
    try:
        import openai
        import os
        
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            # Just check if API key is configured - don't make actual API call
            checks["openai"] = {
                "status": "up",
                "message": "API key configured"
            }
        else:
            checks["openai"] = {
                "status": "degraded",
                "message": "API key not configured"
            }
            overall_status = "degraded"
            
    except Exception as e:
        checks["openai"] = {
            "status": "down",
            "message": f"OpenAI check failed: {str(e)}"
        }
        overall_status = "degraded"
    
    # Google OAuth Check
    try:
        google_client_id = os.getenv("GOOGLE_CLIENT_ID")
        google_client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        
        if google_client_id and google_client_secret:
            checks["google_oauth"] = {
                "status": "up",
                "message": "OAuth credentials configured"
            }
        else:
            checks["google_oauth"] = {
                "status": "degraded",
                "message": "OAuth credentials not configured"
            }
            if overall_status == "up":
                overall_status = "degraded"
                
    except Exception as e:
        checks["google_oauth"] = {
            "status": "degraded",
            "message": f"Google OAuth check failed: {str(e)}"
        }
        if overall_status == "up":
            overall_status = "degraded"
    
    return {
        "status": overall_status,
        "last_check": datetime.utcnow().isoformat(),
        "details": checks
    }


@router.get("/health/ready")
async def readiness_check() -> JSONResponse:
    """
    Kubernetes readiness probe endpoint.
    
    Returns 200 when service is ready to accept traffic.
    """
    try:
        # Check if critical components are ready
        health_status = await _check_critical_components()
        
        if health_status["status"] in ["healthy", "degraded"]:
            return JSONResponse(
                status_code=200,
                content={"ready": True, "timestamp": datetime.utcnow().isoformat()}
            )
        else:
            return JSONResponse(
                status_code=503,
                content={"ready": False, "timestamp": datetime.utcnow().isoformat()}
            )
            
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"ready": False, "error": str(e)}
        )


@router.get("/health/live")
async def liveness_check() -> JSONResponse:
    """
    Kubernetes liveness probe endpoint.
    
    Returns 200 when service is alive (even if degraded).
    """
    return JSONResponse(
        status_code=200,
        content={
            "alive": True,
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": time.time() - startup_time
        }
    )


async def _check_monitoring() -> Dict[str, Any]:
    """Check OpenTelemetry monitoring status."""
    
    if not OTEL_MONITORING_AVAILABLE:
        return {
            "status": "degraded",
            "message": "OpenTelemetry monitoring not available",
            "last_check": datetime.utcnow().isoformat(),
            "details": {
                "otel_available": False,
                "tracing_enabled": False,
                "metrics_enabled": False
            }
        }
    
    try:
        otel_monitor = get_otel_monitor()
        
        if not otel_monitor:
            return {
                "status": "degraded",
                "message": "OpenTelemetry monitor not initialized",
                "last_check": datetime.utcnow().isoformat(),
                "details": {
                    "otel_available": True,
                    "monitor_initialized": False
                }
            }
        
        # Get performance summary
        performance_summary = await otel_monitor.get_performance_summary()
        
        return {
            "status": "up",
            "message": "OpenTelemetry monitoring active",
            "last_check": datetime.utcnow().isoformat(),
            "details": {
                "otel_available": True,
                "monitor_initialized": True,
                "service_name": performance_summary["service"]["name"],
                "service_version": performance_summary["service"]["version"],
                "environment": performance_summary["service"]["environment"],
                "tracing_enabled": performance_summary["monitoring"]["tracing_enabled"],
                "metrics_enabled": performance_summary["monitoring"]["metrics_enabled"],
                "exporters": performance_summary["monitoring"]["exporters"]
            }
        }
        
    except Exception as e:
        return {
            "status": "degraded",
            "message": f"Monitoring check failed: {str(e)}",
            "last_check": datetime.utcnow().isoformat()
        }


@router.get("/health/monitoring")
async def monitoring_status() -> JSONResponse:
    """
    OpenTelemetry monitoring status endpoint.
    
    Returns detailed information about telemetry configuration and status.
    """
    try:
        monitoring_check = await _check_monitoring()
        
        return JSONResponse(
            status_code=200 if monitoring_check["status"] == "up" else 503,
            content=monitoring_check
        )
        
    except Exception as e:
        logger.error(f"Monitoring status check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "down", 
                "message": f"Monitoring status check failed: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@router.get("/health/deployment")
async def deployment_health() -> JSONResponse:
    """
    Deployment-specific health check for Replit and other platforms.
    Verifies essential services and configuration for new deployments.
    """
    import os
    
    try:
        # Check essential environment variables
        essential_vars = {
            "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
            "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        }
        
        # Check optional services
        optional_services = {
            "neo4j": bool(os.getenv("NEO4J_URI")),
            "qdrant": bool(os.getenv("QDRANT_URL")),
            "supabase": bool(os.getenv("SUPABASE_URL")),
            "google_oauth": bool(os.getenv("GOOGLE_CLIENT_ID")),
            "langfuse": bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
        }
        
        # Determine overall status
        has_ai = essential_vars["OPENAI_API_KEY"]
        status = "ready" if has_ai else "limited"
        
        # Generate next steps for setup
        next_steps = []
        if not has_ai:
            next_steps.append("Add OPENAI_API_KEY to your environment secrets for AI features")
        if not any(optional_services.values()):
            next_steps.append("Add database secrets (NEO4J_URI, QDRANT_URL, SUPABASE_URL) for full functionality")
        if not next_steps:
            next_steps.append("All essential services configured! System ready for production use.")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": status,
                "timestamp": datetime.utcnow().isoformat(),
                "deployment": {
                    "platform": "replit" if os.getenv("REPL_ID") else "local",
                    "repl_id": os.getenv("REPL_ID"),
                    "essential_services": essential_vars,
                    "optional_services": optional_services,
                    "features_available": {
                        "ai_content_generation": has_ai,
                        "knowledge_graph": optional_services["neo4j"],
                        "vector_search": optional_services["qdrant"],
                        "user_authentication": optional_services["supabase"],
                        "google_oauth": optional_services["google_oauth"],
                        "llm_monitoring": optional_services["langfuse"]
                    }
                },
                "next_steps": next_steps
            }
        )
        
    except Exception as e:
        logger.error(f"Deployment health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "timestamp": datetime.utcnow().isoformat(),
                "message": f"Deployment health check failed: {str(e)}"
            }
        )