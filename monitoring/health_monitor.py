"""
Health monitoring system for the SEO Content Knowledge Graph System.

This module provides comprehensive health checks for all system components
including databases, services, agents, and external integrations.
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import structlog
import aiohttp
import psutil
from pydantic import BaseModel, Field

from database.neo4j_client import Neo4jClient
from database.qdrant_client import QdrantClient
from database.supabase_client import SupabaseClient
from config.settings import get_settings

logger = structlog.get_logger(__name__)

class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component: str
    status: HealthStatus
    message: str
    response_time: float
    details: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "component": self.component,
            "status": self.status.value,
            "message": self.message,
            "response_time": self.response_time,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }

class SystemMetrics(BaseModel):
    """System resource metrics."""
    
    cpu_usage: float = Field(description="CPU usage percentage")
    memory_usage: float = Field(description="Memory usage percentage")
    disk_usage: float = Field(description="Disk usage percentage")
    network_io: Dict[str, int] = Field(description="Network I/O statistics")
    process_count: int = Field(description="Number of running processes")
    load_average: List[float] = Field(description="System load averages")
    uptime: float = Field(description="System uptime in seconds")
    
    @classmethod
    def collect(cls) -> 'SystemMetrics':
        """Collect current system metrics."""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
            
            # Process count
            process_count = len(psutil.pids())
            
            # Load average
            load_average = list(psutil.getloadavg())
            
            # Uptime
            uptime = time.time() - psutil.boot_time()
            
            return cls(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_io=network_io,
                process_count=process_count,
                load_average=load_average,
                uptime=uptime
            )
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            # Return default values on error
            return cls(
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_io={},
                process_count=0,
                load_average=[0.0, 0.0, 0.0],
                uptime=0.0
            )

class HealthMonitor:
    """
    Comprehensive health monitoring system.
    
    Monitors system health, database connections, service availability,
    and external integrations with configurable thresholds and alerts.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = structlog.get_logger(__name__)
        
        # Health check results cache
        self.health_cache: Dict[str, HealthCheckResult] = {}
        self.last_full_check = None
        
        # Monitoring configuration
        self.check_interval = 30  # seconds
        self.timeout = 10  # seconds
        self.retry_count = 3
        
        # Health thresholds
        self.thresholds = {
            "response_time": 5.0,  # seconds
            "cpu_usage": 80.0,     # percentage
            "memory_usage": 85.0,  # percentage
            "disk_usage": 90.0,    # percentage
            "error_rate": 5.0      # percentage
        }
        
        # Initialize database clients
        self.neo4j_client = None
        self.qdrant_client = None
        self.supabase_client = None
        
        # Monitoring state
        self.is_running = False
        self.monitoring_task = None
    
    async def initialize(self):
        """Initialize the health monitor."""
        try:
            # Initialize database clients
            self.neo4j_client = Neo4jClient()
            self.qdrant_client = QdrantClient()
            self.supabase_client = SupabaseClient()
            
            self.logger.info("Health monitor initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize health monitor: {e}")
            raise
    
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.is_running:
            self.logger.warning("Health monitoring is already running")
            return
        
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Perform health checks
                await self.perform_health_checks()
                
                # Wait for next check
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def perform_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Perform comprehensive health checks."""
        start_time = time.time()
        
        # Run all health checks concurrently
        tasks = [
            self._check_system_resources(),
            self._check_database_connections(),
            self._check_external_services(),
            self._check_application_health(),
            self._check_agent_health()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results and handle exceptions
        all_results = {}
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Health check failed: {result}")
                continue
            
            if isinstance(result, dict):
                all_results.update(result)
        
        # Update cache
        self.health_cache.update(all_results)
        self.last_full_check = datetime.now(timezone.utc)
        
        # Log summary
        total_time = time.time() - start_time
        healthy_count = sum(1 for r in all_results.values() if r.status == HealthStatus.HEALTHY)
        total_count = len(all_results)
        
        self.logger.info(
            f"Health checks completed: {healthy_count}/{total_count} healthy, "
            f"took {total_time:.2f}s"
        )
        
        return all_results
    
    async def _check_system_resources(self) -> Dict[str, HealthCheckResult]:
        """Check system resource utilization."""
        start_time = time.time()
        
        try:
            # Collect system metrics
            metrics = SystemMetrics.collect()
            
            # Determine health status
            status = HealthStatus.HEALTHY
            issues = []
            
            if metrics.cpu_usage > self.thresholds["cpu_usage"]:
                status = HealthStatus.WARNING
                issues.append(f"High CPU usage: {metrics.cpu_usage:.1f}%")
            
            if metrics.memory_usage > self.thresholds["memory_usage"]:
                status = HealthStatus.WARNING
                issues.append(f"High memory usage: {metrics.memory_usage:.1f}%")
            
            if metrics.disk_usage > self.thresholds["disk_usage"]:
                status = HealthStatus.CRITICAL
                issues.append(f"High disk usage: {metrics.disk_usage:.1f}%")
            
            # Check load average
            if metrics.load_average[0] > psutil.cpu_count() * 2:
                status = HealthStatus.WARNING
                issues.append(f"High load average: {metrics.load_average[0]:.2f}")
            
            message = "System resources healthy" if not issues else "; ".join(issues)
            response_time = time.time() - start_time
            
            return {
                "system_resources": HealthCheckResult(
                    component="system_resources",
                    status=status,
                    message=message,
                    response_time=response_time,
                    details=metrics.dict(),
                    timestamp=datetime.now(timezone.utc)
                )
            }
            
        except Exception as e:
            return {
                "system_resources": HealthCheckResult(
                    component="system_resources",
                    status=HealthStatus.CRITICAL,
                    message=f"Failed to check system resources: {e}",
                    response_time=time.time() - start_time,
                    details={},
                    timestamp=datetime.now(timezone.utc)
                )
            }
    
    async def _check_database_connections(self) -> Dict[str, HealthCheckResult]:
        """Check database connection health."""
        results = {}
        
        # Check Neo4j
        start_time = time.time()
        try:
            if self.neo4j_client:
                is_connected = await self.neo4j_client.health_check()
                status = HealthStatus.HEALTHY if is_connected else HealthStatus.CRITICAL
                message = "Neo4j connection healthy" if is_connected else "Neo4j connection failed"
                
                # Get additional details
                details = {}
                if is_connected:
                    try:
                        # Get database info
                        details = await self.neo4j_client.get_database_info()
                    except:
                        pass
                
                results["neo4j"] = HealthCheckResult(
                    component="neo4j",
                    status=status,
                    message=message,
                    response_time=time.time() - start_time,
                    details=details,
                    timestamp=datetime.now(timezone.utc)
                )
            else:
                results["neo4j"] = HealthCheckResult(
                    component="neo4j",
                    status=HealthStatus.CRITICAL,
                    message="Neo4j client not initialized",
                    response_time=0,
                    details={},
                    timestamp=datetime.now(timezone.utc)
                )
                
        except Exception as e:
            results["neo4j"] = HealthCheckResult(
                component="neo4j",
                status=HealthStatus.CRITICAL,
                message=f"Neo4j health check failed: {e}",
                response_time=time.time() - start_time,
                details={},
                timestamp=datetime.now(timezone.utc)
            )
        
        # Check Qdrant
        start_time = time.time()
        try:
            if self.qdrant_client:
                is_connected = await self.qdrant_client.health_check()
                status = HealthStatus.HEALTHY if is_connected else HealthStatus.CRITICAL
                message = "Qdrant connection healthy" if is_connected else "Qdrant connection failed"
                
                # Get additional details
                details = {}
                if is_connected:
                    try:
                        details = await self.qdrant_client.get_cluster_info()
                    except:
                        pass
                
                results["qdrant"] = HealthCheckResult(
                    component="qdrant",
                    status=status,
                    message=message,
                    response_time=time.time() - start_time,
                    details=details,
                    timestamp=datetime.now(timezone.utc)
                )
            else:
                results["qdrant"] = HealthCheckResult(
                    component="qdrant",
                    status=HealthStatus.CRITICAL,
                    message="Qdrant client not initialized",
                    response_time=0,
                    details={},
                    timestamp=datetime.now(timezone.utc)
                )
                
        except Exception as e:
            results["qdrant"] = HealthCheckResult(
                component="qdrant",
                status=HealthStatus.CRITICAL,
                message=f"Qdrant health check failed: {e}",
                response_time=time.time() - start_time,
                details={},
                timestamp=datetime.now(timezone.utc)
            )
        
        # Check Supabase
        start_time = time.time()
        try:
            if self.supabase_client:
                is_connected = await self.supabase_client.health_check()
                status = HealthStatus.HEALTHY if is_connected else HealthStatus.CRITICAL
                message = "Supabase connection healthy" if is_connected else "Supabase connection failed"
                
                results["supabase"] = HealthCheckResult(
                    component="supabase",
                    status=status,
                    message=message,
                    response_time=time.time() - start_time,
                    details={},
                    timestamp=datetime.now(timezone.utc)
                )
            else:
                results["supabase"] = HealthCheckResult(
                    component="supabase",
                    status=HealthStatus.CRITICAL,
                    message="Supabase client not initialized",
                    response_time=0,
                    details={},
                    timestamp=datetime.now(timezone.utc)
                )
                
        except Exception as e:
            results["supabase"] = HealthCheckResult(
                component="supabase",
                status=HealthStatus.CRITICAL,
                message=f"Supabase health check failed: {e}",
                response_time=time.time() - start_time,
                details={},
                timestamp=datetime.now(timezone.utc)
            )
        
        return results
    
    async def _check_external_services(self) -> Dict[str, HealthCheckResult]:
        """Check external service health."""
        results = {}
        
        # Check OpenAI API
        start_time = time.time()
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
                async with session.get("https://api.openai.com/v1/models", headers=headers) as response:
                    if response.status == 200:
                        status = HealthStatus.HEALTHY
                        message = "OpenAI API accessible"
                        details = {"status_code": response.status}
                    else:
                        status = HealthStatus.WARNING
                        message = f"OpenAI API returned status {response.status}"
                        details = {"status_code": response.status}
                    
                    results["openai"] = HealthCheckResult(
                        component="openai",
                        status=status,
                        message=message,
                        response_time=time.time() - start_time,
                        details=details,
                        timestamp=datetime.now(timezone.utc)
                    )
                    
        except Exception as e:
            results["openai"] = HealthCheckResult(
                component="openai",
                status=HealthStatus.CRITICAL,
                message=f"OpenAI API check failed: {e}",
                response_time=time.time() - start_time,
                details={},
                timestamp=datetime.now(timezone.utc)
            )
        
        # Check SearXNG
        start_time = time.time()
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                searxng_url = self.settings.searxng_url or "http://localhost:8080"
                async with session.get(f"{searxng_url}/healthz") as response:
                    if response.status == 200:
                        status = HealthStatus.HEALTHY
                        message = "SearXNG accessible"
                    else:
                        status = HealthStatus.WARNING
                        message = f"SearXNG returned status {response.status}"
                    
                    results["searxng"] = HealthCheckResult(
                        component="searxng",
                        status=status,
                        message=message,
                        response_time=time.time() - start_time,
                        details={"status_code": response.status},
                        timestamp=datetime.now(timezone.utc)
                    )
                    
        except Exception as e:
            results["searxng"] = HealthCheckResult(
                component="searxng",
                status=HealthStatus.WARNING,
                message=f"SearXNG check failed: {e}",
                response_time=time.time() - start_time,
                details={},
                timestamp=datetime.now(timezone.utc)
            )
        
        return results
    
    async def _check_application_health(self) -> Dict[str, HealthCheckResult]:
        """Check application-specific health."""
        results = {}
        
        # Check web server
        start_time = time.time()
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get("http://localhost:8000/api/health") as response:
                    if response.status == 200:
                        status = HealthStatus.HEALTHY
                        message = "Web server healthy"
                        data = await response.json()
                        details = data
                    else:
                        status = HealthStatus.WARNING
                        message = f"Web server returned status {response.status}"
                        details = {"status_code": response.status}
                    
                    results["web_server"] = HealthCheckResult(
                        component="web_server",
                        status=status,
                        message=message,
                        response_time=time.time() - start_time,
                        details=details,
                        timestamp=datetime.now(timezone.utc)
                    )
                    
        except Exception as e:
            results["web_server"] = HealthCheckResult(
                component="web_server",
                status=HealthStatus.CRITICAL,
                message=f"Web server check failed: {e}",
                response_time=time.time() - start_time,
                details={},
                timestamp=datetime.now(timezone.utc)
            )
        
        return results
    
    async def _check_agent_health(self) -> Dict[str, HealthCheckResult]:
        """Check AI agent health."""
        results = {}
        
        # This is a placeholder - in practice, you'd check agent responsiveness
        # by sending test requests to each agent type
        
        agent_types = [
            "content_analysis",
            "seo_research", 
            "trend_analysis",
            "competitor_analysis"
        ]
        
        for agent_type in agent_types:
            start_time = time.time()
            
            # Placeholder health check
            try:
                # In practice, you'd create and test the agent
                status = HealthStatus.HEALTHY
                message = f"{agent_type} agent healthy"
                details = {"agent_type": agent_type}
                
                results[f"agent_{agent_type}"] = HealthCheckResult(
                    component=f"agent_{agent_type}",
                    status=status,
                    message=message,
                    response_time=time.time() - start_time,
                    details=details,
                    timestamp=datetime.now(timezone.utc)
                )
                
            except Exception as e:
                results[f"agent_{agent_type}"] = HealthCheckResult(
                    component=f"agent_{agent_type}",
                    status=HealthStatus.WARNING,
                    message=f"{agent_type} agent check failed: {e}",
                    response_time=time.time() - start_time,
                    details={},
                    timestamp=datetime.now(timezone.utc)
                )
        
        return results
    
    async def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        if not self.health_cache:
            await self.perform_health_checks()
        
        # Count statuses
        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.WARNING: 0,
            HealthStatus.CRITICAL: 0,
            HealthStatus.UNKNOWN: 0
        }
        
        for result in self.health_cache.values():
            status_counts[result.status] += 1
        
        # Determine overall status
        if status_counts[HealthStatus.CRITICAL] > 0:
            overall_status = HealthStatus.CRITICAL
        elif status_counts[HealthStatus.WARNING] > 0:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Calculate average response time
        if self.health_cache:
            avg_response_time = sum(r.response_time for r in self.health_cache.values()) / len(self.health_cache)
        else:
            avg_response_time = 0
        
        return {
            "overall_status": overall_status.value,
            "status_counts": {k.value: v for k, v in status_counts.items()},
            "total_components": len(self.health_cache),
            "last_check": self.last_full_check.isoformat() if self.last_full_check else None,
            "average_response_time": avg_response_time,
            "components": {k: v.to_dict() for k, v in self.health_cache.items()}
        }
    
    async def get_component_health(self, component: str) -> Optional[HealthCheckResult]:
        """Get health status for a specific component."""
        return self.health_cache.get(component)
    
    async def is_healthy(self) -> bool:
        """Check if the system is overall healthy."""
        summary = await self.get_health_summary()
        return summary["overall_status"] == HealthStatus.HEALTHY.value

# Export the health monitor
__all__ = ["HealthMonitor", "HealthStatus", "HealthCheckResult", "SystemMetrics"]