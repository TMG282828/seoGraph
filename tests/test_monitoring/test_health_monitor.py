"""
Tests for the health monitoring system.

This module tests the HealthMonitor class and its various health check capabilities.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

from monitoring.health_monitor import (
    HealthMonitor, 
    HealthStatus, 
    HealthCheckResult, 
    SystemMetrics
)

class TestHealthMonitor:
    """Test cases for HealthMonitor."""
    
    @pytest.fixture
    def health_monitor(self):
        """Create a HealthMonitor instance for testing."""
        return HealthMonitor()
    
    @pytest.mark.asyncio
    async def test_initialization(self, health_monitor):
        """Test health monitor initialization."""
        # Test that monitor can be initialized
        assert health_monitor.settings is not None
        assert health_monitor.health_cache == {}
        assert health_monitor.last_full_check is None
        assert not health_monitor.is_running
        
        # Test initialization method
        await health_monitor.initialize()
        
        # Verify clients are initialized
        assert health_monitor.neo4j_client is not None
        assert health_monitor.qdrant_client is not None
        assert health_monitor.supabase_client is not None
    
    @pytest.mark.asyncio
    async def test_system_metrics_collection(self):
        """Test system metrics collection."""
        # Test that metrics can be collected
        metrics = SystemMetrics.collect()
        
        # Verify all required fields are present
        assert isinstance(metrics.cpu_usage, float)
        assert isinstance(metrics.memory_usage, float)
        assert isinstance(metrics.disk_usage, float)
        assert isinstance(metrics.network_io, dict)
        assert isinstance(metrics.process_count, int)
        assert isinstance(metrics.load_average, list)
        assert isinstance(metrics.uptime, float)
        
        # Verify reasonable values
        assert 0 <= metrics.cpu_usage <= 100
        assert 0 <= metrics.memory_usage <= 100
        assert 0 <= metrics.disk_usage <= 100
        assert metrics.process_count > 0
        assert len(metrics.load_average) == 3
        assert metrics.uptime > 0
    
    @pytest.mark.asyncio
    async def test_health_check_result_creation(self):
        """Test HealthCheckResult creation and serialization."""
        result = HealthCheckResult(
            component="test_component",
            status=HealthStatus.HEALTHY,
            message="Test message",
            response_time=1.23,
            details={"key": "value"},
            timestamp=datetime.now(timezone.utc)
        )
        
        # Test basic properties
        assert result.component == "test_component"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "Test message"
        assert result.response_time == 1.23
        assert result.details == {"key": "value"}
        
        # Test serialization
        result_dict = result.to_dict()
        assert result_dict["component"] == "test_component"
        assert result_dict["status"] == "healthy"
        assert result_dict["message"] == "Test message"
        assert result_dict["response_time"] == 1.23
        assert result_dict["details"] == {"key": "value"}
        assert "timestamp" in result_dict
    
    @pytest.mark.asyncio
    async def test_system_resource_check(self, health_monitor):
        """Test system resource health check."""
        await health_monitor.initialize()
        
        # Test system resource check
        results = await health_monitor._check_system_resources()
        
        # Verify result structure
        assert "system_resources" in results
        result = results["system_resources"]
        
        assert isinstance(result, HealthCheckResult)
        assert result.component == "system_resources"
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.WARNING, HealthStatus.CRITICAL]
        assert isinstance(result.message, str)
        assert result.response_time > 0
        assert isinstance(result.details, dict)
        assert result.timestamp is not None
        
        # Verify details contain system metrics
        details = result.details
        assert "cpu_usage" in details
        assert "memory_usage" in details
        assert "disk_usage" in details
        assert "network_io" in details
        assert "process_count" in details
        assert "load_average" in details
        assert "uptime" in details
    
    @pytest.mark.asyncio
    async def test_database_connection_checks(self, health_monitor):
        """Test database connection health checks."""
        await health_monitor.initialize()
        
        # Mock database clients
        health_monitor.neo4j_client = Mock()
        health_monitor.qdrant_client = Mock()
        health_monitor.supabase_client = Mock()
        
        # Mock successful health checks
        health_monitor.neo4j_client.health_check = AsyncMock(return_value=True)
        health_monitor.neo4j_client.get_database_info = AsyncMock(return_value={"version": "5.15"})
        health_monitor.qdrant_client.health_check = AsyncMock(return_value=True)
        health_monitor.qdrant_client.get_cluster_info = AsyncMock(return_value={"status": "green"})
        health_monitor.supabase_client.health_check = AsyncMock(return_value=True)
        
        # Test database connection checks
        results = await health_monitor._check_database_connections()
        
        # Verify all database checks are present
        assert "neo4j" in results
        assert "qdrant" in results
        assert "supabase" in results
        
        # Verify result structure
        for component in ["neo4j", "qdrant", "supabase"]:
            result = results[component]
            assert isinstance(result, HealthCheckResult)
            assert result.component == component
            assert result.status == HealthStatus.HEALTHY
            assert result.response_time > 0
            assert result.timestamp is not None
    
    @pytest.mark.asyncio
    async def test_database_connection_failure(self, health_monitor):
        """Test database connection failure handling."""
        await health_monitor.initialize()
        
        # Mock database clients with failures
        health_monitor.neo4j_client = Mock()
        health_monitor.neo4j_client.health_check = AsyncMock(return_value=False)
        
        # Test failed connection
        results = await health_monitor._check_database_connections()
        
        # Verify failure is handled properly
        assert "neo4j" in results
        result = results["neo4j"]
        assert result.status == HealthStatus.CRITICAL
        assert "connection failed" in result.message.lower()
    
    @pytest.mark.asyncio
    async def test_external_service_checks(self, health_monitor):
        """Test external service health checks."""
        await health_monitor.initialize()
        
        # Mock successful external service responses
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"status": "ok"})
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            # Test external service checks
            results = await health_monitor._check_external_services()
            
            # Verify results (may vary based on actual service availability)
            assert isinstance(results, dict)
            
            # At minimum, should check OpenAI and SearXNG
            # Results may be empty if services are not reachable
            for component, result in results.items():
                assert isinstance(result, HealthCheckResult)
                assert result.component == component
                assert result.status in [HealthStatus.HEALTHY, HealthStatus.WARNING, HealthStatus.CRITICAL]
                assert result.response_time >= 0
    
    @pytest.mark.asyncio
    async def test_application_health_check(self, health_monitor):
        """Test application health check."""
        await health_monitor.initialize()
        
        # Mock web server response
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"status": "healthy"})
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            # Test application health check
            results = await health_monitor._check_application_health()
            
            # Verify web server check
            assert "web_server" in results
            result = results["web_server"]
            assert isinstance(result, HealthCheckResult)
            assert result.component == "web_server"
            assert result.status == HealthStatus.HEALTHY
            assert result.response_time > 0
    
    @pytest.mark.asyncio
    async def test_agent_health_checks(self, health_monitor):
        """Test AI agent health checks."""
        await health_monitor.initialize()
        
        # Test agent health checks
        results = await health_monitor._check_agent_health()
        
        # Verify all agent types are checked
        expected_agents = [
            "agent_content_analysis",
            "agent_seo_research",
            "agent_trend_analysis",
            "agent_competitor_analysis"
        ]
        
        for agent in expected_agents:
            assert agent in results
            result = results[agent]
            assert isinstance(result, HealthCheckResult)
            assert result.component == agent
            assert result.status in [HealthStatus.HEALTHY, HealthStatus.WARNING, HealthStatus.CRITICAL]
            assert result.response_time >= 0
    
    @pytest.mark.asyncio
    async def test_perform_health_checks(self, health_monitor):
        """Test comprehensive health check execution."""
        await health_monitor.initialize()
        
        # Mock database clients to avoid actual connections
        health_monitor.neo4j_client = Mock()
        health_monitor.qdrant_client = Mock()
        health_monitor.supabase_client = Mock()
        
        health_monitor.neo4j_client.health_check = AsyncMock(return_value=True)
        health_monitor.qdrant_client.health_check = AsyncMock(return_value=True)
        health_monitor.supabase_client.health_check = AsyncMock(return_value=True)
        
        # Perform comprehensive health checks
        results = await health_monitor.perform_health_checks()
        
        # Verify results structure
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Verify all results are HealthCheckResult objects
        for component, result in results.items():
            assert isinstance(result, HealthCheckResult)
            assert result.component == component
            assert result.status in [HealthStatus.HEALTHY, HealthStatus.WARNING, HealthStatus.CRITICAL, HealthStatus.UNKNOWN]
            assert result.response_time >= 0
            assert result.timestamp is not None
        
        # Verify cache is updated
        assert health_monitor.health_cache == results
        assert health_monitor.last_full_check is not None
    
    @pytest.mark.asyncio
    async def test_health_summary(self, health_monitor):
        """Test health summary generation."""
        await health_monitor.initialize()
        
        # Mock some health check results
        health_monitor.health_cache = {
            "component1": HealthCheckResult(
                component="component1",
                status=HealthStatus.HEALTHY,
                message="OK",
                response_time=1.0,
                details={},
                timestamp=datetime.now(timezone.utc)
            ),
            "component2": HealthCheckResult(
                component="component2",
                status=HealthStatus.WARNING,
                message="Warning",
                response_time=2.0,
                details={},
                timestamp=datetime.now(timezone.utc)
            )
        }
        
        # Get health summary
        summary = await health_monitor.get_health_summary()
        
        # Verify summary structure
        assert "overall_status" in summary
        assert "status_counts" in summary
        assert "total_components" in summary
        assert "last_check" in summary
        assert "average_response_time" in summary
        assert "components" in summary
        
        # Verify values
        assert summary["overall_status"] == "warning"  # Due to warning component
        assert summary["status_counts"]["healthy"] == 1
        assert summary["status_counts"]["warning"] == 1
        assert summary["total_components"] == 2
        assert summary["average_response_time"] == 1.5
        assert len(summary["components"]) == 2
    
    @pytest.mark.asyncio
    async def test_component_health_retrieval(self, health_monitor):
        """Test individual component health retrieval."""
        await health_monitor.initialize()
        
        # Mock a health check result
        test_result = HealthCheckResult(
            component="test_component",
            status=HealthStatus.HEALTHY,
            message="Test OK",
            response_time=1.0,
            details={},
            timestamp=datetime.now(timezone.utc)
        )
        
        health_monitor.health_cache["test_component"] = test_result
        
        # Get component health
        result = await health_monitor.get_component_health("test_component")
        
        # Verify result
        assert result == test_result
        assert result.component == "test_component"
        assert result.status == HealthStatus.HEALTHY
        
        # Test non-existent component
        result = await health_monitor.get_component_health("non_existent")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_is_healthy_check(self, health_monitor):
        """Test overall health status check."""
        await health_monitor.initialize()
        
        # Mock healthy system
        health_monitor.health_cache = {
            "component1": HealthCheckResult(
                component="component1",
                status=HealthStatus.HEALTHY,
                message="OK",
                response_time=1.0,
                details={},
                timestamp=datetime.now(timezone.utc)
            )
        }
        
        # Test healthy system
        is_healthy = await health_monitor.is_healthy()
        assert is_healthy is True
        
        # Mock unhealthy system
        health_monitor.health_cache = {
            "component1": HealthCheckResult(
                component="component1",
                status=HealthStatus.CRITICAL,
                message="Failed",
                response_time=1.0,
                details={},
                timestamp=datetime.now(timezone.utc)
            )
        }
        
        # Test unhealthy system
        is_healthy = await health_monitor.is_healthy()
        assert is_healthy is False
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, health_monitor):
        """Test monitoring start/stop lifecycle."""
        await health_monitor.initialize()
        
        # Mock database clients
        health_monitor.neo4j_client = Mock()
        health_monitor.qdrant_client = Mock()
        health_monitor.supabase_client = Mock()
        
        health_monitor.neo4j_client.health_check = AsyncMock(return_value=True)
        health_monitor.qdrant_client.health_check = AsyncMock(return_value=True)
        health_monitor.supabase_client.health_check = AsyncMock(return_value=True)
        
        # Test initial state
        assert not health_monitor.is_running
        assert health_monitor.monitoring_task is None
        
        # Start monitoring
        await health_monitor.start_monitoring()
        
        # Verify monitoring started
        assert health_monitor.is_running
        assert health_monitor.monitoring_task is not None
        
        # Wait a bit for monitoring to run
        await asyncio.sleep(0.1)
        
        # Stop monitoring
        await health_monitor.stop_monitoring()
        
        # Verify monitoring stopped
        assert not health_monitor.is_running
        assert health_monitor.monitoring_task.cancelled()
        
        # Test double start/stop
        await health_monitor.start_monitoring()
        await health_monitor.start_monitoring()  # Should not error
        
        await health_monitor.stop_monitoring()
        await health_monitor.stop_monitoring()  # Should not error
    
    @pytest.mark.asyncio
    async def test_health_check_error_handling(self, health_monitor):
        """Test error handling in health checks."""
        await health_monitor.initialize()
        
        # Mock database client that raises exception
        health_monitor.neo4j_client = Mock()
        health_monitor.neo4j_client.health_check = AsyncMock(side_effect=Exception("Connection failed"))
        
        # Test that exceptions are handled gracefully
        results = await health_monitor._check_database_connections()
        
        # Verify error is handled
        assert "neo4j" in results
        result = results["neo4j"]
        assert result.status == HealthStatus.CRITICAL
        assert "Connection failed" in result.message
    
    def test_health_status_enum(self):
        """Test HealthStatus enum values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.WARNING.value == "warning"
        assert HealthStatus.CRITICAL.value == "critical"
        assert HealthStatus.UNKNOWN.value == "unknown"
    
    def test_system_metrics_validation(self):
        """Test SystemMetrics model validation."""
        # Test valid metrics
        metrics = SystemMetrics(
            cpu_usage=50.0,
            memory_usage=60.0,
            disk_usage=70.0,
            network_io={"bytes_sent": 1000, "bytes_recv": 2000},
            process_count=150,
            load_average=[1.0, 1.5, 2.0],
            uptime=86400.0
        )
        
        assert metrics.cpu_usage == 50.0
        assert metrics.memory_usage == 60.0
        assert metrics.disk_usage == 70.0
        assert metrics.network_io == {"bytes_sent": 1000, "bytes_recv": 2000}
        assert metrics.process_count == 150
        assert metrics.load_average == [1.0, 1.5, 2.0]
        assert metrics.uptime == 86400.0
        
        # Test model serialization
        metrics_dict = metrics.dict()
        assert "cpu_usage" in metrics_dict
        assert "memory_usage" in metrics_dict
        assert "disk_usage" in metrics_dict
        assert "network_io" in metrics_dict
        assert "process_count" in metrics_dict
        assert "load_average" in metrics_dict
        assert "uptime" in metrics_dict