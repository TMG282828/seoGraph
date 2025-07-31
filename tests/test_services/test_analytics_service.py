"""
Tests for the Analytics Service module.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from services.analytics_service import AnalyticsService


class TestAnalyticsService:
    """Test suite for AnalyticsService."""

    def test_init_with_valid_config(self):
        """Test AnalyticsService initialization with valid configuration."""
        service = AnalyticsService()
        assert service is not None
        assert hasattr(service, 'neo4j_client')
        assert hasattr(service, 'redis_client')

    def test_init_sets_default_metrics(self):
        """Test that initialization sets up default metrics."""
        service = AnalyticsService()
        assert hasattr(service, 'metrics')
        assert isinstance(service.metrics, dict)

    @patch('services.analytics_service.Neo4jClient')
    def test_init_with_mocked_neo4j(self, mock_neo4j):
        """Test initialization with mocked Neo4j client."""
        mock_neo4j.return_value = MagicMock()
        service = AnalyticsService()
        assert service.neo4j_client is not None

    def test_validate_time_range_valid(self):
        """Test validation of valid time range."""
        service = AnalyticsService()
        start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 2, tzinfo=timezone.utc)
        
        # This should not raise an exception
        result = service._validate_time_range(start_time, end_time)
        assert result is True

    def test_validate_time_range_invalid(self):
        """Test validation of invalid time range."""
        service = AnalyticsService()
        start_time = datetime(2024, 1, 2, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        
        with pytest.raises(ValueError, match="Start time must be before end time"):
            service._validate_time_range(start_time, end_time)

    def test_get_default_metrics(self):
        """Test getting default metrics structure."""
        service = AnalyticsService()
        metrics = service._get_default_metrics()
        
        assert isinstance(metrics, dict)
        assert 'content_metrics' in metrics
        assert 'seo_metrics' in metrics
        assert 'performance_metrics' in metrics
        
        # Check content metrics structure
        assert 'total_content' in metrics['content_metrics']
        assert 'content_by_type' in metrics['content_metrics']
        assert 'recent_content' in metrics['content_metrics']

    def test_format_metrics_for_response(self):
        """Test metrics formatting for API response."""
        service = AnalyticsService()
        raw_metrics = {
            'content_metrics': {
                'total_content': 100,
                'content_by_type': {'article': 50, 'blog': 30, 'page': 20}
            }
        }
        
        formatted = service._format_metrics_for_response(raw_metrics)
        
        assert isinstance(formatted, dict)
        assert 'content_metrics' in formatted
        assert formatted['content_metrics']['total_content'] == 100