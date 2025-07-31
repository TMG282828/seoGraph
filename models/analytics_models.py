"""
Analytics-related Pydantic models for the SEO Content Knowledge Graph System.

This module defines data models for analytics, performance metrics, 
time-series data, and reporting.
"""

import uuid
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator
import structlog

logger = structlog.get_logger(__name__)


class MetricType(str, Enum):
    """Types of metrics."""
    
    TRAFFIC = "traffic"
    ENGAGEMENT = "engagement"
    CONVERSION = "conversion"
    SEO = "seo"
    CONTENT = "content"
    SOCIAL = "social"
    TECHNICAL = "technical"
    BUSINESS = "business"


class TimeGranularity(str, Enum):
    """Time granularity for analytics data."""
    
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class TrendDirection(str, Enum):
    """Trend direction for metrics."""
    
    UP = "up"
    DOWN = "down"
    FLAT = "flat"
    VOLATILE = "volatile"


class DataSource(str, Enum):
    """Data sources for analytics."""
    
    GOOGLE_ANALYTICS = "google_analytics"
    GOOGLE_SEARCH_CONSOLE = "google_search_console"
    GOOGLE_ADS = "google_ads"
    FACEBOOK_INSIGHTS = "facebook_insights"
    TWITTER_ANALYTICS = "twitter_analytics"
    LINKEDIN_ANALYTICS = "linkedin_analytics"
    INTERNAL_TRACKING = "internal_tracking"
    THIRD_PARTY = "third_party"


class MetricDefinition(BaseModel):
    """Definition of a metric for analytics tracking."""
    
    metric_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Human-readable metric name")
    description: str = Field(..., description="Metric description")
    
    metric_type: MetricType = Field(..., description="Type of metric")
    unit: str = Field(..., description="Unit of measurement")
    
    # Calculation
    formula: Optional[str] = Field(None, description="Formula for calculated metrics")
    source_metrics: List[str] = Field(default_factory=list, description="Source metrics for calculation")
    
    # Data source
    data_source: DataSource = Field(..., description="Primary data source")
    source_field: Optional[str] = Field(None, description="Field name in source system")
    
    # Configuration
    is_active: bool = Field(True, description="Whether metric is actively tracked")
    is_calculated: bool = Field(False, description="Whether metric is calculated from others")
    aggregation_method: str = Field("sum", description="Aggregation method (sum, avg, max, min)")
    
    # Thresholds and alerts
    target_value: Optional[float] = Field(None, description="Target value for metric")
    warning_threshold: Optional[float] = Field(None, description="Warning threshold")
    critical_threshold: Optional[float] = Field(None, description="Critical threshold")
    
    # Metadata
    tenant_id: str = Field(..., description="Tenant identifier")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    def is_threshold_exceeded(self, value: float, threshold_type: str = "warning") -> bool:
        """Check if a value exceeds the specified threshold."""
        if threshold_type == "warning" and self.warning_threshold:
            return value > self.warning_threshold
        elif threshold_type == "critical" and self.critical_threshold:
            return value > self.critical_threshold
        return False


class MetricValue(BaseModel):
    """Individual metric value with timestamp."""
    
    value_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    metric_id: str = Field(..., description="Associated metric ID")
    
    value: float = Field(..., description="Metric value")
    timestamp: datetime = Field(..., description="Timestamp of measurement")
    
    # Dimensions
    dimensions: Dict[str, str] = Field(default_factory=dict, description="Metric dimensions")
    
    # Context
    content_id: Optional[str] = Field(None, description="Associated content ID")
    campaign_id: Optional[str] = Field(None, description="Associated campaign ID")
    user_id: Optional[str] = Field(None, description="Associated user ID")
    
    # Data quality
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence in data quality")
    data_source: DataSource = Field(..., description="Source of this data point")
    
    # Metadata
    tenant_id: str = Field(..., description="Tenant identifier")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        """Ensure timestamp is timezone-aware."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class ContentPerformanceMetrics(BaseModel):
    """Content performance metrics for a specific time period."""
    
    content_id: str = Field(..., description="Content identifier")
    period_start: datetime = Field(..., description="Start of measurement period")
    period_end: datetime = Field(..., description="End of measurement period")
    
    # Traffic metrics
    page_views: int = Field(0, ge=0, description="Total page views")
    unique_visitors: int = Field(0, ge=0, description="Unique visitors")
    sessions: int = Field(0, ge=0, description="Sessions")
    bounce_rate: float = Field(0.0, ge=0.0, le=1.0, description="Bounce rate")
    
    # Engagement metrics
    avg_session_duration: float = Field(0.0, ge=0.0, description="Average session duration (seconds)")
    pages_per_session: float = Field(0.0, ge=0.0, description="Pages per session")
    time_on_page: float = Field(0.0, ge=0.0, description="Average time on page (seconds)")
    
    # SEO metrics
    organic_traffic: int = Field(0, ge=0, description="Organic traffic")
    organic_keywords: int = Field(0, ge=0, description="Number of ranking keywords")
    avg_position: float = Field(0.0, ge=0.0, description="Average search position")
    impressions: int = Field(0, ge=0, description="Search impressions")
    clicks: int = Field(0, ge=0, description="Search clicks")
    ctr: float = Field(0.0, ge=0.0, le=1.0, description="Click-through rate")
    
    # Social metrics
    social_shares: int = Field(0, ge=0, description="Social media shares")
    social_engagement: int = Field(0, ge=0, description="Social engagement")
    social_reach: int = Field(0, ge=0, description="Social reach")
    
    # Conversion metrics
    conversions: int = Field(0, ge=0, description="Total conversions")
    conversion_rate: float = Field(0.0, ge=0.0, le=1.0, description="Conversion rate")
    conversion_value: float = Field(0.0, ge=0.0, description="Total conversion value")
    
    # Technical metrics
    page_load_time: float = Field(0.0, ge=0.0, description="Average page load time (seconds)")
    mobile_visitors: int = Field(0, ge=0, description="Mobile visitors")
    mobile_percentage: float = Field(0.0, ge=0.0, le=1.0, description="Mobile traffic percentage")
    
    # Calculated metrics
    engagement_score: float = Field(0.0, ge=0.0, description="Overall engagement score")
    performance_score: float = Field(0.0, ge=0.0, description="Overall performance score")
    
    # Metadata
    tenant_id: str = Field(..., description="Tenant identifier")
    measured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    def calculate_engagement_score(self) -> float:
        """Calculate overall engagement score."""
        # Weighted combination of engagement metrics
        components = [
            (1 - self.bounce_rate) * 0.3,  # Lower bounce rate is better
            min(self.avg_session_duration / 300, 1.0) * 0.25,  # Max 5 minutes
            min(self.pages_per_session / 5, 1.0) * 0.25,  # Max 5 pages
            min(self.time_on_page / 180, 1.0) * 0.2  # Max 3 minutes
        ]
        
        self.engagement_score = sum(components)
        return self.engagement_score
    
    def calculate_performance_score(self) -> float:
        """Calculate overall performance score."""
        # Weighted combination of key metrics
        components = [
            min(self.page_views / 1000, 1.0) * 0.2,  # Max 1000 views
            min(self.organic_traffic / 500, 1.0) * 0.25,  # Max 500 organic
            self.ctr * 0.2,  # Already 0-1
            min(self.social_shares / 100, 1.0) * 0.15,  # Max 100 shares
            self.conversion_rate * 0.2  # Already 0-1
        ]
        
        self.performance_score = sum(components)
        return self.performance_score


class TimeSeriesData(BaseModel):
    """Time series data for analytics."""
    
    series_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    metric_id: str = Field(..., description="Associated metric ID")
    
    # Time series configuration
    granularity: TimeGranularity = Field(..., description="Time granularity")
    start_time: datetime = Field(..., description="Series start time")
    end_time: datetime = Field(..., description="Series end time")
    
    # Data points
    data_points: List[Dict[str, Union[datetime, float]]] = Field(
        default_factory=list, 
        description="Time series data points"
    )
    
    # Trend analysis
    trend_direction: Optional[TrendDirection] = Field(None, description="Overall trend direction")
    trend_strength: float = Field(0.0, ge=0.0, le=1.0, description="Trend strength")
    
    # Statistical measures
    min_value: Optional[float] = Field(None, description="Minimum value")
    max_value: Optional[float] = Field(None, description="Maximum value")
    mean_value: Optional[float] = Field(None, description="Mean value")
    median_value: Optional[float] = Field(None, description="Median value")
    std_deviation: Optional[float] = Field(None, description="Standard deviation")
    
    # Seasonality
    seasonal_pattern: Optional[Dict[str, float]] = Field(None, description="Seasonal pattern")
    has_seasonality: bool = Field(False, description="Whether data shows seasonality")
    
    # Metadata
    tenant_id: str = Field(..., description="Tenant identifier")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    def add_data_point(self, timestamp: datetime, value: float) -> None:
        """Add a data point to the series."""
        self.data_points.append({
            'timestamp': timestamp,
            'value': value
        })
        
        # Update statistics
        self._update_statistics()
    
    def _update_statistics(self) -> None:
        """Update statistical measures."""
        if not self.data_points:
            return
        
        values = [point['value'] for point in self.data_points]
        
        self.min_value = min(values)
        self.max_value = max(values)
        self.mean_value = sum(values) / len(values)
        
        # Calculate median
        sorted_values = sorted(values)
        n = len(sorted_values)
        if n % 2 == 0:
            self.median_value = (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
        else:
            self.median_value = sorted_values[n//2]
        
        # Calculate standard deviation
        if len(values) > 1:
            variance = sum((x - self.mean_value) ** 2 for x in values) / (len(values) - 1)
            self.std_deviation = variance ** 0.5
    
    def analyze_trend(self) -> TrendDirection:
        """Analyze trend direction."""
        if len(self.data_points) < 2:
            self.trend_direction = TrendDirection.FLAT
            return self.trend_direction
        
        # Simple linear regression to determine trend
        n = len(self.data_points)
        x_values = list(range(n))
        y_values = [point['value'] for point in self.data_points]
        
        # Calculate slope
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            self.trend_direction = TrendDirection.FLAT
            self.trend_strength = 0.0
        else:
            slope = numerator / denominator
            
            # Determine trend direction
            if abs(slope) < 0.1:
                self.trend_direction = TrendDirection.FLAT
            elif slope > 0:
                self.trend_direction = TrendDirection.UP
            else:
                self.trend_direction = TrendDirection.DOWN
            
            # Calculate trend strength (normalized)
            self.trend_strength = min(abs(slope), 1.0)
        
        return self.trend_direction


class AnalyticsReport(BaseModel):
    """Analytics report with insights and recommendations."""
    
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(..., description="Report title")
    description: Optional[str] = Field(None, description="Report description")
    
    # Report configuration
    report_type: str = Field(..., description="Type of report")
    period_start: datetime = Field(..., description="Report period start")
    period_end: datetime = Field(..., description="Report period end")
    
    # Data
    metrics: List[MetricValue] = Field(default_factory=list, description="Metrics in report")
    time_series: List[TimeSeriesData] = Field(default_factory=list, description="Time series data")
    content_performance: List[ContentPerformanceMetrics] = Field(
        default_factory=list, 
        description="Content performance data"
    )
    
    # Insights
    key_insights: List[str] = Field(default_factory=list, description="Key insights")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    alerts: List[str] = Field(default_factory=list, description="Alerts and warnings")
    
    # Summary statistics
    summary_stats: Dict[str, float] = Field(default_factory=dict, description="Summary statistics")
    
    # Metadata
    tenant_id: str = Field(..., description="Tenant identifier")
    created_by: str = Field(..., description="Report creator")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    def add_insight(self, insight: str) -> None:
        """Add a key insight to the report."""
        self.key_insights.append(insight)
    
    def add_recommendation(self, recommendation: str) -> None:
        """Add a recommendation to the report."""
        self.recommendations.append(recommendation)
    
    def add_alert(self, alert: str) -> None:
        """Add an alert to the report."""
        self.alerts.append(alert)
    
    def calculate_summary_stats(self) -> Dict[str, float]:
        """Calculate summary statistics from data."""
        stats = {}
        
        # Aggregate metrics by type
        metric_totals = {}
        for metric in self.metrics:
            metric_type = metric.dimensions.get('type', 'unknown')
            if metric_type not in metric_totals:
                metric_totals[metric_type] = 0
            metric_totals[metric_type] += metric.value
        
        stats.update(metric_totals)
        
        # Content performance aggregates
        if self.content_performance:
            stats['total_page_views'] = sum(cp.page_views for cp in self.content_performance)
            stats['total_organic_traffic'] = sum(cp.organic_traffic for cp in self.content_performance)
            stats['avg_engagement_score'] = sum(cp.engagement_score for cp in self.content_performance) / len(self.content_performance)
            stats['avg_performance_score'] = sum(cp.performance_score for cp in self.content_performance) / len(self.content_performance)
        
        self.summary_stats = stats
        return stats


class AnalyticsDashboard(BaseModel):
    """Analytics dashboard configuration."""
    
    dashboard_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Dashboard name")
    description: Optional[str] = Field(None, description="Dashboard description")
    
    # Layout configuration
    layout: Dict[str, Any] = Field(default_factory=dict, description="Dashboard layout")
    widgets: List[Dict[str, Any]] = Field(default_factory=list, description="Dashboard widgets")
    
    # Data configuration
    metrics: List[str] = Field(default_factory=list, description="Metrics to display")
    time_range: str = Field("30d", description="Default time range")
    refresh_interval: int = Field(300, description="Refresh interval in seconds")
    
    # Access control
    is_public: bool = Field(False, description="Whether dashboard is public")
    allowed_users: List[str] = Field(default_factory=list, description="Allowed user IDs")
    
    # Metadata
    tenant_id: str = Field(..., description="Tenant identifier")
    created_by: str = Field(..., description="Dashboard creator")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    def add_widget(self, widget_config: Dict[str, Any]) -> None:
        """Add a widget to the dashboard."""
        widget_config['widget_id'] = str(uuid.uuid4())
        self.widgets.append(widget_config)
        self.updated_at = datetime.now(timezone.utc)
    
    def remove_widget(self, widget_id: str) -> bool:
        """Remove a widget from the dashboard."""
        for i, widget in enumerate(self.widgets):
            if widget.get('widget_id') == widget_id:
                del self.widgets[i]
                self.updated_at = datetime.now(timezone.utc)
                return True
        return False


class AnalyticsAlert(BaseModel):
    """Analytics alert configuration."""
    
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Alert name")
    description: Optional[str] = Field(None, description="Alert description")
    
    # Alert configuration
    metric_id: str = Field(..., description="Metric to monitor")
    condition: str = Field(..., description="Alert condition (gt, lt, eq, etc.)")
    threshold: float = Field(..., description="Alert threshold")
    
    # Notification configuration
    notification_channels: List[str] = Field(default_factory=list, description="Notification channels")
    notification_frequency: str = Field("immediate", description="Notification frequency")
    
    # State
    is_active: bool = Field(True, description="Whether alert is active")
    last_triggered: Optional[datetime] = Field(None, description="Last trigger time")
    trigger_count: int = Field(0, description="Number of times triggered")
    
    # Metadata
    tenant_id: str = Field(..., description="Tenant identifier")
    created_by: str = Field(..., description="Alert creator")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    def check_condition(self, value: float) -> bool:
        """Check if alert condition is met."""
        if self.condition == "gt":
            return value > self.threshold
        elif self.condition == "lt":
            return value < self.threshold
        elif self.condition == "eq":
            return abs(value - self.threshold) < 0.001
        elif self.condition == "gte":
            return value >= self.threshold
        elif self.condition == "lte":
            return value <= self.threshold
        else:
            return False
    
    def trigger_alert(self) -> None:
        """Trigger the alert."""
        self.last_triggered = datetime.now(timezone.utc)
        self.trigger_count += 1


# =============================================================================
# Request/Response Models
# =============================================================================

class MetricsQuery(BaseModel):
    """Query model for metrics data."""
    
    metric_ids: List[str] = Field(..., description="Metric IDs to query")
    start_time: datetime = Field(..., description="Query start time")
    end_time: datetime = Field(..., description="Query end time")
    granularity: TimeGranularity = Field(TimeGranularity.DAY, description="Data granularity")
    
    # Filters
    content_ids: Optional[List[str]] = Field(None, description="Filter by content IDs")
    dimensions: Optional[Dict[str, str]] = Field(None, description="Dimension filters")
    
    # Options
    include_calculated: bool = Field(True, description="Include calculated metrics")
    limit: int = Field(1000, description="Maximum results")


class ReportRequest(BaseModel):
    """Request model for generating reports."""
    
    report_type: str = Field(..., description="Type of report")
    title: str = Field(..., description="Report title")
    period_start: datetime = Field(..., description="Report period start")
    period_end: datetime = Field(..., description="Report period end")
    
    # Configuration
    metrics: List[str] = Field(..., description="Metrics to include")
    content_ids: Optional[List[str]] = Field(None, description="Content IDs to include")
    include_insights: bool = Field(True, description="Include AI-generated insights")
    include_recommendations: bool = Field(True, description="Include recommendations")
    
    # Format
    format: str = Field("json", description="Report format (json, pdf, csv)")


class DashboardRequest(BaseModel):
    """Request model for dashboard operations."""
    
    name: str = Field(..., description="Dashboard name")
    description: Optional[str] = Field(None, description="Dashboard description")
    
    # Configuration
    metrics: List[str] = Field(..., description="Metrics to display")
    time_range: str = Field("30d", description="Time range")
    layout: Optional[Dict[str, Any]] = Field(None, description="Layout configuration")
    
    # Access
    is_public: bool = Field(False, description="Whether dashboard is public")
    allowed_users: Optional[List[str]] = Field(None, description="Allowed users")


if __name__ == "__main__":
    # Example usage
    metric_def = MetricDefinition(
        name="Page Views",
        description="Total page views for content",
        metric_type=MetricType.TRAFFIC,
        unit="views",
        data_source=DataSource.GOOGLE_ANALYTICS,
        source_field="pageviews",
        tenant_id="example-tenant"
    )
    
    print(f"Created metric: {metric_def.name}")
    print(f"Type: {metric_def.metric_type}")
    print(f"Source: {metric_def.data_source}")
    
    # Example metric value
    metric_value = MetricValue(
        metric_id=metric_def.metric_id,
        value=1250.0,
        timestamp=datetime.now(timezone.utc),
        data_source=DataSource.GOOGLE_ANALYTICS,
        tenant_id="example-tenant"
    )
    
    print(f"Metric value: {metric_value.value}")
    
    # Example content performance
    performance = ContentPerformanceMetrics(
        content_id="content-123",
        period_start=datetime.now(timezone.utc) - timedelta(days=7),
        period_end=datetime.now(timezone.utc),
        page_views=1500,
        unique_visitors=800,
        bounce_rate=0.35,
        avg_session_duration=180.0,
        organic_traffic=600,
        ctr=0.12,
        tenant_id="example-tenant"
    )
    
    engagement_score = performance.calculate_engagement_score()
    performance_score = performance.calculate_performance_score()
    
    print(f"Content performance - Engagement: {engagement_score:.2f}, Performance: {performance_score:.2f}")