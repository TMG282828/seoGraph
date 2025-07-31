"""
Advanced Analytics Service for the SEO Content Knowledge Graph System.

This module provides comprehensive analytics capabilities including time-series analysis,
performance tracking, automated insights, and real-time monitoring.
"""

import asyncio
import statistics
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
import json

import numpy as np
import pandas as pd
import structlog
from cachetools import TTLCache
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from database.neo4j_client import Neo4jClient
from database.qdrant_client import QdrantClient
from models.analytics_models import (
    MetricDefinition,
    MetricValue,
    ContentPerformanceMetrics,
    TimeSeriesData,
    AnalyticsReport,
    AnalyticsDashboard,
    AnalyticsAlert,
    MetricType,
    TimeGranularity,
    TrendDirection,
    DataSource
)
from models.content_models import ContentItem
from config.settings import get_settings

logger = structlog.get_logger(__name__)


class AnalyticsError(Exception):
    """Raised when analytics operations fail."""
    pass


class DataIngestionError(AnalyticsError):
    """Raised when data ingestion fails."""
    pass


class InsightGenerationError(AnalyticsError):
    """Raised when insight generation fails."""
    pass


class AnalyticsService:
    """
    Advanced analytics service with comprehensive tracking and analysis.
    
    Provides:
    - Time-series data collection and analysis
    - Performance metric calculation and trending
    - Automated insight generation
    - Real-time monitoring and alerting
    - Custom dashboard creation
    - Advanced statistical analysis
    """
    
    def __init__(self,
                 neo4j_client: Neo4jClient,
                 qdrant_client: QdrantClient):
        self.neo4j_client = neo4j_client
        self.qdrant_client = qdrant_client
        self.settings = get_settings()
        
        # Metric definitions cache
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        
        # Data caches
        self.metrics_cache = TTLCache(maxsize=10000, ttl=300)  # 5 minutes
        self.reports_cache = TTLCache(maxsize=100, ttl=1800)   # 30 minutes
        self.insights_cache = TTLCache(maxsize=500, ttl=3600)  # 1 hour
        
        # Alert monitoring
        self.active_alerts: Dict[str, AnalyticsAlert] = {}
        self.alert_check_interval = 60  # seconds
        self.alert_task: Optional[asyncio.Task] = None
        
        # Statistical models
        self.trend_models: Dict[str, LinearRegression] = {}
        self.anomaly_thresholds: Dict[str, Dict[str, float]] = {}
        
        logger.info("Analytics service initialized")
    
    async def initialize(self) -> None:
        """Initialize the analytics service."""
        try:
            # Load metric definitions
            await self._load_metric_definitions()
            
            # Load active alerts
            await self._load_active_alerts()
            
            # Start alert monitoring
            await self._start_alert_monitoring()
            
            logger.info("Analytics service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize analytics service: {e}")
            raise AnalyticsError(f"Initialization failed: {e}")
    
    async def close(self) -> None:
        """Close the analytics service."""
        try:
            if self.alert_task:
                self.alert_task.cancel()
                try:
                    await self.alert_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Analytics service closed")
            
        except Exception as e:
            logger.error(f"Error closing analytics service: {e}")
    
    # =============================================================================
    # Metric Management
    # =============================================================================
    
    async def create_metric_definition(self, metric_def: MetricDefinition) -> str:
        """Create a new metric definition."""
        try:
            # Store in database
            query = """
            CREATE (m:MetricDefinition {
                metric_id: $metric_id,
                name: $name,
                description: $description,
                metric_type: $metric_type,
                unit: $unit,
                formula: $formula,
                source_metrics: $source_metrics,
                data_source: $data_source,
                source_field: $source_field,
                is_active: $is_active,
                is_calculated: $is_calculated,
                aggregation_method: $aggregation_method,
                target_value: $target_value,
                warning_threshold: $warning_threshold,
                critical_threshold: $critical_threshold,
                tenant_id: $tenant_id,
                created_at: datetime(),
                updated_at: datetime()
            })
            RETURN m.metric_id as metric_id
            """
            
            result = await self.neo4j_client.run_query(query, metric_def.dict())
            
            if result:
                metric_id = result[0]['metric_id']
                
                # Cache the definition
                self.metric_definitions[metric_id] = metric_def
                
                logger.info(f"Created metric definition: {metric_def.name}", metric_id=metric_id)
                return metric_id
            
            raise AnalyticsError("Failed to create metric definition")
            
        except Exception as e:
            logger.error(f"Failed to create metric definition: {e}")
            raise AnalyticsError(f"Metric creation failed: {e}")
    
    async def get_metric_definition(self, metric_id: str) -> Optional[MetricDefinition]:
        """Get a metric definition by ID."""
        try:
            # Check cache first
            if metric_id in self.metric_definitions:
                return self.metric_definitions[metric_id]
            
            # Query database
            query = """
            MATCH (m:MetricDefinition {metric_id: $metric_id})
            RETURN m
            """
            
            result = await self.neo4j_client.run_query(query, {"metric_id": metric_id})
            
            if result:
                metric_data = result[0]['m']
                metric_def = MetricDefinition(**metric_data)
                
                # Cache it
                self.metric_definitions[metric_id] = metric_def
                
                return metric_def
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get metric definition {metric_id}: {e}")
            return None
    
    # =============================================================================
    # Data Ingestion
    # =============================================================================
    
    async def ingest_metric_value(self, metric_value: MetricValue) -> bool:
        """Ingest a single metric value."""
        try:
            # Validate metric definition exists
            metric_def = await self.get_metric_definition(metric_value.metric_id)
            if not metric_def:
                raise DataIngestionError(f"Metric definition not found: {metric_value.metric_id}")
            
            # Store in database
            query = """
            MATCH (m:MetricDefinition {metric_id: $metric_id})
            CREATE (v:MetricValue {
                value_id: $value_id,
                metric_id: $metric_id,
                value: $value,
                timestamp: $timestamp,
                dimensions: $dimensions,
                content_id: $content_id,
                campaign_id: $campaign_id,
                user_id: $user_id,
                confidence: $confidence,
                data_source: $data_source,
                tenant_id: $tenant_id,
                created_at: datetime()
            })
            CREATE (m)-[:HAS_VALUE]->(v)
            RETURN v.value_id as value_id
            """
            
            result = await self.neo4j_client.run_query(query, metric_value.dict())
            
            if result:
                # Check for alerts
                await self._check_metric_alerts(metric_value)
                
                # Update anomaly detection
                await self._update_anomaly_detection(metric_value)
                
                # Clear relevant caches
                self._clear_metric_caches(metric_value.metric_id)
                
                logger.debug(f"Ingested metric value", 
                           metric_id=metric_value.metric_id,
                           value=metric_value.value)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to ingest metric value: {e}")
            raise DataIngestionError(f"Ingestion failed: {e}")
    
    async def ingest_metric_values_batch(self, metric_values: List[MetricValue]) -> int:
        """Ingest multiple metric values in batch."""
        try:
            successful_count = 0
            
            # Group by metric for efficiency
            grouped_values = defaultdict(list)
            for value in metric_values:
                grouped_values[value.metric_id].append(value)
            
            # Process each metric group
            for metric_id, values in grouped_values.items():
                try:
                    # Validate metric definition
                    metric_def = await self.get_metric_definition(metric_id)
                    if not metric_def:
                        logger.warning(f"Skipping values for unknown metric: {metric_id}")
                        continue
                    
                    # Batch insert
                    query = """
                    UNWIND $values as value_data
                    MATCH (m:MetricDefinition {metric_id: value_data.metric_id})
                    CREATE (v:MetricValue {
                        value_id: value_data.value_id,
                        metric_id: value_data.metric_id,
                        value: value_data.value,
                        timestamp: value_data.timestamp,
                        dimensions: value_data.dimensions,
                        content_id: value_data.content_id,
                        campaign_id: value_data.campaign_id,
                        user_id: value_data.user_id,
                        confidence: value_data.confidence,
                        data_source: value_data.data_source,
                        tenant_id: value_data.tenant_id,
                        created_at: datetime()
                    })
                    CREATE (m)-[:HAS_VALUE]->(v)
                    """
                    
                    values_data = [value.dict() for value in values]
                    await self.neo4j_client.run_query(query, {"values": values_data})
                    
                    successful_count += len(values)
                    
                    # Check alerts for each value
                    for value in values:
                        await self._check_metric_alerts(value)
                    
                    # Clear caches
                    self._clear_metric_caches(metric_id)
                    
                except Exception as e:
                    logger.error(f"Failed to ingest batch for metric {metric_id}: {e}")
                    continue
            
            logger.info(f"Batch ingested {successful_count}/{len(metric_values)} metric values")
            return successful_count
            
        except Exception as e:
            logger.error(f"Failed to ingest metric values batch: {e}")
            raise DataIngestionError(f"Batch ingestion failed: {e}")
    
    async def ingest_content_performance(self, performance: ContentPerformanceMetrics) -> bool:
        """Ingest content performance metrics."""
        try:
            # Calculate derived metrics
            performance.calculate_engagement_score()
            performance.calculate_performance_score()
            
            # Store in database
            query = """
            MATCH (c:Content {id: $content_id, tenant_id: $tenant_id})
            CREATE (p:ContentPerformance {
                content_id: $content_id,
                period_start: $period_start,
                period_end: $period_end,
                page_views: $page_views,
                unique_visitors: $unique_visitors,
                sessions: $sessions,
                bounce_rate: $bounce_rate,
                avg_session_duration: $avg_session_duration,
                pages_per_session: $pages_per_session,
                time_on_page: $time_on_page,
                organic_traffic: $organic_traffic,
                organic_keywords: $organic_keywords,
                avg_position: $avg_position,
                impressions: $impressions,
                clicks: $clicks,
                ctr: $ctr,
                social_shares: $social_shares,
                social_engagement: $social_engagement,
                social_reach: $social_reach,
                conversions: $conversions,
                conversion_rate: $conversion_rate,
                conversion_value: $conversion_value,
                page_load_time: $page_load_time,
                mobile_visitors: $mobile_visitors,
                mobile_percentage: $mobile_percentage,
                engagement_score: $engagement_score,
                performance_score: $performance_score,
                tenant_id: $tenant_id,
                measured_at: datetime()
            })
            CREATE (c)-[:HAS_PERFORMANCE]->(p)
            RETURN p.content_id as content_id
            """
            
            result = await self.neo4j_client.run_query(query, performance.dict())
            
            if result:
                logger.info(f"Ingested content performance for {performance.content_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to ingest content performance: {e}")
            return False
    
    # =============================================================================
    # Time Series Analysis
    # =============================================================================
    
    async def get_time_series(self,
                            metric_id: str,
                            start_time: datetime,
                            end_time: datetime,
                            granularity: TimeGranularity = TimeGranularity.DAY,
                            tenant_id: str = "default") -> Optional[TimeSeriesData]:
        """Get time series data for a metric."""
        try:
            cache_key = f"timeseries_{metric_id}_{start_time}_{end_time}_{granularity}_{tenant_id}"
            
            # Check cache
            if cache_key in self.metrics_cache:
                return self.metrics_cache[cache_key]
            
            # Query database
            query = """
            MATCH (m:MetricDefinition {metric_id: $metric_id})-[:HAS_VALUE]->(v:MetricValue)
            WHERE v.tenant_id = $tenant_id
              AND v.timestamp >= $start_time
              AND v.timestamp <= $end_time
            RETURN v.timestamp as timestamp, v.value as value
            ORDER BY v.timestamp
            """
            
            result = await self.neo4j_client.run_query(query, {
                "metric_id": metric_id,
                "tenant_id": tenant_id,
                "start_time": start_time,
                "end_time": end_time
            })
            
            if not result:
                return None
            
            # Aggregate by granularity
            aggregated_data = await self._aggregate_time_series(result, granularity)
            
            # Create time series object
            time_series = TimeSeriesData(
                metric_id=metric_id,
                granularity=granularity,
                start_time=start_time,
                end_time=end_time,
                data_points=aggregated_data,
                tenant_id=tenant_id
            )
            
            # Analyze trend
            time_series.analyze_trend()
            
            # Detect seasonality
            await self._detect_seasonality(time_series)
            
            # Cache result
            self.metrics_cache[cache_key] = time_series
            
            return time_series
            
        except Exception as e:
            logger.error(f"Failed to get time series for metric {metric_id}: {e}")
            return None
    
    async def _aggregate_time_series(self,
                                   raw_data: List[Dict],
                                   granularity: TimeGranularity) -> List[Dict[str, Union[datetime, float]]]:
        """Aggregate time series data by granularity."""
        try:
            if not raw_data:
                return []
            
            # Convert to DataFrame for easier aggregation
            df = pd.DataFrame(raw_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Define aggregation rules
            agg_rules = {
                TimeGranularity.HOUR: 'H',
                TimeGranularity.DAY: 'D',
                TimeGranularity.WEEK: 'W',
                TimeGranularity.MONTH: 'M',
                TimeGranularity.QUARTER: 'Q',
                TimeGranularity.YEAR: 'Y'
            }
            
            freq = agg_rules.get(granularity, 'D')
            
            # Aggregate data
            aggregated = df.resample(freq).agg({
                'value': 'sum'  # Can be made configurable per metric
            })
            
            # Convert back to list format
            data_points = []
            for timestamp, row in aggregated.iterrows():
                if not pd.isna(row['value']):
                    data_points.append({
                        'timestamp': timestamp.to_pydatetime(),
                        'value': float(row['value'])
                    })
            
            return data_points
            
        except Exception as e:
            logger.error(f"Failed to aggregate time series data: {e}")
            return []
    
    async def _detect_seasonality(self, time_series: TimeSeriesData) -> None:
        """Detect seasonal patterns in time series data."""
        try:
            if len(time_series.data_points) < 24:  # Need sufficient data
                return
            
            values = [point['value'] for point in time_series.data_points]
            
            # Simple seasonal detection using autocorrelation
            # In production, use more sophisticated methods like FFT or STL decomposition
            
            # Check for weekly pattern (7 days)
            if len(values) >= 14:
                weekly_correlation = self._calculate_autocorrelation(values, 7)
                if weekly_correlation > 0.7:
                    time_series.has_seasonality = True
                    time_series.seasonal_pattern = {'weekly': weekly_correlation}
            
            # Check for monthly pattern (30 days)
            if len(values) >= 60:
                monthly_correlation = self._calculate_autocorrelation(values, 30)
                if monthly_correlation > 0.7:
                    time_series.has_seasonality = True
                    if not time_series.seasonal_pattern:
                        time_series.seasonal_pattern = {}
                    time_series.seasonal_pattern['monthly'] = monthly_correlation
            
        except Exception as e:
            logger.error(f"Failed to detect seasonality: {e}")
    
    def _calculate_autocorrelation(self, values: List[float], lag: int) -> float:
        """Calculate autocorrelation for given lag."""
        try:
            if len(values) <= lag:
                return 0.0
            
            series1 = values[:-lag]
            series2 = values[lag:]
            
            if len(series1) != len(series2):
                return 0.0
            
            # Calculate correlation coefficient
            mean1 = statistics.mean(series1)
            mean2 = statistics.mean(series2)
            
            numerator = sum((x - mean1) * (y - mean2) for x, y in zip(series1, series2))
            
            std1 = statistics.stdev(series1) if len(series1) > 1 else 0
            std2 = statistics.stdev(series2) if len(series2) > 1 else 0
            
            if std1 == 0 or std2 == 0:
                return 0.0
            
            denominator = len(series1) * std1 * std2
            
            return numerator / denominator if denominator != 0 else 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate autocorrelation: {e}")
            return 0.0
    
    # =============================================================================
    # Analytics and Insights
    # =============================================================================
    
    async def generate_analytics_report(self,
                                      report_type: str,
                                      period_start: datetime,
                                      period_end: datetime,
                                      metric_ids: List[str],
                                      tenant_id: str,
                                      content_ids: Optional[List[str]] = None) -> AnalyticsReport:
        """Generate comprehensive analytics report."""
        try:
            cache_key = f"report_{report_type}_{period_start}_{period_end}_{tenant_id}"
            
            # Check cache
            if cache_key in self.reports_cache:
                return self.reports_cache[cache_key]
            
            logger.info(f"Generating analytics report: {report_type}")
            
            # Create report
            report = AnalyticsReport(
                title=f"{report_type.title()} Analytics Report",
                report_type=report_type,
                period_start=period_start,
                period_end=period_end,
                tenant_id=tenant_id,
                created_by="analytics_service"
            )
            
            # Collect metrics data
            for metric_id in metric_ids:
                time_series = await self.get_time_series(
                    metric_id, period_start, period_end, TimeGranularity.DAY, tenant_id
                )
                if time_series:
                    report.time_series.append(time_series)
            
            # Collect content performance data
            if content_ids:
                for content_id in content_ids:
                    performance = await self._get_content_performance(
                        content_id, period_start, period_end, tenant_id
                    )
                    if performance:
                        report.content_performance.append(performance)
            
            # Generate insights
            await self._generate_report_insights(report)
            
            # Calculate summary statistics
            report.calculate_summary_stats()
            
            # Cache report
            self.reports_cache[cache_key] = report
            
            logger.info(f"Generated analytics report with {len(report.time_series)} metrics")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate analytics report: {e}")
            raise AnalyticsError(f"Report generation failed: {e}")
    
    async def _generate_report_insights(self, report: AnalyticsReport) -> None:
        """Generate AI-powered insights for the report."""
        try:
            insights = []
            recommendations = []
            alerts = []
            
            # Analyze time series trends
            for ts in report.time_series:
                if ts.trend_direction == TrendDirection.UP and ts.trend_strength > 0.7:
                    insights.append(f"Strong upward trend detected in {ts.metric_id} (strength: {ts.trend_strength:.2f})")
                elif ts.trend_direction == TrendDirection.DOWN and ts.trend_strength > 0.7:
                    alerts.append(f"Strong downward trend in {ts.metric_id} requires attention")
                
                if ts.has_seasonality:
                    insights.append(f"Seasonal pattern detected in {ts.metric_id}")
            
            # Analyze content performance
            if report.content_performance:
                avg_engagement = sum(cp.engagement_score for cp in report.content_performance) / len(report.content_performance)
                avg_performance = sum(cp.performance_score for cp in report.content_performance) / len(report.content_performance)
                
                if avg_engagement > 0.7:
                    insights.append(f"High average engagement score: {avg_engagement:.2f}")
                elif avg_engagement < 0.3:
                    recommendations.append("Consider improving content engagement strategies")
                
                if avg_performance > 0.7:
                    insights.append(f"Strong overall content performance: {avg_performance:.2f}")
                elif avg_performance < 0.3:
                    recommendations.append("Content performance needs improvement")
                
                # Find top and bottom performers
                top_performer = max(report.content_performance, key=lambda x: x.performance_score)
                bottom_performer = min(report.content_performance, key=lambda x: x.performance_score)
                
                insights.append(f"Top performing content: {top_performer.content_id} (score: {top_performer.performance_score:.2f})")
                if bottom_performer.performance_score < 0.3:
                    recommendations.append(f"Review content {bottom_performer.content_id} for optimization opportunities")
            
            # Add comparative insights
            if len(report.time_series) > 1:
                # Compare metric trends
                trending_up = [ts for ts in report.time_series if ts.trend_direction == TrendDirection.UP]
                trending_down = [ts for ts in report.time_series if ts.trend_direction == TrendDirection.DOWN]
                
                if len(trending_up) > len(trending_down):
                    insights.append(f"Overall positive trend: {len(trending_up)} metrics trending up vs {len(trending_down)} down")
                elif len(trending_down) > len(trending_up):
                    alerts.append(f"Concerning trend: {len(trending_down)} metrics trending down vs {len(trending_up)} up")
            
            # Update report
            report.key_insights.extend(insights)
            report.recommendations.extend(recommendations)
            report.alerts.extend(alerts)
            
        except Exception as e:
            logger.error(f"Failed to generate report insights: {e}")
            raise InsightGenerationError(f"Insight generation failed: {e}")
    
    async def _get_content_performance(self,
                                     content_id: str,
                                     period_start: datetime,
                                     period_end: datetime,
                                     tenant_id: str) -> Optional[ContentPerformanceMetrics]:
        """Get content performance metrics for a specific period."""
        try:
            query = """
            MATCH (c:Content {id: $content_id, tenant_id: $tenant_id})-[:HAS_PERFORMANCE]->(p:ContentPerformance)
            WHERE p.period_start >= $period_start AND p.period_end <= $period_end
            RETURN p
            ORDER BY p.measured_at DESC
            LIMIT 1
            """
            
            result = await self.neo4j_client.run_query(query, {
                "content_id": content_id,
                "tenant_id": tenant_id,
                "period_start": period_start,
                "period_end": period_end
            })
            
            if result:
                performance_data = result[0]['p']
                return ContentPerformanceMetrics(**performance_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get content performance for {content_id}: {e}")
            return None
    
    # =============================================================================
    # Alerting
    # =============================================================================
    
    async def create_alert(self, alert: AnalyticsAlert) -> str:
        """Create a new analytics alert."""
        try:
            # Store in database
            query = """
            CREATE (a:AnalyticsAlert {
                alert_id: $alert_id,
                name: $name,
                description: $description,
                metric_id: $metric_id,
                condition: $condition,
                threshold: $threshold,
                notification_channels: $notification_channels,
                notification_frequency: $notification_frequency,
                is_active: $is_active,
                last_triggered: $last_triggered,
                trigger_count: $trigger_count,
                tenant_id: $tenant_id,
                created_by: $created_by,
                created_at: datetime()
            })
            RETURN a.alert_id as alert_id
            """
            
            result = await self.neo4j_client.run_query(query, alert.dict())
            
            if result:
                alert_id = result[0]['alert_id']
                
                # Add to active alerts
                self.active_alerts[alert_id] = alert
                
                logger.info(f"Created analytics alert: {alert.name}", alert_id=alert_id)
                return alert_id
            
            raise AnalyticsError("Failed to create alert")
            
        except Exception as e:
            logger.error(f"Failed to create analytics alert: {e}")
            raise AnalyticsError(f"Alert creation failed: {e}")
    
    async def _check_metric_alerts(self, metric_value: MetricValue) -> None:
        """Check if any alerts should be triggered for a metric value."""
        try:
            # Find alerts for this metric
            metric_alerts = [
                alert for alert in self.active_alerts.values()
                if alert.metric_id == metric_value.metric_id and alert.is_active
            ]
            
            for alert in metric_alerts:
                if alert.check_condition(metric_value.value):
                    # Check notification frequency to avoid spam
                    if await self._should_trigger_alert(alert):
                        await self._trigger_alert(alert, metric_value)
            
        except Exception as e:
            logger.error(f"Failed to check metric alerts: {e}")
    
    async def _should_trigger_alert(self, alert: AnalyticsAlert) -> bool:
        """Check if alert should be triggered based on frequency settings."""
        try:
            if not alert.last_triggered:
                return True
            
            time_since_last = datetime.now(timezone.utc) - alert.last_triggered
            
            if alert.notification_frequency == "immediate":
                return True
            elif alert.notification_frequency == "hourly":
                return time_since_last.total_seconds() >= 3600
            elif alert.notification_frequency == "daily":
                return time_since_last.total_seconds() >= 86400
            elif alert.notification_frequency == "weekly":
                return time_since_last.total_seconds() >= 604800
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check alert trigger frequency: {e}")
            return False
    
    async def _trigger_alert(self, alert: AnalyticsAlert, metric_value: MetricValue) -> None:
        """Trigger an analytics alert."""
        try:
            # Update alert state
            alert.trigger_alert()
            
            # Log alert
            logger.warning(
                f"Analytics alert triggered: {alert.name}",
                alert_id=alert.alert_id,
                metric_id=metric_value.metric_id,
                value=metric_value.value,
                threshold=alert.threshold,
                condition=alert.condition
            )
            
            # Send notifications (implement based on channels)
            for channel in alert.notification_channels:
                await self._send_alert_notification(alert, metric_value, channel)
            
            # Update database
            query = """
            MATCH (a:AnalyticsAlert {alert_id: $alert_id})
            SET a.last_triggered = datetime(),
                a.trigger_count = a.trigger_count + 1
            """
            
            await self.neo4j_client.run_query(query, {"alert_id": alert.alert_id})
            
        except Exception as e:
            logger.error(f"Failed to trigger alert {alert.alert_id}: {e}")
    
    async def _send_alert_notification(self, 
                                     alert: AnalyticsAlert, 
                                     metric_value: MetricValue, 
                                     channel: str) -> None:
        """Send alert notification to specified channel."""
        try:
            # Implement notification logic based on channel
            # This is a placeholder - implement with actual notification systems
            
            message = f"""
            Analytics Alert: {alert.name}
            
            Metric: {metric_value.metric_id}
            Value: {metric_value.value}
            Threshold: {alert.threshold}
            Condition: {alert.condition}
            Time: {metric_value.timestamp}
            """
            
            if channel == "log":
                logger.warning(f"ALERT NOTIFICATION: {message}")
            elif channel == "email":
                # Implement email notification
                pass
            elif channel == "slack":
                # Implement Slack notification
                pass
            elif channel == "webhook":
                # Implement webhook notification
                pass
            
        except Exception as e:
            logger.error(f"Failed to send alert notification to {channel}: {e}")
    
    async def _start_alert_monitoring(self) -> None:
        """Start background task for alert monitoring."""
        async def alert_monitoring_loop():
            try:
                while True:
                    await asyncio.sleep(self.alert_check_interval)
                    
                    # Periodic alert checks can be added here
                    # For now, alerts are checked on data ingestion
                    
            except asyncio.CancelledError:
                logger.info("Alert monitoring loop cancelled")
            except Exception as e:
                logger.error(f"Alert monitoring loop error: {e}")
        
        if not self.alert_task:
            self.alert_task = asyncio.create_task(alert_monitoring_loop())
            logger.info("Started alert monitoring")
    
    # =============================================================================
    # Anomaly Detection
    # =============================================================================
    
    async def _update_anomaly_detection(self, metric_value: MetricValue) -> None:
        """Update anomaly detection models with new data."""
        try:
            metric_id = metric_value.metric_id
            
            # Simple threshold-based anomaly detection
            # In production, use more sophisticated methods like isolation forest or LSTM
            
            if metric_id not in self.anomaly_thresholds:
                # Initialize thresholds
                await self._initialize_anomaly_thresholds(metric_id)
            
            thresholds = self.anomaly_thresholds.get(metric_id, {})
            
            if thresholds:
                # Check for anomalies
                is_anomaly = (
                    metric_value.value > thresholds.get('upper', float('inf')) or
                    metric_value.value < thresholds.get('lower', float('-inf'))
                )
                
                if is_anomaly:
                    logger.warning(
                        f"Anomaly detected in metric {metric_id}",
                        value=metric_value.value,
                        thresholds=thresholds
                    )
                    
                    # Could trigger an anomaly alert here
            
        except Exception as e:
            logger.error(f"Failed to update anomaly detection: {e}")
    
    async def _initialize_anomaly_thresholds(self, metric_id: str) -> None:
        """Initialize anomaly detection thresholds for a metric."""
        try:
            # Get recent metric values
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=30)  # Last 30 days
            
            time_series = await self.get_time_series(
                metric_id, start_time, end_time, TimeGranularity.DAY
            )
            
            if time_series and len(time_series.data_points) >= 10:
                values = [point['value'] for point in time_series.data_points]
                
                # Calculate thresholds using IQR method
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                
                lower_threshold = q1 - 1.5 * iqr
                upper_threshold = q3 + 1.5 * iqr
                
                self.anomaly_thresholds[metric_id] = {
                    'lower': lower_threshold,
                    'upper': upper_threshold,
                    'q1': q1,
                    'q3': q3,
                    'iqr': iqr
                }
                
                logger.info(f"Initialized anomaly thresholds for {metric_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize anomaly thresholds for {metric_id}: {e}")
    
    # =============================================================================
    # Utility Methods
    # =============================================================================
    
    async def _load_metric_definitions(self) -> None:
        """Load metric definitions from database."""
        try:
            query = """
            MATCH (m:MetricDefinition)
            RETURN m
            """
            
            result = await self.neo4j_client.run_query(query)
            
            for record in result:
                metric_data = record['m']
                metric_def = MetricDefinition(**metric_data)
                self.metric_definitions[metric_def.metric_id] = metric_def
            
            logger.info(f"Loaded {len(self.metric_definitions)} metric definitions")
            
        except Exception as e:
            logger.error(f"Failed to load metric definitions: {e}")
    
    async def _load_active_alerts(self) -> None:
        """Load active alerts from database."""
        try:
            query = """
            MATCH (a:AnalyticsAlert {is_active: true})
            RETURN a
            """
            
            result = await self.neo4j_client.run_query(query)
            
            for record in result:
                alert_data = record['a']
                alert = AnalyticsAlert(**alert_data)
                self.active_alerts[alert.alert_id] = alert
            
            logger.info(f"Loaded {len(self.active_alerts)} active alerts")
            
        except Exception as e:
            logger.error(f"Failed to load active alerts: {e}")
    
    def _clear_metric_caches(self, metric_id: str) -> None:
        """Clear caches related to a specific metric."""
        try:
            # Remove metric-related cache entries
            keys_to_remove = [
                key for key in self.metrics_cache.keys()
                if metric_id in str(key)
            ]
            
            for key in keys_to_remove:
                del self.metrics_cache[key]
            
        except Exception as e:
            logger.error(f"Failed to clear metric caches: {e}")
    
    async def get_service_stats(self) -> Dict[str, Any]:
        """Get analytics service statistics."""
        try:
            stats = {
                'metric_definitions_count': len(self.metric_definitions),
                'active_alerts_count': len(self.active_alerts),
                'metrics_cache_size': len(self.metrics_cache),
                'reports_cache_size': len(self.reports_cache),
                'insights_cache_size': len(self.insights_cache),
                'anomaly_thresholds_count': len(self.anomaly_thresholds),
                'alert_monitoring_active': self.alert_task is not None and not self.alert_task.done()
            }
            
            # Add database statistics
            query = """
            MATCH (m:MetricDefinition)
            OPTIONAL MATCH (m)-[:HAS_VALUE]->(v:MetricValue)
            RETURN count(m) as metric_definitions,
                   count(v) as metric_values
            """
            
            result = await self.neo4j_client.run_query(query)
            if result:
                stats.update(result[0])
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get service stats: {e}")
            return {'error': str(e)}


# =============================================================================
# Utility Functions
# =============================================================================

async def track_simple_metric(metric_name: str, 
                            value: float, 
                            tenant_id: str = "default") -> bool:
    """
    Simple function to track a metric value.
    
    Args:
        metric_name: Name of the metric
        value: Metric value
        tenant_id: Tenant identifier
        
    Returns:
        True if successful
    """
    # Initialize required services
    settings = get_settings()
    
    neo4j_client = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_username,
        password=settings.neo4j_password
    )
    
    qdrant_client = QdrantClient(settings.qdrant_url)
    
    # Create analytics service
    analytics_service = AnalyticsService(
        neo4j_client=neo4j_client,
        qdrant_client=qdrant_client
    )
    
    try:
        await analytics_service.initialize()
        
        # Create or get metric definition
        metric_def = MetricDefinition(
            name=metric_name,
            description=f"Simple metric: {metric_name}",
            metric_type=MetricType.BUSINESS,
            unit="count",
            data_source=DataSource.INTERNAL_TRACKING,
            tenant_id=tenant_id
        )
        
        metric_id = await analytics_service.create_metric_definition(metric_def)
        
        # Create metric value
        metric_value = MetricValue(
            metric_id=metric_id,
            value=value,
            timestamp=datetime.now(timezone.utc),
            data_source=DataSource.INTERNAL_TRACKING,
            tenant_id=tenant_id
        )
        
        # Ingest value
        success = await analytics_service.ingest_metric_value(metric_value)
        
        return success
        
    finally:
        await analytics_service.close()


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        # Test simple metric tracking
        success = await track_simple_metric(
            metric_name="test_page_views",
            value=150.0,
            tenant_id="test-tenant"
        )
        
        print(f"Metric tracking: {'success' if success else 'failed'}")

    asyncio.run(main())