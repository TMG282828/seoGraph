# OpenTelemetry Performance Monitoring

This directory contains the OpenTelemetry performance monitoring integration for the SEO Content Knowledge Graph System.

## Overview

The OpenTelemetry monitoring system provides:

- **Distributed Tracing**: Track requests across services and components
- **Custom Metrics**: Monitor agent performance, database queries, and system resources
- **Automatic Instrumentation**: FastAPI, HTTP clients, and database connections
- **Multiple Exporters**: Prometheus, Jaeger, OTLP, and console output

## Quick Start

### 1. Install Dependencies

The required OpenTelemetry packages are included in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Configuration

Copy the example configuration:

```bash
cp .env.otel.example .env
```

Or add these variables to your existing `.env` file:

```bash
# Basic configuration
OTEL_SERVICE_NAME=seo-content-system
OTEL_SERVICE_VERSION=1.0.0
OTEL_ENVIRONMENT=development
OTEL_ENABLE_CONSOLE=true
OTEL_ENABLE_PROMETHEUS=true
```

### 3. Start the Application

The monitoring system initializes automatically when you start the FastAPI application:

```bash
python web/main.py
```

### 4. View Metrics and Traces

- **Prometheus metrics**: http://localhost:9090/metrics
- **Health monitoring**: http://localhost:8000/api/health/monitoring
- **Console traces**: Check application logs

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OTEL_SERVICE_NAME` | Service name identifier | `seo-content-system` |
| `OTEL_SERVICE_VERSION` | Service version | `1.0.0` |
| `OTEL_ENVIRONMENT` | Deployment environment | `development` |
| `OTEL_ENABLE_CONSOLE` | Enable console trace export | `false` |
| `OTEL_ENABLE_PROMETHEUS` | Enable Prometheus metrics | `true` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP endpoint URL | None |
| `OTEL_EXPORTER_JAEGER_ENDPOINT` | Jaeger endpoint URL | None |
| `OTEL_PROMETHEUS_PORT` | Prometheus metrics port | `9090` |
| `OTEL_TRACE_SAMPLE_RATE` | Trace sampling rate (0.0-1.0) | `1.0` |

### Exporters

#### Console Export (Development)
```bash
OTEL_ENABLE_CONSOLE=true
```

#### Prometheus Metrics
```bash
OTEL_ENABLE_PROMETHEUS=true
OTEL_PROMETHEUS_PORT=9090
```

#### Jaeger Tracing
```bash
OTEL_EXPORTER_JAEGER_ENDPOINT=http://localhost:14268/api/traces
```

#### OTLP (Generic)
```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

## Usage in Code

### Agent Tracing

Automatically trace agent operations:

```python
from monitoring.otel_monitor import trace_agent

@trace_agent("content_analysis", "analyze_content")
async def analyze_content(self, content_item):
    # Agent implementation
    return result
```

### Database Tracing

Automatically trace database queries:

```python
from monitoring.otel_monitor import trace_db

@trace_db("neo4j", "execute_query")
async def execute_query(self, query, parameters=None):
    # Database implementation
    return results
```

### Custom Tracing

For custom operations:

```python
from monitoring.otel_monitor import get_otel_monitor

otel_monitor = get_otel_monitor()
if otel_monitor:
    async with otel_monitor.trace_operation("custom_operation") as span:
        # Your custom operation
        span.set_attribute("operation_type", "processing")
        result = await process_data()
        return result
```

### Custom Metrics

Record custom metrics:

```python
from monitoring.otel_monitor import get_otel_monitor

otel_monitor = get_otel_monitor()
if otel_monitor:
    await otel_monitor.record_custom_metric(
        "processing_queue_size", 
        42, 
        {"queue_type": "content_analysis"}
    )
```

## Available Metrics

### Agent Metrics
- `agent_requests_total`: Total agent requests by type and operation
- `agent_request_duration_seconds`: Agent request duration histogram
- `agent_errors_total`: Total agent errors by type and error type
- `agent_tokens_total`: Total tokens consumed by agents

### Database Metrics
- `database_queries_total`: Total database queries by database and operation
- `database_query_duration_seconds`: Database query duration histogram
- `database_connections_active`: Active database connections

### System Metrics
- `system_cpu_usage_percent`: System CPU usage percentage
- `system_memory_usage_percent`: System memory usage percentage
- `system_disk_usage_percent`: System disk usage percentage
- `system_active_connections`: Number of active system connections

## Integration with Observability Platforms

### New Relic
```bash
OTEL_EXPORTER_OTLP_ENDPOINT=https://otlp.nr-data.net:4317
OTEL_EXPORTER_OTLP_HEADERS=api-key=YOUR_NEW_RELIC_LICENSE_KEY
```

### Honeycomb
```bash
OTEL_EXPORTER_OTLP_ENDPOINT=https://api.honeycomb.io:443
OTEL_EXPORTER_OTLP_HEADERS=x-honeycomb-team=YOUR_API_KEY
```

### Grafana Cloud
```bash
OTEL_EXPORTER_OTLP_ENDPOINT=https://tempo-us-central1.grafana.net:443
OTEL_EXPORTER_OTLP_HEADERS=authorization=Basic BASE64_ENCODED_CREDENTIALS
```

## Health Monitoring

Check the monitoring system status:

```bash
curl http://localhost:8000/api/health/monitoring
```

Example response:
```json
{
  "status": "up",
  "message": "OpenTelemetry monitoring active",
  "last_check": "2024-01-15T10:30:00Z",
  "details": {
    "otel_available": true,
    "monitor_initialized": true,
    "service_name": "seo-content-system",
    "service_version": "1.0.0",
    "environment": "development",
    "tracing_enabled": true,
    "metrics_enabled": true,
    "exporters": {
      "otlp": false,
      "jaeger": false,
      "prometheus": true,
      "console": true
    }
  }
}
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure OpenTelemetry packages are installed
2. **No Traces**: Check `OTEL_TRACE_SAMPLE_RATE` is > 0.0
3. **Missing Metrics**: Verify Prometheus endpoint configuration
4. **Connection Issues**: Check exporter endpoint URLs and authentication

### Debug Mode

Enable console output for debugging:

```bash
OTEL_ENABLE_CONSOLE=true
```

This will print traces and spans to the application logs.

### Performance Impact

- **Production**: Use sampling rate < 1.0 (e.g., 0.1 for 10% sampling)
- **Development**: Use sampling rate 1.0 for complete visibility
- **Metrics**: Minimal performance impact, safe for production

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Application   │    │  OTel Monitor    │    │   Exporters     │
│                 │    │                  │    │                 │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│ │   Agents    │ │────│ │   Tracing    │ │────│ │ Prometheus  │ │
│ └─────────────┘ │    │ └──────────────┘ │    │ └─────────────┘ │
│                 │    │                  │    │                 │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│ │ Databases   │ │────│ │   Metrics    │ │────│ │   Jaeger    │ │
│ └─────────────┘ │    │ └──────────────┘ │    │ └─────────────┘ │
│                 │    │                  │    │                 │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│ │  FastAPI    │ │────│ │Instrumentation│ │────│ │   Console   │ │
│ └─────────────┘ │    │ └──────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

The monitoring system automatically instruments the application and exports telemetry data to configured backends.