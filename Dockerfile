# Multi-stage Docker build for SEO Content Knowledge Graph System
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r app && useradd -r -g app app

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs static/uploads && \
    chown -R app:app /app

# Switch to app user
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "production_server.py"]
# Base stage - common dependencies
# =============================================================================
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# =============================================================================
# Dependencies stage - install Python dependencies
# =============================================================================
FROM base as dependencies

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# =============================================================================
# Development stage - includes dev dependencies
# =============================================================================
FROM dependencies as development

# Install development dependencies
RUN pip install \
    pytest>=7.4.0 \
    pytest-asyncio>=0.23.0 \
    pytest-cov>=4.1.0 \
    pytest-mock>=3.12.0 \
    pytest-xdist>=3.5.0 \
    black>=23.12.0 \
    ruff>=0.1.0 \
    mypy>=1.8.0 \
    pre-commit>=3.6.0 \
    bandit>=1.7.0 \
    safety>=2.3.0

# Copy application code
COPY . .

# Change ownership to app user
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Expose port
EXPOSE 8000

# Default command for development
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# =============================================================================
# Production stage - optimized for production
# =============================================================================
FROM dependencies as production

# Copy only necessary files for production
COPY agents/ ./agents/
COPY models/ ./models/
COPY database/ ./database/
COPY services/ ./services/
COPY web/ ./web/
COPY cli/ ./cli/
COPY config/ ./config/
COPY pyproject.toml .

# Create necessary directories
RUN mkdir -p logs backups temp

# Change ownership to app user
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# =============================================================================
# CLI stage - optimized for CLI usage
# =============================================================================
FROM dependencies as cli

# Copy necessary files for CLI
COPY agents/ ./agents/
COPY models/ ./models/
COPY database/ ./database/
COPY services/ ./services/
COPY cli/ ./cli/
COPY config/ ./config/
COPY pyproject.toml .

# Change ownership to app user
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Entry point for CLI
ENTRYPOINT ["python", "-m", "cli.main"]

# =============================================================================
# Worker stage - optimized for background tasks
# =============================================================================
FROM dependencies as worker

# Copy necessary files for worker
COPY agents/ ./agents/
COPY models/ ./models/
COPY database/ ./database/
COPY services/ ./services/
COPY config/ ./config/
COPY pyproject.toml .

# Change ownership to app user
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Default command for worker
CMD ["celery", "-A", "services.celery_app", "worker", "--loglevel=info"]