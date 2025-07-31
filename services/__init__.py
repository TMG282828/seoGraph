"""
Services package for the SEO Content Knowledge Graph System.

DEPRECATED: This package provides backward compatibility for legacy imports.
New code should import directly from src.services.

This package contains various services for content processing, analytics,
AI integration, and workflow management.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from 'services' is deprecated. Use 'src.services' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from consolidated structure
try:
    from src.services.content_ingestion import ContentIngestionService
    from src.services.analytics_service import AnalyticsService
    from src.services.embedding_service import EmbeddingService
    from src.services.workflow_orchestrator import WorkflowOrchestrator
    from src.services.competitor_monitoring import CompetitorMonitoringService
    from src.services.gap_analysis import GapAnalysisService
    from src.services.google_drive_service import GoogleDriveService
    from src.services.auth_service import AuthService
    from src.services.searxng_service import SearXNGService
    
except ImportError as e:
    # Fallback to legacy implementations if new structure not available
    warnings.warn(f"Failed to import from new structure, falling back to legacy: {e}")
    from .content_ingestion import ContentIngestionService
    from .analytics_service import AnalyticsService
    from .embedding_service import EmbeddingService
    from .workflow_orchestrator import WorkflowOrchestrator
    from .competitor_monitoring import CompetitorMonitoringService
    from .gap_analysis import GapAnalysisService
    from .google_drive_service import GoogleDriveService
    from .auth_service import AuthService
    from .searxng_service import SearXNGService

__all__ = [
    "ContentIngestionService",
    "AnalyticsService", 
    "EmbeddingService",
    "WorkflowOrchestrator",
    "CompetitorMonitoringService",
    "GapAnalysisService",
    "GoogleDriveService",
    "AuthService",
    "SearXNGService",
]