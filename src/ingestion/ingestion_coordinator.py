"""
Content Ingestion Coordinator for SEO Content Knowledge Graph System.

This module provides centralized coordination of all content ingestion operations including:
- Management of different content sources (websites, Google Drive, file uploads)
- Job scheduling and queue management
- Progress tracking and monitoring
- Error handling and recovery
- Deduplication and quality control
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum

from .base_ingester import BaseIngester, IngestionJob, ContentSource
from .website_ingester import WebsiteIngester
from .gdrive_ingester import GoogleDriveIngester
from .file_ingester import FileUploadIngester
from .queue_processor import ContentProcessingQueue, TaskPriority
from .deduplication_service import ContentDeduplicationService
from ..database.supabase_client import supabase_client

logger = logging.getLogger(__name__)


class IngestionType(Enum):
    """Types of content ingestion."""
    WEBSITE = "website"
    GOOGLE_DRIVE = "google_drive"
    FILE_UPLOAD = "file_upload"
    BULK_IMPORT = "bulk_import"


class IngestionStatus(Enum):
    """Status of ingestion operations."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ContentIngestionCoordinator:
    """
    Central coordinator for all content ingestion operations.
    
    Manages the complete content ingestion workflow from source setup
    to final knowledge graph integration.
    """
    
    def __init__(self, organization_id: str):
        self.organization_id = organization_id
        self.logger = logging.getLogger(__name__)
        
        # Initialize services
        self.queue_manager = ContentProcessingQueue()
        self.dedup_service = ContentDeduplicationService(organization_id)
        
        # Initialize ingesters
        self.website_ingester = None
        self.gdrive_ingester = None
        self.file_ingester = None
    
    async def setup_website_source(self, website_url: str, config: Dict[str, Any]) -> ContentSource:
        """Set up a new website content source."""
        try:
            if not self.website_ingester:
                self.website_ingester = WebsiteIngester(self.organization_id)
            
            # Validate website access
            async with self.website_ingester:
                if not await self.website_ingester.validate_source({'website_url': website_url, 'config': config}):
                    raise Exception(f"Cannot access website: {website_url}")
                
                # Create content source
                source = await self.website_ingester.create_website_source(website_url, config)
                
                self.logger.info(f"Created website content source: {source.source_id}")
                return source
            
        except Exception as e:
            self.logger.error(f"Failed to setup website source: {e}")
            raise
    
    async def setup_gdrive_source(self, folder_id: str, config: Dict[str, Any], 
                                credentials_file: str) -> ContentSource:
        """Set up a new Google Drive content source."""
        try:
            if not self.gdrive_ingester:
                self.gdrive_ingester = GoogleDriveIngester(self.organization_id, credentials_file)
            
            # Initialize service and validate access
            if not await self.gdrive_ingester.initialize_service():
                raise Exception("Failed to initialize Google Drive service")
            
            if not await self.gdrive_ingester.validate_source({'folder_id': folder_id, 'config': config}):
                raise Exception(f"Cannot access Google Drive folder: {folder_id}")
            
            # Create content source
            source = await self.gdrive_ingester.create_gdrive_source(folder_id, config)
            
            self.logger.info(f"Created Google Drive content source: {source.source_id}")
            return source
            
        except Exception as e:
            self.logger.error(f"Failed to setup Google Drive source: {e}")
            raise
    
    async def setup_file_upload_source(self, source_name: str, config: Dict[str, Any]) -> ContentSource:
        """Set up a new file upload content source."""
        try:
            if not self.file_ingester:
                self.file_ingester = FileUploadIngester(self.organization_id)
            
            # Validate file upload configuration
            if not await self.file_ingester.validate_source(config):
                raise Exception("File upload configuration validation failed")
            
            # Create content source
            source = await self.file_ingester.create_file_upload_source(source_name, config)
            
            self.logger.info(f"Created file upload content source: {source.source_id}")
            return source
            
        except Exception as e:
            self.logger.error(f"Failed to setup file upload source: {e}")
            raise
    
    async def start_ingestion(self, source_id: str, priority: TaskPriority = TaskPriority.MEDIUM) -> str:
        """Start content ingestion for a source."""
        try:
            # Get source information
            source = await self._get_content_source(source_id)
            if not source:
                raise Exception(f"Content source {source_id} not found")
            
            # Queue appropriate ingestion task
            task_id = None
            
            if source["source_type"] == "website":
                source_metadata = json.loads(source["source_metadata"])
                task_id = await self.queue_manager.enqueue_website_processing(
                    self.organization_id,
                    source["source_url"],
                    source_metadata.get("config", {}),
                    priority
                )
            
            elif source["source_type"] == "google_drive":
                source_metadata = json.loads(source["source_metadata"])
                task_id = await self.queue_manager.enqueue_gdrive_processing(
                    self.organization_id,
                    source_metadata["folder_id"],
                    source_metadata.get("config", {}),
                    priority
                )
            
            elif source["source_type"] == "file_upload":
                # File upload sources are typically processed immediately upon upload
                # This would be for reprocessing existing uploads
                raise Exception("File upload sources cannot be restarted via this method")
            
            else:
                raise Exception(f"Unknown source type: {source['source_type']}")
            
            # Update source last crawled timestamp
            await self._update_source_crawl_status(source_id, "running", task_id)
            
            self.logger.info(f"Started ingestion for source {source_id}, task: {task_id}")
            return task_id
            
        except Exception as e:
            self.logger.error(f"Failed to start ingestion for source {source_id}: {e}")
            raise
    
    async def process_file_uploads(self, files_data: List[Dict[str, Any]], 
                                 source_name: str = "Direct Upload",
                                 priority: TaskPriority = TaskPriority.HIGH) -> str:
        """Process uploaded files immediately."""
        try:
            # Enqueue file processing task
            task_id = await self.queue_manager.enqueue_file_processing(
                self.organization_id,
                files_data,
                source_name,
                priority
            )
            
            self.logger.info(f"Started file upload processing, task: {task_id}")
            return task_id
            
        except Exception as e:
            self.logger.error(f"Failed to process file uploads: {e}")
            raise
    
    async def get_ingestion_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of an ingestion task."""
        try:
            return await self.queue_manager.get_task_status(task_id)
        except Exception as e:
            self.logger.error(f"Failed to get ingestion status for {task_id}: {e}")
            return {"error": str(e)}
    
    async def cancel_ingestion(self, task_id: str) -> bool:
        """Cancel a running ingestion task."""
        try:
            return await self.queue_manager.cancel_task(task_id)
        except Exception as e:
            self.logger.error(f"Failed to cancel ingestion {task_id}: {e}")
            return False
    
    async def get_source_statistics(self, source_id: str) -> Dict[str, Any]:
        """Get statistics for a content source."""
        try:
            source = await self._get_content_source(source_id)
            if not source:
                return {"error": "Source not found"}
            
            stats = {
                "source_id": source_id,
                "source_type": source["source_type"],
                "created_at": source["created_at"],
                "last_crawled": source["last_crawled"],
                "is_active": source["is_active"]
            }
            
            # Get content statistics
            raw_content_result = supabase_client.client.table("raw_content").select("file_size, word_count").eq("source_id", source_id).eq("organization_id", self.organization_id).execute()
            
            processed_content_result = supabase_client.client.table("processed_content").select("quality_score, processing_status").eq("source_id", source_id).eq("organization_id", self.organization_id).execute()
            
            raw_data = raw_content_result.data
            processed_data = processed_content_result.data
            
            stats.update({
                "total_raw_content": len(raw_data),
                "total_processed_content": len(processed_data),
                "total_size_bytes": sum(item.get("file_size", 0) for item in raw_data),
                "total_words": sum(item.get("word_count", 0) for item in raw_data),
                "successful_processing": len([item for item in processed_data if item.get("processing_status") == "completed"]),
                "failed_processing": len([item for item in processed_data if item.get("processing_status") == "failed"]),
                "average_quality_score": 0
            })
            
            # Calculate average quality score
            quality_scores = [item.get("quality_score", 0) for item in processed_data if item.get("quality_score")]
            if quality_scores:
                stats["average_quality_score"] = sum(quality_scores) / len(quality_scores)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get source statistics: {e}")
            return {"error": str(e)}
    
    async def get_organization_ingestion_overview(self) -> Dict[str, Any]:
        """Get overview of all ingestion activities for the organization."""
        try:
            # Get all sources
            sources_result = supabase_client.client.table("content_sources").select("*").eq("organization_id", self.organization_id).execute()
            
            # Get ingestion jobs
            jobs_result = supabase_client.client.table("ingestion_jobs").select("*").eq("organization_id", self.organization_id).order("started_at", desc=True).limit(50).execute()
            
            # Get queue statistics
            queue_stats = await self.queue_manager.get_queue_statistics()
            
            # Calculate summary statistics
            sources = sources_result.data
            jobs = jobs_result.data
            
            active_sources = len([s for s in sources if s.get("is_active")])
            
            job_stats = {
                "total_jobs": len(jobs),
                "completed_jobs": len([j for j in jobs if j.get("status") == "completed"]),
                "failed_jobs": len([j for j in jobs if j.get("status") == "failed"]),
                "running_jobs": len([j for j in jobs if j.get("status") == "running"]),
            }
            
            # Recent activity
            recent_jobs = jobs[:10]  # Last 10 jobs
            
            overview = {
                "organization_id": self.organization_id,
                "summary": {
                    "total_sources": len(sources),
                    "active_sources": active_sources,
                    "source_types": {},
                    "job_statistics": job_stats,
                    "queue_health": queue_stats
                },
                "sources": sources,
                "recent_jobs": recent_jobs,
                "generated_at": datetime.now().isoformat()
            }
            
            # Count source types
            for source in sources:
                source_type = source.get("source_type", "unknown")
                overview["summary"]["source_types"][source_type] = overview["summary"]["source_types"].get(source_type, 0) + 1
            
            return overview
            
        except Exception as e:
            self.logger.error(f"Failed to get organization ingestion overview: {e}")
            return {"error": str(e)}
    
    async def run_duplicate_detection(self, content_id: str) -> List[Dict[str, Any]]:
        """Run duplicate detection for a specific piece of content."""
        try:
            # Get content information
            content_info = await self._get_processed_content(content_id)
            if not content_info:
                raise Exception(f"Content {content_id} not found")
            
            # Run duplicate detection
            duplicates = await self.dedup_service.detect_duplicates(
                content_id,
                content_info["processed_text"],
                content_info["title"],
                json.loads(content_info.get("metadata", "{}"))
            )
            
            # Convert to dict format
            duplicate_dicts = []
            for duplicate in duplicates:
                duplicate_dicts.append({
                    "content_id_1": duplicate.content_id_1,
                    "content_id_2": duplicate.content_id_2,
                    "duplicate_type": duplicate.duplicate_type.value,
                    "similarity_score": duplicate.similarity_score,
                    "duplicate_percentage": duplicate.duplicate_percentage,
                    "recommended_action": duplicate.recommended_action.value,
                    "confidence_score": duplicate.confidence_score,
                    "details": duplicate.details
                })
            
            return duplicate_dicts
            
        except Exception as e:
            self.logger.error(f"Failed to run duplicate detection: {e}")
            return []
    
    async def schedule_periodic_sync(self, source_id: str, interval_hours: int = 24) -> bool:
        """Schedule periodic synchronization for a content source."""
        try:
            # This would integrate with a scheduler like Celery Beat
            # For now, just update the source metadata
            
            source = await self._get_content_source(source_id)
            if not source:
                raise Exception(f"Content source {source_id} not found")
            
            source_metadata = json.loads(source["source_metadata"])
            source_metadata["sync_schedule"] = {
                "enabled": True,
                "interval_hours": interval_hours,
                "next_sync": (datetime.now() + timedelta(hours=interval_hours)).isoformat()
            }
            
            supabase_client.client.table("content_sources").update({
                "source_metadata": json.dumps(source_metadata),
                "updated_at": datetime.now().isoformat()
            }).eq("source_id", source_id).execute()
            
            self.logger.info(f"Scheduled periodic sync for source {source_id} every {interval_hours} hours")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to schedule periodic sync: {e}")
            return False
    
    # Helper methods
    
    async def _get_content_source(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get content source information."""
        try:
            result = supabase_client.client.table("content_sources").select("*").eq("source_id", source_id).eq("organization_id", self.organization_id).execute()
            
            if result.data:
                return result.data[0]
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get content source: {e}")
            return None
    
    async def _get_processed_content(self, content_id: str) -> Optional[Dict[str, Any]]:
        """Get processed content information."""
        try:
            result = supabase_client.client.table("processed_content").select("*").eq("content_id", content_id).eq("organization_id", self.organization_id).execute()
            
            if result.data:
                return result.data[0]
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get processed content: {e}")
            return None
    
    async def _update_source_crawl_status(self, source_id: str, status: str, task_id: str = None):
        """Update the crawl status of a content source."""
        try:
            update_data = {
                "last_crawled": datetime.now().isoformat() if status == "completed" else None,
                "updated_at": datetime.now().isoformat()
            }
            
            if task_id:
                source = await self._get_content_source(source_id)
                if source:
                    source_metadata = json.loads(source["source_metadata"])
                    source_metadata["last_crawl_task_id"] = task_id
                    source_metadata["last_crawl_status"] = status
                    update_data["source_metadata"] = json.dumps(source_metadata)
            
            supabase_client.client.table("content_sources").update(update_data).eq("source_id", source_id).execute()
            
        except Exception as e:
            self.logger.error(f"Failed to update source crawl status: {e}")


# Global coordinator instance factory
def create_ingestion_coordinator(organization_id: str) -> ContentIngestionCoordinator:
    """Create a content ingestion coordinator for an organization."""
    return ContentIngestionCoordinator(organization_id)