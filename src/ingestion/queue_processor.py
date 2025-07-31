"""
Content Processing Queue for SEO Content Knowledge Graph System.

This module provides asynchronous content processing using Redis/Celery including:
- Task queue management for ingestion jobs
- Distributed processing across multiple workers
- Progress tracking and status updates
- Error handling and retry logic
- Priority-based job scheduling
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import json
import hashlib
from enum import Enum

from celery import Celery
from celery.result import AsyncResult
from celery import signals
from celery.exceptions import Retry, WorkerLostError
from kombu import Queue
import redis

from .base_ingester import IngestionJob, BaseIngester
from .website_ingester import WebsiteIngester
from .gdrive_ingester import GoogleDriveIngester
from .file_ingester import FileUploadIngester
from ..database.supabase_client import supabase_client

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    MEDIUM = 5
    HIGH = 10
    URGENT = 20


class TaskStatus(Enum):
    """Task status values."""
    PENDING = "pending"
    STARTED = "started"
    PROGRESS = "progress"
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"
    REVOKED = "revoked"


# Redis configuration
REDIS_URL = "redis://localhost:6379/0"  # From environment variables
redis_client = redis.from_url(REDIS_URL)

# Celery configuration
celery_app = Celery(
    'content_processor',
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=['src.ingestion.queue_processor']
)

# Celery settings
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour timeout
    task_soft_time_limit=3300,  # 55 minutes soft timeout
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=False,
    task_default_retry_delay=60,
    task_max_retries=3,
    worker_log_format='[%(asctime)s: %(levelname)s/%(processName)s] %(message)s',
    worker_task_log_format='[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s',
)

# Define task queues with priorities
celery_app.conf.task_routes = {
    'src.ingestion.queue_processor.process_website_content': {'queue': 'content_processing'},
    'src.ingestion.queue_processor.process_gdrive_content': {'queue': 'content_processing'},
    'src.ingestion.queue_processor.process_file_upload': {'queue': 'content_processing'},
    'src.ingestion.queue_processor.run_maintenance_tasks': {'queue': 'maintenance'},
    'src.ingestion.queue_processor.cleanup_failed_jobs': {'queue': 'cleanup'},
}

celery_app.conf.task_queues = (
    Queue('content_processing', routing_key='content_processing'),
    Queue('maintenance', routing_key='maintenance'),
    Queue('cleanup', routing_key='cleanup'),
)


class ContentProcessingQueue:
    """
    Content processing queue manager.
    
    Manages asynchronous content ingestion and processing tasks using Celery/Redis.
    """
    
    def __init__(self):
        self.celery_app = celery_app
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
    
    async def enqueue_website_processing(self, organization_id: str, website_url: str,
                                       source_config: Dict[str, Any], 
                                       priority: TaskPriority = TaskPriority.MEDIUM) -> str:
        """Enqueue website content processing task."""
        try:
            task_id = self._generate_task_id("website", organization_id, website_url)
            
            # Create task payload
            task_payload = {
                'task_id': task_id,
                'organization_id': organization_id,
                'source_type': 'website',
                'source_config': {
                    'website_url': website_url,
                    **source_config
                },
                'priority': priority.value,
                'created_at': datetime.now().isoformat()
            }
            
            # Store task metadata
            await self._store_task_metadata(task_id, task_payload)
            
            # Enqueue Celery task
            result = process_website_content.apply_async(
                args=[task_payload],
                task_id=task_id,
                priority=priority.value,
                queue='content_processing'
            )
            
            self.logger.info(f"Enqueued website processing task: {task_id}")
            return task_id
            
        except Exception as e:
            self.logger.error(f"Failed to enqueue website processing task: {e}")
            raise
    
    async def enqueue_gdrive_processing(self, organization_id: str, folder_id: str,
                                      source_config: Dict[str, Any],
                                      priority: TaskPriority = TaskPriority.MEDIUM) -> str:
        """Enqueue Google Drive content processing task."""
        try:
            task_id = self._generate_task_id("gdrive", organization_id, folder_id)
            
            task_payload = {
                'task_id': task_id,
                'organization_id': organization_id,
                'source_type': 'gdrive',
                'source_config': {
                    'folder_id': folder_id,
                    **source_config
                },
                'priority': priority.value,
                'created_at': datetime.now().isoformat()
            }
            
            await self._store_task_metadata(task_id, task_payload)
            
            result = process_gdrive_content.apply_async(
                args=[task_payload],
                task_id=task_id,
                priority=priority.value,
                queue='content_processing'
            )
            
            self.logger.info(f"Enqueued Google Drive processing task: {task_id}")
            return task_id
            
        except Exception as e:
            self.logger.error(f"Failed to enqueue Google Drive processing task: {e}")
            raise
    
    async def enqueue_file_processing(self, organization_id: str, files_data: List[Dict[str, Any]],
                                    source_name: str, priority: TaskPriority = TaskPriority.HIGH) -> str:
        """Enqueue file upload processing task."""
        try:
            task_id = self._generate_task_id("file_upload", organization_id, source_name)
            
            task_payload = {
                'task_id': task_id,
                'organization_id': organization_id,
                'source_type': 'file_upload',
                'source_config': {
                    'files_data': files_data,
                    'source_name': source_name
                },
                'priority': priority.value,
                'created_at': datetime.now().isoformat()
            }
            
            await self._store_task_metadata(task_id, task_payload)
            
            result = process_file_upload.apply_async(
                args=[task_payload],
                task_id=task_id,
                priority=priority.value,
                queue='content_processing'
            )
            
            self.logger.info(f"Enqueued file upload processing task: {task_id}")
            return task_id
            
        except Exception as e:
            self.logger.error(f"Failed to enqueue file upload processing task: {e}")
            raise
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status and progress information."""
        try:
            # Get Celery task result
            result = AsyncResult(task_id, app=self.celery_app)
            
            # Get task metadata from Redis
            metadata = await self._get_task_metadata(task_id)
            
            # Combine status information
            status_info = {
                'task_id': task_id,
                'status': result.status,
                'progress': 0,
                'result': None,
                'error': None,
                'created_at': metadata.get('created_at') if metadata else None,
                'started_at': None,
                'completed_at': None,
                'organization_id': metadata.get('organization_id') if metadata else None,
                'source_type': metadata.get('source_type') if metadata else None
            }
            
            # Update based on task state
            if result.status == 'PENDING':
                status_info['progress'] = 0
            elif result.status == 'STARTED':
                status_info['started_at'] = result.info.get('started_at') if result.info else None
                status_info['progress'] = result.info.get('progress', 0) if result.info else 0
            elif result.status == 'PROGRESS':
                status_info['progress'] = result.info.get('progress', 0)
                status_info['current_step'] = result.info.get('current_step', '')
            elif result.status == 'SUCCESS':
                status_info['progress'] = 100
                status_info['result'] = result.info
                status_info['completed_at'] = result.date_done.isoformat() if result.date_done else None
            elif result.status == 'FAILURE':
                status_info['progress'] = 0
                status_info['error'] = str(result.info)
                status_info['completed_at'] = result.date_done.isoformat() if result.date_done else None
            
            return status_info
            
        except Exception as e:
            self.logger.error(f"Failed to get task status for {task_id}: {e}")
            return {
                'task_id': task_id,
                'status': 'UNKNOWN',
                'error': str(e)
            }
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        try:
            self.celery_app.control.revoke(task_id, terminate=True)
            
            # Update task metadata
            metadata = await self._get_task_metadata(task_id)
            if metadata:
                metadata['status'] = TaskStatus.REVOKED.value
                metadata['completed_at'] = datetime.now().isoformat()
                await self._store_task_metadata(task_id, metadata)
            
            self.logger.info(f"Cancelled task: {task_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel task {task_id}: {e}")
            return False
    
    async def get_queue_statistics(self) -> Dict[str, Any]:
        """Get queue statistics and health information."""
        try:
            # Get Celery inspect information
            inspect = self.celery_app.control.inspect()
            
            # Get active tasks
            active_tasks = inspect.active() or {}
            total_active = sum(len(tasks) for tasks in active_tasks.values())
            
            # Get scheduled tasks
            scheduled_tasks = inspect.scheduled() or {}
            total_scheduled = sum(len(tasks) for tasks in scheduled_tasks.values())
            
            # Get reserved tasks
            reserved_tasks = inspect.reserved() or {}
            total_reserved = sum(len(tasks) for tasks in reserved_tasks.values())
            
            # Get worker statistics
            worker_stats = inspect.stats() or {}
            
            # Get Redis queue lengths
            queue_lengths = {}
            for queue_name in ['content_processing', 'maintenance', 'cleanup']:
                queue_lengths[queue_name] = self.redis_client.llen(queue_name)
            
            return {
                'active_tasks': total_active,
                'scheduled_tasks': total_scheduled,
                'reserved_tasks': total_reserved,
                'queue_lengths': queue_lengths,
                'workers': list(worker_stats.keys()),
                'worker_count': len(worker_stats),
                'redis_connected': self.redis_client.ping(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get queue statistics: {e}")
            return {'error': str(e)}
    
    async def cleanup_completed_tasks(self, older_than_days: int = 7) -> int:
        """Clean up completed tasks older than specified days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=older_than_days)
            
            # Get all task metadata keys
            task_keys = self.redis_client.keys("task_metadata:*")
            
            cleaned_count = 0
            for key in task_keys:
                try:
                    metadata_json = self.redis_client.get(key)
                    if metadata_json:
                        metadata = json.loads(metadata_json)
                        created_at = datetime.fromisoformat(metadata.get('created_at', ''))
                        
                        if created_at < cutoff_date:
                            # Check if task is completed
                            task_id = key.decode('utf-8').replace('task_metadata:', '')
                            result = AsyncResult(task_id, app=self.celery_app)
                            
                            if result.status in ['SUCCESS', 'FAILURE', 'REVOKED']:
                                self.redis_client.delete(key)
                                result.forget()  # Remove from Celery backend
                                cleaned_count += 1
                
                except Exception as e:
                    self.logger.warning(f"Failed to process task key {key}: {e}")
            
            self.logger.info(f"Cleaned up {cleaned_count} completed tasks")
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup completed tasks: {e}")
            return 0
    
    # Helper methods
    
    def _generate_task_id(self, source_type: str, organization_id: str, identifier: str) -> str:
        """Generate unique task ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{source_type}_{organization_id}_{identifier}_{timestamp}"
        hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"{source_type}_{timestamp}_{hash_suffix}"
    
    async def _store_task_metadata(self, task_id: str, metadata: Dict[str, Any]):
        """Store task metadata in Redis."""
        try:
            key = f"task_metadata:{task_id}"
            self.redis_client.set(key, json.dumps(metadata), ex=604800)  # 7 days expiry
        except Exception as e:
            self.logger.error(f"Failed to store task metadata: {e}")
    
    async def _get_task_metadata(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task metadata from Redis."""
        try:
            key = f"task_metadata:{task_id}"
            metadata_json = self.redis_client.get(key)
            if metadata_json:
                return json.loads(metadata_json)
            return None
        except Exception as e:
            self.logger.error(f"Failed to get task metadata: {e}")
            return None


# Celery Tasks
@celery_app.task(bind=True, name='src.ingestion.queue_processor.process_website_content')
def process_website_content(self, task_payload: Dict[str, Any]):
    """Process website content ingestion task."""
    try:
        organization_id = task_payload['organization_id']
        source_config = task_payload['source_config']
        
        # Update task status
        self.update_state(
            state='PROGRESS',
            meta={'progress': 10, 'current_step': 'Initializing website ingester'}
        )
        
        # Initialize ingester
        ingester = WebsiteIngester(organization_id)
        
        # Validate source
        self.update_state(
            state='PROGRESS',
            meta={'progress': 20, 'current_step': 'Validating website access'}
        )
        
        if not asyncio.run(ingester.validate_source(source_config)):
            raise Exception("Website validation failed")
        
        # Create ingestion job
        self.update_state(
            state='PROGRESS',
            meta={'progress': 30, 'current_step': 'Creating ingestion job'}
        )
        
        job = asyncio.run(ingester.create_ingestion_job(
            "website_crawl",
            "website_source",  # This would be the actual source ID
            source_config
        ))
        
        # Run ingestion
        self.update_state(
            state='PROGRESS',
            meta={'progress': 40, 'current_step': 'Extracting website content'}
        )
        
        results = asyncio.run(ingester.run_ingestion_job(job, source_config))
        
        self.update_state(
            state='PROGRESS',
            meta={'progress': 100, 'current_step': 'Completed'}
        )
        
        return {
            'job_id': job.job_id,
            'results': results,
            'completed_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Website processing task failed: {e}")
        raise self.retry(exc=e, countdown=60, max_retries=3)


@celery_app.task(bind=True, name='src.ingestion.queue_processor.process_gdrive_content')
def process_gdrive_content(self, task_payload: Dict[str, Any]):
    """Process Google Drive content ingestion task."""
    try:
        organization_id = task_payload['organization_id']
        source_config = task_payload['source_config']
        
        # Update task status
        self.update_state(
            state='PROGRESS',
            meta={'progress': 10, 'current_step': 'Initializing Google Drive ingester'}
        )
        
        # Initialize ingester
        credentials_file = "credentials/google_credentials.json"  # From config
        ingester = GoogleDriveIngester(organization_id, credentials_file)
        
        # Validate source
        self.update_state(
            state='PROGRESS',
            meta={'progress': 20, 'current_step': 'Validating Google Drive access'}
        )
        
        if not asyncio.run(ingester.validate_source(source_config)):
            raise Exception("Google Drive validation failed")
        
        # Create ingestion job
        self.update_state(
            state='PROGRESS',
            meta={'progress': 30, 'current_step': 'Creating ingestion job'}
        )
        
        job = asyncio.run(ingester.create_ingestion_job(
            "gdrive_sync",
            "gdrive_source",  # This would be the actual source ID
            source_config
        ))
        
        # Run ingestion
        self.update_state(
            state='PROGRESS',
            meta={'progress': 40, 'current_step': 'Syncing Google Drive content'}
        )
        
        results = asyncio.run(ingester.run_ingestion_job(job, source_config))
        
        self.update_state(
            state='PROGRESS',
            meta={'progress': 100, 'current_step': 'Completed'}
        )
        
        return {
            'job_id': job.job_id,
            'results': results,
            'completed_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Google Drive processing task failed: {e}")
        raise self.retry(exc=e, countdown=60, max_retries=3)


@celery_app.task(bind=True, name='src.ingestion.queue_processor.process_file_upload')
def process_file_upload(self, task_payload: Dict[str, Any]):
    """Process file upload ingestion task."""
    try:
        organization_id = task_payload['organization_id']
        source_config = task_payload['source_config']
        
        # Update task status
        self.update_state(
            state='PROGRESS',
            meta={'progress': 10, 'current_step': 'Initializing file upload ingester'}
        )
        
        # Initialize ingester
        ingester = FileUploadIngester(organization_id)
        
        # Validate source
        self.update_state(
            state='PROGRESS',
            meta={'progress': 20, 'current_step': 'Validating file upload configuration'}
        )
        
        if not asyncio.run(ingester.validate_source(source_config)):
            raise Exception("File upload validation failed")
        
        # Create ingestion job
        self.update_state(
            state='PROGRESS',
            meta={'progress': 30, 'current_step': 'Creating ingestion job'}
        )
        
        job = asyncio.run(ingester.create_ingestion_job(
            "file_upload",
            "file_upload_source",  # This would be the actual source ID
            source_config
        ))
        
        # Process uploaded files
        self.update_state(
            state='PROGRESS',
            meta={'progress': 40, 'current_step': 'Processing uploaded files'}
        )
        
        # Prepare files for processing
        source_config['uploaded_files'] = source_config.get('files_data', [])
        
        results = asyncio.run(ingester.run_ingestion_job(job, source_config))
        
        self.update_state(
            state='PROGRESS',
            meta={'progress': 100, 'current_step': 'Completed'}
        )
        
        return {
            'job_id': job.job_id,
            'results': results,
            'completed_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"File upload processing task failed: {e}")
        raise self.retry(exc=e, countdown=60, max_retries=3)


@celery_app.task(name='src.ingestion.queue_processor.run_maintenance_tasks')
def run_maintenance_tasks():
    """Run periodic maintenance tasks."""
    try:
        logger.info("Starting maintenance tasks")
        
        # Cleanup old tasks
        queue_manager = ContentProcessingQueue()
        cleaned_count = asyncio.run(queue_manager.cleanup_completed_tasks(older_than_days=7))
        
        # Additional maintenance tasks can be added here
        
        return {
            'cleaned_tasks': cleaned_count,
            'completed_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Maintenance tasks failed: {e}")
        raise


@celery_app.task(name='src.ingestion.queue_processor.cleanup_failed_jobs')
def cleanup_failed_jobs():
    """Clean up failed ingestion jobs."""
    try:
        logger.info("Starting cleanup of failed jobs")
        
        # Get failed ingestion jobs older than 24 hours
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        result = supabase_client.client.table("ingestion_jobs").select("*").eq("status", "failed").lt("started_at", cutoff_time.isoformat()).execute()
        
        failed_jobs = result.data
        
        # Mark jobs for cleanup or retry
        cleanup_count = 0
        for job in failed_jobs:
            try:
                # Logic to decide whether to retry or cleanup
                # For now, just mark as cleaned up
                
                supabase_client.client.table("ingestion_jobs").update({
                    "status": "cleaned_up",
                    "updated_at": datetime.now().isoformat()
                }).eq("job_id", job["job_id"]).execute()
                
                cleanup_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to cleanup job {job['job_id']}: {e}")
        
        return {
            'cleaned_jobs': cleanup_count,
            'completed_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed job cleanup failed: {e}")
        raise


# Celery signals
@signals.worker_ready.connect
def worker_ready(sender=None, **kwargs):
    """Called when worker is ready."""
    logger.info(f"Celery worker ready: {sender}")


@signals.task_prerun.connect
def task_prerun(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwargs_extra):
    """Called before task execution."""
    logger.info(f"Starting task: {task.name} [{task_id}]")


@signals.task_postrun.connect
def task_postrun(sender=None, task_id=None, task=None, args=None, kwargs=None, 
                retval=None, state=None, **kwargs_extra):
    """Called after task execution."""
    logger.info(f"Completed task: {task.name} [{task_id}] - State: {state}")


@signals.task_failure.connect
def task_failure(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwargs):
    """Called on task failure."""
    logger.error(f"Task failed: {sender.name} [{task_id}] - Exception: {exception}")


# Global queue manager instance
queue_manager = ContentProcessingQueue()