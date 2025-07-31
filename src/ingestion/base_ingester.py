"""
Base Content Ingester for SEO Content Knowledge Graph System.

This module provides the foundational framework for all content ingestion operations,
including content processing, validation, and integration with the agent pipeline.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
import hashlib
import json

from ..database.supabase_client import supabase_client
from ..services.graph_vector_service import graph_vector_service
from ..agents.agent_coordinator import agent_coordinator, WorkflowRequest, AgentContext

logger = logging.getLogger(__name__)


class ContentSource(BaseModel):
    """Model for content source information."""
    source_id: str
    source_type: str  # website, google_drive, file_upload
    source_url: Optional[str] = None
    source_metadata: Dict[str, Any] = Field(default_factory=dict)
    organization_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    last_crawled: Optional[datetime] = None
    is_active: bool = True


class RawContent(BaseModel):
    """Model for raw ingested content before processing."""
    content_id: str
    source_id: str
    raw_text: str
    content_type: str
    title: Optional[str] = None
    url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    content_hash: str
    extracted_at: datetime = Field(default_factory=datetime.now)
    file_size: Optional[int] = None
    word_count: int = 0


class ProcessedContent(BaseModel):
    """Model for processed content ready for knowledge graph integration."""
    content_id: str
    source_id: str
    processed_text: str
    title: str
    summary: str
    keywords: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    quality_score: float = 0.0
    processing_status: str = "completed"
    processed_at: datetime = Field(default_factory=datetime.now)
    agent_results: Optional[Dict[str, Any]] = None


class IngestionJob(BaseModel):
    """Model for ingestion job tracking."""
    job_id: str
    job_type: str  # crawl, file_upload, drive_sync
    source_id: str
    organization_id: str
    status: str = "pending"  # pending, running, completed, failed, paused
    progress: int = 0
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    job_config: Dict[str, Any] = Field(default_factory=dict)
    results: Dict[str, Any] = Field(default_factory=dict)


class BaseIngester(ABC):
    """
    Base class for all content ingesters.
    
    Provides common functionality for:
    - Content extraction and processing
    - Database operations and storage
    - Agent pipeline integration
    - Error handling and retry logic
    - Progress tracking and logging
    """
    
    def __init__(self, ingester_type: str, organization_id: str):
        """Initialize the base ingester."""
        self.ingester_type = ingester_type
        self.organization_id = organization_id
        self.logger = logging.getLogger(f"{__name__}.{ingester_type}")
        
    @abstractmethod
    async def extract_content(self, source_config: Dict[str, Any]) -> List[RawContent]:
        """Extract raw content from the source."""
        pass
    
    @abstractmethod
    async def validate_source(self, source_config: Dict[str, Any]) -> bool:
        """Validate that the content source is accessible and properly configured."""
        pass
    
    async def create_ingestion_job(self, job_type: str, source_id: str, 
                                 job_config: Dict[str, Any]) -> IngestionJob:
        """Create a new ingestion job."""
        job_id = self._generate_job_id()
        
        job = IngestionJob(
            job_id=job_id,
            job_type=job_type,
            source_id=source_id,
            organization_id=self.organization_id,
            job_config=job_config
        )
        
        # Store job in database
        await self._store_ingestion_job(job)
        
        self.logger.info(f"Created ingestion job: {job_id}")
        return job
    
    async def process_content_pipeline(self, raw_content: RawContent, 
                                     job: IngestionJob) -> ProcessedContent:
        """Process raw content through the AI agent pipeline."""
        try:
            # Create agent context
            context = AgentContext(
                organization_id=self.organization_id,
                user_id="system",  # System-initiated processing
                session_id=job.job_id,
                metadata={
                    "source_id": raw_content.source_id,
                    "content_type": raw_content.content_type,
                    "ingestion_job": job.job_id
                }
            )
            
            # Prepare workflow request for content analysis
            workflow_request = WorkflowRequest(
                workflow_type="content_optimization",
                input_data={
                    "content_text": raw_content.raw_text,
                    "content_title": raw_content.title or "Untitled",
                    "content_url": raw_content.url,
                    "content_type": raw_content.content_type,
                    "source_metadata": raw_content.metadata
                },
                parallel_execution=True,
                include_analytics=True
            )
            
            # Execute agent workflow
            self.logger.info(f"Processing content through agent pipeline: {raw_content.content_id}")
            agent_results = await agent_coordinator.execute_workflow(workflow_request, context)
            
            # Extract processed content from agent results
            processed_content = await self._extract_processed_content(raw_content, agent_results)
            
            # Store processed content
            await self._store_processed_content(processed_content)
            
            # Integrate with knowledge graph
            await self._integrate_with_knowledge_graph(processed_content)
            
            self.logger.info(f"Successfully processed content: {raw_content.content_id}")
            return processed_content
            
        except Exception as e:
            self.logger.error(f"Failed to process content {raw_content.content_id}: {e}")
            
            # Create failed processing result
            processed_content = ProcessedContent(
                content_id=raw_content.content_id,
                source_id=raw_content.source_id,
                processed_text=raw_content.raw_text,
                title=raw_content.title or "Failed Processing",
                summary="Content processing failed",
                processing_status="failed",
                metadata={**raw_content.metadata, "error": str(e)}
            )
            
            await self._store_processed_content(processed_content)
            raise
    
    async def run_ingestion_job(self, job: IngestionJob, source_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run the complete ingestion job."""
        try:
            # Update job status
            job.status = "running"
            job.started_at = datetime.now()
            await self._update_ingestion_job(job)
            
            self.logger.info(f"Starting ingestion job: {job.job_id}")
            
            # Extract content from source
            self.logger.info("Extracting content from source...")
            raw_contents = await self.extract_content(source_config)
            
            job.total_items = len(raw_contents)
            await self._update_ingestion_job(job)
            
            # Process each piece of content
            processed_contents = []
            failed_contents = []
            
            for i, raw_content in enumerate(raw_contents):
                try:
                    # Check for duplicates
                    if await self._is_duplicate_content(raw_content):
                        self.logger.info(f"Skipping duplicate content: {raw_content.content_id}")
                        continue
                    
                    # Store raw content
                    await self._store_raw_content(raw_content)
                    
                    # Process through agent pipeline
                    processed_content = await self.process_content_pipeline(raw_content, job)
                    processed_contents.append(processed_content)
                    
                    job.processed_items += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to process item {i}: {e}")
                    failed_contents.append({
                        "content_id": raw_content.content_id,
                        "error": str(e)
                    })
                    job.failed_items += 1
                
                # Update progress
                job.progress = int((i + 1) / len(raw_contents) * 100)
                await self._update_ingestion_job(job)
            
            # Complete the job
            job.status = "completed"
            job.completed_at = datetime.now()
            job.results = {
                "processed_count": len(processed_contents),
                "failed_count": len(failed_contents),
                "failed_items": failed_contents,
                "total_words_processed": sum(content.metadata.get("word_count", 0) for content in processed_contents)
            }
            
            await self._update_ingestion_job(job)
            
            self.logger.info(f"Completed ingestion job: {job.job_id} - "
                           f"{len(processed_contents)} processed, {len(failed_contents)} failed")
            
            return job.results
            
        except Exception as e:
            # Mark job as failed
            job.status = "failed"
            job.completed_at = datetime.now()
            job.error_message = str(e)
            await self._update_ingestion_job(job)
            
            self.logger.error(f"Ingestion job failed: {job.job_id} - {e}")
            raise
    
    # Helper methods
    
    def _generate_content_id(self, source_id: str, content_identifier: str) -> str:
        """Generate a unique content ID."""
        return f"{source_id}_{hashlib.md5(content_identifier.encode()).hexdigest()[:16]}"
    
    def _generate_job_id(self) -> str:
        """Generate a unique job ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.ingester_type}_{timestamp}_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calculate content hash for duplicate detection."""
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def _extract_processed_content(self, raw_content: RawContent, 
                                       agent_results: Dict[str, Any]) -> ProcessedContent:
        """Extract processed content from agent results."""
        
        # Extract analysis results
        aggregated_results = agent_results.get("aggregated_results", {})
        
        # Extract keywords and topics from analysis
        keywords = []
        topics = []
        entities = []
        quality_score = 0.0
        
        # Process agent-specific results
        if "content_analysis" in agent_results.get("agent_results", {}):
            analysis_result = agent_results["agent_results"]["content_analysis"]
            if analysis_result.success:
                analysis_data = analysis_result.result_data
                keywords.extend(analysis_data.get("keywords", []))
                topics.extend(analysis_data.get("topics", []))
                entities.extend(analysis_data.get("entities", []))
                quality_score = analysis_data.get("quality_score", 0.0)
        
        # Generate summary from first paragraph or use AI-generated summary
        summary = raw_content.raw_text.split('\n')[0][:200] + "..." if len(raw_content.raw_text) > 200 else raw_content.raw_text
        
        return ProcessedContent(
            content_id=raw_content.content_id,
            source_id=raw_content.source_id,
            processed_text=raw_content.raw_text,
            title=raw_content.title or "Untitled Content",
            summary=summary,
            keywords=keywords[:10],  # Limit to top 10 keywords
            topics=topics[:5],      # Limit to top 5 topics
            entities=entities[:20], # Limit to top 20 entities
            metadata={
                **raw_content.metadata,
                "word_count": raw_content.word_count,
                "original_hash": raw_content.content_hash
            },
            quality_score=quality_score,
            agent_results=agent_results
        )
    
    async def _integrate_with_knowledge_graph(self, content: ProcessedContent):
        """Integrate processed content with the knowledge graph."""
        try:
            # Use the graph-vector service to add content to both Neo4j and Qdrant
            await graph_vector_service.add_content_to_graph(
                organization_id=self.organization_id,
                content_id=content.content_id,
                title=content.title,
                content_text=content.processed_text,
                summary=content.summary,
                keywords=content.keywords,
                topics=content.topics,
                entities=content.entities,
                metadata=content.metadata,
                source_url=content.metadata.get("url"),
                content_type=content.metadata.get("content_type", "unknown")
            )
            
            self.logger.info(f"Integrated content with knowledge graph: {content.content_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to integrate content with knowledge graph: {e}")
            raise
    
    # Database operations
    
    async def _store_ingestion_job(self, job: IngestionJob):
        """Store ingestion job in database."""
        try:
            data = {
                "job_id": job.job_id,
                "job_type": job.job_type,
                "source_id": job.source_id,
                "organization_id": job.organization_id,
                "status": job.status,
                "progress": job.progress,
                "total_items": job.total_items,
                "processed_items": job.processed_items,
                "failed_items": job.failed_items,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "error_message": job.error_message,
                "job_config": json.dumps(job.job_config),
                "results": json.dumps(job.results)
            }
            
            supabase_client.client.table("ingestion_jobs").insert(data).execute()
            
        except Exception as e:
            self.logger.error(f"Failed to store ingestion job: {e}")
            raise
    
    async def _update_ingestion_job(self, job: IngestionJob):
        """Update ingestion job in database."""
        try:
            data = {
                "status": job.status,
                "progress": job.progress,
                "total_items": job.total_items,
                "processed_items": job.processed_items,
                "failed_items": job.failed_items,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "error_message": job.error_message,
                "results": json.dumps(job.results),
                "updated_at": datetime.now().isoformat()
            }
            
            supabase_client.client.table("ingestion_jobs").update(data).eq("job_id", job.job_id).execute()
            
        except Exception as e:
            self.logger.error(f"Failed to update ingestion job: {e}")
            raise
    
    async def _store_raw_content(self, content: RawContent):
        """Store raw content in database."""
        try:
            data = {
                "content_id": content.content_id,
                "source_id": content.source_id,
                "raw_text": content.raw_text,
                "content_type": content.content_type,
                "title": content.title,
                "url": content.url,
                "metadata": json.dumps(content.metadata),
                "content_hash": content.content_hash,
                "extracted_at": content.extracted_at.isoformat(),
                "file_size": content.file_size,
                "word_count": content.word_count,
                "organization_id": self.organization_id
            }
            
            supabase_client.client.table("raw_content").insert(data).execute()
            
        except Exception as e:
            self.logger.error(f"Failed to store raw content: {e}")
            raise
    
    async def _store_processed_content(self, content: ProcessedContent):
        """Store processed content in database."""
        try:
            data = {
                "content_id": content.content_id,
                "source_id": content.source_id,
                "processed_text": content.processed_text,
                "title": content.title,
                "summary": content.summary,
                "keywords": json.dumps(content.keywords),
                "topics": json.dumps(content.topics),
                "entities": json.dumps(content.entities),
                "metadata": json.dumps(content.metadata),
                "quality_score": content.quality_score,
                "processing_status": content.processing_status,
                "processed_at": content.processed_at.isoformat(),
                "agent_results": json.dumps(content.agent_results) if content.agent_results else None,
                "organization_id": self.organization_id
            }
            
            supabase_client.client.table("processed_content").insert(data).execute()
            
        except Exception as e:
            self.logger.error(f"Failed to store processed content: {e}")
            raise
    
    async def _is_duplicate_content(self, content: RawContent) -> bool:
        """Check if content is a duplicate based on hash."""
        try:
            result = supabase_client.client.table("raw_content").select("content_id").eq("content_hash", content.content_hash).eq("organization_id", self.organization_id).execute()
            
            return len(result.data) > 0
            
        except Exception as e:
            self.logger.warning(f"Failed to check for duplicate content: {e}")
            return False
    
    # Utility methods for subclasses
    
    def _extract_text_from_html(self, html_content: str) -> str:
        """Extract clean text from HTML content."""
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text and clean it up
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except ImportError:
            self.logger.warning("BeautifulSoup not available, returning raw HTML")
            return html_content
        except Exception as e:
            self.logger.error(f"Failed to extract text from HTML: {e}")
            return html_content
    
    def _count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split()) if text else 0
    
    def _extract_title_from_html(self, html_content: str) -> Optional[str]:
        """Extract title from HTML content."""
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Try to find title tag
            title_tag = soup.find('title')
            if title_tag and title_tag.string:
                return title_tag.string.strip()
            
            # Try to find h1 tag
            h1_tag = soup.find('h1')
            if h1_tag and h1_tag.string:
                return h1_tag.string.strip()
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to extract title from HTML: {e}")
            return None
    
    async def get_job_status(self, job_id: str) -> Optional[IngestionJob]:
        """Get the current status of an ingestion job."""
        try:
            result = supabase_client.client.table("ingestion_jobs").select("*").eq("job_id", job_id).eq("organization_id", self.organization_id).execute()
            
            if result.data:
                job_data = result.data[0]
                return IngestionJob(
                    job_id=job_data["job_id"],
                    job_type=job_data["job_type"],
                    source_id=job_data["source_id"],
                    organization_id=job_data["organization_id"],
                    status=job_data["status"],
                    progress=job_data["progress"],
                    total_items=job_data["total_items"],
                    processed_items=job_data["processed_items"],
                    failed_items=job_data["failed_items"],
                    started_at=datetime.fromisoformat(job_data["started_at"]) if job_data["started_at"] else None,
                    completed_at=datetime.fromisoformat(job_data["completed_at"]) if job_data["completed_at"] else None,
                    error_message=job_data["error_message"],
                    job_config=json.loads(job_data["job_config"]) if job_data["job_config"] else {},
                    results=json.loads(job_data["results"]) if job_data["results"] else {}
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get job status: {e}")
            return None