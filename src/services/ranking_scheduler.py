"""
Ranking Scheduler Service for Automated SerpBear Integration.

This service handles:
- Nightly automated ranking data collection
- Scheduled keyword registration and management
- Performance monitoring and alerting
- Automated reporting and insights generation

Orchestrates the complete SerpBear integration pipeline on a schedule.
"""

import logging
import asyncio
import schedule
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import threading
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field

from .serpbear_client import serpbear_client, test_serpbear_connection
from .keyword_manager import keyword_manager, extract_and_register_keywords
from .rank_processor import rank_processor, daily_ranking_workflow
from .ranking_graph_service import ranking_graph_service, sync_rankings_to_graph

logger = logging.getLogger(__name__)


class ScheduledJob(BaseModel):
    """Model for scheduled job configuration."""
    job_id: str
    name: str
    function: str
    schedule_type: str = Field(description="daily, weekly, hourly, custom")
    schedule_time: str = Field(description="Time specification (e.g., '02:00', 'monday')")
    enabled: bool = True
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    run_count: int = 0
    error_count: int = 0


class JobExecutionResult(BaseModel):
    """Model for job execution results."""
    job_id: str
    execution_time: str
    duration_seconds: float
    success: bool
    result_data: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None


class RankingScheduler:
    """
    Comprehensive scheduling service for SerpBear integration.
    
    This service manages automated execution of:
    1. Daily ranking data collection and processing
    2. Keyword extraction and registration from new content
    3. Performance analysis and alert generation
    4. Neo4j graph synchronization
    5. Automated reporting and insights
    """
    
    def __init__(self):
        """Initialize ranking scheduler."""
        self.scheduled_jobs: Dict[str, ScheduledJob] = {}
        self.execution_history: List[JobExecutionResult] = []
        self.is_running = False
        self.scheduler_thread = None
        logger.info("Ranking scheduler initialized")
    
    def register_job(
        self, 
        job_id: str, 
        name: str, 
        function: Callable, 
        schedule_type: str = "daily",
        schedule_time: str = "02:00",
        enabled: bool = True
    ):
        """
        Register a scheduled job.
        
        Args:
            job_id: Unique job identifier
            name: Human-readable job name
            function: Async function to execute
            schedule_type: Type of schedule (daily, weekly, hourly)
            schedule_time: Time specification
            enabled: Whether job is enabled
        """
        try:
            job = ScheduledJob(
                job_id=job_id,
                name=name,
                function=function.__name__,
                schedule_type=schedule_type,
                schedule_time=schedule_time,
                enabled=enabled
            )
            
            self.scheduled_jobs[job_id] = job
            
            # Configure the actual schedule
            if enabled:
                self._configure_schedule(job, function)
            
            logger.info(f"📅 Registered job: {name} ({schedule_type} at {schedule_time})")
            
        except Exception as e:
            logger.error(f"❌ Failed to register job {job_id}: {e}")
    
    def _configure_schedule(self, job: ScheduledJob, function: Callable):
        """Configure schedule.py for the job."""
        try:
            # Create wrapped function for async execution
            def job_wrapper():
                asyncio.run(self._execute_job(job.job_id, function))
            
            # Configure based on schedule type
            if job.schedule_type == "daily":
                schedule.every().day.at(job.schedule_time).do(job_wrapper)
            elif job.schedule_type == "weekly":
                # Format: "monday", "tuesday", etc.
                getattr(schedule.every(), job.schedule_time.lower()).at("02:00").do(job_wrapper)
            elif job.schedule_type == "hourly":
                schedule.every().hour.do(job_wrapper)
            elif job.schedule_type == "custom":
                # Handle custom schedule formats
                if "every" in job.schedule_time:
                    # Format: "every 30 minutes", "every 2 hours"
                    parts = job.schedule_time.split()
                    if len(parts) >= 3:
                        interval = int(parts[1])
                        unit = parts[2].rstrip('s')  # Remove plural 's'
                        getattr(schedule.every(interval), unit).do(job_wrapper)
            
            logger.debug(f"✅ Configured schedule for {job.name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to configure schedule for {job.job_id}: {e}")
    
    async def _execute_job(self, job_id: str, function: Callable) -> JobExecutionResult:
        """
        Execute a scheduled job and record results.
        
        Args:
            job_id: Job identifier
            function: Function to execute
            
        Returns:
            Job execution result
        """
        start_time = time.time()
        execution_time = str(datetime.now())
        
        try:
            logger.info(f"🚀 Executing scheduled job: {job_id}")
            
            # Execute the function
            if asyncio.iscoroutinefunction(function):
                result_data = await function()
            else:
                result_data = function()
            
            duration = time.time() - start_time
            
            # Create execution result
            result = JobExecutionResult(
                job_id=job_id,
                execution_time=execution_time,
                duration_seconds=round(duration, 2),
                success=True,
                result_data=result_data or {}
            )
            
            # Update job statistics
            if job_id in self.scheduled_jobs:
                job = self.scheduled_jobs[job_id]
                job.last_run = execution_time
                job.run_count += 1
            
            logger.info(f"✅ Job {job_id} completed successfully in {duration:.2f}s")
            
        except Exception as e:
            duration = time.time() - start_time
            error_message = str(e)
            
            # Create error result
            result = JobExecutionResult(
                job_id=job_id,
                execution_time=execution_time,
                duration_seconds=round(duration, 2),
                success=False,
                error_message=error_message
            )
            
            # Update job error statistics
            if job_id in self.scheduled_jobs:
                job = self.scheduled_jobs[job_id]
                job.error_count += 1
            
            logger.error(f"❌ Job {job_id} failed after {duration:.2f}s: {error_message}")
        
        # Store execution history (keep last 100 executions)
        self.execution_history.append(result)
        if len(self.execution_history) > 100:
            self.execution_history.pop(0)
        
        return result
    
    def start_scheduler(self):
        """Start the scheduler in a background thread."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        
        def run_scheduler():
            logger.info("📅 Starting ranking scheduler")
            while self.is_running:
                try:
                    schedule.run_pending()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"❌ Scheduler error: {e}")
                    time.sleep(60)
        
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("✅ Ranking scheduler started")
    
    def stop_scheduler(self):
        """Stop the scheduler."""
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return
        
        self.is_running = False
        schedule.clear()
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)
        
        logger.info("⏹️ Ranking scheduler stopped")
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """
        Get current scheduler status and statistics.
        
        Returns:
            Scheduler status information
        """
        try:
            # Calculate job statistics
            total_jobs = len(self.scheduled_jobs)
            enabled_jobs = len([j for j in self.scheduled_jobs.values() if j.enabled])
            total_executions = sum(j.run_count for j in self.scheduled_jobs.values())
            total_errors = sum(j.error_count for j in self.scheduled_jobs.values())
            
            # Get recent execution results
            recent_executions = sorted(
                self.execution_history, 
                key=lambda x: x.execution_time, 
                reverse=True
            )[:10]
            
            # Calculate success rate
            if total_executions > 0:
                success_rate = ((total_executions - total_errors) / total_executions) * 100
            else:
                success_rate = 100.0
            
            status = {
                "scheduler_running": self.is_running,
                "total_jobs": total_jobs,
                "enabled_jobs": enabled_jobs,
                "total_executions": total_executions,
                "total_errors": total_errors,
                "success_rate": round(success_rate, 1),
                "next_scheduled_jobs": [
                    {
                        "job_id": job.job_id,
                        "name": job.name,
                        "next_run": str(schedule.next_run()) if schedule.jobs else None
                    }
                    for job in self.scheduled_jobs.values() 
                    if job.enabled
                ][:5],
                "recent_executions": [
                    {
                        "job_id": exec.job_id,
                        "execution_time": exec.execution_time,
                        "success": exec.success,
                        "duration": exec.duration_seconds
                    }
                    for exec in recent_executions
                ],
                "timestamp": str(datetime.now())
            }
            
            return status
        
        except Exception as e:
            logger.error(f"❌ Failed to get scheduler status: {e}")
            return {"error": str(e)}
    
    def register_default_jobs(self):
        """Register default ranking-related scheduled jobs."""
        try:
            logger.info("📋 Registering default ranking jobs")
            
            # Job 1: Daily ranking data collection and processing
            self.register_job(
                job_id="daily_ranking_collection",
                name="Daily Ranking Data Collection",
                function=self._job_daily_ranking_collection,
                schedule_type="daily",
                schedule_time="02:00",
                enabled=True
            )
            
            # Job 2: Weekly keyword extraction and registration
            self.register_job(
                job_id="weekly_keyword_registration",
                name="Weekly Keyword Registration",
                function=self._job_keyword_registration,
                schedule_type="weekly",
                schedule_time="sunday",
                enabled=True
            )
            
            # Job 3: Daily graph synchronization
            self.register_job(
                job_id="daily_graph_sync",
                name="Daily Graph Synchronization",
                function=self._job_graph_sync,
                schedule_type="daily",
                schedule_time="03:00",
                enabled=True
            )
            
            # Job 4: Weekly performance reporting
            self.register_job(
                job_id="weekly_performance_report",
                name="Weekly Performance Report",
                function=self._job_performance_report,
                schedule_type="weekly",
                schedule_time="monday",
                enabled=True
            )
            
            # Job 5: Monthly data cleanup
            self.register_job(
                job_id="monthly_data_cleanup",
                name="Monthly Data Cleanup",
                function=self._job_data_cleanup,
                schedule_type="custom",
                schedule_time="every 30 days",
                enabled=True
            )
            
            logger.info("✅ Default ranking jobs registered")
            
        except Exception as e:
            logger.error(f"❌ Failed to register default jobs: {e}")
    
    # Job implementation methods
    async def _job_daily_ranking_collection(self) -> Dict[str, Any]:
        """Daily ranking data collection job."""
        try:
            logger.info("📊 Executing daily ranking collection")
            
            # Test SerpBear connection first
            connection_ok = await test_serpbear_connection()
            if not connection_ok:
                return {"error": "SerpBear connection failed"}
            
            # Execute daily ranking workflow
            result = await daily_ranking_workflow()
            
            logger.info(f"✅ Daily ranking collection completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Daily ranking collection failed: {e}")
            return {"error": str(e)}
    
    async def _job_keyword_registration(self) -> Dict[str, Any]:
        """Weekly keyword extraction and registration job."""
        try:
            logger.info("🔑 Executing keyword registration")
            
            # Extract and register keywords for main domain
            # Note: This should be configured with actual domain
            domain = "example.com"  # Replace with actual domain from config
            
            result = await extract_and_register_keywords(domain, max_keywords=50)
            
            logger.info(f"✅ Keyword registration completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Keyword registration failed: {e}")
            return {"error": str(e)}
    
    async def _job_graph_sync(self) -> Dict[str, Any]:
        """Daily graph synchronization job."""
        try:
            logger.info("🔄 Executing graph synchronization")
            
            # Sync rankings to Neo4j graph
            result = await sync_rankings_to_graph()
            
            logger.info(f"✅ Graph synchronization completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Graph synchronization failed: {e}")
            return {"error": str(e)}
    
    async def _job_performance_report(self) -> Dict[str, Any]:
        """Weekly performance reporting job."""
        try:
            logger.info("📈 Executing performance report generation")
            
            # Generate comprehensive performance insights
            # This would typically send email reports or update dashboards
            
            async with serpbear_client as client:
                domains = await client.get_domains()
                domain_names = [d.domain for d in domains]
            
            reports = {}
            for domain in domain_names[:3]:  # Limit to first 3 domains
                insights = await rank_processor.get_ranking_insights(domain, days=7)
                reports[domain] = insights
            
            result = {
                "report_generated": True,
                "domains_analyzed": len(reports),
                "reports": reports,
                "timestamp": str(datetime.now())
            }
            
            logger.info(f"✅ Performance report generated: {len(reports)} domains")
            return result
            
        except Exception as e:
            logger.error(f"❌ Performance report generation failed: {e}")
            return {"error": str(e)}
    
    async def _job_data_cleanup(self) -> Dict[str, Any]:
        """Monthly data cleanup job."""
        try:
            logger.info("🧹 Executing monthly data cleanup")
            
            # Clean up old ranking data (keep 90 days)
            cleanup_result = await ranking_graph_service.cleanup_old_rankings(days_to_keep=90)
            
            logger.info(f"✅ Data cleanup completed: {cleanup_result}")
            return cleanup_result
            
        except Exception as e:
            logger.error(f"❌ Data cleanup failed: {e}")
            return {"error": str(e)}


# Global scheduler instance
ranking_scheduler = RankingScheduler()


@asynccontextmanager
async def managed_scheduler():
    """Context manager for scheduler lifecycle."""
    try:
        # Register default jobs
        ranking_scheduler.register_default_jobs()
        
        # Start scheduler
        ranking_scheduler.start_scheduler()
        
        yield ranking_scheduler
        
    finally:
        # Stop scheduler on exit
        ranking_scheduler.stop_scheduler()


async def setup_ranking_automation() -> Dict[str, Any]:
    """
    Setup complete ranking automation system.
    
    Returns:
        Setup status and configuration
    """
    try:
        logger.info("🎛️ Setting up ranking automation system")
        
        # Register default jobs
        ranking_scheduler.register_default_jobs()
        
        # Test SerpBear connection
        connection_ok = await test_serpbear_connection()
        
        # Start scheduler
        ranking_scheduler.start_scheduler()
        
        setup_result = {
            "scheduler_started": ranking_scheduler.is_running,
            "serpbear_connection": connection_ok,
            "jobs_registered": len(ranking_scheduler.scheduled_jobs),
            "default_jobs": list(ranking_scheduler.scheduled_jobs.keys()),
            "next_execution": str(schedule.next_run()) if schedule.jobs else None,
            "timestamp": str(datetime.now())
        }
        
        logger.info(f"✅ Ranking automation setup complete: {setup_result}")
        return setup_result
        
    except Exception as e:
        logger.error(f"❌ Ranking automation setup failed: {e}")
        return {"error": str(e)}


def get_automation_status() -> Dict[str, Any]:
    """
    Get current automation system status.
    
    Returns:
        Current status of all automation components
    """
    return ranking_scheduler.get_scheduler_status()


if __name__ == "__main__":
    # Test the scheduler
    async def main():
        print("Setting up ranking automation...")
        result = await setup_ranking_automation()
        print(f"Setup result: {result}")
        
        # Wait a bit to see scheduler in action
        await asyncio.sleep(5)
        
        # Get status
        status = get_automation_status()
        print(f"Status: {status}")
        
        # Stop scheduler
        ranking_scheduler.stop_scheduler()
    
    asyncio.run(main())