"""
Batch processing manager for Knowledge Base.
Handles batch upload operations and status tracking.
"""

from typing import Dict, Any, List
import logging
import uuid
import asyncio
from datetime import datetime
from fastapi import UploadFile

logger = logging.getLogger(__name__)

# In-memory batch status tracking (could be moved to Redis/database for production)
batch_status_store: Dict[str, Dict[str, Any]] = {}


async def initialize_batch(batch_id: str, total_files: int) -> None:
    """Initialize batch tracking data."""
    batch_status_store[batch_id] = {
        "batch_id": batch_id,
        "status": "processing",
        "total_files": total_files,
        "processed_files": [],
        "failed_files": [],
        "summary": {
            "successful": 0,
            "failed": 0,
            "total_word_count": 0
        },
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }


async def process_batch_files(
    batch_id: str,
    files: List[UploadFile],
    current_user: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process multiple files in batch.
    
    Args:
        batch_id: Unique batch identifier
        files: List of uploaded files
        current_user: Current user context
        
    Returns:
        Batch processing results
    """
    from .content_analysis import analyze_file_content
    from .content_storage import store_content
    
    batch_results = {
        "success": True,
        "batch_id": batch_id,
        "total_files": len(files),
        "processed_files": [],
        "failed_files": [],
        "summary": {"successful": 0, "failed": 0, "total_word_count": 0}
    }
    
    for i, file in enumerate(files):
        try:
            # Update batch status
            batch_status_store[batch_id]["status"] = "processing"
            batch_status_store[batch_id]["current_file"] = f"{i+1}/{len(files)}"
            batch_status_store[batch_id]["updated_at"] = datetime.utcnow().isoformat()
            
            if not file.filename:
                raise ValueError("File has no filename")
            
            # Read and process file
            content_bytes = await file.read()
            if len(content_bytes) == 0:
                raise ValueError("Empty file")
            
            # Decode content
            try:
                content_text = content_bytes.decode('utf-8')
            except UnicodeDecodeError:
                content_text = content_bytes.decode('latin-1', errors='ignore')
            
            # Analyze content
            analysis_result = await analyze_file_content(
                file=file,
                content_text=content_text,
                content_bytes=content_bytes,
                current_user=current_user
            )
            
            # Store in database
            storage_success = await store_content(
                filename=file.filename,
                content_text=content_text,
                content_type=file.content_type,
                analysis_result=analysis_result,
                current_user=current_user
            )
            
            if storage_success:
                # Success
                processed_file = {
                    "filename": file.filename,
                    "status": "success",
                    "word_count": analysis_result.get("word_count", 0),
                    "seo_score": analysis_result.get("analysis", {}).get("seo_metrics", {}).get("overall_seo_score", 0)
                }
                batch_results["processed_files"].append(processed_file)
                batch_status_store[batch_id]["processed_files"].append(processed_file)
                batch_results["summary"]["successful"] += 1
                batch_results["summary"]["total_word_count"] += processed_file["word_count"]
                batch_status_store[batch_id]["summary"]["successful"] += 1
                batch_status_store[batch_id]["summary"]["total_word_count"] += processed_file["word_count"]
            else:
                raise Exception("Database storage failed")
                
        except Exception as e:
            # Failure
            failed_file = {
                "filename": file.filename if file.filename else f"file_{i+1}",
                "status": "failed",
                "error": str(e)
            }
            batch_results["failed_files"].append(failed_file)
            batch_status_store[batch_id]["failed_files"].append(failed_file)
            batch_results["summary"]["failed"] += 1
            batch_status_store[batch_id]["summary"]["failed"] += 1
            logger.error(f"Batch file processing failed: {file.filename}: {e}")
    
    # Update final batch status
    final_status = "completed" if batch_results["summary"]["failed"] == 0 else "partial"
    batch_status_store[batch_id]["status"] = final_status
    batch_status_store[batch_id]["updated_at"] = datetime.utcnow().isoformat()
    batch_status_store[batch_id].pop("current_file", None)
    
    return batch_results


async def get_batch_status_data(batch_id: str) -> Dict[str, Any]:
    """Get current batch status."""
    return batch_status_store.get(batch_id)


async def clear_batch_data(batch_id: str) -> bool:
    """Clear batch data from memory."""
    if batch_id in batch_status_store:
        del batch_status_store[batch_id]
        return True
    return False