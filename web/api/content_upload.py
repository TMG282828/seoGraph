"""
Content upload module for Knowledge Base.
Handles file uploads and processing.
"""

from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from typing import List, Dict, Any
import logging
import uuid
from datetime import datetime

from .content_auth import get_current_user_safe

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/upload")
async def upload_content(
    request: Request,
    file: UploadFile = File(...)
):
    """Upload and analyze a single content file."""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file uploaded")
        
        # Get current user for context
        current_user = await get_current_user_safe(request)
        
        # Read file content
        content_bytes = await file.read()
        
        if len(content_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Process content based on file type
        try:
            content_text = content_bytes.decode('utf-8')
        except UnicodeDecodeError:
            try:
                content_text = content_bytes.decode('latin-1')
            except UnicodeDecodeError as decode_error:
                logger.warning(f"Failed to decode file content: {decode_error}")
                content_text = str(content_bytes)[:1000]  # Use first 1000 chars as fallback
        
        # Generate analysis
        logger.info(f"Starting analysis for file: {file.filename}")
        from .content_analysis import analyze_file_content
        analysis_result = await analyze_file_content(
            file=file,
            content_text=content_text,
            content_bytes=content_bytes,
            current_user=current_user
        )
        logger.info(f"Analysis completed for file: {file.filename}, type: {type(analysis_result)}")
        
        # Store in database
        try:
            from .content_storage import store_content
            logger.info(f"Attempting to store content for file: {file.filename}")
            storage_success = await store_content(
                filename=file.filename,
                content_text=content_text,
                content_type=file.content_type,
                analysis_result=analysis_result,
                current_user=current_user
            )
            logger.info(f"Storage result for {file.filename}: {storage_success}")
            
            # Add storage status to response
            analysis_result["storage_success"] = storage_success
            if storage_success:
                analysis_result["content_id"] = "stored"
            else:
                analysis_result["storage_error"] = "Failed to store content in database"
                
        except Exception as storage_error:
            logger.error(f"Storage exception for {file.filename}: {storage_error}")
            analysis_result["storage_success"] = False
            analysis_result["storage_error"] = f"Storage exception: {str(storage_error)}"
        
        return analysis_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/batch-upload")
async def batch_upload_content(
    request: Request,
    files: List[UploadFile] = File(...)
):
    """Upload and analyze multiple content files in batch."""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
    
    # Get current user
    current_user = await get_current_user_safe(request)
    
    # Generate batch ID
    batch_id = str(uuid.uuid4())
    
    # Initialize batch tracking
    from .batch_manager import initialize_batch, process_batch_files
    
    await initialize_batch(batch_id, len(files))
    
    # Process files
    results = await process_batch_files(
        batch_id=batch_id,
        files=files,
        current_user=current_user
    )
    
    return results


@router.get("/batch-status/{batch_id}")
async def get_batch_status(batch_id: str):
    """Get status of batch upload operation."""
    from .batch_manager import get_batch_status_data
    
    status = await get_batch_status_data(batch_id)
    if not status:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    return status


@router.delete("/batch-status/{batch_id}")
async def clear_batch_status(batch_id: str):
    """Clear batch status data."""
    from .batch_manager import clear_batch_data
    
    success = await clear_batch_data(batch_id)
    if not success:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    return {"success": True, "message": "Batch status cleared"}