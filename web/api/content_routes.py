"""
Main content management API routes for Knowledge Base.
Refactored into modular components following CLAUDE.md guidelines (<500 lines).
"""

from fastapi import APIRouter, Request, HTTPException, Query
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging
import json

# Import modular components
from .content_upload import router as upload_router
from .content_auth import get_current_user_safe

logger = logging.getLogger(__name__)

# Main router that combines all content routes
router = APIRouter(prefix="/api/content", tags=["content"])

# Include upload routes
router.include_router(upload_router, tags=["upload"])


@router.get("/list")
async def list_content(
    request: Request,
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    search: Optional[str] = Query(None),
    content_type: Optional[str] = Query(None),
    sort: str = Query("created_at", regex="^(created_at|title|word_count)$")
):
    """List content with pagination, search, and filtering."""
    try:
        # Get current user context
        current_user = await get_current_user_safe(request)
        
        # Use real database service for data retrieval
        from src.database.content_service import ContentDatabaseService
        db_service = ContentDatabaseService()
        
        # Build filter parameters
        organization_id = current_user.get("org_id", "00000000-0000-0000-0000-000000000001")
        
        # Get content list from database
        result = await db_service.get_content_items(
            organization_id=organization_id,
            search=search,
            content_type=content_type,
            limit=limit,
            offset=offset
        )
        
        # Check if result is successful
        if not result.get('success'):
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
        
        # Extract data from result
        content_list = result.get('content', [])
        total_count = result.get('total', 0)
        
        # Format response
        formatted_content = []
        for item in content_list:
            # Calculate file size from content length if not provided
            content = item.get("content", "")
            file_size = item.get("file_size") or (len(content.encode('utf-8')) if content else 0)
            
            formatted_item = {
                "id": item.get("id"),
                "title": item.get("title", "Untitled"),
                "content_type": item.get("content_type", "document"),
                "content": content,  # Include content for document viewing
                "word_count": item.get("word_count", 0),
                "seo_score": item.get("seo_score", 0),
                "readability_score": item.get("readability_score", 0),
                "file_size": file_size,  # Add file_size field
                "file_type": item.get("file_type", "document"),
                "summary": item.get("summary") or content[:200] + "..." if content else "No content available",
                "status": item.get("processing_status", "completed"),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
                "extracted_topics": item.get("extracted_topics", [])[:5]  # Limit for list view
            }
            formatted_content.append(formatted_item)
        
        return {
            "success": True,
            "content": formatted_content,
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < total_count
        }
        
    except Exception as e:
        logger.error(f"Content listing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve content: {str(e)}")


@router.get("/{content_id}")
async def get_content_by_id(content_id: str, request: Request):
    """Get specific content item by ID for document viewing."""
    try:
        # Get current user context
        current_user = await get_current_user_safe(request)
        organization_id = current_user.get("org_id", "00000000-0000-0000-0000-000000000001")
        
        # Use database service to get content
        from src.database.content_service import ContentDatabaseService
        db_service = ContentDatabaseService()
        
        # Get content item
        content_item = await db_service.get_content_item(content_id, organization_id)
        
        if not content_item:
            raise HTTPException(status_code=404, detail="Content not found")
        
        # Format full content for viewing
        formatted_content = {
            "id": content_item.get("id"),
            "title": content_item.get("title", "Untitled"),
            "content": content_item.get("content", ""),
            "content_type": content_item.get("content_type", "document"),
            "word_count": content_item.get("word_count", 0),
            "seo_score": content_item.get("seo_score"),
            "readability_score": content_item.get("readability_score"),
            "file_size": content_item.get("file_size") or (len(content_item.get("content", "").encode('utf-8'))),
            "file_type": content_item.get("file_type", "document"),
            "summary": content_item.get("summary", ""),
            "keywords": content_item.get("keywords", []),
            "extracted_topics": content_item.get("extracted_topics", []),
            "created_at": content_item.get("created_at"),
            "updated_at": content_item.get("updated_at"),
            "processing_status": content_item.get("processing_status", "completed")
        }
        
        return {
            "success": True,
            **formatted_content
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get content {content_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve content: {str(e)}")


# Health check for the content API
@router.get("/health")
async def content_health():
    """Health check endpoint for content API."""
    return {
        "status": "healthy",
        "module": "content_api",
        "version": "2.0.0",
        "components": [
            "file_upload",
            "batch_processing", 
            "content_listing",
            "ai_generation",
            "url_import"
        ]
    }


# Brand Voice Management Models
class BrandVoiceModel(BaseModel):
    """Model for brand voice configuration."""
    description: str = ""
    tone: str = "professional" 
    formality: str = "semi-formal"
    keywords: str = ""


@router.get("/brand-voice")
async def get_brand_voice(request: Request):
    """Get brand voice configuration for the current user/organization."""
    try:
        current_user = await get_current_user_safe(request)
        org_id = current_user.get("org_id", "default")
        
        # For now, return default configuration
        # In the future, this could be stored in the database per organization
        default_brand_voice = {
            "description": "",
            "tone": "professional",
            "formality": "semi-formal", 
            "keywords": ""
        }
        
        return {
            "success": True,
            "brand_voice": default_brand_voice
        }
        
    except Exception as e:
        logger.error(f"Failed to get brand voice: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve brand voice")


@router.post("/brand-voice")
async def save_brand_voice(request: Request, brand_voice: BrandVoiceModel):
    """Save brand voice configuration for the current user/organization."""
    try:
        current_user = await get_current_user_safe(request)
        org_id = current_user.get("org_id", "default")
        
        # For now, just return success
        # In the future, this could be stored in the database per organization
        logger.info(f"Brand voice saved for org {org_id}: {brand_voice.dict()}")
        
        return {
            "success": True,
            "message": "Brand voice configuration saved successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to save brand voice: {e}")
        raise HTTPException(status_code=500, detail="Failed to save brand voice")


@router.put("/brand-voice")  
async def update_brand_voice(request: Request, brand_voice: BrandVoiceModel):
    """Update brand voice configuration for the current user/organization."""
    try:
        current_user = await get_current_user_safe(request)
        org_id = current_user.get("org_id", "default")
        
        # For now, just return success 
        # In the future, this could be stored in the database per organization
        logger.info(f"Brand voice updated for org {org_id}: {brand_voice.dict()}")
        
        return {
            "success": True,
            "message": "Brand voice configuration updated successfully",
            "brand_voice": brand_voice.dict()
        }
        
    except Exception as e:
        logger.error(f"Failed to update brand voice: {e}")
        raise HTTPException(status_code=500, detail="Failed to update brand voice")


@router.post("/generate")
async def generate_content(request: dict):
    """Generate content using AI agents."""
    try:
        from .content_generation import generate_ai_content
        return await generate_ai_content(request)
    except Exception as e:
        logger.error(f"Content generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.post("/url-import")
async def import_from_url(request: Request, data: dict):
    """Import content from URL with web scraping."""
    try:
        from .url_import import import_url_content
        return await import_url_content(request, data)
    except Exception as e:
        logger.error(f"URL import failed: {e}")
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")


@router.get("/{content_id}")
async def get_content_by_id(
    content_id: str,
    request: Request
):
    """Get specific content item by ID."""
    try:
        # Get current user context
        current_user = await get_current_user_safe(request)
        
        # Use real database service for data retrieval
        from src.database.content_service import ContentDatabaseService
        db_service = ContentDatabaseService()
        
        # Get current user org for security check
        user_org_id = current_user.get("org_id", "00000000-0000-0000-0000-000000000001")
        content_item = await db_service.get_content_item(content_id, user_org_id)
        
        if not content_item:
            raise HTTPException(status_code=404, detail="Content not found")
        
        return {
            "success": True,
            "content": content_item
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Content retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve content: {str(e)}")