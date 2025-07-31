"""
Debug router for graph API issues.
"""
from fastapi import APIRouter
import logging

logger = logging.getLogger(__name__)
debug_graph_router = APIRouter(prefix="/debug-graph", tags=["debug-graph"])

@debug_graph_router.get("/test")
async def debug_test():
    """Test if debug router works."""
    return {"status": "debug router working", "message": "This is a new debug endpoint"}

@debug_graph_router.get("/graph-test")
async def debug_graph_test():
    """Test the graph functionality in isolation."""
    try:
        from src.database.content_service import ContentDatabaseService
        db_service = ContentDatabaseService()
        
        result = await db_service.get_content_items(
            organization_id="demo-org",
            limit=5
        )
        
        if result.get("success"):
            content_list = result.get("content", [])
            return {
                "success": True,
                "message": f"Found {len(content_list)} documents",
                "content_sample": content_list[:1] if content_list else []
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "Unknown error")
            }
            
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }