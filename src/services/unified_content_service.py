"""
Unified Content Service for Knowledge Base.
Provides a centralized interface for content operations.
"""

from typing import Dict, List, Any, Optional
import logging
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

class UnifiedContentService:
    """Unified service for all content operations."""
    
    def __init__(self):
        # In-memory storage for demo purposes
        self._content_store = []
        logger.info("UnifiedContentService initialized")
    
    async def list_content(
        self, 
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
        sort_by: str = "created_at"
    ) -> List[Dict[str, Any]]:
        """List content with filtering, pagination, and sorting."""
        try:
            # Apply filters
            filtered_content = self._content_store.copy()
            
            if filters:
                if "search" in filters:
                    search_term = filters["search"].lower()
                    filtered_content = [
                        item for item in filtered_content
                        if search_term in item.get("title", "").lower() or
                           search_term in item.get("content", "").lower()
                    ]
                
                if "content_type" in filters:
                    filtered_content = [
                        item for item in filtered_content
                        if item.get("content_type") == filters["content_type"]
                    ]
                
                if "organization_id" in filters:
                    # Filter by organization if specified
                    filtered_content = [
                        item for item in filtered_content
                        if item.get("organization_id") == filters["organization_id"]
                    ]
            
            # Sort
            if sort_by == "created_at":
                filtered_content.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            elif sort_by == "title":
                filtered_content.sort(key=lambda x: x.get("title", ""))
            elif sort_by == "word_count":
                filtered_content.sort(key=lambda x: x.get("word_count", 0), reverse=True)
            
            # Paginate
            start = offset
            end = offset + limit
            paginated_content = filtered_content[start:end]
            
            logger.info(f"Listed {len(paginated_content)} content items (filtered from {len(filtered_content)})")
            return paginated_content
            
        except Exception as e:
            logger.error(f"Failed to list content: {e}")
            return []
    
    async def count_content(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count total content items matching filters."""
        try:
            if not filters:
                return len(self._content_store)
            
            # Apply same filtering logic as list_content
            filtered_content = self._content_store.copy()
            
            if "search" in filters:
                search_term = filters["search"].lower()
                filtered_content = [
                    item for item in filtered_content
                    if search_term in item.get("title", "").lower() or
                       search_term in item.get("content", "").lower()
                ]
            
            if "content_type" in filters:
                filtered_content = [
                    item for item in filtered_content
                    if item.get("content_type") == filters["content_type"]
                ]
            
            if "organization_id" in filters:
                filtered_content = [
                    item for item in filtered_content
                    if item.get("organization_id") == filters["organization_id"]
                ]
            
            return len(filtered_content)
            
        except Exception as e:
            logger.error(f"Failed to count content: {e}")
            return 0
    
    async def store_content(
        self,
        title: str,
        content: str,
        content_type: str = "document",
        organization_id: str = "demo-org",
        **kwargs
    ) -> Dict[str, Any]:
        """Store new content item."""
        try:
            content_item = {
                "id": str(uuid.uuid4()),
                "title": title,
                "content": content,
                "content_type": content_type,
                "organization_id": organization_id,
                "word_count": len(content.split()),
                "seo_score": 75,  # Mock score
                "readability_score": 80,  # Mock score
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "extracted_topics": ["demo", "content", "analysis"]  # Mock topics
            }
            
            # Add any additional fields from kwargs
            content_item.update(kwargs)
            
            self._content_store.append(content_item)
            
            logger.info(f"Stored content item: {content_item['id']} - {title}")
            return content_item
            
        except Exception as e:
            logger.error(f"Failed to store content: {e}")
            raise
    
    async def get_content_by_id(self, content_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve content by ID."""
        try:
            for item in self._content_store:
                if item.get("id") == content_id:
                    return item
            return None
            
        except Exception as e:
            logger.error(f"Failed to get content by ID {content_id}: {e}")
            return None
    
    async def delete_content(self, content_id: str) -> bool:
        """Delete content by ID."""
        try:
            for i, item in enumerate(self._content_store):
                if item.get("id") == content_id:
                    del self._content_store[i]
                    logger.info(f"Deleted content item: {content_id}")
                    return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete content {content_id}: {e}")
            return False

# Global instance
unified_content_service = UnifiedContentService()