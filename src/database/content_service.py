"""
Unified Content Database Service for Knowledge Base.

Provides a single interface for content operations that works with both
SQLAlchemy (local) and Supabase (production) databases.
"""

import logging
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import both database systems
from .database import get_db_session
from .models import ContentItem
from sqlalchemy.orm import Session
from sqlalchemy import desc

try:
    from .supabase_client import supabase_client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    supabase_client = None

logger = logging.getLogger('database')


class ContentDatabaseService:
    """
    Unified service for content database operations.
    
    Automatically chooses between SQLAlchemy (local) and Supabase (production)
    based on availability and configuration.
    """
    
    def __init__(self):
        # For now, prefer SQLAlchemy for reliable local development
        # In production, this would check for valid Supabase credentials
        self.use_supabase = False  # Force SQLAlchemy for demo
        logger.info(f"Content DB Service initialized: {'Supabase' if self.use_supabase else 'SQLAlchemy'}")
    
    async def create_content_item(self, content_data: Dict[str, Any], organization_id: str) -> Dict[str, Any]:
        """Create a new content item in the appropriate database."""
        try:
            # Generate UUID if not provided
            if 'id' not in content_data:
                content_data['id'] = str(uuid.uuid4())
            
            # Set organization context
            content_data['organization_id'] = organization_id
            
            if self.use_supabase:
                # Use Supabase for production
                supabase_client._current_organization_id = organization_id
                result = await supabase_client.create_content_item(content_data)
                return result
            else:
                # Use SQLAlchemy for local/fallback
                db = get_db_session()
                try:
                    # Create ContentItem instance
                    content_item = ContentItem(
                        id=content_data['id'],
                        organization_id=organization_id,
                        title=content_data.get('title', 'Untitled'),
                        content=content_data.get('content', ''),
                        content_type=content_data.get('content_type', 'document'),
                        source_type=content_data.get('source_type', 'manual'),
                        source_url=content_data.get('source_url'),
                        original_filename=content_data.get('original_filename'),
                        word_count=content_data.get('word_count'),
                        seo_score=content_data.get('seo_score'),
                        readability_score=content_data.get('readability_score'),
                        keywords=content_data.get('keywords', []),
                        processing_status=content_data.get('processing_status', 'completed'),
                        file_size=content_data.get('file_size'),
                        file_type=content_data.get('file_type'),
                        created_by=content_data.get('created_by')
                    )
                    
                    db.add(content_item)
                    db.commit()
                    db.refresh(content_item)
                    
                    return {
                        'success': True,
                        'data': content_item.to_dict(),
                        'id': content_item.id
                    }
                finally:
                    db.close()
                    
        except Exception as e:
            logger.error(f"Failed to create content item: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_content_items(self, organization_id: str, limit: int = 50, 
                               offset: int = 0, search: Optional[str] = None,
                               content_type: Optional[str] = None) -> Dict[str, Any]:
        """Get content items with optional filtering."""
        try:
            logger.info(f"🔍 ContentDatabaseService.get_content_items called with org_id: '{organization_id}', limit: {limit}, offset: {offset}")
            
            if self.use_supabase:
                # Use Supabase
                logger.info("Using Supabase client")
                items = await supabase_client.get_content_items(organization_id, limit, offset)
                total = len(items)  # Simple approximation
                
                return {
                    'success': True,
                    'content': items,
                    'total': total,
                    'limit': limit,
                    'offset': offset
                }
            else:
                # Use SQLAlchemy
                logger.info("Using SQLAlchemy database")
                db = get_db_session()
                try:
                    # Debug: Check total items in table first
                    total_items_in_db = db.query(ContentItem).count()
                    logger.info(f"🔢 Total ContentItems in database: {total_items_in_db}")
                    
                    # Debug: Check items with this organization_id
                    org_items_count = db.query(ContentItem).filter(ContentItem.organization_id == organization_id).count()
                    logger.info(f"🏢 ContentItems with organization_id='{organization_id}': {org_items_count}")
                    
                    # Debug: Show sample organization_ids in database
                    sample_orgs = db.query(ContentItem.organization_id).distinct().limit(5).all()
                    logger.info(f"📋 Sample organization_ids in database: {[org[0] for org in sample_orgs]}")
                    
                    query = db.query(ContentItem).filter(ContentItem.organization_id == organization_id)
                    logger.info(f"🔍 Base query created with filter: organization_id == '{organization_id}'")
                    
                    # Apply search filter
                    if search:
                        logger.info(f"🔎 Applying search filter: '{search}'")
                        query = query.filter(
                            ContentItem.title.contains(search) | 
                            ContentItem.content.contains(search)
                        )
                    
                    # Apply content type filter
                    if content_type:
                        logger.info(f"📁 Applying content_type filter: '{content_type}'")
                        query = query.filter(ContentItem.content_type == content_type)
                    
                    # Get total count
                    total = query.count()
                    logger.info(f"📊 Query count result: {total}")
                    
                    # Apply pagination and ordering
                    items = query.order_by(desc(ContentItem.created_at)).offset(offset).limit(limit).all()
                    logger.info(f"📦 Retrieved {len(items)} items after pagination")
                    
                    # Debug: Log details of retrieved items
                    for i, item in enumerate(items):
                        logger.info(f"  Item {i+1}: id={item.id}, title='{item.title[:50]}...', org_id='{item.organization_id}'")
                    
                    result = {
                        'success': True,
                        'content': [item.to_dict() for item in items],
                        'total': total,
                        'limit': limit,
                        'offset': offset
                    }
                    logger.info(f"✅ Returning result: success={result['success']}, content_length={len(result['content'])}, total={result['total']}")
                    return result
                    
                finally:
                    db.close()
                    
        except Exception as e:
            logger.error(f"❌ Failed to get content items: {e}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e),
                'content': [],
                'total': 0,
                'limit': limit,
                'offset': offset
            }
    
    async def get_content_item(self, content_id: str, organization_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific content item by ID."""
        try:
            if self.use_supabase:
                # Use Supabase
                items = await supabase_client.get_content_items(organization_id)
                for item in items:
                    if item.get('id') == content_id:
                        return item
                return None
            else:
                # Use SQLAlchemy
                db = get_db_session()
                try:
                    item = db.query(ContentItem).filter(
                        ContentItem.id == content_id,
                        ContentItem.organization_id == organization_id
                    ).first()
                    
                    return item.to_dict() if item else None
                finally:
                    db.close()
                    
        except Exception as e:
            logger.error(f"Failed to get content item {content_id}: {e}")
            return None
    
    async def update_content_item(self, content_id: str, organization_id: str, 
                                 updates: Dict[str, Any]) -> bool:
        """Update a content item."""
        try:
            if self.use_supabase:
                # Use Supabase (would need implementation in supabase_client)
                logger.warning("Content item updates not implemented for Supabase yet")
                return False
            else:
                # Use SQLAlchemy
                db = get_db_session()
                try:
                    item = db.query(ContentItem).filter(
                        ContentItem.id == content_id,
                        ContentItem.organization_id == organization_id
                    ).first()
                    
                    if not item:
                        return False
                    
                    # Apply updates
                    for key, value in updates.items():
                        if hasattr(item, key):
                            setattr(item, key, value)
                    
                    item.updated_at = datetime.utcnow()
                    db.commit()
                    return True
                finally:
                    db.close()
                    
        except Exception as e:
            logger.error(f"Failed to update content item {content_id}: {e}")
            return False
    
    async def delete_content_item(self, content_id: str, organization_id: str) -> bool:
        """Delete a content item."""
        try:
            if self.use_supabase:
                # Use Supabase (would need implementation in supabase_client)
                logger.warning("Content item deletion not implemented for Supabase yet")
                return False
            else:
                # Use SQLAlchemy
                db = get_db_session()
                try:
                    item = db.query(ContentItem).filter(
                        ContentItem.id == content_id,
                        ContentItem.organization_id == organization_id
                    ).first()
                    
                    if not item:
                        return False
                    
                    db.delete(item)
                    db.commit()
                    return True
                finally:
                    db.close()
                    
        except Exception as e:
            logger.error(f"Failed to delete content item {content_id}: {e}")
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the current database configuration."""
        return {
            'database_type': 'supabase' if self.use_supabase else 'sqlalchemy',
            'supabase_available': SUPABASE_AVAILABLE,
            'using_supabase': self.use_supabase
        }


# Global content database service instance
content_db_service = ContentDatabaseService()