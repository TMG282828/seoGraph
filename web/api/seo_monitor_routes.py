"""
SEO Monitor API routes for keyword tracking and performance monitoring.
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

from src.database.database import get_db
from src.database.models import TrackedKeyword, KeywordHistory
from src.services.google_ads_service import google_ads_service
from src.services.keyword_sync_service import keyword_sync_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/seo-monitor", tags=["seo-monitor"])

@router.post("/add-keyword")
async def add_keyword_to_tracker(request: dict, db: Session = Depends(get_db)):
    """Add a keyword to the tracking system."""
    try:
        keyword_term = request.get("keyword", "").strip()
        domain = request.get("domain", "").strip()
        target_url = request.get("target_url", "").strip()
        notes = request.get("notes", "").strip()
        
        if not keyword_term:
            raise HTTPException(status_code=400, detail="Keyword is required")
        
        # Check if keyword already exists for this domain
        existing = db.query(TrackedKeyword).filter(
            TrackedKeyword.keyword == keyword_term.lower(),
            TrackedKeyword.domain == domain
        ).first()
        
        if existing:
            if not existing.is_active:
                # Reactivate if it was previously deactivated
                existing.is_active = True
                existing.updated_at = datetime.now()
                db.commit()
                db.refresh(existing)
                
                return {
                    "success": True,
                    "message": "Keyword reactivated in tracker",
                    "keyword": existing.to_dict()
                }
            else:
                return {
                    "success": False,
                    "message": "Keyword already being tracked for this domain",
                    "keyword": existing.to_dict()
                }
        
        # Get initial keyword data from Google Ads service
        keyword_data = await google_ads_service.get_keyword_ideas([keyword_term])
        initial_data = keyword_data.get("keywords", [{}])[0] if keyword_data.get("keywords") else {}
        
        # Create new tracked keyword
        tracked_keyword = TrackedKeyword(
            keyword=keyword_term.lower(),
            user_id="default_user",  # TODO: Use actual user ID when auth is implemented
            domain=domain or None,
            target_url=target_url or None,
            search_volume=initial_data.get("volume"),
            difficulty=initial_data.get("difficulty"),
            cpc=initial_data.get("cpc"),
            data_source=initial_data.get("data_source", "algorithmic"),
            notes=notes or None,
            is_active=True,
            last_checked=datetime.now()
        )
        
        db.add(tracked_keyword)
        db.commit()
        db.refresh(tracked_keyword)
        
        # Add initial history record
        history_record = KeywordHistory(
            tracked_keyword_id=tracked_keyword.id,
            search_volume=initial_data.get("volume"),
            difficulty=initial_data.get("difficulty"),
            cpc=initial_data.get("cpc"),
            data_source=initial_data.get("data_source", "algorithmic"),
            confidence_score=initial_data.get("confidence_score", 0.75)
        )
        
        db.add(history_record)
        db.commit()
        
        logger.info(f"Added keyword '{keyword_term}' to tracker for domain '{domain}'")
        
        # Automatically sync to SerpBear for ranking tracking
        serpbear_sync_success = False
        try:
            serpbear_sync_success = await keyword_sync_service.sync_keyword_to_serpbear(tracked_keyword, db)
            if serpbear_sync_success:
                logger.info(f"✅ Synced '{keyword_term}' to SerpBear for ranking tracking")
            else:
                logger.warning(f"⚠️ Failed to sync '{keyword_term}' to SerpBear")
        except Exception as sync_error:
            logger.error(f"SerpBear sync error for '{keyword_term}': {sync_error}")
        
        return {
            "success": True,
            "message": "Keyword added to tracker successfully",
            "keyword": tracked_keyword.to_dict(),
            "data_source": keyword_data.get("data_source", "Enhanced Algorithmic"),
            "api_status": keyword_data.get("api_status", "fallback_mode"),
            "serpbear_synced": serpbear_sync_success
        }
        
    except Exception as e:
        logger.error(f"Failed to add keyword to tracker: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add keyword: {str(e)}")

@router.get("/tracked-keywords")
async def get_tracked_keywords(
    domain: Optional[str] = None,
    active_only: bool = True,
    db: Session = Depends(get_db)
):
    """Get all tracked keywords."""
    try:
        query = db.query(TrackedKeyword)
        
        if domain:
            query = query.filter(TrackedKeyword.domain == domain)
        
        if active_only:
            query = query.filter(TrackedKeyword.is_active == True)
        
        keywords = query.order_by(TrackedKeyword.created_at.desc()).all()
        
        return {
            "success": True,
            "keywords": [kw.to_dict() for kw in keywords],
            "total_count": len(keywords),
            "domain_filter": domain,
            "active_only": active_only
        }
        
    except Exception as e:
        logger.error(f"Failed to get tracked keywords: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get keywords: {str(e)}")

@router.delete("/tracked-keywords/{keyword_id}")
async def remove_keyword_from_tracker(keyword_id: int, db: Session = Depends(get_db)):
    """Remove a keyword from tracking (soft delete)."""
    try:
        keyword = db.query(TrackedKeyword).filter(TrackedKeyword.id == keyword_id).first()
        
        if not keyword:
            raise HTTPException(status_code=404, detail="Keyword not found")
        
        # Soft delete by setting is_active to False
        keyword.is_active = False
        keyword.updated_at = datetime.now()
        
        db.commit()
        
        return {
            "success": True,
            "message": "Keyword removed from tracking",
            "keyword": keyword.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Failed to remove keyword from tracker: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove keyword: {str(e)}")

@router.get("/tracked-keywords/{keyword_id}/history")
async def get_keyword_history(keyword_id: int, db: Session = Depends(get_db)):
    """Get historical data for a tracked keyword."""
    try:
        keyword = db.query(TrackedKeyword).filter(TrackedKeyword.id == keyword_id).first()
        
        if not keyword:
            raise HTTPException(status_code=404, detail="Keyword not found")
        
        history = db.query(KeywordHistory).filter(
            KeywordHistory.tracked_keyword_id == keyword_id
        ).order_by(KeywordHistory.recorded_at.desc()).limit(100).all()
        
        return {
            "success": True,
            "keyword": keyword.to_dict(),
            "history": [h.to_dict() for h in history],
            "total_records": len(history)
        }
        
    except Exception as e:
        logger.error(f"Failed to get keyword history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

@router.post("/refresh-keyword-data")
async def refresh_keyword_data(request: dict, db: Session = Depends(get_db)):
    """Refresh data for tracked keywords."""
    try:
        keyword_ids = request.get("keyword_ids", [])
        
        if not keyword_ids:
            # Refresh all active keywords
            keywords = db.query(TrackedKeyword).filter(TrackedKeyword.is_active == True).all()
        else:
            keywords = db.query(TrackedKeyword).filter(
                TrackedKeyword.id.in_(keyword_ids),
                TrackedKeyword.is_active == True
            ).all()
        
        if not keywords:
            return {
                "success": True,
                "message": "No keywords to refresh",
                "refreshed_count": 0
            }
        
        # Get updated data from Google Ads service
        keyword_terms = [kw.keyword for kw in keywords]
        updated_data = await google_ads_service.get_keyword_ideas(keyword_terms)
        keyword_data_map = {
            data["term"]: data for data in updated_data.get("keywords", [])
        }
        
        refreshed_count = 0
        
        for keyword in keywords:
            data = keyword_data_map.get(keyword.keyword, {})
            
            if data:
                # Update keyword record
                keyword.previous_position = keyword.current_position
                keyword.search_volume = data.get("volume", keyword.search_volume)
                keyword.difficulty = data.get("difficulty", keyword.difficulty)
                keyword.cpc = data.get("cpc", keyword.cpc)
                keyword.data_source = data.get("data_source", keyword.data_source)
                keyword.last_checked = datetime.now()
                keyword.updated_at = datetime.now()
                
                # Add history record
                history_record = KeywordHistory(
                    tracked_keyword_id=keyword.id,
                    position=keyword.current_position,
                    search_volume=data.get("volume"),
                    difficulty=data.get("difficulty"),
                    cpc=data.get("cpc"),
                    data_source=data.get("data_source", "algorithmic"),
                    confidence_score=data.get("confidence_score", 0.75)
                )
                
                db.add(history_record)
                refreshed_count += 1
        
        db.commit()
        
        return {
            "success": True,
            "message": f"Refreshed data for {refreshed_count} keywords",
            "refreshed_count": refreshed_count,
            "data_source": updated_data.get("data_source", "Enhanced Algorithmic"),
            "api_status": updated_data.get("api_status", "fallback_mode")
        }
        
    except Exception as e:
        logger.error(f"Failed to refresh keyword data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh data: {str(e)}")

@router.get("/dashboard-metrics")
async def get_dashboard_metrics(domain: Optional[str] = None, db: Session = Depends(get_db)):
    """Get SEO Monitor dashboard metrics."""
    try:
        query = db.query(TrackedKeyword).filter(TrackedKeyword.is_active == True)
        
        if domain:
            query = query.filter(TrackedKeyword.domain == domain)
        
        keywords = query.all()
        
        total_keywords = len(keywords)
        avg_position = 0
        avg_difficulty = 0
        total_volume = 0
        top_10_count = 0
        
        if keywords:
            positions = [kw.current_position for kw in keywords if kw.current_position]
            difficulties = [kw.difficulty for kw in keywords if kw.difficulty]
            volumes = [kw.search_volume for kw in keywords if kw.search_volume]
            
            if positions:
                avg_position = sum(positions) / len(positions)
                top_10_count = len([p for p in positions if p <= 10])
            
            if difficulties:
                avg_difficulty = sum(difficulties) / len(difficulties)
            
            if volumes:
                total_volume = sum(volumes)
        
        return {
            "success": True,
            "metrics": {
                "total_keywords": total_keywords,
                "avg_position": round(avg_position, 1) if avg_position else None,
                "avg_difficulty": round(avg_difficulty, 1) if avg_difficulty else None,
                "total_search_volume": total_volume,
                "top_10_keywords": top_10_count,
                "domain": domain
            },
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get dashboard metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.post("/sync-to-serpbear")
async def sync_keywords_to_serpbear(db: Session = Depends(get_db)):
    """Bulk sync all tracked keywords to SerpBear for ranking tracking."""
    try:
        logger.info("🔄 Starting bulk sync of tracked keywords to SerpBear")
        
        sync_result = await keyword_sync_service.bulk_sync_to_serpbear(db)
        
        if sync_result.success:
            message = f"Sync completed: {sync_result.added_to_serpbear} added, {sync_result.already_synced} already synced"
            if sync_result.failed_syncs > 0:
                message += f", {sync_result.failed_syncs} failed"
            
            return {
                "success": True,
                "message": message,
                "stats": {
                    "added_to_serpbear": sync_result.added_to_serpbear,
                    "already_synced": sync_result.already_synced,
                    "failed_syncs": sync_result.failed_syncs,
                    "total_processed": sync_result.added_to_serpbear + sync_result.already_synced + sync_result.failed_syncs
                },
                "errors": sync_result.errors
            }
        else:
            return {
                "success": False,
                "message": "Sync failed",
                "errors": sync_result.errors
            }
        
    except Exception as e:
        logger.error(f"Failed to sync keywords to SerpBear: {e}")
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")


@router.get("/serpbear-sync-status")
async def get_serpbear_sync_status(db: Session = Depends(get_db)):
    """Get sync status between tracked keywords and SerpBear."""
    try:
        status = await keyword_sync_service.get_sync_status(db)
        
        return {
            "success": True,
            "status": status
        }
        
    except Exception as e:
        logger.error(f"Failed to get SerpBear sync status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.delete("/tracked-keywords/{keyword_id}/from-serpbear")
async def remove_keyword_from_serpbear(keyword_id: int, db: Session = Depends(get_db)):
    """Remove a tracked keyword from SerpBear tracking."""
    try:
        # Get the tracked keyword
        keyword = db.query(TrackedKeyword).filter(TrackedKeyword.id == keyword_id).first()
        
        if not keyword:
            raise HTTPException(status_code=404, detail="Keyword not found")
        
        # Extract domain
        domain = keyword.domain
        if not domain and keyword.target_url:
            from urllib.parse import urlparse
            parsed = urlparse(keyword.target_url)
            domain = parsed.netloc.replace('www.', '')
        
        if not domain:
            raise HTTPException(status_code=400, detail="No domain found for keyword")
        
        # Remove from SerpBear
        success = await keyword_sync_service.remove_from_serpbear(keyword.keyword, domain)
        
        if success:
            # Update keyword notes to remove SerpBear ID
            if keyword.notes and "SerpBear ID:" in keyword.notes:
                import re
                keyword.notes = re.sub(r'\s*\[SerpBear ID: \d+\]', '', keyword.notes)
                keyword.updated_at = datetime.now()
                db.commit()
            
            return {
                "success": True,
                "message": f"Removed '{keyword.keyword}' from SerpBear tracking"
            }
        else:
            return {
                "success": False,
                "message": f"Failed to remove '{keyword.keyword}' from SerpBear"
            }
        
    except Exception as e:
        logger.error(f"Failed to remove keyword from SerpBear: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove keyword: {str(e)}")