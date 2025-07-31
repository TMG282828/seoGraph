"""
Content Deduplication and Versioning Service for SEO Content Knowledge Graph System.

This module provides comprehensive content deduplication and version management including:
- Content similarity detection and duplicate identification
- Version tracking and change management
- Content merging and consolidation strategies
- Semantic similarity analysis for near-duplicates
- Content history and audit trail maintenance
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
import hashlib
import json
from difflib import SequenceMatcher
from dataclasses import dataclass
from enum import Enum

from ..database.supabase_client import supabase_client
from ..database.qdrant_client import qdrant_client
from ..services.graph_vector_service import graph_vector_service

logger = logging.getLogger(__name__)


class DuplicateType(Enum):
    """Types of duplicate content."""
    EXACT_DUPLICATE = "exact_duplicate"
    NEAR_DUPLICATE = "near_duplicate"
    SEMANTIC_SIMILAR = "semantic_similar"
    CONTENT_UPDATE = "content_update"
    TITLE_VARIATION = "title_variation"


class ContentAction(Enum):
    """Actions to take on duplicate content."""
    KEEP_ORIGINAL = "keep_original"
    REPLACE_WITH_NEW = "replace_with_new"
    MERGE_CONTENT = "merge_content"
    CREATE_VERSION = "create_version"
    MARK_DUPLICATE = "mark_duplicate"


@dataclass
class ContentVersion:
    """Content version information."""
    version_id: str
    content_id: str
    version_number: int
    content_text: str
    title: str
    metadata: Dict[str, Any]
    content_hash: str
    created_at: datetime
    created_by: str
    change_summary: str
    is_active: bool = True


@dataclass
class DuplicateMatch:
    """Duplicate content match information."""
    content_id_1: str
    content_id_2: str
    duplicate_type: DuplicateType
    similarity_score: float
    duplicate_percentage: float
    semantic_similarity: float
    recommended_action: ContentAction
    confidence_score: float
    detected_at: datetime
    details: Dict[str, Any]


class ContentDeduplicationService:
    """
    Content deduplication and versioning service.
    
    Provides comprehensive duplicate detection, content versioning, and merge management.
    """
    
    def __init__(self, organization_id: str):
        self.organization_id = organization_id
        self.logger = logging.getLogger(__name__)
        
        # Similarity thresholds
        self.exact_duplicate_threshold = 0.98
        self.near_duplicate_threshold = 0.85
        self.semantic_similarity_threshold = 0.80
        self.title_similarity_threshold = 0.90
        
        # Minimum content length for comparison
        self.min_content_length = 100
        
        # Content change thresholds
        self.significant_change_threshold = 0.30
        self.minor_change_threshold = 0.10
    
    async def detect_duplicates(self, content_id: str, content_text: str, 
                              title: str, metadata: Dict[str, Any]) -> List[DuplicateMatch]:
        """Detect duplicate content for a given piece of content."""
        try:
            duplicates = []
            
            # Skip very short content
            if len(content_text.strip()) < self.min_content_length:
                return duplicates
            
            # Generate content hash
            content_hash = self._calculate_content_hash(content_text)
            
            # 1. Check for exact duplicates by hash
            exact_duplicates = await self._find_exact_duplicates(content_hash, content_id)
            duplicates.extend(exact_duplicates)
            
            # 2. Check for near duplicates by text similarity
            near_duplicates = await self._find_near_duplicates(content_text, title, content_id)
            duplicates.extend(near_duplicates)
            
            # 3. Check for semantic similarity using vector embeddings
            semantic_duplicates = await self._find_semantic_duplicates(content_text, content_id)
            duplicates.extend(semantic_duplicates)
            
            # 4. Check for title variations with similar content
            title_variations = await self._find_title_variations(title, content_text, content_id)
            duplicates.extend(title_variations)
            
            # Remove duplicates from results and sort by confidence
            unique_duplicates = self._remove_duplicate_matches(duplicates)
            unique_duplicates.sort(key=lambda x: x.confidence_score, reverse=True)
            
            self.logger.info(f"Found {len(unique_duplicates)} potential duplicates for content {content_id}")
            
            return unique_duplicates
            
        except Exception as e:
            self.logger.error(f"Failed to detect duplicates for content {content_id}: {e}")
            return []
    
    async def process_duplicate_content(self, duplicate_match: DuplicateMatch, 
                                      auto_resolve: bool = False) -> Dict[str, Any]:
        """Process and resolve duplicate content."""
        try:
            content_1_id = duplicate_match.content_id_1
            content_2_id = duplicate_match.content_id_2
            
            # Get full content information
            content_1 = await self._get_content_info(content_1_id)
            content_2 = await self._get_content_info(content_2_id)
            
            if not content_1 or not content_2:
                raise Exception(f"Content not found: {content_1_id} or {content_2_id}")
            
            # Determine action based on duplicate type and auto-resolve setting
            action = duplicate_match.recommended_action
            if not auto_resolve:
                # Store duplicate for manual review
                await self._store_duplicate_for_review(duplicate_match, content_1, content_2)
                return {
                    "action": "stored_for_review",
                    "duplicate_id": f"{content_1_id}_{content_2_id}",
                    "requires_manual_review": True
                }
            
            # Execute recommended action
            result = await self._execute_duplicate_action(action, content_1, content_2, duplicate_match)
            
            # Log the resolution
            await self._log_duplicate_resolution(duplicate_match, action, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process duplicate content: {e}")
            return {"action": "failed", "error": str(e)}
    
    async def create_content_version(self, content_id: str, new_content_text: str, 
                                   new_title: str, change_summary: str, 
                                   created_by: str) -> ContentVersion:
        """Create a new version of existing content."""
        try:
            # Get current content
            current_content = await self._get_content_info(content_id)
            if not current_content:
                raise Exception(f"Content {content_id} not found")
            
            # Get current version number
            current_version = await self._get_latest_version_number(content_id)
            new_version_number = current_version + 1
            
            # Generate version ID
            version_id = f"{content_id}_v{new_version_number}"
            
            # Create new version
            version = ContentVersion(
                version_id=version_id,
                content_id=content_id,
                version_number=new_version_number,
                content_text=new_content_text,
                title=new_title,
                metadata=current_content.get("metadata", {}),
                content_hash=self._calculate_content_hash(new_content_text),
                created_at=datetime.now(),
                created_by=created_by,
                change_summary=change_summary
            )
            
            # Store version
            await self._store_content_version(version)
            
            # Update main content with new version
            await self._update_main_content(content_id, new_content_text, new_title, version_id)
            
            # Update knowledge graph if significant change
            change_score = await self._calculate_change_significance(
                current_content.get("processed_text", ""), 
                new_content_text
            )
            
            if change_score > self.significant_change_threshold:
                await self._update_knowledge_graph(content_id, new_content_text, new_title)
            
            self.logger.info(f"Created version {new_version_number} for content {content_id}")
            
            return version
            
        except Exception as e:
            self.logger.error(f"Failed to create content version: {e}")
            raise
    
    async def merge_duplicate_content(self, content_id_1: str, content_id_2: str, 
                                    merge_strategy: str = "combine") -> Dict[str, Any]:
        """Merge two duplicate pieces of content."""
        try:
            # Get content information
            content_1 = await self._get_content_info(content_id_1)
            content_2 = await self._get_content_info(content_id_2)
            
            if not content_1 or not content_2:
                raise Exception("One or both content pieces not found")
            
            # Determine primary content (keep the one with better quality or more recent)
            primary_content, secondary_content = await self._determine_primary_content(content_1, content_2)
            
            # Apply merge strategy
            if merge_strategy == "combine":
                merged_content = await self._combine_content(primary_content, secondary_content)
            elif merge_strategy == "append":
                merged_content = await self._append_content(primary_content, secondary_content)
            elif merge_strategy == "best_sections":
                merged_content = await self._merge_best_sections(primary_content, secondary_content)
            else:
                raise Exception(f"Unknown merge strategy: {merge_strategy}")
            
            # Create new version with merged content
            version = await self.create_content_version(
                primary_content["content_id"],
                merged_content["text"],
                merged_content["title"],
                f"Merged with {secondary_content['content_id']} using {merge_strategy} strategy",
                "system_merge"
            )
            
            # Mark secondary content as merged
            await self._mark_content_as_merged(secondary_content["content_id"], primary_content["content_id"])
            
            # Update knowledge graph to remove secondary content references
            await self._remove_from_knowledge_graph(secondary_content["content_id"])
            
            return {
                "action": "merged",
                "primary_content_id": primary_content["content_id"],
                "secondary_content_id": secondary_content["content_id"],
                "merged_version": version.version_id,
                "merge_strategy": merge_strategy
            }
            
        except Exception as e:
            self.logger.error(f"Failed to merge duplicate content: {e}")
            return {"action": "failed", "error": str(e)}
    
    async def get_content_versions(self, content_id: str) -> List[ContentVersion]:
        """Get all versions of a piece of content."""
        try:
            result = supabase_client.client.table("content_versions").select("*").eq("content_id", content_id).eq("organization_id", self.organization_id).order("version_number", desc=True).execute()
            
            versions = []
            for version_data in result.data:
                version = ContentVersion(
                    version_id=version_data["version_id"],
                    content_id=version_data["content_id"],
                    version_number=version_data["version_number"],
                    content_text=version_data["content_text"],
                    title=version_data["title"],
                    metadata=json.loads(version_data["metadata"]),
                    content_hash=version_data["content_hash"],
                    created_at=datetime.fromisoformat(version_data["created_at"]),
                    created_by=version_data["created_by"],
                    change_summary=version_data["change_summary"],
                    is_active=version_data["is_active"]
                )
                versions.append(version)
            
            return versions
            
        except Exception as e:
            self.logger.error(f"Failed to get content versions: {e}")
            return []
    
    async def get_duplicate_review_queue(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get duplicates awaiting manual review."""
        try:
            result = supabase_client.client.table("duplicate_reviews").select("*").eq("organization_id", self.organization_id).eq("status", "pending").order("detected_at", desc=True).limit(limit).execute()
            
            return result.data
            
        except Exception as e:
            self.logger.error(f"Failed to get duplicate review queue: {e}")
            return []
    
    async def resolve_duplicate_review(self, review_id: str, action: str, 
                                     resolved_by: str, notes: str = "") -> Dict[str, Any]:
        """Resolve a duplicate content review."""
        try:
            # Get review information
            result = supabase_client.client.table("duplicate_reviews").select("*").eq("review_id", review_id).eq("organization_id", self.organization_id).execute()
            
            if not result.data:
                raise Exception(f"Duplicate review {review_id} not found")
            
            review_data = result.data[0]
            
            # Execute the chosen action
            if action == "merge":
                result = await self.merge_duplicate_content(
                    review_data["content_id_1"], 
                    review_data["content_id_2"]
                )
            elif action == "keep_both":
                result = {"action": "keep_both"}
            elif action == "remove_duplicate":
                # Mark one as duplicate
                await self._mark_content_as_duplicate(review_data["content_id_2"], review_data["content_id_1"])
                result = {"action": "removed_duplicate", "removed_id": review_data["content_id_2"]}
            else:
                raise Exception(f"Unknown action: {action}")
            
            # Update review status
            supabase_client.client.table("duplicate_reviews").update({
                "status": "resolved",
                "resolution_action": action,
                "resolved_by": resolved_by,
                "resolved_at": datetime.now().isoformat(),
                "resolution_notes": notes,
                "resolution_result": json.dumps(result)
            }).eq("review_id", review_id).execute()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to resolve duplicate review: {e}")
            return {"action": "failed", "error": str(e)}
    
    # Helper methods for duplicate detection
    
    async def _find_exact_duplicates(self, content_hash: str, exclude_content_id: str) -> List[DuplicateMatch]:
        """Find exact duplicates by content hash."""
        try:
            result = supabase_client.client.table("raw_content").select("content_id").eq("content_hash", content_hash).eq("organization_id", self.organization_id).neq("content_id", exclude_content_id).execute()
            
            duplicates = []
            for row in result.data:
                duplicate = DuplicateMatch(
                    content_id_1=exclude_content_id,
                    content_id_2=row["content_id"],
                    duplicate_type=DuplicateType.EXACT_DUPLICATE,
                    similarity_score=1.0,
                    duplicate_percentage=100.0,
                    semantic_similarity=1.0,
                    recommended_action=ContentAction.KEEP_ORIGINAL,
                    confidence_score=1.0,
                    detected_at=datetime.now(),
                    details={"detection_method": "content_hash"}
                )
                duplicates.append(duplicate)
            
            return duplicates
            
        except Exception as e:
            self.logger.error(f"Failed to find exact duplicates: {e}")
            return []
    
    async def _find_near_duplicates(self, content_text: str, title: str, exclude_content_id: str) -> List[DuplicateMatch]:
        """Find near duplicates using text similarity."""
        try:
            # Get all content for comparison (this could be optimized with indexing)
            result = supabase_client.client.table("raw_content").select("content_id, raw_text, title").eq("organization_id", self.organization_id).neq("content_id", exclude_content_id).execute()
            
            duplicates = []
            content_words = set(content_text.lower().split())
            
            for row in result.data:
                if not row["raw_text"] or len(row["raw_text"]) < self.min_content_length:
                    continue
                
                # Calculate text similarity
                similarity = SequenceMatcher(None, content_text.lower(), row["raw_text"].lower()).ratio()
                
                if similarity >= self.near_duplicate_threshold:
                    # Calculate additional metrics
                    other_words = set(row["raw_text"].lower().split())
                    word_overlap = len(content_words & other_words) / len(content_words | other_words) if content_words | other_words else 0
                    
                    title_similarity = SequenceMatcher(None, title.lower(), row["title"].lower()).ratio() if row["title"] else 0
                    
                    # Determine duplicate type and action
                    duplicate_type = DuplicateType.NEAR_DUPLICATE
                    recommended_action = ContentAction.MARK_DUPLICATE if similarity > 0.95 else ContentAction.CREATE_VERSION
                    
                    duplicate = DuplicateMatch(
                        content_id_1=exclude_content_id,
                        content_id_2=row["content_id"],
                        duplicate_type=duplicate_type,
                        similarity_score=similarity,
                        duplicate_percentage=similarity * 100,
                        semantic_similarity=word_overlap,
                        recommended_action=recommended_action,
                        confidence_score=similarity * 0.9,  # Slightly lower confidence than exact matches
                        detected_at=datetime.now(),
                        details={
                            "detection_method": "text_similarity",
                            "title_similarity": title_similarity,
                            "word_overlap": word_overlap
                        }
                    )
                    duplicates.append(duplicate)
            
            return duplicates
            
        except Exception as e:
            self.logger.error(f"Failed to find near duplicates: {e}")
            return []
    
    async def _find_semantic_duplicates(self, content_text: str, exclude_content_id: str) -> List[DuplicateMatch]:
        """Find semantically similar content using vector embeddings."""
        try:
            # Search for similar content using Qdrant
            search_results = qdrant_client.search_similar_content(
                organization_id=self.organization_id,
                query_text=content_text,
                limit=20,
                score_threshold=self.semantic_similarity_threshold
            )
            
            duplicates = []
            for result in search_results:
                if result["content_id"] == exclude_content_id:
                    continue
                
                similarity_score = result["score"]
                
                if similarity_score >= self.semantic_similarity_threshold:
                    duplicate = DuplicateMatch(
                        content_id_1=exclude_content_id,
                        content_id_2=result["content_id"],
                        duplicate_type=DuplicateType.SEMANTIC_SIMILAR,
                        similarity_score=similarity_score,
                        duplicate_percentage=similarity_score * 100,
                        semantic_similarity=similarity_score,
                        recommended_action=ContentAction.CREATE_VERSION if similarity_score > 0.9 else ContentAction.KEEP_ORIGINAL,
                        confidence_score=similarity_score * 0.8,
                        detected_at=datetime.now(),
                        details={
                            "detection_method": "semantic_similarity",
                            "vector_score": similarity_score
                        }
                    )
                    duplicates.append(duplicate)
            
            return duplicates
            
        except Exception as e:
            self.logger.error(f"Failed to find semantic duplicates: {e}")
            return []
    
    async def _find_title_variations(self, title: str, content_text: str, exclude_content_id: str) -> List[DuplicateMatch]:
        """Find content with very similar titles but different content."""
        try:
            result = supabase_client.client.table("raw_content").select("content_id, title, raw_text").eq("organization_id", self.organization_id).neq("content_id", exclude_content_id).execute()
            
            duplicates = []
            
            for row in result.data:
                if not row["title"] or not row["raw_text"]:
                    continue
                
                # Calculate title similarity
                title_similarity = SequenceMatcher(None, title.lower(), row["title"].lower()).ratio()
                
                if title_similarity >= self.title_similarity_threshold:
                    # Check content similarity
                    content_similarity = SequenceMatcher(None, content_text.lower(), row["raw_text"].lower()).ratio()
                    
                    # If titles are very similar but content is different, it might be a variation
                    if content_similarity < 0.5:  # Content is significantly different
                        duplicate = DuplicateMatch(
                            content_id_1=exclude_content_id,
                            content_id_2=row["content_id"],
                            duplicate_type=DuplicateType.TITLE_VARIATION,
                            similarity_score=title_similarity,
                            duplicate_percentage=title_similarity * 100,
                            semantic_similarity=content_similarity,
                            recommended_action=ContentAction.KEEP_ORIGINAL,  # Usually keep both for title variations
                            confidence_score=title_similarity * 0.6,  # Lower confidence as content differs
                            detected_at=datetime.now(),
                            details={
                                "detection_method": "title_similarity",
                                "title_similarity": title_similarity,
                                "content_similarity": content_similarity
                            }
                        )
                        duplicates.append(duplicate)
            
            return duplicates
            
        except Exception as e:
            self.logger.error(f"Failed to find title variations: {e}")
            return []
    
    # Helper methods for content management
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calculate SHA256 hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _remove_duplicate_matches(self, duplicates: List[DuplicateMatch]) -> List[DuplicateMatch]:
        """Remove duplicate matches from results."""
        seen_pairs = set()
        unique_duplicates = []
        
        for duplicate in duplicates:
            # Create a pair identifier (order-independent)
            pair = tuple(sorted([duplicate.content_id_1, duplicate.content_id_2]))
            
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                unique_duplicates.append(duplicate)
        
        return unique_duplicates
    
    async def _get_content_info(self, content_id: str) -> Optional[Dict[str, Any]]:
        """Get content information from database."""
        try:
            result = supabase_client.client.table("processed_content").select("*").eq("content_id", content_id).eq("organization_id", self.organization_id).execute()
            
            if result.data:
                return result.data[0]
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get content info: {e}")
            return None
    
    async def _get_latest_version_number(self, content_id: str) -> int:
        """Get the latest version number for content."""
        try:
            result = supabase_client.client.table("content_versions").select("version_number").eq("content_id", content_id).eq("organization_id", self.organization_id).order("version_number", desc=True).limit(1).execute()
            
            if result.data:
                return result.data[0]["version_number"]
            return 0
            
        except Exception as e:
            self.logger.error(f"Failed to get latest version number: {e}")
            return 0
    
    async def _store_content_version(self, version: ContentVersion):
        """Store content version in database."""
        try:
            data = {
                "version_id": version.version_id,
                "content_id": version.content_id,
                "organization_id": self.organization_id,
                "version_number": version.version_number,
                "content_text": version.content_text,
                "title": version.title,
                "metadata": json.dumps(version.metadata),
                "content_hash": version.content_hash,
                "created_at": version.created_at.isoformat(),
                "created_by": version.created_by,
                "change_summary": version.change_summary,
                "is_active": version.is_active
            }
            
            supabase_client.client.table("content_versions").insert(data).execute()
            
        except Exception as e:
            self.logger.error(f"Failed to store content version: {e}")
            raise
    
    async def _calculate_change_significance(self, old_content: str, new_content: str) -> float:
        """Calculate how significant the change is between two content versions."""
        try:
            if not old_content or not new_content:
                return 1.0
            
            similarity = SequenceMatcher(None, old_content, new_content).ratio()
            change_score = 1.0 - similarity
            
            return change_score
            
        except Exception as e:
            self.logger.error(f"Failed to calculate change significance: {e}")
            return 0.0
    
    # Additional helper methods would be implemented here for:
    # - _store_duplicate_for_review
    # - _execute_duplicate_action
    # - _log_duplicate_resolution
    # - _update_main_content
    # - _update_knowledge_graph
    # - _determine_primary_content
    # - _combine_content
    # - _append_content
    # - _merge_best_sections
    # - _mark_content_as_merged
    # - _remove_from_knowledge_graph
    # - _mark_content_as_duplicate