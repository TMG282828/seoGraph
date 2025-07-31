"""
Content storage module for Knowledge Base.
Handles database persistence and retrieval.
"""

from typing import Dict, Any
import logging
import time

# Use database-specific logger
logger = logging.getLogger('database')


async def store_content(
    filename: str,
    content_text: str, 
    content_type: str,
    analysis_result: Dict[str, Any],
    current_user: Dict[str, Any]
) -> bool:
    """
    Store content and analysis results using the comprehensive RAG/Graph pipeline.
    
    This now processes content through:
    1. Basic database storage (SQLAlchemy/Supabase)
    2. Vector embeddings generation and storage (Qdrant)
    3. Graph relationships creation (Neo4j)
    4. AI analysis enhancement
    
    Args:
        filename: Name of the uploaded file
        content_text: File content text
        content_type: MIME type
        analysis_result: Analysis results from AI agent
        current_user: Current user context
        
    Returns:
        True if successful, False otherwise
    """
    try:
        start_time = time.time()
        organization_id = current_user.get("org_id", "demo-org")
        
        file_name = filename  # Avoid logging conflicts with LogRecord.filename
        logger.info(f"🚀 Starting comprehensive RAG/Graph storage for file: {file_name}")
        
        # Step 1: Store in basic database first (existing functionality)
        from src.database.content_service import ContentDatabaseService
        db_service = ContentDatabaseService()
        
        # Prepare content data for storage
        content_data = {
            "title": filename,
            "content": content_text,
            "content_type": "document",
            "file_type": content_type,
            "word_count": analysis_result.get("word_count", 0),
            "analysis_data": analysis_result.get("analysis", {}),
            "extracted_topics": analysis_result.get("extracted_topics", []),
            "recommendations": analysis_result.get("recommendations", []),
            "seo_score": analysis_result.get("analysis", {}).get("seo_metrics", {}).get("overall_seo_score", 0),
            "readability_score": analysis_result.get("analysis", {}).get("seo_metrics", {}).get("readability_score", 0),
            "summary": analysis_result.get("summary", ""),
            "topics": analysis_result.get("extracted_topics", []),
            "keywords": analysis_result.get("keywords", [])
        }
        
        # Store in basic database
        stored_content = await db_service.create_content_item(
            content_data=content_data,
            organization_id=organization_id
        )
        
        if not stored_content:
            logger.error(f"❌ Failed to store content in basic database: {file_name}")
            return False
        
        content_id = stored_content.get('id')
        logger.info(f"✅ Basic storage complete for {file_name} (ID: {content_id})")
        
        # Step 2: Process through comprehensive RAG/Graph pipeline
        try:
            logger.info(f"🔄 Starting RAG/Graph processing for content {content_id}")
            
            # Import and initialize GraphVectorService
            from src.services.graph_vector_service import graph_vector_service
            
            # Set organization context
            graph_vector_service.set_organization_context(organization_id)
            
            # Prepare enhanced content data for comprehensive processing
            enhanced_content_data = {
                **content_data,
                'id': content_id,
                'organization_id': organization_id,
                'source': 'knowledge_base_upload',
                'publish_date': stored_content.get('created_at', ''),
                'url': f"/knowledge-base/content/{content_id}"
            }
            
            # Process through comprehensive pipeline
            processing_result = await graph_vector_service.process_content_comprehensive(enhanced_content_data)
            
            if processing_result.get('success'):
                logger.info(f"🎯 RAG/Graph processing successful for {file_name}")
                logger.info(f"   - Neo4j node created: {processing_result.get('neo4j_id', 'N/A')}")
                logger.info(f"   - Vector embedding stored: {processing_result.get('embedding_stored', False)}")
                logger.info(f"   - Topics processed: {processing_result.get('topics_processed', 0)}")
                logger.info(f"   - Keywords processed: {processing_result.get('keywords_processed', 0)}")
            else:
                logger.error(f"❌ RAG/Graph processing failed for {file_name}: {processing_result.get('error', 'Unknown error')}")
                # Don't fail the entire upload if RAG/Graph processing fails
                logger.warning(f"⚠️ Content {file_name} stored in basic database but RAG/Graph processing incomplete")
                
        except Exception as rag_error:
            logger.error(f"❌ RAG/Graph processing exception for {file_name}: {rag_error}")
            # Don't fail the entire upload if RAG/Graph processing fails
            logger.warning(f"⚠️ Content {file_name} stored in basic database but RAG/Graph processing failed")
        
        duration = time.time() - start_time
        
        # Log successful storage
        from config.logging_config import log_database_operation
        log_database_operation("INSERT", "content_items", duration, True, 
                             file_name=file_name, content_id=content_id)
        
        logger.info(f"🏁 Complete storage pipeline finished for {file_name} in {duration:.3f}s")
        return True
            
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ Complete storage pipeline failed for {file_name}: {e} after {duration:.3f}s")
        from config.logging_config import log_database_operation
        log_database_operation("INSERT", "content_items", duration, False, 
                             file_name=file_name, error=str(e))
        return False