"""
URL import module for Knowledge Base.
Handles web scraping and content extraction from URLs.
"""

from fastapi import Request
from typing import Dict, Any
import logging
import asyncio

from .content_auth import get_current_user_safe

logger = logging.getLogger(__name__)


async def import_url_content(request: Request, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Import content from URL with web scraping.
    
    Args:
        request: FastAPI request object
        data: Request data containing URL
        
    Returns:
        Imported content result
    """
    try:
        url = data.get("url", "")
        if not url:
            return {"success": False, "error": "URL is required"}
        
        # Validate URL format
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Get current user
        current_user = await get_current_user_safe(request)
        
        # Extract content using web scraping
        extracted_data = await _extract_web_content(url)
        
        if not extracted_data.get("success"):
            return extracted_data
        
        # Analyze extracted content with timeout
        analysis_result = await _analyze_url_content(
            extracted_data, 
            current_user, 
            timeout_seconds=6.0
        )
        
        # Store in database
        from .content_storage import store_content
        storage_success = await store_content(
            filename=f"URL_Import_{extracted_data.get('title', 'Untitled')}",
            content_text=extracted_data.get('content', ''),
            content_type="webpage",
            analysis_result=analysis_result,
            current_user=current_user
        )
        
        return {
            "success": True,
            "url": url,
            "title": extracted_data.get("title", "Untitled"),
            "content_preview": extracted_data.get("content", "")[:200] + "...",
            "word_count": len(extracted_data.get("content", "").split()),
            "analysis": analysis_result.get("analysis", {}),
            "stored_in_database": storage_success,
            "extraction_method": extracted_data.get("method", "unknown"),
            "note": "Content successfully imported from URL"
        }
        
    except Exception as e:
        logger.error(f"URL import failed: {e}")
        return {"success": False, "error": str(e)}


async def _extract_web_content(url: str) -> Dict[str, Any]:
    """Extract content from web URL using Crawl4AI."""
    try:
        from crawl4ai import AsyncWebCrawler
        from bs4 import BeautifulSoup
        
        # Initialize Crawl4AI crawler with fast configuration
        async with AsyncWebCrawler(
            verbose=False,
            headless=True,
            browser_type="chromium",
            page_timeout=8000,  # 8 second timeout
            delay_before_return_html=0.5
        ) as crawler:
            
            logger.info(f"Starting web crawl for URL: {url}")
            result = await crawler.arun(url=url)
            
            # If crawling succeeds, parse with BeautifulSoup
            if result.success and result.cleaned_html:
                logger.info("Web crawling successful, parsing content")
                
                soup = BeautifulSoup(result.cleaned_html, 'html.parser')
                
                # Extract title
                title = soup.find('title')
                title_text = title.get_text().strip() if title else "Untitled"
                
                # Extract meta description
                meta_desc = soup.find('meta', {'name': 'description'})
                meta_description = meta_desc.get('content', '').strip() if meta_desc else ''
                
                # Remove unwanted elements
                for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                    tag.decompose()
                
                # Extract main content
                main_content = soup.get_text(separator=' ', strip=True)
                
                return {
                    "success": True,
                    "title": title_text,
                    "meta_description": meta_description,
                    "content": main_content[:5000],  # Limit content length
                    "method": "crawl4ai_beautifulsoup"
                }
            else:
                logger.warning(f"Crawl4AI failed for {url}, trying basic fallback")
                return await _basic_url_import(url)
                
    except Exception as e:
        logger.error(f"Crawl4AI extraction failed: {e}")
        return await _basic_url_import(url)


async def _basic_url_import(url: str) -> Dict[str, Any]:
    """Basic URL import fallback using requests."""
    try:
        import requests
        from bs4 import BeautifulSoup
        
        # Simple HTTP request with timeout
        response = requests.get(url, timeout=5, headers={
            'User-Agent': 'Mozilla/5.0 (compatible; ContentBot/1.0)'
        })
        response.raise_for_status()
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title = soup.find('title')
        title_text = title.get_text().strip() if title else "Untitled"
        
        # Extract meta description
        meta_desc = soup.find('meta', {'name': 'description'})
        meta_description = meta_desc.get('content', '').strip() if meta_desc else ''
        
        # Remove unwanted elements
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()
        
        # Extract text content
        main_content = soup.get_text(separator=' ', strip=True)
        
        return {
            "success": True,
            "title": title_text,
            "meta_description": meta_description,
            "content": main_content[:3000],  # Smaller limit for basic mode
            "method": "requests_beautifulsoup"
        }
        
    except Exception as e:
        logger.error(f"Basic URL import failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "method": "failed"
        }


async def _analyze_url_content(
    extracted_data: Dict[str, Any], 
    current_user: Dict[str, Any],
    timeout_seconds: float = 6.0
) -> Dict[str, Any]:
    """Analyze extracted URL content with AI agents (with timeout)."""
    try:
        async def run_ai_analysis():
            from src.agents.content_analysis_agent import content_analysis_agent
            from src.agents.base_agent import AgentContext
            
            analysis_request = {
                "content_text": extracted_data.get("content", ""),
                "title": extracted_data.get("title", ""),
                "analysis_type": "seo",
                "include_recommendations": True,
                "target_keywords": []
            }
            
            context = AgentContext(
                organization_id=current_user.get('org_id', '00000000-0000-0000-0000-000000000001'),
                user_id=current_user.get('id', 'anonymous'),
                session_id="url-analysis-session",
                task_type="content_analysis"
            )
            
            return await content_analysis_agent.execute(analysis_request, context)
        
        # Run with timeout
        result = await asyncio.wait_for(run_ai_analysis(), timeout=timeout_seconds)
        
        if result.success and result.result_data:
            analysis_data = result.result_data
            return {
                "success": True,
                "analysis": {
                    "ai_seo_score": analysis_data.get("seo_metrics", {}).get("overall_seo_score", 0),
                    "ai_readability": analysis_data.get("seo_metrics", {}).get("readability_score", 0),
                    "ai_recommendations": analysis_data.get("recommendations", []),
                    "confidence_score": analysis_data.get("confidence_score", 0.8)
                }
            }
        else:
            logger.info("AI analysis returned no data, using basic analysis")
            return await _basic_content_analysis(extracted_data)
            
    except asyncio.TimeoutError:
        logger.info("AI analysis timed out for URL import - using basic analysis")
        return await _basic_content_analysis(extracted_data)
    except Exception as ai_error:
        logger.warning(f"AI analysis failed for URL import: {ai_error}")
        return await _basic_content_analysis(extracted_data)


async def _basic_content_analysis(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """Basic content analysis without AI."""
    content = extracted_data.get("content", "")
    title = extracted_data.get("title", "")
    
    # Basic metrics
    words = content.split()
    word_count = len(words)
    char_count = len(content)
    
    # Simple readability score
    sentences = content.split('. ')
    sentence_count = len(sentences)
    avg_sentence_length = word_count / max(sentence_count, 1)
    readability_score = max(0, min(100, 206.835 - (1.015 * avg_sentence_length)))
    
    # Basic SEO score
    seo_score = 0
    if word_count >= 300: seo_score += 30
    if len(title) >= 30: seo_score += 20
    if char_count >= 1000: seo_score += 25
    seo_score += min(25, readability_score / 4)
    
    return {
        "success": True,
        "analysis": {
            "basic_seo_score": round(seo_score, 1),
            "basic_readability": round(readability_score, 1),
            "word_count": word_count,
            "character_count": char_count,
            "basic_recommendations": [
                f"Content has {word_count} words - {'good length' if word_count >= 300 else 'consider longer content'}",
                f"Readability score: {readability_score:.1f} - {'readable' if readability_score > 60 else 'could be clearer'}",
                "Basic analysis completed - AI analysis was not available"
            ]
        }
    }