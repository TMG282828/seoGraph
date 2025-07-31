"""
Content analysis module for Knowledge Base.
Handles AI-powered content analysis and processing.
"""

from typing import Dict, Any
import logging
from fastapi import UploadFile

logger = logging.getLogger(__name__)


async def analyze_file_content(
    file: UploadFile,
    content_text: str,
    content_bytes: bytes,
    current_user: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyze uploaded file content with AI agents.
    
    Args:
        file: Uploaded file object
        content_text: Decoded content text
        content_bytes: Raw file bytes
        current_user: Current user context
        
    Returns:
        Analysis result dictionary
    """
    try:
        # Try AI analysis first
        from src.agents.content_analysis_agent import content_analysis_agent
        from src.agents.base_agent import AgentContext
        
        # Prepare analysis request
        analysis_request = {
            "content_text": content_text,
            "filename": file.filename,
            "content_type": file.content_type or "text/plain",
            "analysis_type": "comprehensive",
            "include_seo_analysis": True,
            "include_topic_analysis": True,
            "include_keyword_analysis": True,
            "target_keywords": []  # Can be enhanced later
        }
        
        # Create agent context
        context = AgentContext(
            organization_id=current_user.get('org_id', '00000000-0000-0000-0000-000000000001'),
            user_id=current_user.get('id', 'anonymous'),
            session_id=f"file-upload-{file.filename}",
            task_type="content_analysis"
        )
        
        # Execute content analysis with real AI agent
        logger.info(f"Analyzing uploaded file: {file.filename} with AI agent")
        result = await content_analysis_agent.execute(analysis_request, context)
        
        if result.success and result.result_data:
            analysis_data = result.result_data
            
            # Extract key metrics from AI analysis
            structural_analysis = analysis_data.get("structural_analysis", {})
            seo_metrics = analysis_data.get("seo_metrics", {})
            keyword_analysis = analysis_data.get("keyword_analysis", {})
            topic_analysis = analysis_data.get("topic_analysis", {})
            
            return {
                "success": True,
                "file_name": file.filename,
                "file_size": len(content_bytes),
                "content_type": file.content_type,
                "word_count": seo_metrics.get("word_count", len(content_text.split())),
                "analysis": {
                    "type": "ai_powered",
                    "agent_used": "content_analysis_agent",
                    "structural_analysis": structural_analysis,
                    "seo_metrics": seo_metrics,
                    "keyword_analysis": keyword_analysis,
                    "topic_analysis": topic_analysis,
                    "brand_voice_compliance": analysis_data.get("brand_voice_compliance", {}),
                    "confidence_score": analysis_data.get("confidence_score", 0.8)
                },
                "recommendations": analysis_data.get("recommendations", []),
                "extracted_topics": topic_analysis.get("primary_keywords", []),
                "named_entities": topic_analysis.get("named_entities", []),
                "note": "Real AI-powered content analysis using advanced content analysis agent"
            }
        else:
            # Fallback to enhanced analysis if AI agent fails
            logger.warning("AI agent failed, using enhanced analysis fallback")
            return await _generate_enhanced_file_analysis(file.filename, content_text, len(content_bytes), file.content_type)
            
    except ImportError as import_error:
        logger.warning(f"AI agents not available: {import_error}")
        return await _generate_enhanced_file_analysis(file.filename, content_text, len(content_bytes), file.content_type)
    except Exception as agent_error:
        logger.error(f"AI agent execution failed: {agent_error}")
        return await _generate_enhanced_file_analysis(file.filename, content_text, len(content_bytes), file.content_type)


async def _generate_enhanced_file_analysis(filename: str, content_text: str, file_size: int, content_type: str) -> Dict[str, Any]:
    """
    Generate enhanced file analysis without AI agents.
    
    Args:
        filename: Name of the file
        content_text: File content text
        file_size: Size of file in bytes
        content_type: MIME type of file
        
    Returns:
        Analysis result dictionary
    """
    import re
    from collections import Counter
    
    # Basic text analysis
    words = content_text.split()
    word_count = len(words)
    char_count = len(content_text)
    paragraph_count = len([p for p in content_text.split('\n\n') if p.strip()])
    
    # Keyword extraction (simple frequency-based)
    word_freq = Counter(word.lower().strip('.,!?;:"()[]{}') for word in words if len(word) > 3)
    top_keywords = [{"word": word, "frequency": freq, "relevance": min(freq/10, 1.0)} 
                   for word, freq in word_freq.most_common(10)]
    
    # SEO metrics
    title_match = re.search(r'^#\s+(.+)$', content_text, re.MULTILINE)
    title = title_match.group(1) if title_match else filename
    
    h1_count = len(re.findall(r'^#+\s', content_text, re.MULTILINE))
    link_count = len(re.findall(r'\[([^\]]+)\]\([^)]+\)', content_text))
    
    # Basic readability (simplified Flesch reading ease approximation)
    sentences = len(re.findall(r'[.!?]+', content_text))
    avg_sentence_length = word_count / max(sentences, 1)
    readability_score = max(0, min(100, 206.835 - (1.015 * avg_sentence_length)))
    
    # SEO score calculation
    seo_score = 0
    if word_count >= 300: seo_score += 20
    elif word_count >= 150: seo_score += 10
    if h1_count >= 1: seo_score += 15
    if paragraph_count >= 3: seo_score += 10
    if link_count >= 2: seo_score += 10
    if len(title) >= 30: seo_score += 15
    seo_score += min(30, readability_score / 3)  # Readability bonus
    
    return {
        "success": True,
        "file_name": filename,
        "file_size": file_size,
        "content_type": content_type,
        "word_count": word_count,
        "analysis": {
            "type": "enhanced_statistical",
            "structural_analysis": {
                "word_count": word_count,
                "character_count": char_count,
                "paragraph_count": paragraph_count,
                "heading_count": h1_count,
                "link_count": link_count,
                "estimated_reading_time": max(1, word_count // 200)
            },
            "seo_metrics": {
                "overall_seo_score": round(seo_score, 1),
                "readability_score": round(readability_score, 1),
                "keyword_density": len(top_keywords),
                "content_length_score": min(100, (word_count / 10)),
                "structure_score": min(100, (h1_count * 20) + (paragraph_count * 5))
            },
            "keyword_analysis": {
                "primary_keywords": top_keywords[:5],
                "all_keywords": top_keywords,
                "keyword_diversity": len(set(word.lower() for word in words if len(word) > 3))
            },
            "topic_analysis": {
                "primary_topics": [kw["word"] for kw in top_keywords[:3]],
                "content_focus": "general" if len(top_keywords) > 8 else "focused",
                "topic_coherence": min(1.0, top_keywords[0]["frequency"] / max(word_count/100, 1) if top_keywords else 0)
            }
        },
        "recommendations": [
            f"Document has {word_count} words - {'good length' if word_count >= 300 else 'consider expanding'}",
            f"Readability score: {readability_score:.1f} - {'easy to read' if readability_score > 60 else 'consider simplifying'}",
            f"Found {h1_count} headings - {'good structure' if h1_count >= 3 else 'add more headings for better organization'}",
            f"SEO score: {seo_score:.1f}/100 - {'good optimization' if seo_score >= 70 else 'needs SEO improvement'}"
        ],
        "extracted_topics": [kw["word"] for kw in top_keywords[:5]],
        "named_entities": [],  # Would need NLP library for proper NER
        "note": "Enhanced statistical analysis with SEO optimization insights"
    }