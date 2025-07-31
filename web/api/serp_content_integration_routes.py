"""
API routes for SERP-Content Integration.

This module provides endpoints for connecting SERP analysis with content generation
workflows, enabling SEO-optimized content creation based on ranking data.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field

try:
    from src.services.serp_content_integration_service import (
        serp_content_integration_service, 
        SERPContentStrategy,
        SERPContentInsight
    )
    SERP_INTEGRATION_AVAILABLE = True
except ImportError as e:
    SERP_INTEGRATION_AVAILABLE = False
    # Create production-safe mock service that returns only empty/null responses
    class MockSERPService:
        async def get_content_recommendations(self, domain, limit=20):
            return []  # Always return empty list - no fake data
        async def analyze_keyword_opportunity(self, keyword, domain):
            return None  # Always return None - no fake data
        async def initialize(self):
            pass  # No-op for mock service
    serp_content_integration_service = MockSERPService()

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/serp-content", tags=["SERP Content Integration"])


# Request/Response Models

class ContentOpportunityRequest(BaseModel):
    """Request to analyze content opportunities."""
    domain: str = Field(..., description="Domain to analyze")
    keywords: Optional[List[str]] = Field(None, description="Specific keywords to analyze")
    days_back: int = Field(default=30, description="Days to look back for trend analysis")


class SERPContentGenerationRequest(BaseModel):
    """Request to generate SERP-optimized content."""
    keyword: str = Field(..., description="Target keyword")
    domain: str = Field(..., description="Domain to optimize for")
    content_type: str = Field(default="blog_post", description="Type of content to generate")
    include_competitors: bool = Field(default=True, description="Include competitor analysis")


class ContentRecommendationResponse(BaseModel):
    """Response with content recommendations."""
    success: bool
    domain: str
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    total_opportunities: int = 0
    total_opportunity_score: float = 0.0
    last_updated: str


@router.post("/analyze-opportunities", response_model=Dict[str, Any])
async def analyze_content_opportunities(request: ContentOpportunityRequest):
    """Analyze SERP data to identify content creation opportunities."""
    
    if not SERP_INTEGRATION_AVAILABLE:
        # Return clear service unavailable status for production transparency
        return {
            "success": False,
            "domain": request.domain,
            "strategy": {
                "primary_keywords": [],
                "secondary_keywords": [],
                "total_opportunities": 0,
                "total_opportunity_score": 0.0,
                "content_calendar": [],
                "top_opportunities": []
            },
            "analysis_timestamp": "N/A",
            "error": "SERP integration service not available",
            "message": "Google Ads and SERP analysis features are currently unavailable. Please contact support to enable these services."
        }
    
    try:
        logger.info(f"Analyzing content opportunities for domain: {request.domain}")
        
        # Initialize service if needed
        await serp_content_integration_service.initialize()
        
        # Analyze opportunities
        strategy = await serp_content_integration_service.analyze_content_opportunities(
            domain=request.domain,
            keywords=request.keywords,
            days_back=request.days_back
        )
        
        return {
            "success": True,
            "domain": request.domain,
            "strategy": {
                "primary_keywords": strategy.primary_keywords,
                "secondary_keywords": strategy.secondary_keywords,
                "total_opportunities": len(strategy.content_priorities),
                "total_opportunity_score": strategy.total_opportunity_score,
                "content_calendar": strategy.recommended_content_calendar,
                "top_opportunities": [
                    {
                        "keyword": insight.keyword,
                        "opportunity_score": insight.opportunity_score,
                        "current_position": insight.current_position,
                        "target_position": insight.target_position,
                        "estimated_traffic": insight.estimated_traffic_potential,
                        "content_type": insight.recommended_content_type,
                        "difficulty": insight.difficulty_score,
                        "content_gaps": insight.content_gaps
                    }
                    for insight in strategy.content_priorities[:10]
                ]
            },
            "analysis_timestamp": "2024-01-15T10:30:00Z"  # Would be dynamic
        }
        
    except Exception as e:
        logger.error(f"Error analyzing content opportunities: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/recommendations/{domain}", response_model=ContentRecommendationResponse)
async def get_content_recommendations(
    domain: str,
    limit: int = Query(default=10, ge=1, le=50, description="Maximum number of recommendations")
):
    """Get prioritized content recommendations based on SERP analysis."""
    
    if not SERP_INTEGRATION_AVAILABLE:
        # Return clear service unavailable status for production transparency
        return ContentRecommendationResponse(
            success=False,
            domain=domain,
            recommendations=[],
            total_opportunities=0,
            total_opportunity_score=0.0,
            last_updated="Service unavailable"
        )
    
    try:
        # Initialize service if needed
        await serp_content_integration_service.initialize()
        
        # Get recommendations
        recommendations = await serp_content_integration_service.get_content_recommendations(
            domain=domain,
            limit=limit
        )
        
        # Calculate totals
        total_opportunity_score = sum(rec.get("opportunity_score", 0) for rec in recommendations)
        
        return ContentRecommendationResponse(
            success=True,
            domain=domain,
            recommendations=recommendations,
            total_opportunities=len(recommendations),
            total_opportunity_score=total_opportunity_score,
            last_updated="2024-01-15T10:30:00Z"  # Would be dynamic
        )
        
    except Exception as e:
        logger.error(f"Error getting content recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")


@router.post("/generate-content", response_model=Dict[str, Any])
async def generate_serp_optimized_content(
    request: SERPContentGenerationRequest,
    background_tasks: BackgroundTasks
):
    """Generate content optimized for SERP performance."""
    
    if not SERP_INTEGRATION_AVAILABLE:
        # Return clear service unavailable status for production transparency
        return {
            "success": False,
            "content": None,
            "generation_timestamp": "N/A",
            "optimization_applied": False,
            "error": "SERP content generation service not available",
            "message": "SERP-optimized content generation requires Google Ads integration. Please contact support to enable this feature."
        }
    
    try:
        logger.info(f"Generating SERP-optimized content for keyword: {request.keyword}")
        
        # Initialize service if needed
        await serp_content_integration_service.initialize()
        
        # Generate content
        result = await serp_content_integration_service.generate_serp_optimized_content(
            keyword=request.keyword,
            domain=request.domain,
            content_type=request.content_type,
            include_competitors=request.include_competitors
        )
        
        if not result:
            raise HTTPException(status_code=400, detail="Content generation failed")
        
        return {
            "success": True,
            "content": result,
            "generation_timestamp": "2024-01-15T10:30:00Z",  # Would be dynamic
            "optimization_applied": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating SERP-optimized content: {e}")
        raise HTTPException(status_code=500, detail=f"Content generation failed: {str(e)}")


@router.get("/keyword-analysis/{domain}/{keyword}")
async def analyze_keyword_opportunity(domain: str, keyword: str):
    """Analyze a specific keyword for content opportunity."""
    
    if not SERP_INTEGRATION_AVAILABLE:
        # Return clear service unavailable status for production transparency
        return {
            "success": False,
            "keyword": keyword,
            "domain": domain,
            "analysis": None,
            "serp_data": None,
            "error": "Keyword analysis service not available",
            "message": "Keyword opportunity analysis requires Google Ads and SERP data integration. Please contact support to enable this feature."
        }
    
    try:
        # Initialize service if needed  
        await serp_content_integration_service.initialize()
        
        # Get unified keyword data
        from src.services.unified_seo_data_service import UnifiedSEODataService
        seo_service = UnifiedSEODataService()
        
        keyword_data = await seo_service.get_unified_keyword_data(domain, keyword)
        if not keyword_data:
            raise HTTPException(status_code=404, detail="Keyword data not found")
        
        # Analyze opportunity
        insight = await serp_content_integration_service._analyze_keyword_opportunity(keyword_data)
        if not insight:
            raise HTTPException(status_code=400, detail="Could not analyze keyword opportunity")
        
        return {
            "success": True,
            "keyword": keyword,
            "domain": domain,
            "analysis": {
                "current_position": insight.current_position,
                "target_position": insight.target_position,
                "opportunity_score": insight.opportunity_score,
                "estimated_traffic": insight.estimated_traffic_potential,
                "difficulty_score": insight.difficulty_score,
                "content_gaps": insight.content_gaps,
                "recommended_content_type": insight.recommended_content_type,
                "competitor_insights": insight.competitor_insights
            },
            "serp_data": {
                "search_volume": keyword_data.search_volume,
                "competition": keyword_data.competition,
                "cpc": keyword_data.cpc,
                "position_history": keyword_data.historical_positions
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing keyword opportunity: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/integration-status")
async def get_integration_status():
    """Get the status of SERP-Content integration services."""
    
    try:
        status = {
            "service_available": SERP_INTEGRATION_AVAILABLE,
            "components": {
                "serp_integration_service": SERP_INTEGRATION_AVAILABLE,
                "unified_seo_service": False,
                "content_generation_agent": False
            }
        }
        
        if SERP_INTEGRATION_AVAILABLE:
            # Check component availability
            try:
                from src.services.unified_seo_data_service import UnifiedSEODataService
                status["components"]["unified_seo_service"] = True
            except ImportError:
                pass
            
            try:
                from src.agents.content_generation.agent import ContentGenerationAgent
                status["components"]["content_generation_agent"] = True
            except ImportError:
                pass
        
        return {
            "success": True,
            "status": status,
            "message": "SERP-Content integration ready" if SERP_INTEGRATION_AVAILABLE else "Google Ads and SERP integration services are not configured. Contact support to enable these features.",
            "service_available": SERP_INTEGRATION_AVAILABLE
        }
        
    except Exception as e:
        logger.error(f"Error checking integration status: {e}")
        return {
            "success": False,
            "status": {"service_available": False},
            "error": str(e)
        }


@router.post("/workflow/start")
async def start_serp_content_workflow(
    domain: str,
    target_keywords: List[str],
    content_goals: Dict[str, Any] = None
):
    """Start an integrated SERP analysis and content generation workflow."""
    
    if not SERP_INTEGRATION_AVAILABLE:
        # Return clear service unavailable status for production transparency
        return {
            "success": False,
            "workflow": None,
            "error": "SERP workflow service not available",
            "message": "SERP-Content workflows require Google Ads integration and keyword data services. Please contact support to enable these features."
        }
    
    try:
        logger.info(f"Starting SERP-Content workflow for domain: {domain}")
        
        # Initialize service
        await serp_content_integration_service.initialize()
        
        # Step 1: Analyze opportunities for target keywords
        strategy = await serp_content_integration_service.analyze_content_opportunities(
            domain=domain,
            keywords=target_keywords
        )
        
        # Step 2: Get prioritized recommendations
        recommendations = await serp_content_integration_service.get_content_recommendations(
            domain=domain,
            limit=10
        )
        
        # Step 3: Create workflow plan
        workflow_plan = {
            "workflow_id": f"serp_workflow_{domain}_{int(datetime.now().timestamp())}",
            "domain": domain,
            "target_keywords": target_keywords,
            "phase": "analysis_complete",
            "strategy": {
                "primary_keywords": strategy.primary_keywords,
                "secondary_keywords": strategy.secondary_keywords,
                "total_opportunity_score": strategy.total_opportunity_score
            },
            "recommendations": recommendations,
            "next_steps": [
                {
                    "step": "content_generation",
                    "description": "Generate content for top-priority keywords",
                    "keywords": [rec["keyword"] for rec in recommendations[:3]]
                },
                {
                    "step": "optimization_review",
                    "description": "Review and optimize generated content",
                    "timeline": "after_generation"
                }
            ]
        }
        
        return {
            "success": True,
            "workflow": workflow_plan,
            "message": f"SERP-Content workflow started with {len(recommendations)} opportunities identified"
        }
        
    except Exception as e:
        logger.error(f"Error starting SERP-Content workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Workflow start failed: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check for SERP-Content integration service."""
    return {
        "status": "healthy" if SERP_INTEGRATION_AVAILABLE else "unavailable",
        "service": "SERP Content Integration",
        "available": SERP_INTEGRATION_AVAILABLE,
        "components": {
            "integration_service": SERP_INTEGRATION_AVAILABLE,
            "content_agent": SERP_INTEGRATION_AVAILABLE,
            "seo_data_service": SERP_INTEGRATION_AVAILABLE
        }
    }