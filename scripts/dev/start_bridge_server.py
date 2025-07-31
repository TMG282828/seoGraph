#!/usr/bin/env python3
"""
Start the SerpBear Bridge Server.

This script starts a FastAPI server with the SerpBear bridge service
to act as a custom scraper for SerpBear.
"""

import uvicorn
from fastapi import FastAPI
from src.services.serpbear_bridge import router
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SerpBear Bridge Service",
    description="Custom scraper bridge for SerpBear integration",
    version="1.0.0"
)

# Include the bridge router
app.include_router(router)

# Add health check at root
@app.get("/health")
async def root_health():
    """Root health check endpoint."""
    return {"status": "healthy", "service": "serpbear-bridge"}

# Add specific endpoint that SerpBear might expect for custom scrapers
from fastapi import Request
from src.services.serpbear_bridge import SerpBearKeywordRequest, SerpBearScrapingResponse
import json

@app.post("/", response_model=SerpBearScrapingResponse)
async def root_scraper_endpoint(request: SerpBearKeywordRequest):
    """Root endpoint that SerpBear might call for custom scraping."""
    logger.info(f"🎯 Root scraper called for keyword: {request.keyword}")
    from src.services.serpbear_bridge import serpbear_bridge
    return await serpbear_bridge.scrape_keyword(request)

# Add catch-all for any requests to see what SerpBear is calling
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def catch_all_root(path: str, request: Request):
    """Catch all requests to see what SerpBear is calling."""
    method = request.method
    headers = dict(request.headers)
    query_params = dict(request.query_params)
    
    try:
        body = await request.body()
        body_text = body.decode() if body else ""
    except:
        body_text = ""
    
    logger.info(f"🔍 SerpBear Request: {method} /{path}")
    logger.info(f"   Query: {query_params}")
    logger.info(f"   Body: {body_text}")
    
    # Return a basic scraper response format
    return {
        "status": "received", 
        "method": method,
        "path": path,
        "success": True,
        "position": 5,
        "url": "https://example.com/test",
        "title": "Test Result"
    }

if __name__ == "__main__":
    logger.info("🌉 Starting SerpBear Bridge Server on port 8002")
    uvicorn.run(
        "start_bridge_server:app",
        host="0.0.0.0",
        port=8002,
        reload=False,
        log_level="info"
    )