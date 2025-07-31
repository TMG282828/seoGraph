#!/usr/bin/env python3
"""
ScrapingAnt API Proxy Service

This service acts as a proxy/mock for ScrapingAnt API calls from SerpBear,
redirecting them to our custom SERP scraper bridge.
"""

import asyncio
import logging
import json
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import aiohttp
from urllib.parse import parse_qs, unquote
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ScrapingAnt Proxy", description="Proxy ScrapingAnt API calls to custom bridge")

# Our bridge service URL
BRIDGE_URL = "http://localhost:8000/api/serp-bridge/"

@app.post("/v2/general")
@app.get("/v2/general")
async def scrapingant_proxy(request: Request):
    """
    Mock ScrapingAnt API endpoint that redirects to our bridge.
    
    ScrapingAnt expects parameters like:
    - url: The URL to scrape (Google search URL)
    - x-api-key: API key
    """
    try:
        logger.info(f"📞 ScrapingAnt API call received: {request.method} {request.url}")
        
        # Get query parameters
        query_params = dict(request.query_params)
        logger.info(f"   Query params: {query_params}")
        
        # Get request body if POST
        body = {}
        if request.method == "POST":
            body = await request.json() if await request.body() else {}
            logger.info(f"   Body: {body}")
        
        # Extract Google search URL from parameters
        google_url = query_params.get("url") or body.get("url", "")
        logger.info(f"   Google URL: {google_url}")
        
        if not google_url:
            logger.warning("   No URL provided in ScrapingAnt request")
            return JSONResponse(
                status_code=400,
                content={"error": "Missing URL parameter"}
            )
        
        # Parse Google search URL to extract keyword and other parameters
        search_params = extract_google_search_params(google_url)
        logger.info(f"   Extracted search params: {search_params}")
        
        if not search_params:
            logger.warning("   Could not parse Google search URL")
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid Google search URL"}
            )
        
        # Make request to our bridge service
        bridge_response = await call_bridge_service(search_params)
        
        if bridge_response:
            # Convert bridge response to ScrapingAnt format
            scrapingant_response = convert_to_scrapingant_format(bridge_response, google_url)
            logger.info(f"   ✅ Returning ScrapingAnt formatted response")
            return JSONResponse(content=scrapingant_response)
        else:
            logger.error("   ❌ Bridge service failed")
            return JSONResponse(
                status_code=500,
                content={"error": "Bridge service unavailable"}
            )
            
    except Exception as e:
        logger.error(f"   ❌ Error in ScrapingAnt proxy: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

def extract_google_search_params(google_url: str) -> Dict[str, str]:
    """Extract search parameters from Google search URL."""
    try:
        from urllib.parse import urlparse, parse_qs
        
        parsed = urlparse(google_url)
        query_params = parse_qs(parsed.query)
        
        # Extract keyword (q parameter)
        keyword = query_params.get("q", [""])[0]
        if not keyword:
            return {}
        
        # Extract other parameters
        gl = query_params.get("gl", ["US"])[0]  # Country
        hl = query_params.get("hl", ["en"])[0]  # Language
        
        return {
            "keyword": keyword,
            "domain": "example.com",  # We'll need to determine this somehow
            "country": gl,
            "device": "desktop",
            "engine": "google"
        }
        
    except Exception as e:
        logger.error(f"Error parsing Google URL: {e}")
        return {}

async def call_bridge_service(search_params: Dict[str, str]) -> Dict[str, Any]:
    """Call our bridge service with the search parameters."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(BRIDGE_URL, json=search_params, timeout=30) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Bridge service error: {response.status}")
                    return None
                    
    except Exception as e:
        logger.error(f"Error calling bridge service: {e}")
        return None

def convert_to_scrapingant_format(bridge_response: Dict[str, Any], original_url: str) -> Dict[str, Any]:
    """Convert our bridge response to ScrapingAnt format."""
    try:
        # Create a mock HTML response that SerpBear can parse
        position = bridge_response.get("position")
        domain = bridge_response.get("domain", "")
        keyword = bridge_response.get("keyword", "")
        
        # Create fake Google SERP HTML with our result
        if position:
            html_content = f"""
            <html>
            <body>
                <div id="search">
                    <div class="g">
                        <h3>Result {position}</h3>
                        <a href="https://{domain}/">{bridge_response.get('title', 'Title')}</a>
                        <div class="s">
                            <span class="st">{bridge_response.get('snippet', 'Description')}</span>
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """
        else:
            html_content = "<html><body><div id='search'>No results found</div></body></html>"
        
        return {
            "html": html_content,
            "text": bridge_response.get("snippet", ""),
            "status_code": 200,
            "cookies": "",
            "headers": [
                {"name": "Content-Type", "value": "text/html; charset=utf-8"}
            ]
        }
        
    except Exception as e:
        logger.error(f"Error converting to ScrapingAnt format: {e}")
        return {
            "html": "<html><body>Error</body></html>",
            "text": "Error",
            "status_code": 500,
            "cookies": "",
            "headers": []
        }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ScrapingAnt Proxy"}

if __name__ == "__main__":
    print("🕷️ Starting ScrapingAnt Proxy Service")
    print("=" * 50)
    print("This service will intercept ScrapingAnt API calls")
    print("and redirect them to our custom bridge service.")
    print()
    print("Listening on: http://localhost:8081")
    print("ScrapingAnt endpoint: http://localhost:8081/v2/general")
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8081, log_level="info")