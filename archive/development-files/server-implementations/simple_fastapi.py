#!/usr/bin/env python3
"""Simplified FastAPI server to test basic functionality."""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Create FastAPI app
app = FastAPI(title="SEO Content Knowledge Graph System - Simple Mode")

# Setup templates
templates = Jinja2Templates(directory="web/templates")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return """
    <html>
    <head><title>SEO Content System</title></head>
    <body>
    <h1>🚀 SEO Content Knowledge Graph System</h1>
    <p>✅ FastAPI is working!</p>
    <ul>
        <li><a href="/content-studio">Content Studio</a></li>
        <li><a href="/health">Health Check</a></li>
    </ul>
    </body>
    </html>
    """

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "FastAPI server is running"}

@app.get("/content-studio", response_class=HTMLResponse)
async def content_studio(request: Request):
    try:
        return templates.TemplateResponse("content.html", {"request": request})
    except Exception as e:
        return f"""
        <html>
        <body>
        <h1>Content Studio</h1>
        <p>Template loading error: {e}</p>
        <p>The core FastAPI server is working. Template needs adjustment for simplified mode.</p>
        </body>
        </html>
        """

if __name__ == "__main__":
    print("🚀 Starting Simplified FastAPI Server...")
    print("📍 Will be available at: http://localhost:8888")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8888,
        log_level="info"
    )