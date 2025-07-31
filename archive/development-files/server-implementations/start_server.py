#!/usr/bin/env python3
"""Simple server startup script."""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import uvicorn
    from web.main import app
    
    print("Starting SEO Content Knowledge Graph System...")
    print("Server will be available at: http://localhost:9000")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=9000, 
        log_level="info",
        access_log=True
    )
    
except Exception as e:
    print(f"Error starting server: {e}")
    print("Make sure virtual environment is activated and dependencies are installed.")