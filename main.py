#!/usr/bin/env python3
"""
Main entry point for SEO Content Knowledge Graph System on Replit.

This file starts the FastAPI server with production-ready configuration
for Replit deployment.
"""

import os
import sys
import uvicorn
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Start the FastAPI server."""
    try:
        # Import the FastAPI app
        from web.main import app
        
        # Get port from environment (Replit sets this automatically)
        port = int(os.getenv("PORT", 8000))
        host = os.getenv("HOST", "0.0.0.0")
        
        logger.info(f"🚀 Starting SEO Content Knowledge Graph System")
        logger.info(f"📡 Server will run on {host}:{port}")
        logger.info(f"🔧 Environment: {'Production' if os.getenv('REPL_ID') else 'Development'}")
        
        # Check for required environment variables
        required_vars = ["OPENAI_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.warning(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
            logger.warning("🔑 Please add these to your Replit Secrets for full functionality")
        else:
            logger.info("✅ All required environment variables found")
        
        # Start the server
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True,
            reload=False  # Disable reload in production
        )
        
    except ImportError as e:
        logger.error(f"❌ Failed to import application: {e}")
        logger.error("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()