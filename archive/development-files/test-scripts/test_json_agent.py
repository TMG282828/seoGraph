#!/usr/bin/env python3
"""
Quick test script for JsonAnalysisAgent.

Tests the new JSON analysis agent with a real brief from the server logs.
"""

import asyncio
import logging
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.json_analysis.agent import JsonAnalysisAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Test brief content from server logs
BRIEF_CONTENT = """Objectives
Promote trending health tips, and upsell relevant products on the online store.
Call to Action
Calls to relevant product pages.
Target Audience
People curious about the trending health tips and wellness enthusiasts.
Content Goals
Create engaging articles about health trends that naturally incorporate product recommendations.
Brand Voice
Authoritative yet approachable, expert-driven content that builds trust with readers.
SEO Focus
Target keywords related to trending health topics with high search volume.
Timeline
Content should be published within 2-3 days to capitalize on trending topics.
Success Metrics
Measure engagement, click-through rates to product pages, and conversion rates."""

async def test_json_analysis():
    """Test the JsonAnalysisAgent with real brief content."""
    try:
        logger.info("🧪 Testing JsonAnalysisAgent with real brief content...")
        
        # Create the agent
        agent = JsonAnalysisAgent()
        
        # Test brief analysis
        result = await agent.analyze_brief(BRIEF_CONTENT, "Manual Brief - 28/07/2025")
        
        logger.info(f"✅ JSON Analysis successful!")
        logger.info(f"📊 Result keys: {list(result.keys())}")
        logger.info(f"📋 Main topic: {result.get('main_topic')}")
        logger.info(f"🎯 Key themes: {result.get('key_themes')}")
        logger.info(f"👥 Target audience: {result.get('target_audience')}")
        logger.info(f"📝 Content type: {result.get('content_type')}")
        logger.info(f"⚡ Complexity: {result.get('estimated_complexity')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ JSON Analysis failed: {e}")
        return False

async def main():
    """Main test function."""
    logger.info("🚀 Starting JsonAnalysisAgent test...")
    
    success = await test_json_analysis()
    
    if success:
        logger.info("✅ All tests passed! JsonAnalysisAgent is working correctly.")
        return 0
    else:
        logger.error("❌ Tests failed! Check the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)