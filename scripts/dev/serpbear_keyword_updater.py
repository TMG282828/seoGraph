#!/usr/bin/env python3
"""
SerpBear Keyword Position Updater

This service directly updates keyword positions in the SerpBear database
using our custom scraper, bypassing SerpBear's built-in scraper system.
"""

import asyncio
import sqlite3
import json
import logging
from datetime import datetime, date
from typing import Dict, List, Any, Optional
import sys
sys.path.append('.')

from src.services.custom_serp_scraper import custom_serp_scraper

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SerpBearKeywordUpdater:
    """Direct keyword position updater for SerpBear database."""
    
    def __init__(self, db_path: str = "./serpbear_db_current.sqlite"):
        """Initialize the updater."""
        self.db_path = db_path
        self.scraper = custom_serp_scraper
        
        logger.info(f"SerpBear Keyword Updater initialized - DB: {db_path}")
    
    def get_all_keywords(self) -> List[Dict[str, Any]]:
        """Get all keywords from SerpBear database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all keywords with their domain info
            cursor.execute("""
                SELECT k.id, k.keyword, k.device, k.country, d.domain, k.position, k.lastResult
                FROM keyword k
                JOIN domain d ON k.domain = d.domain
                WHERE k.keyword IS NOT NULL AND k.keyword != ''
            """)
            
            results = cursor.fetchall()
            conn.close()
            
            keywords = []
            for row in results:
                keywords.append({
                    "id": row[0],
                    "keyword": row[1],
                    "device": row[2],
                    "country": row[3],
                    "domain": row[4],
                    "current_position": row[5],
                    "last_result": row[6]
                })
            
            logger.info(f"Found {len(keywords)} keywords in SerpBear database")
            return keywords
            
        except Exception as e:
            logger.error(f"Error getting keywords: {e}")
            return []
    
    def update_keyword_position(self, keyword_id: int, position: Optional[int], 
                               url: str = "") -> bool:
        """Update a keyword's position in the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create history entry
            today = date.today().isoformat()
            history_data = {today: position} if position else {today: -1}
            
            # Update keyword record  
            cursor.execute("""
                UPDATE keyword 
                SET position = ?, 
                    url = ?, 
                    lastResult = ?,
                    lastUpdated = ?,
                    history = ?,
                    updating = 0,
                    lastUpdateError = NULL
                WHERE id = ?
            """, (
                position or -1,
                url,
                json.dumps({"position": position, "url": url, "timestamp": datetime.now().isoformat()}),
                datetime.now().isoformat(),
                json.dumps(history_data),
                keyword_id
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Updated keyword ID {keyword_id} - Position: {position}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating keyword {keyword_id}: {e}")
            return False
    
    async def refresh_all_keywords(self) -> Dict[str, Any]:
        """Refresh positions for all keywords."""
        try:
            logger.info("🔄 Starting keyword position refresh")
            
            # Get all keywords
            keywords = self.get_all_keywords()
            if not keywords:
                return {"success": False, "error": "No keywords found"}
            
            successful_updates = 0
            failed_updates = 0
            results = []
            
            for keyword_data in keywords:
                try:
                    logger.info(f"   Processing: '{keyword_data['keyword']}' for {keyword_data['domain']}")
                    
                    # Search using our custom scraper
                    result = await self.scraper.search_keyword(
                        keyword=keyword_data["keyword"],
                        domain=keyword_data["domain"],
                        country=keyword_data["country"],
                        device=keyword_data["device"]
                    )
                    
                    if result:
                        # Update database with new position
                        success = self.update_keyword_position(
                            keyword_id=keyword_data["id"],
                            position=result.get("position"),
                            url=result.get("url", "")
                        )
                        
                        if success:
                            successful_updates += 1
                            logger.info(f"      ✅ Updated position: {result.get('position', 'Not found')}")
                        else:
                            failed_updates += 1
                            logger.error(f"      ❌ Database update failed")
                    else:
                        # Update with no position found
                        success = self.update_keyword_position(
                            keyword_id=keyword_data["id"],
                            position=None
                        )
                        
                        if success:
                            successful_updates += 1
                            logger.info(f"      ⚠️ Not found in results")
                        else:
                            failed_updates += 1
                            logger.error(f"      ❌ Database update failed")
                    
                    results.append({
                        "keyword": keyword_data["keyword"],
                        "domain": keyword_data["domain"],
                        "position": result.get("position") if result else None,
                        "success": success
                    })
                    
                    # Small delay between requests
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"      ❌ Error processing keyword: {e}")
                    failed_updates += 1
            
            summary = {
                "success": True,
                "total_keywords": len(keywords),
                "successful_updates": successful_updates,
                "failed_updates": failed_updates,
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"🎉 Refresh completed: {successful_updates}/{len(keywords)} successful")
            return summary
            
        except Exception as e:
            logger.error(f"Error in refresh_all_keywords: {e}")
            return {"success": False, "error": str(e)}

async def main():
    """Main function to test the keyword updater."""
    print("🔄 SerpBear Keyword Position Updater")
    print("=" * 50)
    
    # Copy current database from container
    import subprocess
    try:
        subprocess.run(["docker", "cp", "seo-serpbear:/app/data/database.sqlite", 
                       "./serpbear_db_current.sqlite"], check=True)
        print("✅ Database copied from SerpBear container")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to copy database: {e}")
        return
    
    # Initialize updater
    updater = SerpBearKeywordUpdater()
    
    # Show current keywords
    keywords = updater.get_all_keywords()
    print(f"\n📋 Found {len(keywords)} keywords to update:")
    for kw in keywords:
        print(f"   • {kw['keyword']} ({kw['domain']}) - Current position: {kw['current_position']}")
    
    if not keywords:
        print("⚠️ No keywords found to update")
        return
    
    print(f"\n🚀 Starting position refresh...")
    
    # Refresh all positions
    result = await updater.refresh_all_keywords()
    
    if result["success"]:
        print(f"\n🎉 Refresh Summary:")
        print(f"   Total keywords: {result['total_keywords']}")
        print(f"   Successful updates: {result['successful_updates']}")
        print(f"   Failed updates: {result['failed_updates']}")
        print(f"   Success rate: {(result['successful_updates']/result['total_keywords']*100):.1f}%")
        
        print(f"\n📊 Position Results:")
        for res in result["results"]:
            status = "✅" if res["success"] else "❌"
            pos_str = f"Position {res['position']}" if res['position'] else "Not found"
            print(f"   {status} {res['keyword']} ({res['domain']}): {pos_str}")
    else:
        print(f"❌ Refresh failed: {result.get('error', 'Unknown error')}")
    
    # Copy updated database back to container
    try:
        subprocess.run(["docker", "cp", "./serpbear_db_current.sqlite", 
                       "seo-serpbear:/app/data/database.sqlite"], check=True)
        print(f"\n✅ Updated database copied back to SerpBear container")
        print("🔄 You may need to refresh the SerpBear web interface to see changes")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to copy database back: {e}")

if __name__ == "__main__":
    asyncio.run(main())