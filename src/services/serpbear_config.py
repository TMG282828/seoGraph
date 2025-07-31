"""
SerpBear Configuration Service.

This service configures SerpBear to use our local custom scraper
by modifying its settings and providing compatible API endpoints.
"""

import json
import sqlite3
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class SerpBearConfigurator:
    """
    Configure SerpBear to use our local custom scraper.
    
    This service modifies SerpBear's database settings to point
    to our local scraper bridge instead of third-party services.
    """
    
    def __init__(self, serpbear_data_path: str = "/app/data"):
        """Initialize SerpBear configurator."""
        self.data_path = serpbear_data_path
        self.db_path = f"{serpbear_data_path}/database.sqlite"
        self.settings_path = f"{serpbear_data_path}/settings.json"
        
        logger.info(f"SerpBear configurator initialized - DB: {self.db_path}")
    
    def configure_local_scraper(self, bridge_url: str = "http://host.docker.internal:8000/api/serp-bridge") -> bool:
        """
        Configure SerpBear to use our local scraper bridge.
        
        Args:
            bridge_url: URL of our scraper bridge API
            
        Returns:
            True if configuration successful
        """
        try:
            # Update settings.json
            settings_updated = self._update_settings_json(bridge_url)
            
            # Update database configuration if needed
            db_updated = self._update_database_config(bridge_url)
            
            if settings_updated and db_updated:
                logger.info("SerpBear successfully configured to use local scraper")
                return True
            else:
                logger.error("Failed to configure SerpBear for local scraper")
                return False
                
        except Exception as e:
            logger.error(f"Error configuring SerpBear: {e}")
            return False
    
    def _update_settings_json(self, bridge_url: str) -> bool:
        """Update SerpBear settings.json file."""
        try:
            # Read current settings
            settings = {}
            if os.path.exists(self.settings_path):
                with open(self.settings_path, 'r') as f:
                    settings = json.load(f)
            
            # Update scraper configuration
            settings.update({
                "scraper_type": "custom",
                "custom_scraper_url": bridge_url,
                "custom_scraper_enabled": True,
                "scrape_retry": True,
                "notification_interval": "daily",
                "last_updated": datetime.utcnow().isoformat()
            })
            
            # Write updated settings
            with open(self.settings_path, 'w') as f:
                json.dump(settings, f, indent=2)
            
            logger.info("Updated SerpBear settings.json for local scraper")
            return True
            
        except Exception as e:
            logger.error(f"Error updating settings.json: {e}")
            return False
    
    def _update_database_config(self, bridge_url: str) -> bool:
        """Update SerpBear database configuration."""
        try:
            if not os.path.exists(self.db_path):
                logger.warning(f"SerpBear database not found at {self.db_path}")
                return True  # Database might not be created yet
            
            # Connect to SerpBear database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if settings table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='settings'")
            if not cursor.fetchone():
                # Create settings table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS settings (
                        id INTEGER PRIMARY KEY,
                        key TEXT UNIQUE,
                        value TEXT,
                        created_at TEXT,
                        updated_at TEXT
                    )
                """)
            
            # Update or insert scraper configuration
            now = datetime.utcnow().isoformat()
            
            scraper_config = {
                "type": "custom",
                "url": bridge_url,
                "enabled": True,
                "retry_failed": True
            }
            
            cursor.execute("""
                INSERT OR REPLACE INTO settings (key, value, created_at, updated_at)
                VALUES (?, ?, ?, ?)
            """, ("scraper_config", json.dumps(scraper_config), now, now))
            
            # Commit changes
            conn.commit()
            conn.close()
            
            logger.info("Updated SerpBear database configuration")
            return True
            
        except Exception as e:
            logger.error(f"Error updating database config: {e}")
            return False
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current SerpBear configuration."""
        try:
            config = {
                "settings_file": {},
                "database_config": {},
                "status": "unknown"
            }
            
            # Read settings.json
            if os.path.exists(self.settings_path):
                with open(self.settings_path, 'r') as f:
                    config["settings_file"] = json.load(f)
            
            # Read database config
            if os.path.exists(self.db_path):
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("SELECT key, value FROM settings WHERE key = 'scraper_config'")
                result = cursor.fetchone()
                
                if result:
                    config["database_config"] = json.loads(result[1])
                
                conn.close()
            
            # Determine status
            if (config["settings_file"].get("scraper_type") == "custom" or 
                config["database_config"].get("type") == "custom"):
                config["status"] = "configured_for_local_scraper"
            else:
                config["status"] = "default_configuration"
            
            return config
            
        except Exception as e:
            logger.error(f"Error reading current config: {e}")
            return {"status": "error", "error": str(e)}
    
    def is_configured_for_local_scraper(self) -> bool:
        """Check if SerpBear is configured for local scraper."""
        try:
            config = self.get_current_config()
            return config.get("status") == "configured_for_local_scraper"
        except:
            return False
    
    def reset_configuration(self) -> bool:
        """Reset SerpBear to default configuration."""
        try:
            # Reset settings.json
            default_settings = {
                "scraper_type": "none",
                "notification_interval": "never",
                "notification_email": "",
                "smtp_server": "",
                "smtp_port": "",
                "smtp_username": "",
                "smtp_password": "",
                "scrape_retry": False,
                "screenshot_key": "69408-serpbear",
                "search_console": True,
                "search_console_client_email": "",
                "search_console_private_key": "",
                "keywordsColumns": ["Best", "History", "Volume", "Search Console"]
            }
            
            with open(self.settings_path, 'w') as f:
                json.dump(default_settings, f, indent=2)
            
            # Reset database config
            if os.path.exists(self.db_path):
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM settings WHERE key = 'scraper_config'")
                
                conn.commit()
                conn.close()
            
            logger.info("Reset SerpBear configuration to defaults")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting configuration: {e}")
            return False


# Global configurator instance (will be initialized when Docker volume is available)
serpbear_configurator = None


def initialize_configurator(data_path: str = "/app/data") -> SerpBearConfigurator:
    """Initialize the SerpBear configurator with the correct data path."""
    global serpbear_configurator
    serpbear_configurator = SerpBearConfigurator(data_path)
    return serpbear_configurator


def configure_serpbear_for_local_scraper(bridge_url: str = "http://host.docker.internal:8000/api/serp-bridge") -> bool:
    """
    Configure SerpBear to use local scraper.
    
    This is a convenience function that can be called from other services.
    """
    if not serpbear_configurator:
        # Try to initialize with default Docker volume path
        initialize_configurator()
    
    if serpbear_configurator:
        return serpbear_configurator.configure_local_scraper(bridge_url)
    else:
        logger.error("SerpBear configurator not initialized")
        return False


if __name__ == "__main__":
    # Test the configurator
    configurator = SerpBearConfigurator("/tmp/test_serpbear")
    
    print("🔧 Testing SerpBear Configurator")
    print("=" * 40)
    
    # Create test directories
    os.makedirs("/tmp/test_serpbear", exist_ok=True)
    
    # Test configuration
    print("\n1. Configuring for local scraper...")
    result = configurator.configure_local_scraper("http://localhost:8000/api/serp-bridge")
    print(f"   Result: {'✅ Success' if result else '❌ Failed'}")
    
    # Test reading config
    print("\n2. Reading current configuration...")
    config = configurator.get_current_config()
    print(f"   Status: {config.get('status')}")
    print(f"   Scraper Type: {config.get('settings_file', {}).get('scraper_type', 'none')}")
    
    # Test check function
    print("\n3. Checking local scraper configuration...")
    is_configured = configurator.is_configured_for_local_scraper()
    print(f"   Configured for local scraper: {'✅ Yes' if is_configured else '❌ No'}")
    
    print("\n🎉 SerpBear Configurator test completed!")
    
    # Cleanup
    import shutil
    shutil.rmtree("/tmp/test_serpbear", ignore_errors=True)