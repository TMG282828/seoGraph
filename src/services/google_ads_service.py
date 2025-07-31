"""
Google Ads API service for real keyword data.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)

class GoogleAdsService:
    """Service for Google Ads Keyword Planner API integration."""
    
    def __init__(self):
        """Initialize Google Ads service."""
        self.client_id = os.getenv("GOOGLE_ADS_CLIENT_ID")
        self.client_secret = os.getenv("GOOGLE_ADS_CLIENT_SECRET") 
        self.refresh_token = os.getenv("GOOGLE_ADS_REFRESH_TOKEN")
        self.developer_token = os.getenv("GOOGLE_ADS_DEVELOPER_TOKEN")
        self.customer_id = os.getenv("GOOGLE_ADS_CUSTOMER_ID")
        
        self.is_configured = all([
            self.client_id,
            self.client_secret, 
            self.refresh_token,
            self.developer_token,
            self.customer_id
        ])
        
        if not self.is_configured:
            logger.warning("Google Ads API not fully configured - using fallback data")
    
    async def get_keyword_ideas(
        self, 
        keywords: List[str], 
        country: str = "US", 
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Get keyword ideas and search volume data from Google Ads API.
        
        Args:
            keywords: List of seed keywords
            country: Country code (US, GB, etc.)
            language: Language code (en, es, etc.)
            
        Returns:
            Dict with keyword data including volumes and competition
        """
        try:
            if not self.is_configured:
                logger.warning("Google Ads API not configured - real data unavailable")
                return {
                    "success": False,
                    "keywords": [],
                    "total_results": 0,
                    "country": country,
                    "language": language,
                    "data_source": "Not Available",
                    "api_status": "not_configured",
                    "note": "❌ Google Ads API not configured - no search volume data available"
                }
            
            # Only use real Google Ads API - no fallbacks
            try:
                return await self._get_real_google_ads_data(keywords, country, language)
            except Exception as api_error:
                logger.error(f"Google Ads API call failed: {api_error}")
                return {
                    "success": False,
                    "keywords": [],
                    "total_results": 0,
                    "country": country,
                    "language": language,
                    "data_source": "API Error",
                    "api_status": "api_error",
                    "note": f"❌ Google Ads API error: {str(api_error)}",
                    "error": str(api_error)
                }
            
        except Exception as e:
            logger.error(f"Keyword research failed: {e}")
            return {
                "success": False,
                "keywords": [],
                "total_results": 0,
                "country": country,
                "language": language,
                "data_source": "Service Error",
                "api_status": "service_error",
                "note": f"❌ Service error: {str(e)}",
                "error": str(e)
            }
    
    async def _get_real_google_ads_data(
        self, 
        keywords: List[str], 
        country: str, 
        language: str
    ) -> Dict[str, Any]:
        """Get real keyword data from Google Ads API."""
        try:
            from google.ads.googleads.client import GoogleAdsClient
            
            # Create client
            client = GoogleAdsClient.load_from_env()
            keyword_plan_idea_service = client.get_service("KeywordPlanIdeaService")
            
            # Create request
            request = client.get_type("GenerateKeywordIdeasRequest")
            request.customer_id = self.customer_id
            
            # Set seed keywords
            request.keyword_and_url_seed.keywords = keywords
            
            # Set geo targeting (country)
            geo_target_map = {
                "US": "2840", "GB": "2826", "CA": "2124", "AU": "2036",
                "DE": "2276", "FR": "2250", "ES": "2724", "IT": "2380"
            }
            geo_target_id = geo_target_map.get(country, "2840")  # Default to US
            request.geo_target_constants.append(f"geoTargetConstants/{geo_target_id}")
            
            # Set language targeting
            language_map = {
                "en": "1000", "es": "1003", "fr": "1002", "de": "1001", "it": "1004"
            }
            language_id = language_map.get(language, "1000")  # Default to English
            request.language = f"languageConstants/{language_id}"
            
            # Set other parameters
            request.include_adult_keywords = False
            request.page_size = 1000  # Maximum allowed
            
            # Make the API call
            logger.info(f"Making Google Ads API call for {len(keywords)} keywords")
            response = keyword_plan_idea_service.generate_keyword_ideas(request=request)
            
            # Process results
            results = []
            for idea in response:
                keyword_text = idea.text
                metrics = idea.keyword_idea_metrics
                
                # Extract data
                volume = metrics.avg_monthly_searches if metrics.avg_monthly_searches else 0
                competition_level = metrics.competition.name if metrics.competition else "UNKNOWN"
                
                # Convert competition to difficulty score
                competition_difficulty = {
                    "LOW": 25,
                    "MEDIUM": 55,
                    "HIGH": 85,
                    "UNKNOWN": 50
                }.get(competition_level, 50)
                
                # Extract CPC data
                low_top_of_page_bid = 0
                high_top_of_page_bid = 0
                if metrics.low_top_of_page_bid_micros:
                    low_top_of_page_bid = metrics.low_top_of_page_bid_micros / 1_000_000
                if metrics.high_top_of_page_bid_micros:
                    high_top_of_page_bid = metrics.high_top_of_page_bid_micros / 1_000_000
                
                avg_cpc = (low_top_of_page_bid + high_top_of_page_bid) / 2 if high_top_of_page_bid > 0 else low_top_of_page_bid
                
                # Calculate opportunity score
                opportunity = self._calculate_opportunity_from_real_data(volume, competition_difficulty)
                
                # Classify intent (we still need to analyze the keyword text)
                intent = self._classify_intent(keyword_text, False, keyword_text.startswith(('how', 'what', 'why')))
                
                # Generate related keywords (from API or our logic)
                related_keywords = self._generate_related_keywords(keyword_text, "buy" in keyword_text, keyword_text.startswith(('how', 'what', 'why')))
                
                results.append({
                    "term": keyword_text,
                    "volume": volume,
                    "difficulty": int(competition_difficulty),
                    "cpc": round(avg_cpc, 2),
                    "competition": competition_level,
                    "opportunity": opportunity,
                    "intent": intent,
                    "related_keywords": related_keywords,
                    "data_source": "google_ads_api",
                    "confidence_score": 0.95,
                    "last_updated": datetime.now().isoformat()
                })
            
            logger.info(f"Retrieved {len(results)} keywords from Google Ads API")
            
            return {
                "success": True,
                "keywords": results,
                "total_results": len(results),
                "country": country,
                "language": language,
                "data_source": "Google Ads API",
                "api_status": "google_ads_api",
                "note": "✅ Real search volume data from Google Keyword Planner"
            }
            
        except ImportError:
            raise Exception("Google Ads library not installed")
        except Exception as e:
            raise Exception(f"Google Ads API error: {str(e)}")
    
    def _calculate_opportunity_from_real_data(self, volume: int, difficulty: int) -> str:
        """Calculate opportunity from real API data."""
        if volume > 10000 and difficulty < 40:
            return "High"
        elif volume > 1000 and difficulty < 60:
            return "Medium"
        else:
            return "Low"
    
    # REMOVED: Enhanced algorithmic fallback method - we only use real API data now
    
    def _classify_intent(self, keyword: str, has_commercial: bool, is_question: bool) -> str:
        """Classify search intent."""
        if has_commercial:
            return "Commercial"
        elif is_question:
            return "Informational"
        elif any(word in keyword for word in ['how to', 'guide', 'tutorial', 'learn']):
            return "Informational"
        elif any(word in keyword for word in ['brand', 'company', 'website']):
            return "Navigational"
        else:
            return "Informational"
    
    def _calculate_opportunity(self, volume: int, difficulty: int) -> str:
        """Calculate opportunity score."""
        score = (volume / 1000) / max(difficulty, 1) * 100
        
        if score > 50:
            return "High"
        elif score > 20:
            return "Medium"
        else:
            return "Low"
    
    def _generate_related_keywords(self, keyword: str, has_commercial: bool, is_question: bool) -> List[str]:
        """Generate contextual related keywords."""
        related = []
        
        if has_commercial:
            related.extend([
                f"best {keyword}",
                f"{keyword} price",
                f"{keyword} review",
                f"cheap {keyword}",
                f"{keyword} deals"
            ])
        elif is_question:
            related.extend([
                f"{keyword} guide",
                f"{keyword} tutorial", 
                f"{keyword} explained",
                f"{keyword} tips",
                f"learn {keyword}"
            ])
        else:
            related.extend([
                f"{keyword} guide",
                f"how to {keyword}",
                f"{keyword} tips",
                f"{keyword} benefits",
                f"{keyword} examples"
            ])
        
        return related[:5]  # Limit to 5 related keywords

# Global service instance
google_ads_service = GoogleAdsService()