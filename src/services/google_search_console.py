"""
Google Search Console API integration service.

This service handles authentication and data retrieval from Google Search Console
for multi-domain SEO monitoring and performance analysis.
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)


class GoogleSearchConsoleService:
    """Service for Google Search Console API operations."""
    
    def __init__(self):
        """Initialize the GSC service with API credentials."""
        self.client_config = {
            "web": {
                "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")]
            }
        }
        self.scopes = ['https://www.googleapis.com/auth/webmasters.readonly']
        self.service = None
        
    def get_authorization_url(self) -> str:
        """
        Get authorization URL for Google Search Console access.
        
        Returns:
            str: Authorization URL for user consent
        """
        try:
            flow = Flow.from_client_config(
                self.client_config,
                scopes=self.scopes,
                redirect_uri=self.client_config["web"]["redirect_uris"][0]
            )
            
            auth_url, _ = flow.authorization_url(
                access_type='offline',
                include_granted_scopes='true'
            )
            
            return auth_url
            
        except Exception as e:
            logger.error(f"Failed to generate authorization URL: {e}")
            raise Exception(f"Authorization URL generation failed: {str(e)}")
    
    def authenticate_with_code(self, authorization_code: str) -> Dict[str, Any]:
        """
        Exchange authorization code for access tokens.
        
        Args:
            authorization_code (str): Authorization code from Google OAuth callback
            
        Returns:
            Dict[str, Any]: Token information including access token and refresh token
        """
        try:
            flow = Flow.from_client_config(
                self.client_config,
                scopes=self.scopes,
                redirect_uri=self.client_config["web"]["redirect_uris"][0]
            )
            
            flow.fetch_token(code=authorization_code)
            credentials = flow.credentials
            
            self.service = build('webmasters', 'v3', credentials=credentials)
            
            return {
                "access_token": credentials.token,
                "refresh_token": credentials.refresh_token,
                "expires_at": credentials.expiry.isoformat() if credentials.expiry else None
            }
            
        except Exception as e:
            logger.error(f"Failed to authenticate with code: {e}")
            raise Exception(f"Authentication failed: {str(e)}")
    
    def initialize_service(self, access_token: str, refresh_token: str) -> None:
        """
        Initialize GSC service with stored credentials.
        
        Args:
            access_token (str): Valid access token
            refresh_token (str): Refresh token for token renewal
        """
        try:
            credentials = Credentials(
                token=access_token,
                refresh_token=refresh_token,
                token_uri=self.client_config["web"]["token_uri"],
                client_id=self.client_config["web"]["client_id"],
                client_secret=self.client_config["web"]["client_secret"],
                scopes=self.scopes
            )
            
            self.service = build('webmasters', 'v3', credentials=credentials)
            
        except Exception as e:
            logger.error(f"Failed to initialize service: {e}")
            raise Exception(f"Service initialization failed: {str(e)}")
    
    def get_verified_sites(self) -> List[Dict[str, Any]]:
        """
        Get list of verified sites in Google Search Console.
        
        Returns:
            List[Dict[str, Any]]: List of verified sites with their verification status
        """
        if not self.service:
            raise Exception("GSC service not initialized. Please authenticate first.")
        
        try:
            sites_list = self.service.sites().list().execute()
            
            verified_sites = []
            for site in sites_list.get('siteEntry', []):
                site_url = site.get('siteUrl', '')
                permission_level = site.get('permissionLevel', 'NONE')
                
                verified_sites.append({
                    "domain": site_url.replace('https://', '').replace('http://', '').replace('sc-domain:', ''),
                    "site_url": site_url,
                    "verified": permission_level in ['siteOwner', 'siteFullUser', 'siteRestrictedUser'],
                    "permission_level": permission_level,
                    "verification_methods": self._get_verification_methods(site_url) if permission_level == 'NONE' else []
                })
            
            return verified_sites
            
        except Exception as e:
            logger.error(f"Failed to get verified sites: {e}")
            raise Exception(f"Failed to retrieve sites: {str(e)}")
    
    def get_search_analytics(
        self, 
        site_url: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        dimensions: List[str] = None,
        row_limit: int = 1000
    ) -> Dict[str, Any]:
        """
        Get search analytics data for a specific site.
        
        Args:
            site_url (str): Site URL to get data for
            start_date (datetime, optional): Start date for data range
            end_date (datetime, optional): End date for data range
            dimensions (List[str], optional): Dimensions to group by (query, page, country, device, date)
            row_limit (int): Maximum number of rows to return
            
        Returns:
            Dict[str, Any]: Search analytics data
        """
        if not self.service:
            raise Exception("GSC service not initialized. Please authenticate first.")
        
        try:
            # Default date range (last 30 days)
            if not end_date:
                end_date = datetime.now() - timedelta(days=3)  # GSC has 3-day delay
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            # Default dimensions
            if not dimensions:
                dimensions = ['query', 'page']
            
            request_body = {
                'startDate': start_date.strftime('%Y-%m-%d'),
                'endDate': end_date.strftime('%Y-%m-%d'),
                'dimensions': dimensions,
                'rowLimit': row_limit,
                'startRow': 0
            }
            
            response = self.service.searchanalytics().query(
                siteUrl=site_url,
                body=request_body
            ).execute()
            
            return {
                "success": True,
                "data": response.get('rows', []),
                "total_rows": len(response.get('rows', [])),
                "date_range": {
                    "start_date": start_date.strftime('%Y-%m-%d'),
                    "end_date": end_date.strftime('%Y-%m-%d')
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get search analytics for {site_url}: {e}")
            raise Exception(f"Search analytics retrieval failed: {str(e)}")
    
    def get_performance_metrics(self, site_url: str) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for a site.
        
        Args:
            site_url (str): Site URL to get metrics for
            
        Returns:
            Dict[str, Any]: Performance metrics including clicks, impressions, CTR, position
        """
        try:
            # Get overall metrics
            overall_data = self.get_search_analytics(
                site_url=site_url,
                dimensions=[],  # No dimensions for overall metrics
                row_limit=1
            )
            
            # Get top queries
            query_data = self.get_search_analytics(
                site_url=site_url,
                dimensions=['query'],
                row_limit=50
            )
            
            # Get top pages
            page_data = self.get_search_analytics(
                site_url=site_url,
                dimensions=['page'],
                row_limit=50
            )
            
            # Calculate metrics
            overall_row = overall_data.get('data', [{}])[0] if overall_data.get('data') else {}
            
            return {
                "success": True,
                "domain": site_url.replace('https://', '').replace('http://', ''),
                "metrics": {
                    "total_clicks": overall_row.get('clicks', 0),
                    "total_impressions": overall_row.get('impressions', 0),
                    "average_ctr": round(overall_row.get('ctr', 0) * 100, 2),
                    "average_position": round(overall_row.get('position', 0), 1)
                },
                "top_queries": [
                    {
                        "query": row.get('keys', [''])[0] if row.get('keys') else '',
                        "clicks": row.get('clicks', 0),
                        "impressions": row.get('impressions', 0),
                        "ctr": round(row.get('ctr', 0) * 100, 2),
                        "position": round(row.get('position', 0), 1)
                    }
                    for row in query_data.get('data', [])[:10]
                ],
                "top_pages": [
                    {
                        "page": row.get('keys', [''])[0] if row.get('keys') else '',
                        "clicks": row.get('clicks', 0),
                        "impressions": row.get('impressions', 0),
                        "ctr": round(row.get('ctr', 0) * 100, 2),
                        "position": round(row.get('position', 0), 1)
                    }
                    for row in page_data.get('data', [])[:10]
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics for {site_url}: {e}")
            return {
                "success": False,
                "error": f"Performance metrics retrieval failed: {str(e)}",
                "domain": site_url.replace('https://', '').replace('http://', '')
            }
    
    def _get_verification_methods(self, site_url: str) -> List[str]:
        """
        Get available verification methods for a site.
        
        Args:
            site_url (str): Site URL to get verification methods for
            
        Returns:
            List[str]: Available verification methods
        """
        # Standard verification methods for different site types
        if site_url.startswith('sc-domain:'):
            return ['DNS record verification']
        else:
            return [
                'HTML file upload',
                'HTML tag in head section',
                'Google Analytics tracking code',
                'Google Tag Manager container',
                'DNS record (for domain properties)'
            ]
    
    def add_site(self, site_url: str) -> Dict[str, Any]:
        """
        Add a new site to Google Search Console.
        
        Args:
            site_url (str): Site URL to add
            
        Returns:
            Dict[str, Any]: Result of site addition
        """
        if not self.service:
            raise Exception("GSC service not initialized. Please authenticate first.")
        
        try:
            # Normalize URL format
            if not site_url.startswith(('http://', 'https://', 'sc-domain:')):
                # For domain properties, use sc-domain: prefix
                if '/' not in site_url or site_url.count('/') == 0:
                    site_url = f'sc-domain:{site_url}'
                else:
                    site_url = f'https://{site_url}' if not site_url.startswith('http') else site_url
            
            # Add site to Search Console
            self.service.sites().add(siteUrl=site_url).execute()
            
            return {
                "success": True,
                "domain": site_url.replace('https://', '').replace('http://', '').replace('sc-domain:', ''),
                "site_url": site_url,
                "status": "verification_pending",
                "verification_methods": self._get_verification_methods(site_url)
            }
            
        except Exception as e:
            logger.error(f"Failed to add site {site_url}: {e}")
            return {
                "success": False,
                "error": f"Failed to add site: {str(e)}",
                "domain": site_url
            }


# Initialize global service instance
gsc_service = GoogleSearchConsoleService()