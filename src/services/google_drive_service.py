"""
Google Drive Integration Service for the SEO Content Knowledge Graph System.

This module provides Google Drive integration for content brief management
with OAuth2 authentication, file synchronization, and webhook notifications.
"""

import asyncio
import json
import mimetypes
import os
import re
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlencode, urlparse

import aiofiles
import aiohttp
import structlog
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from pydantic_ai import Agent

from models.workflow_models import ContentBrief, ContentBriefType, Priority, WorkflowStatus
from config.settings import get_settings

logger = structlog.get_logger(__name__)


class GoogleDriveError(Exception):
    """Raised when Google Drive operations fail."""
    pass


class GoogleDriveAuthError(GoogleDriveError):
    """Raised when authentication fails."""
    pass


class GoogleDriveService:
    """
    Google Drive integration service for content brief management.
    
    Provides OAuth2 authentication, file synchronization, content brief parsing,
    and webhook notifications for Google Drive file changes.
    """
    
    # Google Drive API scopes
    SCOPES = [
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/drive.file',
        'https://www.googleapis.com/auth/drive.metadata'
    ]
    
    # Supported file types for content briefs
    SUPPORTED_MIME_TYPES = {
        'application/vnd.google-apps.document': 'google_doc',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
        'text/plain': 'txt',
        'text/markdown': 'md'
    }
    
    def __init__(self, credentials_file: Optional[str] = None):
        """
        Initialize Google Drive service.
        
        Args:
            credentials_file: Path to OAuth2 credentials file
        """
        self.settings = get_settings()
        self.credentials_file = credentials_file or os.getenv('GOOGLE_DRIVE_CREDENTIALS_FILE')
        self.token_file = os.getenv('GOOGLE_DRIVE_TOKEN_FILE', 'google_drive_token.json')
        
        # Service instances
        self.credentials: Optional[Credentials] = None
        self.service = None
        
        # Webhook configuration
        self.webhook_url: Optional[str] = None
        self.webhook_token: Optional[str] = None
        
        # AI agent for content brief parsing
        self.parsing_agent = Agent(
            'openai:gpt-4o-mini',
            system_prompt=self._get_parsing_system_prompt()
        )
        
        # Cache for processed files
        self.processed_files_cache: Dict[str, datetime] = {}
        
        logger.info("Google Drive service initialized")
    
    def _get_parsing_system_prompt(self) -> str:
        """Get system prompt for content brief parsing."""
        return """
You are an AI assistant specialized in extracting structured content brief information from documents.

Your task is to parse content briefs from various document formats (Google Docs, Word, text files) 
and extract the following information:

1. Title: The main title or heading of the content brief
2. Content Type: Type of content (blog_post, article, guide, etc.)
3. Target Keywords: List of SEO keywords to target
4. Target Audience: Description of the intended audience
5. Tone: Writing tone and style (professional, casual, etc.)
6. Word Count: Target word count for the content
7. Outline Requirements: Specific outline or structure requirements
8. Key Points: Important points that must be covered
9. Meta Description: SEO meta description if specified
10. Reference URLs: Any reference links or sources
11. Deadline: Content deadline if mentioned
12. Priority: Priority level (high, medium, low)

Parse the document content and return a structured JSON response with these fields.
If a field is not clearly specified, use reasonable defaults or leave it null.
Be intelligent about inferring content type from context and keywords from the content.

Example response format:
{
    "title": "Complete Guide to Content Marketing",
    "content_type": "guide",
    "target_keywords": ["content marketing", "digital marketing", "SEO"],
    "target_audience": "Marketing professionals and business owners",
    "tone": "professional and informative",
    "word_count": 2000,
    "outline_requirements": ["Introduction", "Core strategies", "Best practices", "Conclusion"],
    "key_points": ["Define content marketing", "Explain benefits", "Provide examples"],
    "meta_description": null,
    "reference_urls": ["https://example.com/reference"],
    "deadline": null,
    "priority": "medium"
}
"""
    
    async def initialize(self) -> None:
        """Initialize Google Drive service with authentication."""
        try:
            await self._authenticate()
            await self._build_service()
            logger.info("Google Drive service authenticated successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google Drive service: {e}")
            raise GoogleDriveAuthError(f"Authentication failed: {e}")
    
    async def _authenticate(self) -> None:
        """Authenticate with Google Drive API using OAuth2."""
        try:
            # Try to load existing credentials
            if os.path.exists(self.token_file):
                self.credentials = Credentials.from_authorized_user_file(
                    self.token_file, self.SCOPES
                )
            
            # If credentials are not valid, refresh or re-authenticate
            if not self.credentials or not self.credentials.valid:
                if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                    # Refresh credentials
                    self.credentials.refresh(Request())
                else:
                    # Run OAuth2 flow
                    if not self.credentials_file or not os.path.exists(self.credentials_file):
                        raise GoogleDriveAuthError("Credentials file not found")
                    
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_file, self.SCOPES
                    )
                    self.credentials = flow.run_local_server(port=0)
                
                # Save credentials for future use
                with open(self.token_file, 'w') as token:
                    token.write(self.credentials.to_json())
            
        except Exception as e:
            raise GoogleDriveAuthError(f"Authentication failed: {e}")
    
    async def _build_service(self) -> None:
        """Build Google Drive service instance."""
        try:
            self.service = build('drive', 'v3', credentials=self.credentials)
        except Exception as e:
            raise GoogleDriveError(f"Failed to build Drive service: {e}")
    
    async def sync_content_briefs(self, 
                                tenant_id: str, 
                                folder_id: Optional[str] = None,
                                folder_name: Optional[str] = None) -> List[ContentBrief]:
        """
        Sync content briefs from Google Drive folder.
        
        Args:
            tenant_id: Tenant identifier
            folder_id: Specific folder ID to sync
            folder_name: Folder name to search for
            
        Returns:
            List of synchronized content briefs
        """
        try:
            if not self.service:
                await self.initialize()
            
            # Find folder if not specified
            if not folder_id and folder_name:
                folder_id = await self._find_folder_by_name(folder_name)
            
            if not folder_id:
                raise GoogleDriveError("No folder specified for sync")
            
            # Get files from folder
            files = await self._get_files_from_folder(folder_id)
            
            # Filter supported file types
            supported_files = [
                file for file in files 
                if file.get('mimeType') in self.SUPPORTED_MIME_TYPES
            ]
            
            logger.info(
                f"Found {len(supported_files)} supported files in folder",
                folder_id=folder_id,
                total_files=len(files)
            )
            
            # Process each file
            briefs = []
            for file_metadata in supported_files:
                try:
                    brief = await self._process_file_to_brief(file_metadata, tenant_id)
                    if brief:
                        briefs.append(brief)
                        
                        # Update cache
                        self.processed_files_cache[file_metadata['id']] = datetime.now(timezone.utc)
                        
                except Exception as e:
                    logger.warning(
                        f"Failed to process file {file_metadata.get('name', 'unknown')}: {e}",
                        file_id=file_metadata.get('id')
                    )
                    continue
            
            logger.info(
                f"Successfully synced {len(briefs)} content briefs",
                tenant_id=tenant_id,
                folder_id=folder_id
            )
            
            return briefs
            
        except Exception as e:
            logger.error(f"Failed to sync content briefs: {e}")
            raise GoogleDriveError(f"Sync failed: {e}")
    
    async def _find_folder_by_name(self, folder_name: str) -> Optional[str]:
        """Find folder ID by name."""
        try:
            query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
            results = self.service.files().list(
                q=query,
                fields='files(id, name)'
            ).execute()
            
            folders = results.get('files', [])
            if folders:
                return folders[0]['id']
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to find folder '{folder_name}': {e}")
            return None
    
    async def _get_files_from_folder(self, folder_id: str) -> List[Dict[str, Any]]:
        """Get all files from a Google Drive folder."""
        try:
            query = f"'{folder_id}' in parents and trashed=false"
            results = self.service.files().list(
                q=query,
                fields='files(id, name, mimeType, modifiedTime, createdTime, owners)',
                pageSize=100
            ).execute()
            
            files = results.get('files', [])
            
            # Handle pagination
            while 'nextPageToken' in results:
                results = self.service.files().list(
                    q=query,
                    fields='files(id, name, mimeType, modifiedTime, createdTime, owners)',
                    pageSize=100,
                    pageToken=results['nextPageToken']
                ).execute()
                files.extend(results.get('files', []))
            
            return files
            
        except Exception as e:
            logger.error(f"Failed to get files from folder {folder_id}: {e}")
            raise GoogleDriveError(f"Failed to get files: {e}")
    
    async def _process_file_to_brief(self, 
                                   file_metadata: Dict[str, Any], 
                                   tenant_id: str) -> Optional[ContentBrief]:
        """Process a Google Drive file into a content brief."""
        try:
            file_id = file_metadata['id']
            file_name = file_metadata['name']
            mime_type = file_metadata['mimeType']
            
            logger.info(f"Processing file: {file_name}", file_id=file_id)
            
            # Download file content
            content = await self._download_file_content(file_id, mime_type)
            
            if not content:
                logger.warning(f"No content extracted from file: {file_name}")
                return None
            
            # Parse content brief using AI
            brief_data = await self._parse_content_brief_with_ai(content, file_name)
            
            if not brief_data:
                logger.warning(f"Failed to parse brief from file: {file_name}")
                return None
            
            # Create content brief
            brief = ContentBrief(
                title=brief_data.get('title', file_name),
                content_type=self._map_content_type(brief_data.get('content_type', 'article')),
                target_keywords=brief_data.get('target_keywords', []),
                target_audience=brief_data.get('target_audience', ''),
                tone=brief_data.get('tone', 'professional'),
                word_count=brief_data.get('word_count', 1000),
                outline_requirements=brief_data.get('outline_requirements', []),
                key_points=brief_data.get('key_points', []),
                meta_description=brief_data.get('meta_description'),
                reference_urls=brief_data.get('reference_urls', []),
                deadline=self._parse_deadline(brief_data.get('deadline')),
                priority=self._map_priority(brief_data.get('priority', 'medium')),
                google_drive_id=file_id,
                google_drive_url=f"https://drive.google.com/file/d/{file_id}/view",
                last_synced=datetime.now(timezone.utc),
                tenant_id=tenant_id,
                created_by=self._get_file_owner(file_metadata),
                status=WorkflowStatus.DRAFT
            )
            
            # Calculate completion percentage
            brief.calculate_completion_percentage()
            
            return brief
            
        except Exception as e:
            logger.error(f"Failed to process file {file_metadata.get('name', 'unknown')}: {e}")
            return None
    
    async def _download_file_content(self, file_id: str, mime_type: str) -> Optional[str]:
        """Download content from Google Drive file."""
        try:
            if mime_type == 'application/vnd.google-apps.document':
                # Export Google Doc as plain text
                request = self.service.files().export_media(
                    fileId=file_id,
                    mimeType='text/plain'
                )
            else:
                # Download file directly
                request = self.service.files().get_media(fileId=file_id)
            
            # Download content
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                downloader = MediaIoBaseDownload(temp_file, request)
                done = False
                
                while not done:
                    status, done = downloader.next_chunk()
                
                temp_file_path = temp_file.name
            
            # Read content
            try:
                async with aiofiles.open(temp_file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
            except UnicodeDecodeError:
                # Try with different encoding
                async with aiofiles.open(temp_file_path, 'r', encoding='latin-1') as f:
                    content = await f.read()
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
            return content
            
        except Exception as e:
            logger.error(f"Failed to download file content {file_id}: {e}")
            return None
    
    async def _parse_content_brief_with_ai(self, content: str, file_name: str) -> Optional[Dict[str, Any]]:
        """Parse content brief using AI agent."""
        try:
            # Prepare prompt
            prompt = f"""
            Parse the following content brief document and extract structured information:

            File name: {file_name}
            
            Content:
            {content[:5000]}  # Limit content to avoid token limits
            
            Please provide a JSON response with the structured brief information.
            """
            
            # Run AI agent
            result = await self.parsing_agent.run(prompt)
            
            if result.data:
                try:
                    # Try to parse as JSON
                    if isinstance(result.data, str):
                        brief_data = json.loads(result.data)
                    else:
                        brief_data = result.data
                    
                    return brief_data
                    
                except json.JSONDecodeError:
                    # If not valid JSON, try to extract from text
                    return self._extract_brief_from_text(result.data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to parse content brief with AI: {e}")
            return None
    
    def _extract_brief_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract brief data from text response."""
        try:
            # Simple regex-based extraction as fallback
            brief_data = {}
            
            # Extract title
            title_match = re.search(r'title["\']?\s*:\s*["\']?([^"\'\\n]+)', text, re.IGNORECASE)
            if title_match:
                brief_data['title'] = title_match.group(1).strip()
            
            # Extract keywords
            keywords_match = re.search(r'keywords["\']?\s*:\s*\[([^\]]+)\]', text, re.IGNORECASE)
            if keywords_match:
                keywords_str = keywords_match.group(1)
                keywords = [kw.strip().strip('"\'') for kw in keywords_str.split(',')]
                brief_data['target_keywords'] = keywords
            
            # Extract word count
            word_count_match = re.search(r'word_count["\']?\s*:\s*(\d+)', text, re.IGNORECASE)
            if word_count_match:
                brief_data['word_count'] = int(word_count_match.group(1))
            
            return brief_data if brief_data else None
            
        except Exception as e:
            logger.error(f"Failed to extract brief from text: {e}")
            return None
    
    def _map_content_type(self, content_type: str) -> ContentBriefType:
        """Map content type string to enum."""
        type_mapping = {
            'blog_post': ContentBriefType.BLOG_POST,
            'blog': ContentBriefType.BLOG_POST,
            'article': ContentBriefType.ARTICLE,
            'guide': ContentBriefType.GUIDE,
            'tutorial': ContentBriefType.TUTORIAL,
            'case_study': ContentBriefType.CASE_STUDY,
            'whitepaper': ContentBriefType.WHITEPAPER,
            'landing_page': ContentBriefType.LANDING_PAGE,
            'product_page': ContentBriefType.PRODUCT_PAGE,
            'news': ContentBriefType.NEWS,
            'press_release': ContentBriefType.PRESS_RELEASE
        }
        
        return type_mapping.get(content_type.lower(), ContentBriefType.ARTICLE)
    
    def _map_priority(self, priority: str) -> Priority:
        """Map priority string to enum."""
        priority_mapping = {
            'low': Priority.LOW,
            'medium': Priority.MEDIUM,
            'high': Priority.HIGH,
            'urgent': Priority.URGENT
        }
        
        return priority_mapping.get(priority.lower(), Priority.MEDIUM)
    
    def _parse_deadline(self, deadline_str: Optional[str]) -> Optional[datetime]:
        """Parse deadline string to datetime."""
        if not deadline_str:
            return None
        
        try:
            # Try common date formats
            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']:
                try:
                    return datetime.strptime(deadline_str, fmt).replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to parse deadline '{deadline_str}': {e}")
            return None
    
    def _get_file_owner(self, file_metadata: Dict[str, Any]) -> str:
        """Get file owner from metadata."""
        owners = file_metadata.get('owners', [])
        if owners:
            return owners[0].get('emailAddress', 'unknown')
        return 'unknown'
    
    async def setup_webhook(self, webhook_url: str, folder_id: str) -> bool:
        """
        Set up webhook for Google Drive file changes.
        
        Args:
            webhook_url: Webhook URL to receive notifications
            folder_id: Folder ID to monitor
            
        Returns:
            True if webhook was set up successfully
        """
        try:
            if not self.service:
                await self.initialize()
            
            # Create webhook channel
            channel_id = f"channel_{folder_id}_{int(datetime.now().timestamp())}"
            
            body = {
                'id': channel_id,
                'type': 'web_hook',
                'address': webhook_url,
                'payload': True
            }
            
            # Set up file watch
            response = self.service.files().watch(
                fileId=folder_id,
                body=body
            ).execute()
            
            self.webhook_url = webhook_url
            self.webhook_token = response.get('resourceId')
            
            logger.info(
                f"Webhook set up successfully",
                channel_id=channel_id,
                resource_id=self.webhook_token
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set up webhook: {e}")
            return False
    
    async def handle_webhook_notification(self, notification_data: Dict[str, Any]) -> bool:
        """
        Handle webhook notification from Google Drive.
        
        Args:
            notification_data: Notification data from webhook
            
        Returns:
            True if notification was handled successfully
        """
        try:
            # Extract file information
            file_id = notification_data.get('fileId')
            if not file_id:
                logger.warning("No file ID in webhook notification")
                return False
            
            # Check if file was recently processed
            if file_id in self.processed_files_cache:
                last_processed = self.processed_files_cache[file_id]
                if (datetime.now(timezone.utc) - last_processed).total_seconds() < 300:
                    logger.info(f"File {file_id} processed recently, skipping")
                    return True
            
            # Get file metadata
            file_metadata = self.service.files().get(
                fileId=file_id,
                fields='id, name, mimeType, modifiedTime, createdTime, owners'
            ).execute()
            
            # Process file if it's a supported type
            if file_metadata.get('mimeType') in self.SUPPORTED_MIME_TYPES:
                # This would typically trigger a re-sync process
                logger.info(
                    f"File {file_metadata.get('name')} changed, triggering sync",
                    file_id=file_id
                )
                
                # Update cache
                self.processed_files_cache[file_id] = datetime.now(timezone.utc)
                
                # Here you would typically trigger a background task to re-sync
                # For now, just log the event
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to handle webhook notification: {e}")
            return False
    
    async def upload_content_brief(self, 
                                 brief: ContentBrief, 
                                 folder_id: str,
                                 file_format: str = 'google_doc') -> Optional[str]:
        """
        Upload content brief to Google Drive.
        
        Args:
            brief: Content brief to upload
            folder_id: Destination folder ID
            file_format: File format (google_doc, docx, txt)
            
        Returns:
            File ID if successful, None otherwise
        """
        try:
            if not self.service:
                await self.initialize()
            
            # Generate content brief document
            content = self._generate_brief_document(brief)
            
            # Prepare file metadata
            file_metadata = {
                'name': f"{brief.title} - Content Brief",
                'parents': [folder_id]
            }
            
            # Set MIME type based on format
            if file_format == 'google_doc':
                file_metadata['mimeType'] = 'application/vnd.google-apps.document'
                upload_mime_type = 'text/plain'
            elif file_format == 'docx':
                file_metadata['mimeType'] = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                upload_mime_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            else:
                file_metadata['mimeType'] = 'text/plain'
                upload_mime_type = 'text/plain'
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            # Upload file
            try:
                media = MediaIoBaseUpload(
                    open(temp_file_path, 'rb'),
                    mimetype=upload_mime_type
                )
                
                file_result = self.service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id'
                ).execute()
                
                file_id = file_result.get('id')
                
                logger.info(
                    f"Content brief uploaded successfully",
                    brief_id=brief.id,
                    file_id=file_id
                )
                
                return file_id
                
            finally:
                # Clean up temp file
                os.unlink(temp_file_path)
            
        except Exception as e:
            logger.error(f"Failed to upload content brief: {e}")
            return None
    
    def _generate_brief_document(self, brief: ContentBrief) -> str:
        """Generate document content from content brief."""
        content = f"""
# {brief.title}

## Content Brief

**Content Type:** {brief.content_type.value}
**Target Audience:** {brief.target_audience}
**Tone:** {brief.tone}
**Word Count:** {brief.word_count}
**Priority:** {brief.priority.value}

### Target Keywords
{chr(10).join(f'- {keyword}' for keyword in brief.target_keywords)}

### Outline Requirements
{chr(10).join(f'- {req}' for req in brief.outline_requirements)}

### Key Points
{chr(10).join(f'- {point}' for point in brief.key_points)}

### Meta Description
{brief.meta_description or 'Not specified'}

### Reference URLs
{chr(10).join(f'- {url}' for url in brief.reference_urls)}

### Deadline
{brief.deadline.strftime('%Y-%m-%d') if brief.deadline else 'Not specified'}

### Additional Notes
{brief.metadata.get('notes', '')}

---
Generated by SEO Content Knowledge Graph System
Brief ID: {brief.id}
Created: {brief.created_at.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return content
    
    async def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        try:
            stats = {
                'authenticated': self.credentials is not None and self.credentials.valid,
                'service_initialized': self.service is not None,
                'webhook_configured': self.webhook_url is not None,
                'processed_files_count': len(self.processed_files_cache),
                'supported_mime_types': list(self.SUPPORTED_MIME_TYPES.keys())
            }
            
            # Add quota information if available
            if self.service:
                try:
                    about = self.service.about().get(fields='storageQuota').execute()
                    storage_quota = about.get('storageQuota', {})
                    stats['storage_quota'] = storage_quota
                except:
                    pass
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get service stats: {e}")
            return {'error': str(e)}


# =============================================================================
# Utility Functions
# =============================================================================

async def sync_briefs_from_drive(tenant_id: str, 
                                folder_name: str = "Content Briefs") -> List[ContentBrief]:
    """
    Simple function to sync content briefs from Google Drive.
    
    Args:
        tenant_id: Tenant identifier
        folder_name: Name of the folder containing briefs
        
    Returns:
        List of synchronized content briefs
    """
    service = GoogleDriveService()
    await service.initialize()
    
    return await service.sync_content_briefs(
        tenant_id=tenant_id,
        folder_name=folder_name
    )


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        # Initialize service
        service = GoogleDriveService()
        
        try:
            await service.initialize()
            
            # Get service stats
            stats = await service.get_service_stats()
            print(f"Service stats: {stats}")
            
            # Sync content briefs
            briefs = await service.sync_content_briefs(
                tenant_id="test-tenant",
                folder_name="Content Briefs"
            )
            
            print(f"Synced {len(briefs)} content briefs")
            
            for brief in briefs:
                print(f"- {brief.title} ({brief.content_type.value})")
                print(f"  Keywords: {', '.join(brief.target_keywords)}")
                print(f"  Status: {brief.status.value}")
                print(f"  Completion: {brief.completion_percentage:.1f}%")
                print()
            
        except Exception as e:
            print(f"Error: {e}")

    asyncio.run(main())