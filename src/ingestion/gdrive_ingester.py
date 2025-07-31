"""
Google Drive Content Ingester for SEO Content Knowledge Graph System.

This module provides comprehensive Google Drive integration including:
- OAuth2 authentication and token management
- Document discovery and syncing
- Support for Google Docs, Sheets, Slides, and PDFs
- Incremental updates and change detection
- Folder-based organization and filtering
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
import json
import base64
from pathlib import Path

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import io

from .base_ingester import BaseIngester, RawContent, ContentSource
from ..database.supabase_client import supabase_client

logger = logging.getLogger(__name__)


class GoogleDriveAuth:
    """Google Drive OAuth2 authentication manager."""
    
    def __init__(self, credentials_file: str, token_storage_path: str):
        self.credentials_file = credentials_file
        self.token_storage_path = token_storage_path
        self.scopes = [
            'https://www.googleapis.com/auth/drive.readonly',
            'https://www.googleapis.com/auth/drive.metadata.readonly'
        ]
    
    def get_authorization_url(self, redirect_uri: str, state: str = None) -> str:
        """Get OAuth2 authorization URL."""
        flow = Flow.from_client_secrets_file(
            self.credentials_file,
            scopes=self.scopes,
            redirect_uri=redirect_uri
        )
        
        authorization_url, _ = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            state=state
        )
        
        return authorization_url
    
    def exchange_code_for_tokens(self, code: str, redirect_uri: str) -> Credentials:
        """Exchange authorization code for access tokens."""
        flow = Flow.from_client_secrets_file(
            self.credentials_file,
            scopes=self.scopes,
            redirect_uri=redirect_uri
        )
        
        flow.fetch_token(code=code)
        return flow.credentials
    
    def save_credentials(self, credentials: Credentials, organization_id: str):
        """Save credentials to storage."""
        try:
            token_data = {
                'organization_id': organization_id,
                'access_token': credentials.token,
                'refresh_token': credentials.refresh_token,
                'token_uri': credentials.token_uri,
                'client_id': credentials.client_id,
                'client_secret': credentials.client_secret,
                'scopes': credentials.scopes,
                'expires_at': credentials.expiry.isoformat() if credentials.expiry else None,
                'created_at': datetime.now().isoformat()
            }
            
            # Store in database
            supabase_client.client.table("gdrive_tokens").upsert(token_data, on_conflict="organization_id").execute()
            
            logger.info(f"Saved Google Drive credentials for organization {organization_id}")
            
        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")
            raise
    
    def load_credentials(self, organization_id: str) -> Optional[Credentials]:
        """Load credentials from storage."""
        try:
            result = supabase_client.client.table("gdrive_tokens").select("*").eq("organization_id", organization_id).execute()
            
            if not result.data:
                return None
            
            token_data = result.data[0]
            
            credentials = Credentials(
                token=token_data['access_token'],
                refresh_token=token_data['refresh_token'],
                token_uri=token_data['token_uri'],
                client_id=token_data['client_id'],
                client_secret=token_data['client_secret'],
                scopes=token_data['scopes']
            )
            
            if token_data['expires_at']:
                credentials.expiry = datetime.fromisoformat(token_data['expires_at'])
            
            # Refresh if needed
            if credentials.expired:
                credentials.refresh(Request())
                self.save_credentials(credentials, organization_id)
            
            return credentials
            
        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")
            return None


class GoogleDriveIngester(BaseIngester):
    """
    Google Drive content ingester with comprehensive document support.
    
    Features:
    - OAuth2 authentication and token management
    - Support for Google Docs, Sheets, Slides, and uploaded files
    - Incremental sync with change detection
    - Folder-based filtering and organization
    - Metadata extraction and preservation
    - Export format handling (PDF, DOCX, etc.)
    """
    
    def __init__(self, organization_id: str, credentials_file: str):
        super().__init__("gdrive", organization_id)
        self.auth_manager = GoogleDriveAuth(credentials_file, f"tokens/{organization_id}")
        self.drive_service = None
        self.credentials = None
        
        # Supported Google Workspace MIME types
        self.supported_google_types = {
            'application/vnd.google-apps.document': {
                'name': 'Google Docs',
                'export_format': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'file_extension': '.docx'
            },
            'application/vnd.google-apps.spreadsheet': {
                'name': 'Google Sheets',
                'export_format': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                'file_extension': '.xlsx'
            },
            'application/vnd.google-apps.presentation': {
                'name': 'Google Slides',
                'export_format': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                'file_extension': '.pptx'
            }
        }
        
        # Supported uploaded file types
        self.supported_file_types = {
            'application/pdf': {'name': 'PDF', 'extension': '.pdf'},
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': {'name': 'Word Document', 'extension': '.docx'},
            'application/msword': {'name': 'Word Document (Legacy)', 'extension': '.doc'},
            'text/plain': {'name': 'Text File', 'extension': '.txt'},
            'text/markdown': {'name': 'Markdown File', 'extension': '.md'}
        }
    
    async def initialize_service(self) -> bool:
        """Initialize Google Drive service with authentication."""
        try:
            self.credentials = self.auth_manager.load_credentials(self.organization_id)
            
            if not self.credentials:
                logger.warning(f"No Google Drive credentials found for organization {self.organization_id}")
                return False
            
            self.drive_service = build('drive', 'v3', credentials=self.credentials)
            
            # Test the connection
            about = self.drive_service.about().get(fields="user").execute()
            logger.info(f"Connected to Google Drive for user: {about['user']['emailAddress']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Google Drive service: {e}")
            return False
    
    async def validate_source(self, source_config: Dict[str, Any]) -> bool:
        """Validate Google Drive access and folder permissions."""
        try:
            if not await self.initialize_service():
                return False
            
            folder_id = source_config.get('folder_id', 'root')
            
            # Try to list files in the folder
            results = self.drive_service.files().list(
                q=f"'{folder_id}' in parents",
                pageSize=1,
                fields="files(id, name)"
            ).execute()
            
            return True
            
        except HttpError as e:
            logger.error(f"Google Drive validation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error validating Google Drive: {e}")
            return False
    
    async def extract_content(self, source_config: Dict[str, Any]) -> List[RawContent]:
        """Extract content from Google Drive."""
        if not self.drive_service:
            if not await self.initialize_service():
                raise Exception("Failed to initialize Google Drive service")
        
        folder_id = source_config.get('folder_id', 'root')
        config = source_config.get('config', {})
        
        # Discover files to process
        files_to_process = await self._discover_files(folder_id, config)
        
        self.logger.info(f"Found {len(files_to_process)} files to process in Google Drive")
        
        # Extract content from each file
        raw_contents = []
        
        for file_info in files_to_process:
            try:
                content = await self._extract_file_content(file_info, config)
                if content:
                    raw_contents.append(content)
                
            except Exception as e:
                self.logger.error(f"Failed to extract content from file {file_info['name']}: {e}")
        
        self.logger.info(f"Successfully extracted content from {len(raw_contents)} Google Drive files")
        return raw_contents
    
    async def _discover_files(self, folder_id: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Discover files in Google Drive folder."""
        files = []
        page_token = None
        
        try:
            while True:
                # Build query
                query_parts = [f"'{folder_id}' in parents"]
                
                # Filter by supported types
                supported_types = list(self.supported_google_types.keys()) + list(self.supported_file_types.keys())
                mime_type_query = " or ".join([f"mimeType='{mime_type}'" for mime_type in supported_types])
                query_parts.append(f"({mime_type_query})")
                
                # Exclude trashed files
                query_parts.append("trashed=false")
                
                query = " and ".join(query_parts)
                
                # Execute query
                results = self.drive_service.files().list(
                    q=query,
                    pageSize=100,
                    pageToken=page_token,
                    fields="nextPageToken, files(id, name, mimeType, size, modifiedTime, createdTime, webViewLink, parents, owners, description)"
                ).execute()
                
                files.extend(results.get('files', []))
                
                page_token = results.get('nextPageToken')
                if not page_token:
                    break
            
            # Filter files based on config
            if config.get('modified_since'):
                modified_since = datetime.fromisoformat(config['modified_since'])
                files = [f for f in files if datetime.fromisoformat(f['modifiedTime'].replace('Z', '+00:00')) > modified_since]
            
            if config.get('max_files'):
                files = files[:config['max_files']]
            
            # Sort by modification time (newest first)
            files.sort(key=lambda x: x['modifiedTime'], reverse=True)
            
            return files
            
        except HttpError as e:
            self.logger.error(f"Failed to discover Google Drive files: {e}")
            return []
    
    async def _extract_file_content(self, file_info: Dict[str, Any], config: Dict[str, Any]) -> Optional[RawContent]:
        """Extract content from a single Google Drive file."""
        try:
            file_id = file_info['id']
            file_name = file_info['name']
            mime_type = file_info['mimeType']
            
            self.logger.info(f"Extracting content from {file_name} ({mime_type})")
            
            # Extract content based on file type
            if mime_type in self.supported_google_types:
                return await self._extract_google_workspace_content(file_info)
            elif mime_type in self.supported_file_types:
                return await self._extract_uploaded_file_content(file_info)
            else:
                self.logger.warning(f"Unsupported file type: {mime_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to extract content from file {file_info['name']}: {e}")
            return None
    
    async def _extract_google_workspace_content(self, file_info: Dict[str, Any]) -> Optional[RawContent]:
        """Extract content from Google Workspace files (Docs, Sheets, Slides)."""
        try:
            file_id = file_info['id']
            mime_type = file_info['mimeType']
            type_info = self.supported_google_types[mime_type]
            
            # Export as text for content analysis
            if mime_type == 'application/vnd.google-apps.document':
                # Export Google Doc as plain text
                export_result = self.drive_service.files().export(
                    fileId=file_id,
                    mimeType='text/plain'
                ).execute()
                
                content_text = export_result.decode('utf-8')
                
            elif mime_type == 'application/vnd.google-apps.spreadsheet':
                # Export Google Sheet as CSV and extract text
                export_result = self.drive_service.files().export(
                    fileId=file_id,
                    mimeType='text/csv'
                ).execute()
                
                # Convert CSV to readable text
                content_text = self._convert_csv_to_text(export_result.decode('utf-8'))
                
            elif mime_type == 'application/vnd.google-apps.presentation':
                # Export Google Slides as plain text
                export_result = self.drive_service.files().export(
                    fileId=file_id,
                    mimeType='text/plain'
                ).execute()
                
                content_text = export_result.decode('utf-8')
            
            else:
                self.logger.warning(f"Unsupported Google Workspace type: {mime_type}")
                return None
            
            if not content_text or len(content_text.strip()) < 50:
                self.logger.info(f"Insufficient content in {file_info['name']}")
                return None
            
            # Generate content ID
            content_id = self._generate_content_id("gdrive", file_id)
            
            # Calculate metadata
            word_count = self._count_words(content_text)
            content_hash = self._calculate_content_hash(content_text)
            
            return RawContent(
                content_id=content_id,
                source_id="gdrive",  # Will be updated with actual source ID
                raw_text=content_text,
                content_type=type_info['name'].lower().replace(' ', '_'),
                title=file_info['name'],
                url=file_info['webViewLink'],
                metadata={
                    "file_id": file_id,
                    "file_name": file_info['name'],
                    "mime_type": mime_type,
                    "google_drive_type": type_info['name'],
                    "created_time": file_info['createdTime'],
                    "modified_time": file_info['modifiedTime'],
                    "web_view_link": file_info['webViewLink'],
                    "owners": file_info.get('owners', []),
                    "description": file_info.get('description', ''),
                    "parents": file_info.get('parents', [])
                },
                content_hash=content_hash,
                word_count=word_count,
                file_size=int(file_info.get('size', 0)) if file_info.get('size') else len(content_text.encode('utf-8'))
            )
            
        except HttpError as e:
            if e.resp.status == 403:
                self.logger.warning(f"No permission to export {file_info['name']}")
            else:
                self.logger.error(f"HTTP error extracting Google Workspace content: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to extract Google Workspace content: {e}")
            return None
    
    async def _extract_uploaded_file_content(self, file_info: Dict[str, Any]) -> Optional[RawContent]:
        """Extract content from uploaded files (PDF, Word, etc.)."""
        try:
            file_id = file_info['id']
            mime_type = file_info['mimeType']
            
            # Download file content
            file_content = self.drive_service.files().get_media(fileId=file_id).execute()
            
            # Extract text based on file type
            if mime_type == 'text/plain':
                content_text = file_content.decode('utf-8')
            
            elif mime_type == 'text/markdown':
                content_text = file_content.decode('utf-8')
            
            elif mime_type == 'application/pdf':
                # PDF text extraction would require PyPDF2 or similar
                # For now, mark for later processing
                content_text = "[PDF content requires processing]"
            
            elif mime_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/msword']:
                # Word document processing would require python-docx
                # For now, mark for later processing
                content_text = "[Word document content requires processing]"
            
            else:
                self.logger.warning(f"Unsupported uploaded file type: {mime_type}")
                return None
            
            # Generate content ID
            content_id = self._generate_content_id("gdrive", file_id)
            
            # Calculate metadata
            word_count = self._count_words(content_text) if not content_text.startswith('[') else 0
            content_hash = self._calculate_content_hash(content_text)
            
            return RawContent(
                content_id=content_id,
                source_id="gdrive",
                raw_text=content_text,
                content_type=self.supported_file_types[mime_type]['name'].lower().replace(' ', '_'),
                title=file_info['name'],
                url=file_info['webViewLink'],
                metadata={
                    "file_id": file_id,
                    "file_name": file_info['name'],
                    "mime_type": mime_type,
                    "file_size": int(file_info.get('size', 0)),
                    "created_time": file_info['createdTime'],
                    "modified_time": file_info['modifiedTime'],
                    "web_view_link": file_info['webViewLink'],
                    "owners": file_info.get('owners', []),
                    "description": file_info.get('description', ''),
                    "parents": file_info.get('parents', []),
                    "requires_processing": content_text.startswith('[')
                },
                content_hash=content_hash,
                word_count=word_count,
                file_size=int(file_info.get('size', 0))
            )
            
        except HttpError as e:
            if e.resp.status == 403:
                self.logger.warning(f"No permission to download {file_info['name']}")
            else:
                self.logger.error(f"HTTP error extracting uploaded file: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to extract uploaded file content: {e}")
            return None
    
    def _convert_csv_to_text(self, csv_content: str) -> str:
        """Convert CSV content to readable text."""
        try:
            import csv
            import io
            
            text_parts = []
            csv_reader = csv.reader(io.StringIO(csv_content))
            
            for row_num, row in enumerate(csv_reader):
                if row_num == 0:
                    # Header row
                    text_parts.append("Spreadsheet Headers: " + " | ".join(row))
                elif row_num < 50:  # Limit to first 50 rows
                    # Data rows
                    non_empty_cells = [cell for cell in row if cell.strip()]
                    if non_empty_cells:
                        text_parts.append("Row {}: {}".format(row_num + 1, " | ".join(non_empty_cells)))
                else:
                    break
            
            return "\n".join(text_parts)
            
        except Exception as e:
            self.logger.warning(f"Failed to convert CSV to text: {e}")
            return csv_content
    
    async def create_gdrive_source(self, folder_id: str, source_config: Dict[str, Any]) -> ContentSource:
        """Create a new Google Drive content source."""
        try:
            if not self.drive_service:
                if not await self.initialize_service():
                    raise Exception("Failed to initialize Google Drive service")
            
            # Get folder information
            if folder_id == 'root':
                folder_name = "My Drive"
            else:
                folder_info = self.drive_service.files().get(fileId=folder_id, fields="name, webViewLink").execute()
                folder_name = folder_info['name']
            
            source_id = f"gdrive_{folder_id}"
            
            source = ContentSource(
                source_id=source_id,
                source_type="google_drive",
                source_url=f"https://drive.google.com/drive/folders/{folder_id}",
                source_metadata={
                    "folder_id": folder_id,
                    "folder_name": folder_name,
                    "config": source_config,
                    "last_sync_status": "not_started",
                    "total_files_discovered": 0,
                    "total_files_processed": 0
                },
                organization_id=self.organization_id
            )
            
            # Store in database
            await self._store_content_source(source)
            
            return source
            
        except Exception as e:
            self.logger.error(f"Failed to create Google Drive source: {e}")
            raise
    
    async def _store_content_source(self, source: ContentSource):
        """Store content source in database."""
        try:
            data = {
                "source_id": source.source_id,
                "source_type": source.source_type,
                "source_url": source.source_url,
                "source_metadata": json.dumps(source.source_metadata),
                "organization_id": source.organization_id,
                "created_at": source.created_at.isoformat(),
                "last_crawled": source.last_crawled.isoformat() if source.last_crawled else None,
                "is_active": source.is_active
            }
            
            supabase_client.client.table("content_sources").upsert(data, on_conflict="source_id").execute()
            
        except Exception as e:
            self.logger.error(f"Failed to store content source: {e}")
            raise
    
    async def get_folder_list(self, parent_folder_id: str = 'root') -> List[Dict[str, Any]]:
        """Get list of folders for UI selection."""
        try:
            if not self.drive_service:
                if not await self.initialize_service():
                    return []
            
            results = self.drive_service.files().list(
                q=f"'{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
                pageSize=50,
                fields="files(id, name, modifiedTime, webViewLink)"
            ).execute()
            
            folders = results.get('files', [])
            
            return [
                {
                    "id": folder['id'],
                    "name": folder['name'],
                    "modified_time": folder['modifiedTime'],
                    "web_view_link": folder['webViewLink']
                }
                for folder in folders
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to get folder list: {e}")
            return []
    
    async def sync_incremental_changes(self, source_id: str, last_sync_time: datetime) -> List[str]:
        """Sync only files that have changed since last sync."""
        try:
            if not self.drive_service:
                if not await self.initialize_service():
                    return []
            
            # Get source configuration
            result = supabase_client.client.table("content_sources").select("*").eq("source_id", source_id).execute()
            
            if not result.data:
                raise Exception(f"Content source {source_id} not found")
            
            source_data = result.data[0]
            source_metadata = json.loads(source_data['source_metadata'])
            folder_id = source_metadata['folder_id']
            
            # Query for files modified since last sync
            last_sync_iso = last_sync_time.isoformat() + 'Z'
            
            results = self.drive_service.files().list(
                q=f"'{folder_id}' in parents and modifiedTime > '{last_sync_iso}' and trashed=false",
                pageSize=100,
                fields="files(id, name, mimeType, modifiedTime)"
            ).execute()
            
            changed_files = results.get('files', [])
            
            self.logger.info(f"Found {len(changed_files)} changed files in Google Drive source {source_id}")
            
            return [f['id'] for f in changed_files]
            
        except Exception as e:
            self.logger.error(f"Failed to sync incremental changes: {e}")
            return []