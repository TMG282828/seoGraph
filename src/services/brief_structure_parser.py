"""
Brief Structure Parser Service.

Parses structured content briefs to extract titles, headings, meta data,
and other structured information for improved content generation.
"""

import logging
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ParsedBrief:
    """Structured data extracted from a content brief."""
    title: Optional[str] = None
    title_tag: Optional[str] = None
    meta_description: Optional[str] = None
    objectives: Optional[str] = None
    call_to_action: Optional[str] = None
    target_audience: Optional[str] = None
    tone_of_voice: Optional[str] = None
    keywords: List[str] = None
    competitor_articles: List[str] = None
    url_slug: Optional[str] = None
    word_count_range: Optional[str] = None
    search_intent: Optional[str] = None
    internal_links: Optional[str] = None
    heading_structure: List[Dict[str, str]] = None
    raw_content: Optional[str] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.competitor_articles is None:
            self.competitor_articles = []
        if self.heading_structure is None:
            self.heading_structure = []


class BriefStructureParser:
    """
    Parses structured content briefs to extract organized data.
    
    Handles various brief formats and extracts key fields for
    improved topic extraction and content generation.
    """
    
    def __init__(self):
        """Initialize the brief structure parser."""
        logger.info("BriefStructureParser initialized")
    
    def parse_brief(self, brief_content: str) -> ParsedBrief:
        """
        Parse a structured content brief into organized data.
        
        Args:
            brief_content: Raw brief content to parse
            
        Returns:
            ParsedBrief: Structured data extracted from the brief
        """
        if not brief_content or not brief_content.strip():
            logger.warning("Empty brief content provided")
            return ParsedBrief(raw_content=brief_content)
        
        logger.info(f"📋 Parsing brief content: {len(brief_content)} characters")
        
        parsed = ParsedBrief(raw_content=brief_content)
        
        # Parse different field types
        parsed.title = self._extract_title(brief_content)
        parsed.title_tag = self._extract_title_tag(brief_content)
        parsed.meta_description = self._extract_meta_description(brief_content)
        parsed.objectives = self._extract_objectives(brief_content)
        parsed.call_to_action = self._extract_call_to_action(brief_content)
        parsed.target_audience = self._extract_target_audience(brief_content)
        parsed.tone_of_voice = self._extract_tone_of_voice(brief_content)
        parsed.keywords = self._extract_keywords(brief_content)
        parsed.competitor_articles = self._extract_competitor_articles(brief_content)
        parsed.url_slug = self._extract_url_slug(brief_content)
        parsed.word_count_range = self._extract_word_count_range(brief_content)
        parsed.search_intent = self._extract_search_intent(brief_content)
        parsed.internal_links = self._extract_internal_links(brief_content)
        parsed.heading_structure = self._extract_heading_structure(brief_content)
        
        logger.info(f"✅ Brief parsing complete - Title: '{parsed.title}', Headings: {len(parsed.heading_structure)}")
        
        return parsed
    
    def _extract_title(self, content: str) -> Optional[str]:
        """Extract the main title/H1 from the brief."""
        # Try simple field parsing first for "Title / H1" header format
        result = self._extract_field_simple(content, ['title / h1', 'title', 'h1', 'subject'])
        if result and len(result) > 10:
            logger.info(f"📄 Extracted title via simple parsing: '{result}'")
            return result
        
        # Fallback to regex patterns for colon-separated format
        title_patterns = [
            r'^title\s*/\s*h1:\s*(.+)$',      # "Title / H1: content"
            r'^title\s*/\s*h1\s+(.+)$',       # "Title / H1 content"
            r'^h1:\s*(.+)$',                  # "H1: content"
            r'^title:\s*(.+)$',               # "Title: content"
            r'^subject:\s*(.+)$',             # "Subject: content"
        ]
        
        lines = content.strip().split('\n')
        
        for line in lines[:20]:  # Check first 20 lines
            line = line.strip()
            if not line:
                continue
                
            for pattern in title_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    title = match.group(1).strip()
                    if title and len(title) > 10:  # Reasonable title length
                        logger.info(f"📄 Extracted title via regex: '{title}'")
                        return title
        
        return None
    
    def _extract_title_tag(self, content: str) -> Optional[str]:
        """Extract title tag metadata."""
        return self._extract_field(content, [
            r'^title\s+tag:?\s*(.+)$',
            r'^title\s+tag\s*#\s*(.+)$',
        ])
    
    def _extract_meta_description(self, content: str) -> Optional[str]:
        """Extract meta description."""
        return self._extract_field(content, [
            r'^meta\s+description:?\s*(.+)$',
            r'^meta\s+description\s*#?\s*(.+)$',
            r'^description:?\s*(.+)$',
        ])
    
    def _extract_objectives(self, content: str) -> Optional[str]:
        """Extract objectives/goals."""
        # Try simple field parsing first for "Objectives" header format
        result = self._extract_field_simple(content, ['objectives', 'goals', 'purpose'])
        if result:
            return result
        
        # Fallback to regex patterns
        return self._extract_field(content, [
            r'^objectives?:?\s*(.+)$',
            r'^goals?:?\s*(.+)$',
            r'^purpose:?\s*(.+)$',
        ])
    
    def _extract_call_to_action(self, content: str) -> Optional[str]:
        """Extract call to action."""
        # Try simple field parsing first
        result = self._extract_field_simple(content, ['call to action', 'cta'])
        if result:
            return result
        
        return self._extract_field(content, [
            r'^call\s+to\s+action:?\s*(.+)$',
            r'^cta:?\s*(.+)$',
        ])
    
    def _extract_target_audience(self, content: str) -> Optional[str]:
        """Extract target audience information."""
        # Try simple field parsing first
        result = self._extract_field_simple(content, ['target audience', 'audience'])
        if result:
            return result
        
        return self._extract_field(content, [
            r'^target\s+audience:?\s*(.+)$',
            r'^audience:?\s*(.+)$',
        ])
    
    def _extract_tone_of_voice(self, content: str) -> Optional[str]:
        """Extract tone of voice guidelines."""
        return self._extract_field(content, [
            r'^tone\s+of\s+voice:?\s*(.+)$',
            r'^tone\s+of\s+voice\s*#\s*(.+)$',
            r'^tone:?\s*(.+)$',
            r'^voice:?\s*(.+)$',
        ])
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords as a list."""
        keywords_text = self._extract_field(content, [
            r'^keywords?:?\s*(.+)$',
            r'^keywords?\s*#\s*(.+)$',
            r'^target\s+keywords?:?\s*(.+)$',
        ])
        
        if not keywords_text:
            return []
        
        # Split keywords by common delimiters
        keywords = re.split(r'[,;\n]+', keywords_text)
        return [kw.strip() for kw in keywords if kw.strip()]
    
    def _extract_competitor_articles(self, content: str) -> List[str]:
        """Extract competitor article URLs."""
        competitor_text = self._extract_field(content, [
            r'^competitor\s+articles?:?\s*(.+)$',
            r'^competitors?:?\s*(.+)$',
            r'^competition:?\s*(.+)$',
        ])
        
        if not competitor_text:
            return []
        
        # Extract URLs from the text
        urls = re.findall(r'https?://[^\s\n]+', competitor_text)
        return [url.strip() for url in urls if url.strip()]
    
    def _extract_url_slug(self, content: str) -> Optional[str]:
        """Extract URL slug information."""
        return self._extract_field(content, [
            r'^url:?\s*(.+)$',
            r'^url\s+slug:?\s*(.+)$',
            r'^slug:?\s*(.+)$',
        ])
    
    def _extract_word_count_range(self, content: str) -> Optional[str]:
        """Extract word count range."""
        return self._extract_field(content, [
            r'^range:?\s*(.+)$',
            r'^word\s+count:?\s*(.+)$',
            r'^length:?\s*(.+)$',
        ])
    
    def _extract_search_intent(self, content: str) -> Optional[str]:
        """Extract search intent information."""
        return self._extract_field(content, [
            r'^search\s+intent:?\s*(.+)$',
            r'^intent:?\s*(.+)$',
        ])
    
    def _extract_internal_links(self, content: str) -> Optional[str]:
        """Extract internal links guidance."""
        return self._extract_field(content, [
            r'^internal\s+links?:?\s*(.+)$',
            r'^links?:?\s*(.+)$',
        ])
    
    def _extract_heading_structure(self, content: str) -> List[Dict[str, str]]:
        """Extract heading structure from the brief."""
        headings = []
        lines = content.strip().split('\n')
        
        current_heading = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for heading patterns
            h1_match = re.match(r'^h1:\s*(.+)$', line, re.IGNORECASE)
            h2_match = re.match(r'^h2:\s*(.+)$', line, re.IGNORECASE)
            h3_match = re.match(r'^h3:\s*(.+)$', line, re.IGNORECASE)
            
            if h1_match:
                if current_heading:
                    headings.append({
                        "level": current_heading["level"],
                        "title": current_heading["title"],
                        "content": '\n'.join(current_content)
                    })
                current_heading = {"level": "h1", "title": h1_match.group(1).strip()}
                current_content = []
            elif h2_match:
                if current_heading:
                    headings.append({
                        "level": current_heading["level"],
                        "title": current_heading["title"],
                        "content": '\n'.join(current_content)
                    })
                current_heading = {"level": "h2", "title": h2_match.group(1).strip()}
                current_content = []
            elif h3_match:
                if current_heading:
                    headings.append({
                        "level": current_heading["level"],
                        "title": current_heading["title"],
                        "content": '\n'.join(current_content)
                    })
                current_heading = {"level": "h3", "title": h3_match.group(1).strip()}
                current_content = []
            elif current_heading:
                current_content.append(line)
        
        # Add the last heading
        if current_heading:
            headings.append({
                "level": current_heading["level"],
                "title": current_heading["title"],
                "content": '\n'.join(current_content)
            })
        
        if headings:
            logger.info(f"📋 Extracted {len(headings)} headings from brief structure")
        
        return headings
    
    def _extract_field_simple(self, content: str, field_names: List[str]) -> Optional[str]:
        """Extract field content using simple field header detection."""
        lines = content.strip().split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check if this line is a field header
            for field_name in field_names:
                if line.lower().replace('#', '').strip() == field_name.lower():
                    # Found field header, get content from next lines
                    content_lines = []
                    for j in range(i + 1, len(lines)):
                        next_line = lines[j].strip()
                        if not next_line:
                            continue
                        
                        # Stop if we hit another field header
                        is_field_header = any(
                            next_line.lower().replace('#', '').strip() == fname.lower()
                            for fname in ['objectives', 'call to action', 'target audience', 
                                        'title / h1', 'title tag', 'meta description', 'keywords',
                                        'tone of voice', 'competitor articles', 'url', 'range',
                                        'search intent', 'internal links', 'h1', 'h2', 'h3']
                        )
                        
                        if is_field_header:
                            break
                        
                        content_lines.append(next_line)
                    
                    if content_lines:
                        return '\n'.join(content_lines).strip()
        
        return None
    
    def _extract_field(self, content: str, patterns: List[str]) -> Optional[str]:
        """Extract a field using regex patterns or simple field detection."""
        # First try regex patterns for fields with colons
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            for pattern in patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    if value:
                        return value
        
        # If no regex match, try simple field header approach
        field_names = []
        for pattern in patterns:
            # Extract field name from pattern
            if 'objectives' in pattern.lower():
                field_names.append('objectives')
            elif 'call to action' in pattern.lower():
                field_names.append('call to action')
            elif 'target audience' in pattern.lower():
                field_names.append('target audience')
            elif 'title' in pattern.lower() and 'h1' in pattern.lower():
                field_names.append('title / h1')
            elif 'title tag' in pattern.lower():
                field_names.append('title tag')
            elif 'meta description' in pattern.lower():
                field_names.append('meta description')
        
        if field_names:
            return self._extract_field_simple(content, field_names)
        
        return None
    
    def get_content_title(self, parsed_brief: ParsedBrief) -> Optional[str]:
        """
        Get the best content title from a parsed brief.
        
        Prioritizes actual content titles over business objectives.
        """
        if parsed_brief.title and len(parsed_brief.title) > 10:
            return parsed_brief.title
        
        # Look for H1 in heading structure
        for heading in parsed_brief.heading_structure:
            if heading["level"] == "h1" and heading["title"]:
                return heading["title"]
        
        return None


# Global instance for use across the application
brief_structure_parser = BriefStructureParser()