"""
SEO rule validators.

Contains validation functions for different types of SEO rules including
keyword density, title optimization, meta descriptions, heading structure,
content length, internal linking, and more.
"""

import re
from typing import Any, Dict, List
from urllib.parse import urlparse

import structlog
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

from .models import SEORule, RuleViolation, RuleSeverity

logger = structlog.get_logger(__name__)

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class SEOValidators:
    """
    Collection of SEO validation functions.
    
    Provides specialized validation methods for different types of SEO rules
    including content analysis, technical SEO checks, and optimization validation.
    """
    
    def __init__(self):
        """Initialize SEO validators."""
        self.stop_words = set(stopwords.words('english'))
    
    def _clean_content(self, content: str) -> str:
        """
        Clean HTML content for analysis.
        
        Args:
            content: Raw HTML content
            
        Returns:
            Cleaned text content
        """
        # Remove HTML tags
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(['script', 'style']):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def _extract_keywords_from_content(self, content: str) -> List[str]:
        """
        Extract keywords from content.
        
        Args:
            content: Content to analyze
            
        Returns:
            List of extracted keywords
        """
        clean_content = self._clean_content(content)
        words = word_tokenize(clean_content.lower())
        
        # Filter out stop words and short words
        keywords = [
            word for word in words 
            if word.isalpha() and len(word) > 3 and word not in self.stop_words
        ]
        
        return keywords
    
    async def validate_keyword_density(self, 
                                     rule: SEORule,
                                     content: str,
                                     content_id: str,
                                     metadata: Dict[str, Any]) -> List[RuleViolation]:
        """
        Validate keyword density in content.
        
        Args:
            rule: SEO rule to validate
            content: Content to analyze
            content_id: Content identifier
            metadata: Additional metadata
            
        Returns:
            List of rule violations
        """
        violations = []
        
        try:
            target_keyword = metadata.get('target_keyword', '')
            if not target_keyword:
                # Skip if no target keyword specified
                return violations
            
            # Clean content and calculate density
            clean_content = self._clean_content(content)
            total_words = len(word_tokenize(clean_content))
            
            if total_words == 0:
                return violations
            
            # Count keyword occurrences (case insensitive)
            keyword_count = clean_content.lower().count(target_keyword.lower())
            density = (keyword_count / total_words) * 100
            
            min_density = rule.parameters.get('min_density', 1.0)
            max_density = rule.parameters.get('max_density', 3.0)
            
            if density < min_density:
                violations.append(RuleViolation(
                    rule_id=rule.rule_id,
                    content_id=content_id,
                    severity=rule.severity,
                    message="Keyword density too low",
                    description=f"Keyword '{target_keyword}' density is {density:.2f}%, should be at least {min_density}%",
                    current_value=density,
                    expected_value=min_density,
                    recommendation=f"Include '{target_keyword}' more naturally in the content",
                    priority=6,
                    tenant_id=rule.tenant_id
                ))
            elif density > max_density:
                violations.append(RuleViolation(
                    rule_id=rule.rule_id,
                    content_id=content_id,
                    severity=rule.severity,
                    message="Keyword density too high",
                    description=f"Keyword '{target_keyword}' density is {density:.2f}%, should be at most {max_density}%",
                    current_value=density,
                    expected_value=max_density,
                    recommendation=f"Reduce usage of '{target_keyword}' to avoid over-optimization",
                    priority=7,
                    tenant_id=rule.tenant_id
                ))
            
        except Exception as e:
            logger.error(f"Failed to validate keyword density: {e}")
            
        return violations
    
    async def validate_title_optimization(self, 
                                        rule: SEORule,
                                        content: str,
                                        content_id: str,
                                        metadata: Dict[str, Any]) -> List[RuleViolation]:
        """
        Validate title optimization.
        
        Args:
            rule: SEO rule to validate
            content: Content to analyze
            content_id: Content identifier
            metadata: Additional metadata
            
        Returns:
            List of rule violations
        """
        violations = []
        
        try:
            title = metadata.get('title', '')
            target_keyword = metadata.get('target_keyword', '')
            
            if not title:
                violations.append(RuleViolation(
                    rule_id=rule.rule_id,
                    content_id=content_id,
                    severity=rule.severity,
                    message="No title found",
                    description="Content must have a title",
                    current_value="None",
                    expected_value="Valid title",
                    recommendation="Add a descriptive title to the content",
                    priority=10,
                    tenant_id=rule.tenant_id
                ))
                return violations
            
            # Check title length
            title_length = len(title)
            min_length = rule.parameters.get('min_length', 30)
            max_length = rule.parameters.get('max_length', 60)
            
            if title_length < min_length:
                violations.append(RuleViolation(
                    rule_id=rule.rule_id,
                    content_id=content_id,
                    severity=rule.severity,
                    message="Title too short",
                    description=f"Title is {title_length} characters, should be at least {min_length}",
                    current_value=title_length,
                    expected_value=min_length,
                    recommendation="Expand title with more descriptive words",
                    priority=6,
                    tenant_id=rule.tenant_id
                ))
            elif title_length > max_length:
                violations.append(RuleViolation(
                    rule_id=rule.rule_id,
                    content_id=content_id,
                    severity=rule.severity,
                    message="Title too long",
                    description=f"Title is {title_length} characters, should be at most {max_length}",
                    current_value=title_length,
                    expected_value=max_length,
                    recommendation="Shorten title to fit in search results",
                    priority=5,
                    tenant_id=rule.tenant_id
                ))
            
            # Check if target keyword is in title
            if target_keyword and target_keyword.lower() not in title.lower():
                violations.append(RuleViolation(
                    rule_id=rule.rule_id,
                    content_id=content_id,
                    severity=rule.severity,
                    message="Target keyword not in title",
                    description=f"Title should include the target keyword '{target_keyword}'",
                    recommendation=f"Include '{target_keyword}' in the title",
                    priority=8,
                    tenant_id=rule.tenant_id
                ))
            
        except Exception as e:
            logger.error(f"Failed to validate title optimization: {e}")
            
        return violations
    
    async def validate_meta_description(self, 
                                      rule: SEORule,
                                      content: str,
                                      content_id: str,
                                      metadata: Dict[str, Any]) -> List[RuleViolation]:
        """
        Validate meta description.
        
        Args:
            rule: SEO rule to validate
            content: Content to analyze
            content_id: Content identifier
            metadata: Additional metadata
            
        Returns:
            List of rule violations
        """
        violations = []
        
        try:
            meta_description = metadata.get('meta_description', '')
            
            if not meta_description:
                violations.append(RuleViolation(
                    rule_id=rule.rule_id,
                    content_id=content_id,
                    severity=rule.severity,
                    message="No meta description found",
                    description="Content must have a meta description",
                    current_value="None",
                    expected_value="Valid meta description",
                    recommendation="Add a compelling meta description",
                    priority=8,
                    tenant_id=rule.tenant_id
                ))
                return violations
            
            # Check meta description length
            desc_length = len(meta_description)
            min_length = rule.parameters.get('min_length', 150)
            max_length = rule.parameters.get('max_length', 160)
            
            if desc_length < min_length:
                violations.append(RuleViolation(
                    rule_id=rule.rule_id,
                    content_id=content_id,
                    severity=rule.severity,
                    message="Meta description too short",
                    description=f"Meta description is {desc_length} characters, should be at least {min_length}",
                    current_value=desc_length,
                    expected_value=min_length,
                    recommendation="Expand meta description with more details",
                    priority=5,
                    tenant_id=rule.tenant_id
                ))
            elif desc_length > max_length:
                violations.append(RuleViolation(
                    rule_id=rule.rule_id,
                    content_id=content_id,
                    severity=rule.severity,
                    message="Meta description too long",
                    description=f"Meta description is {desc_length} characters, should be at most {max_length}",
                    current_value=desc_length,
                    expected_value=max_length,
                    recommendation="Shorten meta description to fit in search results",
                    priority=4,
                    tenant_id=rule.tenant_id
                ))
            
        except Exception as e:
            logger.error(f"Failed to validate meta description: {e}")
            
        return violations
    
    async def validate_heading_structure(self, 
                                       rule: SEORule,
                                       content: str,
                                       content_id: str,
                                       metadata: Dict[str, Any]) -> List[RuleViolation]:
        """
        Validate heading structure.
        
        Args:
            rule: SEO rule to validate
            content: Content to analyze
            content_id: Content identifier
            metadata: Additional metadata
            
        Returns:
            List of rule violations
        """
        violations = []
        
        try:
            # Parse HTML
            soup = BeautifulSoup(content, 'html.parser')
            
            # Check H1 tags
            h1_tags = soup.find_all('h1')
            h1_count = len(h1_tags)
            
            required_count = rule.parameters.get('required_count', 1)
            max_count = rule.parameters.get('max_count', 1)
            
            if h1_count == 0:
                violations.append(RuleViolation(
                    rule_id=rule.rule_id,
                    content_id=content_id,
                    severity=rule.severity,
                    message="No H1 tag found",
                    description="Content must have exactly one H1 tag",
                    current_value=0,
                    expected_value=1,
                    recommendation="Add an H1 tag with your main heading",
                    priority=9,
                    tenant_id=rule.tenant_id
                ))
            elif h1_count > max_count:
                violations.append(RuleViolation(
                    rule_id=rule.rule_id,
                    content_id=content_id,
                    severity=rule.severity,
                    message="Multiple H1 tags found",
                    description=f"Found {h1_count} H1 tags, should have exactly {max_count}",
                    current_value=h1_count,
                    expected_value=max_count,
                    recommendation="Use only one H1 tag and convert others to H2-H6",
                    priority=6,
                    tenant_id=rule.tenant_id
                ))
            
            # Check heading hierarchy
            headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            if headings:
                prev_level = 0
                for heading in headings:
                    level = int(heading.name[1])
                    if level > prev_level + 1:
                        violations.append(RuleViolation(
                            rule_id=rule.rule_id,
                            content_id=content_id,
                            severity=RuleSeverity.WARNING,
                            message="Heading hierarchy skipped",
                            description=f"Heading {heading.name} appears without proper hierarchy",
                            location=str(heading),
                            recommendation="Maintain proper heading hierarchy (H1->H2->H3->etc)",
                            priority=3,
                            tenant_id=rule.tenant_id
                        ))
                    prev_level = level
            
        except Exception as e:
            logger.error(f"Failed to validate heading structure: {e}")
            
        return violations
    
    async def validate_content_length(self, 
                                    rule: SEORule,
                                    content: str,
                                    content_id: str,
                                    metadata: Dict[str, Any]) -> List[RuleViolation]:
        """
        Validate content length.
        
        Args:
            rule: SEO rule to validate
            content: Content to analyze
            content_id: Content identifier
            metadata: Additional metadata
            
        Returns:
            List of rule violations
        """
        violations = []
        
        try:
            # Clean content and count words
            clean_content = self._clean_content(content)
            words = word_tokenize(clean_content)
            word_count = len([w for w in words if w.isalpha()])
            
            min_words = rule.parameters.get('min_words', 300)
            
            if word_count < min_words:
                violations.append(RuleViolation(
                    rule_id=rule.rule_id,
                    content_id=content_id,
                    severity=rule.severity,
                    message="Content too short",
                    description=f"Content has {word_count} words, should have at least {min_words}",
                    current_value=word_count,
                    expected_value=min_words,
                    recommendation="Expand content with more detailed information",
                    priority=5,
                    tenant_id=rule.tenant_id
                ))
            
        except Exception as e:
            logger.error(f"Failed to validate content length: {e}")
            
        return violations
    
    async def validate_internal_linking(self, 
                                      rule: SEORule,
                                      content: str,
                                      content_id: str,
                                      metadata: Dict[str, Any]) -> List[RuleViolation]:
        """
        Validate internal linking.
        
        Args:
            rule: SEO rule to validate
            content: Content to analyze
            content_id: Content identifier
            metadata: Additional metadata
            
        Returns:
            List of rule violations
        """
        violations = []
        
        try:
            # Parse HTML and find internal links
            soup = BeautifulSoup(content, 'html.parser')
            links = soup.find_all('a', href=True)
            
            # Filter internal links (assuming same domain)
            internal_links = []
            for link in links:
                href = link['href']
                if href.startswith('/') or href.startswith('#') or not href.startswith('http'):
                    internal_links.append(link)
            
            link_count = len(internal_links)
            min_links = rule.parameters.get('min_links', 2)
            max_links = rule.parameters.get('max_links', 5)
            
            if link_count < min_links:
                violations.append(RuleViolation(
                    rule_id=rule.rule_id,
                    content_id=content_id,
                    severity=rule.severity,
                    message="Too few internal links",
                    description=f"Found {link_count} internal links, should have at least {min_links}",
                    current_value=link_count,
                    expected_value=min_links,
                    recommendation="Add more internal links to related content",
                    priority=4,
                    tenant_id=rule.tenant_id
                ))
            elif link_count > max_links:
                violations.append(RuleViolation(
                    rule_id=rule.rule_id,
                    content_id=content_id,
                    severity=rule.severity,
                    message="Too many internal links",
                    description=f"Found {link_count} internal links, should have at most {max_links}",
                    current_value=link_count,
                    expected_value=max_links,
                    recommendation="Remove some internal links to avoid over-optimization",
                    priority=3,
                    tenant_id=rule.tenant_id
                ))
            
        except Exception as e:
            logger.error(f"Failed to validate internal linking: {e}")
            
        return violations
    
    async def validate_image_optimization(self, 
                                        rule: SEORule,
                                        content: str,
                                        content_id: str,
                                        metadata: Dict[str, Any]) -> List[RuleViolation]:
        """
        Validate image optimization.
        
        Args:
            rule: SEO rule to validate
            content: Content to analyze
            content_id: Content identifier
            metadata: Additional metadata
            
        Returns:
            List of rule violations
        """
        violations = []
        
        try:
            # Parse HTML and find images
            soup = BeautifulSoup(content, 'html.parser')
            images = soup.find_all('img')
            
            for img in images:
                # Check for alt text
                if not img.get('alt'):
                    violations.append(RuleViolation(
                        rule_id=rule.rule_id,
                        content_id=content_id,
                        severity=rule.severity,
                        message="Image missing alt text",
                        description="All images should have descriptive alt text",
                        location=str(img),
                        recommendation="Add descriptive alt text to the image",
                        priority=6,
                        tenant_id=rule.tenant_id
                    ))
                
                # Check for empty alt text
                elif img.get('alt').strip() == '':
                    violations.append(RuleViolation(
                        rule_id=rule.rule_id,
                        content_id=content_id,
                        severity=rule.severity,
                        message="Image has empty alt text",
                        description="Alt text should be descriptive, not empty",
                        location=str(img),
                        recommendation="Add meaningful alt text describing the image",
                        priority=5,
                        tenant_id=rule.tenant_id
                    ))
            
        except Exception as e:
            logger.error(f"Failed to validate image optimization: {e}")
            
        return violations
    
    async def validate_readability(self, 
                                 rule: SEORule,
                                 content: str,
                                 content_id: str,
                                 metadata: Dict[str, Any]) -> List[RuleViolation]:
        """
        Validate content readability.
        
        Args:
            rule: SEO rule to validate
            content: Content to analyze
            content_id: Content identifier
            metadata: Additional metadata
            
        Returns:
            List of rule violations
        """
        violations = []
        
        try:
            # Clean content and analyze readability
            clean_content = self._clean_content(content)
            sentences = sent_tokenize(clean_content)
            words = word_tokenize(clean_content)
            
            if not sentences or not words:
                return violations
            
            # Calculate average sentence length
            avg_sentence_length = len(words) / len(sentences)
            max_sentence_length = rule.parameters.get('max_avg_sentence_length', 20)
            
            if avg_sentence_length > max_sentence_length:
                violations.append(RuleViolation(
                    rule_id=rule.rule_id,
                    content_id=content_id,
                    severity=rule.severity,
                    message="Sentences too long",
                    description=f"Average sentence length is {avg_sentence_length:.1f} words, should be at most {max_sentence_length}",
                    current_value=avg_sentence_length,
                    expected_value=max_sentence_length,
                    recommendation="Break long sentences into shorter ones for better readability",
                    priority=3,
                    tenant_id=rule.tenant_id
                ))
            
            # Check for passive voice (simplified check)
            passive_indicators = ['was', 'were', 'been', 'being', 'be']
            passive_count = sum(1 for word in words if word.lower() in passive_indicators)
            passive_ratio = passive_count / len(words) if words else 0
            max_passive_ratio = rule.parameters.get('max_passive_ratio', 0.1)
            
            if passive_ratio > max_passive_ratio:
                violations.append(RuleViolation(
                    rule_id=rule.rule_id,
                    content_id=content_id,
                    severity=RuleSeverity.SUGGESTION,
                    message="Too much passive voice",
                    description=f"Passive voice ratio is {passive_ratio:.1%}, should be at most {max_passive_ratio:.1%}",
                    current_value=passive_ratio,
                    expected_value=max_passive_ratio,
                    recommendation="Use more active voice for better readability",
                    priority=2,
                    tenant_id=rule.tenant_id
                ))
            
        except Exception as e:
            logger.error(f"Failed to validate readability: {e}")
            
        return violations