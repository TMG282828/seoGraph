"""
Data models for SEO rules engine.

Contains all Pydantic models, enums, and type definitions used
across the SEO rules system.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field


class SEORuleError(Exception):
    """Raised when SEO rule operations fail."""
    pass


class RuleViolationError(SEORuleError):
    """Raised when content violates SEO rules."""
    pass


class RuleType(str, Enum):
    """Types of SEO rules."""
    
    KEYWORD_DENSITY = "keyword_density"
    TITLE_OPTIMIZATION = "title_optimization"
    META_DESCRIPTION = "meta_description"
    HEADING_STRUCTURE = "heading_structure"
    CONTENT_LENGTH = "content_length"
    INTERNAL_LINKING = "internal_linking"
    IMAGE_OPTIMIZATION = "image_optimization"
    URL_STRUCTURE = "url_structure"
    READABILITY = "readability"
    SEMANTIC_KEYWORDS = "semantic_keywords"
    CONTENT_FRESHNESS = "content_freshness"
    TECHNICAL_SEO = "technical_seo"


class RuleSeverity(str, Enum):
    """Severity levels for rule violations."""
    
    CRITICAL = "critical"
    WARNING = "warning"
    SUGGESTION = "suggestion"
    INFO = "info"


class RuleCategory(str, Enum):
    """Categories of SEO rules."""
    
    ON_PAGE = "on_page"
    TECHNICAL = "technical"
    CONTENT = "content"
    KEYWORDS = "keywords"
    STRUCTURE = "structure"
    PERFORMANCE = "performance"
    ACCESSIBILITY = "accessibility"


class SEORule(BaseModel):
    """
    Individual SEO rule definition.
    
    Represents a single SEO optimization rule with its configuration,
    parameters, and applicability constraints.
    """
    
    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Rule name")
    description: str = Field(..., description="Rule description")
    
    # Rule classification
    rule_type: RuleType = Field(..., description="Type of rule")
    category: RuleCategory = Field(..., description="Rule category")
    severity: RuleSeverity = Field(..., description="Rule severity")
    
    # Rule configuration
    enabled: bool = Field(True, description="Whether rule is enabled")
    weight: float = Field(1.0, ge=0.0, le=10.0, description="Rule weight for scoring")
    
    # Rule parameters
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Rule parameters")
    thresholds: Dict[str, Union[int, float]] = Field(default_factory=dict, description="Rule thresholds")
    
    # Content type applicability
    applicable_content_types: List[str] = Field(default_factory=list, description="Applicable content types")
    
    # Validation function (stored as string for serialization)
    validation_function: Optional[str] = Field(None, description="Validation function name")
    
    # Metadata
    tenant_id: str = Field(..., description="Tenant identifier")
    created_by: str = Field(..., description="Rule creator")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    def is_applicable_to_content_type(self, content_type: str) -> bool:
        """
        Check if rule applies to content type.
        
        Args:
            content_type: Content type to check
            
        Returns:
            True if rule applies to the content type
        """
        if not self.applicable_content_types:
            return True  # Apply to all types if none specified
        return content_type in self.applicable_content_types


class RuleViolation(BaseModel):
    """
    SEO rule violation.
    
    Represents a specific violation of an SEO rule found in content,
    including details about the violation and recommendations for fixing it.
    """
    
    violation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    rule_id: str = Field(..., description="Associated rule ID")
    content_id: str = Field(..., description="Content identifier")
    
    # Violation details
    severity: RuleSeverity = Field(..., description="Violation severity")
    message: str = Field(..., description="Violation message")
    description: str = Field(..., description="Detailed description")
    
    # Location information
    location: Optional[str] = Field(None, description="Location in content")
    element: Optional[str] = Field(None, description="HTML element if applicable")
    
    # Metrics
    current_value: Optional[Union[int, float, str]] = Field(None, description="Current value")
    expected_value: Optional[Union[int, float, str]] = Field(None, description="Expected value")
    
    # Recommendation
    recommendation: str = Field(..., description="How to fix violation")
    priority: int = Field(1, ge=1, le=10, description="Fix priority")
    
    # Metadata
    tenant_id: str = Field(..., description="Tenant identifier")
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert violation to dictionary.
        
        Returns:
            Dictionary representation of the violation
        """
        return {
            'violation_id': self.violation_id,
            'rule_id': self.rule_id,
            'severity': self.severity.value,
            'message': self.message,
            'description': self.description,
            'location': self.location,
            'current_value': self.current_value,
            'expected_value': self.expected_value,
            'recommendation': self.recommendation,
            'priority': self.priority
        }


class SEOAuditResult(BaseModel):
    """
    SEO audit result for content.
    
    Contains comprehensive results of SEO analysis including scores,
    violations, and recommendations.
    """
    
    audit_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content_id: str = Field(..., description="Content identifier")
    
    # Overall scores
    overall_score: float = Field(0.0, ge=0.0, le=100.0, description="Overall SEO score")
    category_scores: Dict[str, float] = Field(default_factory=dict, description="Category scores")
    
    # Violations
    violations: List[RuleViolation] = Field(default_factory=list, description="Rule violations")
    critical_violations: int = Field(0, ge=0, description="Critical violations count")
    warning_violations: int = Field(0, ge=0, description="Warning violations count")
    suggestion_violations: int = Field(0, ge=0, description="Suggestion violations count")
    
    # Recommendations
    top_recommendations: List[str] = Field(default_factory=list, description="Top recommendations")
    quick_fixes: List[str] = Field(default_factory=list, description="Quick fixes")
    
    # Analysis details
    analyzed_rules: int = Field(0, ge=0, description="Number of rules analyzed")
    passed_rules: int = Field(0, ge=0, description="Number of rules passed")
    
    # Performance metrics
    analysis_duration: Optional[float] = Field(None, description="Analysis duration in seconds")
    rules_engine_version: str = Field("1.0", description="Rules engine version")
    
    # Metadata
    tenant_id: str = Field(..., description="Tenant identifier")
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    def get_severity_counts(self) -> Dict[str, int]:
        """
        Get count of violations by severity.
        
        Returns:
            Dictionary with severity counts
        """
        return {
            'critical': self.critical_violations,
            'warning': self.warning_violations,
            'suggestion': self.suggestion_violations,
            'info': len([v for v in self.violations if v.severity == RuleSeverity.INFO])
        }
    
    def get_category_violation_counts(self) -> Dict[str, int]:
        """
        Get violation counts by category.
        
        Returns:
            Dictionary with category violation counts
        """
        category_counts = {}
        for violation in self.violations:
            # Would need to access rule to get category - simplified for now
            category = "unknown"
            category_counts[category] = category_counts.get(category, 0) + 1
        return category_counts
    
    def get_pass_rate(self) -> float:
        """
        Get the pass rate for analyzed rules.
        
        Returns:
            Pass rate as a percentage (0.0-100.0)
        """
        if self.analyzed_rules == 0:
            return 0.0
        return (self.passed_rules / self.analyzed_rules) * 100.0


class RuleRequest(BaseModel):
    """Request for creating or updating SEO rules."""
    
    name: str = Field(..., description="Rule name")
    description: str = Field(..., description="Rule description")
    rule_type: RuleType = Field(..., description="Type of rule")
    category: RuleCategory = Field(..., description="Rule category")
    severity: RuleSeverity = Field(..., description="Rule severity")
    enabled: bool = Field(True, description="Whether rule is enabled")
    weight: float = Field(1.0, ge=0.0, le=10.0, description="Rule weight")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Rule parameters")
    thresholds: Dict[str, Union[int, float]] = Field(default_factory=dict, description="Rule thresholds")
    applicable_content_types: List[str] = Field(default_factory=list, description="Applicable content types")
    tenant_id: str = Field(..., description="Tenant identifier")
    created_by: str = Field(..., description="Rule creator")


class AuditRequest(BaseModel):
    """Request for SEO content audit."""
    
    content_id: str = Field(..., description="Content identifier")
    content_type: str = Field(..., description="Content type")
    tenant_id: str = Field(..., description="Tenant identifier")
    rule_categories: Optional[List[RuleCategory]] = Field(None, description="Specific categories to audit")
    rule_types: Optional[List[RuleType]] = Field(None, description="Specific rule types to audit")
    include_suggestions: bool = Field(True, description="Include suggestion-level violations")
    max_violations: Optional[int] = Field(None, description="Maximum violations to return")
    
    def should_include_severity(self, severity: RuleSeverity) -> bool:
        """
        Check if severity should be included in audit.
        
        Args:
            severity: Rule severity to check
            
        Returns:
            True if severity should be included
        """
        if severity in [RuleSeverity.CRITICAL, RuleSeverity.WARNING]:
            return True
        return severity == RuleSeverity.SUGGESTION and self.include_suggestions