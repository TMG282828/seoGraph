"""
SEO Rules Engine Package.

Provides comprehensive SEO rule management and content auditing capabilities
including rule-based validation, content optimization recommendations, and
customizable rule management.

This package is organized into focused modules:
- models: Data models and type definitions
- validators: SEO validation functions
- engine: Main rules engine and orchestration
- default_rules: Predefined SEO rules
- api: API endpoints for rule management (if needed)

Example usage:
    from src.config.seo_rules import SEORulesEngine, SEORule, RuleType
    
    # Initialize engine
    engine = SEORulesEngine(neo4j_client)
    await engine.initialize()
    
    # Audit content
    result = await engine.audit_content(
        content="<html>Content to audit</html>",
        content_id="content-123",
        content_type="blog_post",
        metadata={"title": "My Title"},
        tenant_id="tenant-123"
    )
"""

from .engine import SEORulesEngine
from .models import (
    # Exceptions
    SEORuleError,
    RuleViolationError,
    
    # Enums
    RuleType,
    RuleSeverity,
    RuleCategory,
    
    # Models
    SEORule,
    RuleViolation,
    SEOAuditResult,
    RuleRequest,
    AuditRequest
)
from .validators import SEOValidators
from .default_rules import (
    create_default_rules,
    get_rule_templates_by_category,
    get_rule_templates_by_type
)

__all__ = [
    # Main engine
    'SEORulesEngine',
    
    # Exceptions
    'SEORuleError',
    'RuleViolationError',
    
    # Enums
    'RuleType',
    'RuleSeverity',
    'RuleCategory',
    
    # Models
    'SEORule',
    'RuleViolation',
    'SEOAuditResult',
    'RuleRequest',
    'AuditRequest',
    
    # Validators
    'SEOValidators',
    
    # Default rules
    'create_default_rules',
    'get_rule_templates_by_category',
    'get_rule_templates_by_type'
]