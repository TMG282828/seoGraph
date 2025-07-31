"""
SEO Rules Engine for the SEO Content Knowledge Graph System.

This module provides rule-based SEO validation, content optimization rules,
keyword optimization guidelines, and customizable rule management.

REFACTORED: This file now imports from the modularized seo_rules package
for better maintainability and organization.
"""

# Import all functionality from the new modular structure
from src.config.seo_rules import (
    # Main engine
    SEORulesEngine,
    
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
    AuditRequest,
    
    # Validators
    SEOValidators,
    
    # Default rules
    create_default_rules,
    get_rule_templates_by_category,
    get_rule_templates_by_type
)

# Backward compatibility - create main engine instance function
def create_seo_rules_engine(neo4j_client):
    """Create SEO rules engine instance for backward compatibility."""
    return SEORulesEngine(neo4j_client)


# Main execution for backward compatibility
if __name__ == "__main__":
    print("SEO Rules Engine - Modularized")
    print("Available components:")
    print("- SEORulesEngine: Main rules engine")
    print("- SEORule, RuleViolation, SEOAuditResult: Core models")
    print("- RuleType, RuleSeverity, RuleCategory: Enums")
    print("- SEOValidators: Validation functions")
    print("- create_default_rules: Default rule templates")