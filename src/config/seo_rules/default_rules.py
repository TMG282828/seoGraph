"""
Default SEO rules configuration.

Contains predefined SEO rules that are automatically loaded when the
rules engine is initialized.
"""

from typing import List

from .models import SEORule, RuleType, RuleCategory, RuleSeverity


def create_default_rules() -> List[SEORule]:
    """
    Create default SEO rules.
    
    Returns:
        List of default SEO rules covering common optimization requirements
    """
    return [
        # Keyword density rules
        SEORule(
            name="Primary Keyword Density",
            description="Primary keyword should appear 1-3% of total content",
            rule_type=RuleType.KEYWORD_DENSITY,
            category=RuleCategory.KEYWORDS,
            severity=RuleSeverity.WARNING,
            parameters={
                'min_density': 1.0,
                'max_density': 3.0,
                'keyword_type': 'primary'
            },
            validation_function='keyword_density',
            tenant_id='default',
            created_by='system'
        ),
        
        # Title optimization rules
        SEORule(
            name="Title Length",
            description="Title should be 30-60 characters for optimal display",
            rule_type=RuleType.TITLE_OPTIMIZATION,
            category=RuleCategory.ON_PAGE,
            severity=RuleSeverity.WARNING,
            parameters={
                'min_length': 30,
                'max_length': 60,
                'optimal_length': 55
            },
            validation_function='title_optimization',
            tenant_id='default',
            created_by='system'
        ),
        
        SEORule(
            name="Title Keyword Inclusion",
            description="Title should include the target keyword",
            rule_type=RuleType.TITLE_OPTIMIZATION,
            category=RuleCategory.ON_PAGE,
            severity=RuleSeverity.CRITICAL,
            weight=2.0,
            parameters={
                'require_target_keyword': True,
                'keyword_position_weight': True
            },
            validation_function='title_optimization',
            tenant_id='default',
            created_by='system'
        ),
        
        # Meta description rules
        SEORule(
            name="Meta Description Length",
            description="Meta description should be 150-160 characters",
            rule_type=RuleType.META_DESCRIPTION,
            category=RuleCategory.ON_PAGE,
            severity=RuleSeverity.WARNING,
            parameters={
                'min_length': 150,
                'max_length': 160,
                'optimal_length': 155
            },
            validation_function='meta_description',
            tenant_id='default',
            created_by='system'
        ),
        
        SEORule(
            name="Meta Description Presence",
            description="Content must have a meta description",
            rule_type=RuleType.META_DESCRIPTION,
            category=RuleCategory.ON_PAGE,
            severity=RuleSeverity.CRITICAL,
            weight=1.5,
            parameters={
                'required': True
            },
            validation_function='meta_description',
            tenant_id='default',
            created_by='system'
        ),
        
        # Heading structure rules
        SEORule(
            name="H1 Tag Presence",
            description="Content should have exactly one H1 tag",
            rule_type=RuleType.HEADING_STRUCTURE,
            category=RuleCategory.STRUCTURE,
            severity=RuleSeverity.CRITICAL,
            weight=2.0,
            parameters={
                'required_count': 1,
                'max_count': 1
            },
            validation_function='heading_structure',
            tenant_id='default',
            created_by='system'
        ),
        
        SEORule(
            name="Heading Hierarchy",
            description="Headings should follow proper hierarchy (H1->H2->H3->etc)",
            rule_type=RuleType.HEADING_STRUCTURE,
            category=RuleCategory.STRUCTURE,
            severity=RuleSeverity.WARNING,
            parameters={
                'enforce_hierarchy': True,
                'allow_skipping': False
            },
            validation_function='heading_structure',
            tenant_id='default',
            created_by='system'
        ),
        
        # Content length rules
        SEORule(
            name="Minimum Content Length",
            description="Content should be at least 300 words for SEO value",
            rule_type=RuleType.CONTENT_LENGTH,
            category=RuleCategory.CONTENT,
            severity=RuleSeverity.WARNING,
            parameters={
                'min_words': 300,
                'recommended_words': 1000
            },
            validation_function='content_length',
            tenant_id='default',
            created_by='system'
        ),
        
        SEORule(
            name="Optimal Content Length",
            description="Content should be 800+ words for better rankings",
            rule_type=RuleType.CONTENT_LENGTH,
            category=RuleCategory.CONTENT,
            severity=RuleSeverity.SUGGESTION,
            parameters={
                'min_words': 800,
                'optimal_words': 1500
            },
            validation_function='content_length',
            tenant_id='default',
            created_by='system'
        ),
        
        # Internal linking rules
        SEORule(
            name="Internal Links Count",
            description="Content should have 2-5 internal links",
            rule_type=RuleType.INTERNAL_LINKING,
            category=RuleCategory.STRUCTURE,
            severity=RuleSeverity.SUGGESTION,
            parameters={
                'min_links': 2,
                'max_links': 5,
                'optimal_links': 3
            },
            validation_function='internal_linking',
            tenant_id='default',
            created_by='system'
        ),
        
        # Image optimization rules
        SEORule(
            name="Image Alt Text",
            description="All images should have descriptive alt text",
            rule_type=RuleType.IMAGE_OPTIMIZATION,
            category=RuleCategory.ACCESSIBILITY,
            severity=RuleSeverity.WARNING,
            weight=1.5,
            parameters={
                'require_alt_text': True,
                'min_alt_length': 10,
                'max_alt_length': 125
            },
            validation_function='image_optimization',
            tenant_id='default',
            created_by='system'
        ),
        
        # Readability rules
        SEORule(
            name="Sentence Length",
            description="Average sentence length should be reasonable for readability",
            rule_type=RuleType.READABILITY,
            category=RuleCategory.CONTENT,
            severity=RuleSeverity.SUGGESTION,
            parameters={
                'max_avg_sentence_length': 20,
                'optimal_avg_sentence_length': 15
            },
            validation_function='readability',
            tenant_id='default',
            created_by='system'
        ),
        
        SEORule(
            name="Passive Voice Usage",
            description="Limit passive voice usage for better readability",
            rule_type=RuleType.READABILITY,
            category=RuleCategory.CONTENT,
            severity=RuleSeverity.SUGGESTION,
            parameters={
                'max_passive_ratio': 0.1,
                'optimal_passive_ratio': 0.05
            },
            validation_function='readability',
            tenant_id='default',
            created_by='system'
        ),
        
        # Semantic keywords rules
        SEORule(
            name="Keyword Variety",
            description="Use semantic variations of target keywords",
            rule_type=RuleType.SEMANTIC_KEYWORDS,
            category=RuleCategory.KEYWORDS,
            severity=RuleSeverity.SUGGESTION,
            parameters={
                'min_variations': 3,
                'semantic_density': 0.5
            },
            validation_function='semantic_keywords',
            tenant_id='default',
            created_by='system'
        ),
        
        # Technical SEO rules
        SEORule(
            name="Content Freshness",
            description="Content should be updated regularly",
            rule_type=RuleType.CONTENT_FRESHNESS,
            category=RuleCategory.TECHNICAL,
            severity=RuleSeverity.INFO,
            parameters={
                'max_age_days': 365,
                'recommended_update_frequency': 180
            },
            validation_function='content_freshness',
            tenant_id='default',
            created_by='system'
        ),
    ]


def get_rule_templates_by_category() -> dict:
    """
    Get rule templates organized by category.
    
    Returns:
        Dictionary of rule templates organized by category
    """
    rules = create_default_rules()
    templates = {}
    
    for rule in rules:
        category = rule.category.value
        if category not in templates:
            templates[category] = []
        
        templates[category].append({
            'name': rule.name,
            'description': rule.description,
            'rule_type': rule.rule_type.value,
            'severity': rule.severity.value,
            'parameters': rule.parameters,
            'validation_function': rule.validation_function
        })
    
    return templates


def get_rule_templates_by_type() -> dict:
    """
    Get rule templates organized by type.
    
    Returns:
        Dictionary of rule templates organized by type
    """
    rules = create_default_rules()
    templates = {}
    
    for rule in rules:
        rule_type = rule.rule_type.value
        if rule_type not in templates:
            templates[rule_type] = []
        
        templates[rule_type].append({
            'name': rule.name,
            'description': rule.description,
            'category': rule.category.value,
            'severity': rule.severity.value,
            'parameters': rule.parameters,
            'validation_function': rule.validation_function
        })
    
    return templates