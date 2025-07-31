"""
SEO Rules Engine.

Main engine for managing and executing SEO rules, including rule storage,
content auditing, and violation reporting.
"""

import asyncio
from typing import Any, Dict, List, Callable, Optional
from datetime import datetime, timezone

import structlog
from cachetools import TTLCache

from database.neo4j_client import Neo4jClient
from config.settings import get_settings
from .models import (
    SEORuleError,
    SEORule,
    RuleViolation,
    SEOAuditResult,
    RuleType,
    RuleCategory,
    RuleSeverity
)
from .validators import SEOValidators
from .default_rules import create_default_rules

logger = structlog.get_logger(__name__)


class SEORulesEngine:
    """
    SEO rules engine for content optimization.
    
    Provides comprehensive SEO rule management and content auditing capabilities:
    - Rule-based SEO validation
    - Content optimization recommendations
    - Keyword optimization guidelines
    - Customizable rule management
    - Performance-optimized caching
    """
    
    def __init__(self, neo4j_client: Neo4jClient):
        """
        Initialize SEO rules engine.
        
        Args:
            neo4j_client: Neo4j database client for rule storage
        """
        self.neo4j_client = neo4j_client
        self.settings = get_settings()
        
        # Caches for performance optimization
        self.rules_cache = TTLCache(maxsize=500, ttl=3600)    # 1 hour
        self.audit_cache = TTLCache(maxsize=1000, ttl=1800)   # 30 minutes
        
        # Initialize validators
        self.validators = SEOValidators()
        
        # Rule validators mapping
        self.rule_validators: Dict[str, Callable] = {
            'keyword_density': self.validators.validate_keyword_density,
            'title_optimization': self.validators.validate_title_optimization,
            'meta_description': self.validators.validate_meta_description,
            'heading_structure': self.validators.validate_heading_structure,
            'content_length': self.validators.validate_content_length,
            'internal_linking': self.validators.validate_internal_linking,
            'image_optimization': self.validators.validate_image_optimization,
            'readability': self.validators.validate_readability,
        }
        
        logger.info("SEO rules engine initialized")
    
    async def initialize(self) -> None:
        """
        Initialize the SEO rules engine.
        
        Sets up database constraints and loads default rules if needed.
        """
        try:
            # Set up Neo4j constraints
            await self._setup_constraints()
            
            # Load default rules if needed
            await self._load_default_rules()
            
            logger.info("SEO rules engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize SEO rules engine: {e}")
            raise SEORuleError(f"Initialization failed: {e}")
    
    async def _setup_constraints(self) -> None:
        """Set up Neo4j constraints for SEO rules data."""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:SEORule) REQUIRE r.rule_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (v:RuleViolation) REQUIRE v.violation_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:SEOAuditResult) REQUIRE a.audit_id IS UNIQUE"
        ]
        
        for constraint in constraints:
            try:
                await self.neo4j_client.execute_query(constraint)
            except Exception as e:
                logger.warning(f"Failed to create constraint: {constraint}, error: {e}")
    
    async def _load_default_rules(self) -> None:
        """Load default rules if they don't exist."""
        try:
            # Check if default rules already exist
            existing_rules = await self.get_rules("default")
            
            if not existing_rules:
                logger.info("Loading default SEO rules")
                default_rules = create_default_rules()
                
                for rule in default_rules:
                    await self.save_rule(rule)
                
                logger.info(f"Loaded {len(default_rules)} default SEO rules")
            
        except Exception as e:
            logger.error(f"Failed to load default rules: {e}")
    
    async def save_rule(self, rule: SEORule) -> str:
        """
        Save an SEO rule to the database.
        
        Args:
            rule: SEO rule to save
            
        Returns:
            Rule ID of saved rule
        """
        try:
            # Update timestamp
            rule.updated_at = datetime.now(timezone.utc)
            
            # Save to Neo4j
            query = """
            MERGE (r:SEORule {rule_id: $rule_id})
            SET r.name = $name,
                r.description = $description,
                r.rule_type = $rule_type,
                r.category = $category,
                r.severity = $severity,
                r.enabled = $enabled,
                r.weight = $weight,
                r.parameters = $parameters,
                r.thresholds = $thresholds,
                r.applicable_content_types = $applicable_content_types,
                r.validation_function = $validation_function,
                r.tenant_id = $tenant_id,
                r.created_by = $created_by,
                r.created_at = $created_at,
                r.updated_at = $updated_at
            RETURN r.rule_id as rule_id
            """
            
            result = await self.neo4j_client.execute_query(query, {
                'rule_id': rule.rule_id,
                'name': rule.name,
                'description': rule.description,
                'rule_type': rule.rule_type.value,
                'category': rule.category.value,
                'severity': rule.severity.value,
                'enabled': rule.enabled,
                'weight': rule.weight,
                'parameters': rule.parameters,
                'thresholds': rule.thresholds,
                'applicable_content_types': rule.applicable_content_types,
                'validation_function': rule.validation_function,
                'tenant_id': rule.tenant_id,
                'created_by': rule.created_by,
                'created_at': rule.created_at.isoformat(),
                'updated_at': rule.updated_at.isoformat() if rule.updated_at else None
            })
            
            # Clear cache
            cache_key = f"rules_{rule.tenant_id}"
            if cache_key in self.rules_cache:
                del self.rules_cache[cache_key]
            
            logger.info(f"Saved SEO rule: {rule.name} ({rule.rule_id})")
            return rule.rule_id
            
        except Exception as e:
            logger.error(f"Failed to save rule: {e}")
            raise SEORuleError(f"Failed to save rule: {e}")
    
    async def get_rules(self, tenant_id: str, enabled_only: bool = True) -> List[SEORule]:
        """
        Get SEO rules for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            enabled_only: Whether to return only enabled rules
            
        Returns:
            List of SEO rules
        """
        try:
            cache_key = f"rules_{tenant_id}_{enabled_only}"
            if cache_key in self.rules_cache:
                return self.rules_cache[cache_key]
            
            # Build query
            where_clause = "r.tenant_id = $tenant_id"
            if enabled_only:
                where_clause += " AND r.enabled = true"
            
            query = f"""
            MATCH (r:SEORule)
            WHERE {where_clause}
            RETURN r
            ORDER BY r.category, r.weight DESC
            """
            
            result = await self.neo4j_client.execute_query(query, {'tenant_id': tenant_id})
            
            rules = []
            for record in result:
                rule_data = dict(record['r'])
                
                # Convert timestamps
                if 'created_at' in rule_data:
                    rule_data['created_at'] = datetime.fromisoformat(rule_data['created_at'])
                if 'updated_at' in rule_data and rule_data['updated_at']:
                    rule_data['updated_at'] = datetime.fromisoformat(rule_data['updated_at'])
                
                # Convert enums
                rule_data['rule_type'] = RuleType(rule_data['rule_type'])
                rule_data['category'] = RuleCategory(rule_data['category'])
                rule_data['severity'] = RuleSeverity(rule_data['severity'])
                
                rules.append(SEORule(**rule_data))
            
            # Cache results
            self.rules_cache[cache_key] = rules
            
            return rules
            
        except Exception as e:
            logger.error(f"Failed to get rules: {e}")
            raise SEORuleError(f"Failed to get rules: {e}")
    
    async def audit_content(self, 
                          content: str,
                          content_id: str,
                          content_type: str,
                          metadata: Dict[str, Any],
                          tenant_id: str,
                          rule_categories: Optional[List[RuleCategory]] = None) -> SEOAuditResult:
        """
        Audit content against SEO rules.
        
        Args:
            content: Content to audit
            content_id: Content identifier
            content_type: Type of content
            metadata: Content metadata (title, description, etc.)
            tenant_id: Tenant identifier
            rule_categories: Specific categories to audit (optional)
            
        Returns:
            SEO audit result with violations and recommendations
        """
        try:
            start_time = datetime.now()
            
            # Check cache
            cache_key = f"audit_{content_id}_{hash(content)}_{tenant_id}"
            if cache_key in self.audit_cache:
                return self.audit_cache[cache_key]
            
            # Get applicable rules
            all_rules = await self.get_rules(tenant_id, enabled_only=True)
            
            # Filter by content type and categories
            applicable_rules = []
            for rule in all_rules:
                if rule.is_applicable_to_content_type(content_type):
                    if rule_categories is None or rule.category in rule_categories:
                        applicable_rules.append(rule)
            
            # Run validation for each rule
            all_violations = []
            category_scores = {}
            
            for rule in applicable_rules:
                try:
                    violations = await self._validate_rule(rule, content, content_id, metadata)
                    all_violations.extend(violations)
                    
                    # Calculate category scores
                    category = rule.category.value
                    if category not in category_scores:
                        category_scores[category] = {'total': 0, 'passed': 0}
                    
                    category_scores[category]['total'] += 1
                    if not violations:
                        category_scores[category]['passed'] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to validate rule {rule.name}: {e}")
                    continue
            
            # Calculate scores
            overall_score = self._calculate_overall_score(all_violations, applicable_rules)
            
            # Convert category scores to percentages
            final_category_scores = {}
            for category, scores in category_scores.items():
                if scores['total'] > 0:
                    final_category_scores[category] = (scores['passed'] / scores['total']) * 100
            
            # Count violations by severity
            critical_count = len([v for v in all_violations if v.severity == RuleSeverity.CRITICAL])
            warning_count = len([v for v in all_violations if v.severity == RuleSeverity.WARNING])
            suggestion_count = len([v for v in all_violations if v.severity == RuleSeverity.SUGGESTION])
            
            # Generate recommendations
            top_recommendations = self._generate_recommendations(all_violations)
            quick_fixes = self._generate_quick_fixes(all_violations)
            
            # Create audit result
            audit_result = SEOAuditResult(
                content_id=content_id,
                overall_score=overall_score,
                category_scores=final_category_scores,
                violations=all_violations,
                critical_violations=critical_count,
                warning_violations=warning_count,
                suggestion_violations=suggestion_count,
                top_recommendations=top_recommendations,
                quick_fixes=quick_fixes,
                analyzed_rules=len(applicable_rules),
                passed_rules=len(applicable_rules) - len([v.rule_id for v in all_violations]),
                analysis_duration=(datetime.now() - start_time).total_seconds(),
                tenant_id=tenant_id
            )
            
            # Cache result
            self.audit_cache[cache_key] = audit_result
            
            logger.info(f"Audited content {content_id}: score {overall_score:.1f}, {len(all_violations)} violations")
            
            return audit_result
            
        except Exception as e:
            logger.error(f"Failed to audit content: {e}")
            raise SEORuleError(f"Content audit failed: {e}")
    
    async def _validate_rule(self, 
                           rule: SEORule,
                           content: str,
                           content_id: str,
                           metadata: Dict[str, Any]) -> List[RuleViolation]:
        """
        Validate content against a single rule.
        
        Args:
            rule: SEO rule to validate
            content: Content to validate
            content_id: Content identifier
            metadata: Content metadata
            
        Returns:
            List of rule violations
        """
        try:
            # Get validator function
            validator_name = rule.validation_function
            if not validator_name or validator_name not in self.rule_validators:
                logger.warning(f"No validator found for rule {rule.name}")
                return []
            
            validator = self.rule_validators[validator_name]
            
            # Run validation
            violations = await validator(rule, content, content_id, metadata)
            
            return violations if violations else []
            
        except Exception as e:
            logger.error(f"Failed to validate rule {rule.name}: {e}")
            return []
    
    def _calculate_overall_score(self, violations: List[RuleViolation], rules: List[SEORule]) -> float:
        """
        Calculate overall SEO score.
        
        Args:
            violations: List of rule violations
            rules: List of rules that were checked
            
        Returns:
            Overall score (0-100)
        """
        if not rules:
            return 0.0
        
        # Weight violations by severity
        penalty = 0.0
        max_penalty = 0.0
        
        for rule in rules:
            max_penalty += rule.weight
            
            # Find violations for this rule
            rule_violations = [v for v in violations if v.rule_id == rule.rule_id]
            
            if rule_violations:
                # Apply penalty based on severity
                severity_penalty = {
                    RuleSeverity.CRITICAL: 1.0,
                    RuleSeverity.WARNING: 0.6,
                    RuleSeverity.SUGGESTION: 0.3,
                    RuleSeverity.INFO: 0.1
                }
                
                for violation in rule_violations:
                    penalty += rule.weight * severity_penalty.get(violation.severity, 0.3)
        
        if max_penalty == 0:
            return 100.0
        
        score = max(0.0, 100.0 - (penalty / max_penalty * 100.0))
        return round(score, 1)
    
    def _generate_recommendations(self, violations: List[RuleViolation]) -> List[str]:
        """
        Generate top recommendations from violations.
        
        Args:
            violations: List of rule violations
            
        Returns:
            List of top recommendations
        """
        if not violations:
            return ["Content meets all SEO requirements!"]
        
        # Sort by priority and get top recommendations
        sorted_violations = sorted(violations, key=lambda v: v.priority, reverse=True)
        
        recommendations = []
        seen_recommendations = set()
        
        for violation in sorted_violations[:5]:  # Top 5
            if violation.recommendation not in seen_recommendations:
                recommendations.append(violation.recommendation)
                seen_recommendations.add(violation.recommendation)
        
        return recommendations
    
    def _generate_quick_fixes(self, violations: List[RuleViolation]) -> List[str]:
        """
        Generate quick fixes from violations.
        
        Args:
            violations: List of rule violations
            
        Returns:
            List of quick fixes
        """
        quick_fixes = []
        
        # Look for common quick fixes
        for violation in violations:
            if violation.severity in [RuleSeverity.CRITICAL, RuleSeverity.WARNING]:
                if "title" in violation.message.lower():
                    quick_fixes.append("Optimize page title length and keyword placement")
                elif "meta description" in violation.message.lower():
                    quick_fixes.append("Add or improve meta description")
                elif "h1" in violation.message.lower():
                    quick_fixes.append("Add or fix H1 heading tag")
                elif "alt text" in violation.message.lower():
                    quick_fixes.append("Add alt text to images")
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(quick_fixes))
    
    async def get_rule_by_id(self, rule_id: str, tenant_id: str) -> Optional[SEORule]:
        """
        Get a specific rule by ID.
        
        Args:
            rule_id: Rule identifier
            tenant_id: Tenant identifier
            
        Returns:
            SEO rule if found, None otherwise
        """
        try:
            query = """
            MATCH (r:SEORule {rule_id: $rule_id, tenant_id: $tenant_id})
            RETURN r
            """
            
            result = await self.neo4j_client.execute_query(query, {
                'rule_id': rule_id,
                'tenant_id': tenant_id
            })
            
            if not result:
                return None
            
            rule_data = dict(result[0]['r'])
            
            # Convert timestamps and enums as in get_rules
            if 'created_at' in rule_data:
                rule_data['created_at'] = datetime.fromisoformat(rule_data['created_at'])
            if 'updated_at' in rule_data and rule_data['updated_at']:
                rule_data['updated_at'] = datetime.fromisoformat(rule_data['updated_at'])
            
            rule_data['rule_type'] = RuleType(rule_data['rule_type'])
            rule_data['category'] = RuleCategory(rule_data['category'])
            rule_data['severity'] = RuleSeverity(rule_data['severity'])
            
            return SEORule(**rule_data)
            
        except Exception as e:
            logger.error(f"Failed to get rule by ID: {e}")
            return None
    
    async def delete_rule(self, rule_id: str, tenant_id: str) -> bool:
        """
        Delete a rule.
        
        Args:
            rule_id: Rule identifier
            tenant_id: Tenant identifier
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            query = """
            MATCH (r:SEORule {rule_id: $rule_id, tenant_id: $tenant_id})
            DELETE r
            RETURN count(r) as deleted_count
            """
            
            result = await self.neo4j_client.execute_query(query, {
                'rule_id': rule_id,
                'tenant_id': tenant_id
            })
            
            deleted = result[0]['deleted_count'] > 0 if result else False
            
            if deleted:
                # Clear cache
                cache_key = f"rules_{tenant_id}"
                if cache_key in self.rules_cache:
                    del self.rules_cache[cache_key]
                
                logger.info(f"Deleted SEO rule: {rule_id}")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Failed to delete rule: {e}")
            return False