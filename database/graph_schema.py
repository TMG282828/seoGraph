"""
Neo4j graph schema definitions for the SEO Content Knowledge Graph System.

This module defines the graph database schema including node labels,
relationship types, constraints, and indexes for optimal performance.
"""

from enum import Enum
from typing import Dict, List, Any
import structlog

logger = structlog.get_logger(__name__)


class NodeLabel(str, Enum):
    """Node labels for the knowledge graph."""
    
    # Core Content Nodes
    CONTENT = "Content"
    TOPIC = "Topic"
    KEYWORD = "Keyword"
    
    # User and Tenant Nodes
    USER = "User"
    TENANT = "Tenant"
    
    # SEO and Analytics Nodes
    SEARCH_QUERY = "SearchQuery"
    TREND = "Trend"
    COMPETITOR = "Competitor"
    
    # Content Structure Nodes
    SECTION = "Section"
    PARAGRAPH = "Paragraph"
    SENTENCE = "Sentence"
    
    # External Entities
    DOMAIN = "Domain"
    URL = "Url"
    AUTHOR = "Author"
    
    # Semantic Entities
    ENTITY = "Entity"
    CONCEPT = "Concept"
    CATEGORY = "Category"


class RelationshipType(str, Enum):
    """Relationship types for the knowledge graph."""
    
    # Content Relationships
    RELATES_TO = "RELATES_TO"
    CONTAINS = "CONTAINS"
    REFERENCES = "REFERENCES"
    LINKS_TO = "LINKS_TO"
    SIMILAR_TO = "SIMILAR_TO"
    DERIVED_FROM = "DERIVED_FROM"
    
    # Topic and Keyword Relationships
    HAS_TOPIC = "HAS_TOPIC"
    HAS_KEYWORD = "HAS_KEYWORD"
    KEYWORD_IN_TOPIC = "KEYWORD_IN_TOPIC"
    TOPIC_HIERARCHY = "TOPIC_HIERARCHY"
    SUBTOPIC_OF = "SUBTOPIC_OF"
    
    # User and Tenant Relationships
    OWNS = "OWNS"
    MEMBER_OF = "MEMBER_OF"
    CREATED_BY = "CREATED_BY"
    EDITED_BY = "EDITED_BY"
    
    # SEO Relationships
    RANKS_FOR = "RANKS_FOR"
    COMPETES_WITH = "COMPETES_WITH"
    OPTIMIZES_FOR = "OPTIMIZES_FOR"
    TARGETS = "TARGETS"
    
    # Temporal Relationships
    PRECEDES = "PRECEDES"
    FOLLOWS = "FOLLOWS"
    UPDATED_FROM = "UPDATED_FROM"
    
    # Semantic Relationships
    IS_A = "IS_A"
    PART_OF = "PART_OF"
    MENTIONS = "MENTIONS"
    DEFINES = "DEFINES"
    EXPLAINS = "EXPLAINS"
    
    # Performance Relationships
    PERFORMS_BETTER_THAN = "PERFORMS_BETTER_THAN"
    GENERATES_TRAFFIC_TO = "GENERATES_TRAFFIC_TO"
    CONVERTS_VIA = "CONVERTS_VIA"


class GraphConstraints:
    """Graph database constraints for data integrity."""
    
    UNIQUE_CONSTRAINTS = [
        # Unique identifiers
        {
            "label": NodeLabel.CONTENT,
            "property": "id",
            "name": "content_id_unique"
        },
        {
            "label": NodeLabel.TOPIC,
            "property": "name",
            "name": "topic_name_unique"
        },
        {
            "label": NodeLabel.KEYWORD,
            "property": "text",
            "name": "keyword_text_unique"
        },
        {
            "label": NodeLabel.USER,
            "property": "id",
            "name": "user_id_unique"
        },
        {
            "label": NodeLabel.TENANT,
            "property": "id",
            "name": "tenant_id_unique"
        },
        {
            "label": NodeLabel.URL,
            "property": "url",
            "name": "url_unique"
        },
        {
            "label": NodeLabel.DOMAIN,
            "property": "domain",
            "name": "domain_unique"
        },
        {
            "label": NodeLabel.ENTITY,
            "property": "name",
            "name": "entity_name_unique"
        },
    ]
    
    EXISTENCE_CONSTRAINTS = [
        # Required properties
        {
            "label": NodeLabel.CONTENT,
            "property": "title",
            "name": "content_title_exists"
        },
        {
            "label": NodeLabel.CONTENT,
            "property": "tenant_id",
            "name": "content_tenant_exists"
        },
        {
            "label": NodeLabel.TOPIC,
            "property": "name",
            "name": "topic_name_exists"
        },
        {
            "label": NodeLabel.KEYWORD,
            "property": "text",
            "name": "keyword_text_exists"
        },
        {
            "label": NodeLabel.USER,
            "property": "email",
            "name": "user_email_exists"
        },
        {
            "label": NodeLabel.TENANT,
            "property": "name",
            "name": "tenant_name_exists"
        },
    ]


class GraphIndexes:
    """Graph database indexes for performance optimization."""
    
    RANGE_INDEXES = [
        # Content indexes
        {
            "label": NodeLabel.CONTENT,
            "property": "tenant_id",
            "name": "content_tenant_idx"
        },
        {
            "label": NodeLabel.CONTENT,
            "property": "status",
            "name": "content_status_idx"
        },
        {
            "label": NodeLabel.CONTENT,
            "property": "content_type",
            "name": "content_type_idx"
        },
        {
            "label": NodeLabel.CONTENT,
            "property": "created_at",
            "name": "content_created_idx"
        },
        {
            "label": NodeLabel.CONTENT,
            "property": "word_count",
            "name": "content_word_count_idx"
        },
        
        # Topic indexes
        {
            "label": NodeLabel.TOPIC,
            "property": "tenant_id",
            "name": "topic_tenant_idx"
        },
        {
            "label": NodeLabel.TOPIC,
            "property": "category",
            "name": "topic_category_idx"
        },
        
        # Keyword indexes
        {
            "label": NodeLabel.KEYWORD,
            "property": "tenant_id",
            "name": "keyword_tenant_idx"
        },
        {
            "label": NodeLabel.KEYWORD,
            "property": "search_volume",
            "name": "keyword_search_volume_idx"
        },
        {
            "label": NodeLabel.KEYWORD,
            "property": "competition",
            "name": "keyword_competition_idx"
        },
        
        # User and tenant indexes
        {
            "label": NodeLabel.USER,
            "property": "created_at",
            "name": "user_created_idx"
        },
        {
            "label": NodeLabel.TENANT,
            "property": "created_at",
            "name": "tenant_created_idx"
        },
        
        # URL and domain indexes
        {
            "label": NodeLabel.URL,
            "property": "domain",
            "name": "url_domain_idx"
        },
        {
            "label": NodeLabel.URL,
            "property": "last_crawled",
            "name": "url_last_crawled_idx"
        },
    ]
    
    FULLTEXT_INDEXES = [
        # Content full-text search
        {
            "name": "content_fulltext",
            "label": NodeLabel.CONTENT,
            "properties": ["title", "content", "meta_description"]
        },
        {
            "name": "topic_fulltext",
            "label": NodeLabel.TOPIC,
            "properties": ["name", "description"]
        },
        {
            "name": "keyword_fulltext",
            "label": NodeLabel.KEYWORD,
            "properties": ["text", "variations"]
        },
        {
            "name": "entity_fulltext",
            "label": NodeLabel.ENTITY,
            "properties": ["name", "description", "aliases"]
        },
    ]


class GraphSchema:
    """Complete graph schema definition and management."""
    
    def __init__(self):
        """Initialize graph schema manager."""
        self.constraints = GraphConstraints()
        self.indexes = GraphIndexes()
    
    def get_schema_creation_queries(self) -> List[str]:
        """
        Get all schema creation queries.
        
        Returns:
            List of Cypher queries for schema creation
        """
        queries = []
        
        # Unique constraints
        for constraint in self.constraints.UNIQUE_CONSTRAINTS:
            query = (
                f"CREATE CONSTRAINT {constraint['name']} IF NOT EXISTS "
                f"FOR (n:{constraint['label']}) "
                f"REQUIRE n.{constraint['property']} IS UNIQUE"
            )
            queries.append(query)
        
        # Existence constraints
        for constraint in self.constraints.EXISTENCE_CONSTRAINTS:
            query = (
                f"CREATE CONSTRAINT {constraint['name']} IF NOT EXISTS "
                f"FOR (n:{constraint['label']}) "
                f"REQUIRE n.{constraint['property']} IS NOT NULL"
            )
            queries.append(query)
        
        # Range indexes
        for index in self.indexes.RANGE_INDEXES:
            query = (
                f"CREATE INDEX {index['name']} IF NOT EXISTS "
                f"FOR (n:{index['label']}) "
                f"ON (n.{index['property']})"
            )
            queries.append(query)
        
        # Full-text indexes
        for index in self.indexes.FULLTEXT_INDEXES:
            properties_str = ", ".join([f"n.{prop}" for prop in index['properties']])
            query = (
                f"CREATE FULLTEXT INDEX {index['name']} IF NOT EXISTS "
                f"FOR (n:{index['label']}) "
                f"ON EACH [{properties_str}]"
            )
            queries.append(query)
        
        return queries
    
    def get_node_properties(self, label: NodeLabel) -> Dict[str, Any]:
        """
        Get expected properties for a node label.
        
        Args:
            label: Node label
            
        Returns:
            Dictionary of property definitions
        """
        properties = {
            NodeLabel.CONTENT: {
                "id": {"type": "string", "required": True, "unique": True},
                "title": {"type": "string", "required": True},
                "content": {"type": "string", "required": False},
                "content_type": {"type": "string", "required": True},
                "status": {"type": "string", "required": True},
                "tenant_id": {"type": "string", "required": True},
                "author_id": {"type": "string", "required": True},
                "word_count": {"type": "integer", "required": False},
                "readability_score": {"type": "float", "required": False},
                "seo_score": {"type": "float", "required": False},
                "created_at": {"type": "datetime", "required": True},
                "updated_at": {"type": "datetime", "required": False},
                "published_at": {"type": "datetime", "required": False},
                "meta_description": {"type": "string", "required": False},
                "meta_keywords": {"type": "list", "required": False},
                "slug": {"type": "string", "required": False},
                "canonical_url": {"type": "string", "required": False},
                "language": {"type": "string", "required": False, "default": "en"},
            },
            
            NodeLabel.TOPIC: {
                "name": {"type": "string", "required": True, "unique": True},
                "description": {"type": "string", "required": False},
                "category": {"type": "string", "required": False},
                "tenant_id": {"type": "string", "required": True},
                "search_volume": {"type": "integer", "required": False},
                "trend_direction": {"type": "string", "required": False},
                "importance_score": {"type": "float", "required": False},
                "created_at": {"type": "datetime", "required": True},
                "updated_at": {"type": "datetime", "required": False},
            },
            
            NodeLabel.KEYWORD: {
                "text": {"type": "string", "required": True, "unique": True},
                "tenant_id": {"type": "string", "required": True},
                "search_volume": {"type": "integer", "required": False},
                "competition": {"type": "float", "required": False},
                "cpc": {"type": "float", "required": False},
                "trend_direction": {"type": "string", "required": False},
                "variations": {"type": "list", "required": False},
                "intent": {"type": "string", "required": False},
                "difficulty": {"type": "float", "required": False},
                "created_at": {"type": "datetime", "required": True},
                "updated_at": {"type": "datetime", "required": False},
            },
            
            NodeLabel.USER: {
                "id": {"type": "string", "required": True, "unique": True},
                "email": {"type": "string", "required": True},
                "name": {"type": "string", "required": False},
                "role": {"type": "string", "required": False},
                "created_at": {"type": "datetime", "required": True},
                "last_login": {"type": "datetime", "required": False},
                "preferences": {"type": "map", "required": False},
            },
            
            NodeLabel.TENANT: {
                "id": {"type": "string", "required": True, "unique": True},
                "name": {"type": "string", "required": True},
                "description": {"type": "string", "required": False},
                "domain": {"type": "string", "required": False},
                "industry": {"type": "string", "required": False},
                "settings": {"type": "map", "required": False},
                "created_at": {"type": "datetime", "required": True},
                "updated_at": {"type": "datetime", "required": False},
            },
            
            NodeLabel.URL: {
                "url": {"type": "string", "required": True, "unique": True},
                "domain": {"type": "string", "required": True},
                "title": {"type": "string", "required": False},
                "description": {"type": "string", "required": False},
                "status_code": {"type": "integer", "required": False},
                "last_crawled": {"type": "datetime", "required": False},
                "content_hash": {"type": "string", "required": False},
                "is_internal": {"type": "boolean", "required": False},
            },
            
            NodeLabel.ENTITY: {
                "name": {"type": "string", "required": True, "unique": True},
                "type": {"type": "string", "required": True},
                "description": {"type": "string", "required": False},
                "aliases": {"type": "list", "required": False},
                "confidence": {"type": "float", "required": False},
                "source": {"type": "string", "required": False},
                "created_at": {"type": "datetime", "required": True},
            },
        }
        
        return properties.get(label, {})
    
    def get_relationship_properties(self, rel_type: RelationshipType) -> Dict[str, Any]:
        """
        Get expected properties for a relationship type.
        
        Args:
            rel_type: Relationship type
            
        Returns:
            Dictionary of property definitions
        """
        common_properties = {
            "created_at": {"type": "datetime", "required": True},
            "weight": {"type": "float", "required": False, "default": 1.0},
            "confidence": {"type": "float", "required": False},
            "source": {"type": "string", "required": False},
        }
        
        specific_properties = {
            RelationshipType.SIMILAR_TO: {
                "similarity_score": {"type": "float", "required": True},
                "algorithm": {"type": "string", "required": False},
            },
            
            RelationshipType.RANKS_FOR: {
                "position": {"type": "integer", "required": True},
                "search_engine": {"type": "string", "required": False, "default": "google"},
                "location": {"type": "string", "required": False},
                "device": {"type": "string", "required": False, "default": "desktop"},
                "date_checked": {"type": "datetime", "required": True},
            },
            
            RelationshipType.HAS_KEYWORD: {
                "density": {"type": "float", "required": False},
                "prominence": {"type": "float", "required": False},
                "position": {"type": "string", "required": False},
            },
            
            RelationshipType.LINKS_TO: {
                "anchor_text": {"type": "string", "required": False},
                "link_type": {"type": "string", "required": False},
                "nofollow": {"type": "boolean", "required": False, "default": False},
            },
            
            RelationshipType.PERFORMS_BETTER_THAN: {
                "metric": {"type": "string", "required": True},
                "value_difference": {"type": "float", "required": False},
                "percentage_improvement": {"type": "float", "required": False},
                "measurement_date": {"type": "datetime", "required": True},
            },
        }
        
        properties = common_properties.copy()
        if rel_type in specific_properties:
            properties.update(specific_properties[rel_type])
        
        return properties
    
    def validate_node_data(self, label: NodeLabel, data: Dict[str, Any]) -> List[str]:
        """
        Validate node data against schema.
        
        Args:
            label: Node label
            data: Node data to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        properties = self.get_node_properties(label)
        
        # Check required properties
        for prop_name, prop_def in properties.items():
            if prop_def.get("required", False) and prop_name not in data:
                errors.append(f"Missing required property: {prop_name}")
        
        # Check data types (basic validation)
        for prop_name, value in data.items():
            if prop_name in properties:
                expected_type = properties[prop_name].get("type")
                if expected_type == "string" and not isinstance(value, str):
                    errors.append(f"Property {prop_name} should be string, got {type(value)}")
                elif expected_type == "integer" and not isinstance(value, int):
                    errors.append(f"Property {prop_name} should be integer, got {type(value)}")
                elif expected_type == "float" and not isinstance(value, (int, float)):
                    errors.append(f"Property {prop_name} should be float, got {type(value)}")
                elif expected_type == "boolean" and not isinstance(value, bool):
                    errors.append(f"Property {prop_name} should be boolean, got {type(value)}")
                elif expected_type == "list" and not isinstance(value, list):
                    errors.append(f"Property {prop_name} should be list, got {type(value)}")
                elif expected_type == "map" and not isinstance(value, dict):
                    errors.append(f"Property {prop_name} should be map, got {type(value)}")
        
        return errors
    
    def get_migration_queries(self, from_version: str, to_version: str) -> List[str]:
        """
        Get migration queries between schema versions.
        
        Args:
            from_version: Current schema version
            to_version: Target schema version
            
        Returns:
            List of migration queries
        """
        # This would contain version-specific migration logic
        # For now, return empty list
        logger.info(
            "Schema migration requested",
            from_version=from_version,
            to_version=to_version,
        )
        return []


# =============================================================================
# Utility Functions
# =============================================================================

def get_schema_manager() -> GraphSchema:
    """
    Get graph schema manager instance.
    
    Returns:
        GraphSchema instance
    """
    return GraphSchema()


def get_all_node_labels() -> List[str]:
    """
    Get all defined node labels.
    
    Returns:
        List of node label strings
    """
    return [label.value for label in NodeLabel]


def get_all_relationship_types() -> List[str]:
    """
    Get all defined relationship types.
    
    Returns:
        List of relationship type strings
    """
    return [rel_type.value for rel_type in RelationshipType]


def generate_schema_documentation() -> str:
    """
    Generate documentation for the graph schema.
    
    Returns:
        Formatted schema documentation
    """
    schema = get_schema_manager()
    
    doc = "# SEO Content Knowledge Graph Schema\n\n"
    
    # Node labels section
    doc += "## Node Labels\n\n"
    for label in NodeLabel:
        properties = schema.get_node_properties(label)
        doc += f"### {label.value}\n"
        if properties:
            doc += "Properties:\n"
            for prop_name, prop_def in properties.items():
                required = " (required)" if prop_def.get("required") else ""
                unique = " (unique)" if prop_def.get("unique") else ""
                doc += f"- `{prop_name}`: {prop_def['type']}{required}{unique}\n"
        doc += "\n"
    
    # Relationship types section
    doc += "## Relationship Types\n\n"
    for rel_type in RelationshipType:
        properties = schema.get_relationship_properties(rel_type)
        doc += f"### {rel_type.value}\n"
        if properties:
            doc += "Properties:\n"
            for prop_name, prop_def in properties.items():
                required = " (required)" if prop_def.get("required") else ""
                default = f" (default: {prop_def['default']})" if "default" in prop_def else ""
                doc += f"- `{prop_name}`: {prop_def['type']}{required}{default}\n"
        doc += "\n"
    
    return doc


if __name__ == "__main__":
    # Example usage
    schema = get_schema_manager()
    
    # Print all schema creation queries
    queries = schema.get_schema_creation_queries()
    print(f"Generated {len(queries)} schema queries:")
    for i, query in enumerate(queries, 1):
        print(f"{i}. {query}")
    
    # Generate and print documentation
    documentation = generate_schema_documentation()
    print("\n" + "="*50)
    print(documentation)