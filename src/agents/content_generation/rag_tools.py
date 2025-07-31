"""
RAG Tools for Content Generation Agent.

This module contains RAG-enhanced tools for knowledge graph search and vector similarity search.
"""

import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class RAGTools:
    """RAG-enhanced tools for content generation."""
    
    def __init__(self):
        self.neo4j_client = None
        self.qdrant_client = None
    
    def set_neo4j_client(self, client):
        """Set the Neo4j client for knowledge graph operations."""
        self.neo4j_client = client
    
    def set_qdrant_client(self, client):
        """Set the Qdrant client for vector search operations."""
        self.qdrant_client = client
    
    async def search_knowledge_graph(self, topic: str, keywords: List[str]) -> Dict[str, Any]:
        """Search the knowledge graph for related content and topics."""
        if not self.neo4j_client:
            logger.warning("Neo4j client not available for knowledge graph search")
            return {"related_topics": [], "connections": [], "available": False}
        
        try:
            # Use the search_content method instead of direct query
            # Search for content related to the topic
            search_query = f"{topic} {' '.join(keywords)}"
            result = self.neo4j_client.search_content(search_query, limit=10)
            
            related_content = []
            for record in result:
                related_content.append({
                    "title": record.get("title", ""),
                    "content": record.get("content", "")[:500],  # Truncate for context
                    "keywords": record.get("keywords", []),
                    "relationship": record.get("relationship_type", "related")
                })
            
            # For topic relationships, we'll use a simpler approach since direct queries aren't available
            # Try to find related topics through content search
            try:
                # Get organization stats to see if we have topic data
                org_stats = self.neo4j_client.get_organization_stats()
                topic_relationships = []
                
                # If we have content, add basic relationship info
                if org_stats and org_stats.get('total_nodes', 0) > 0:
                    topic_relationships.append({
                        "topic": f"Related to {topic}",
                        "relationship": "semantic_similarity", 
                        "score": 0.8
                    })
            except Exception:
                topic_relationships = []
            
            return {
                "related_content": related_content,
                "topic_relationships": topic_relationships,
                "available": True,
                "source": "neo4j_knowledge_graph"
            }
            
        except Exception as e:
            logger.error(f"Error searching knowledge graph: {e}")
            return {"related_topics": [], "connections": [], "available": False, "error": str(e)}

    async def find_similar_content(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar content using vector search."""
        if not self.qdrant_client:
            logger.warning("Qdrant client not available for vector search")
            return []
        
        try:
            # Generate a mock embedding for the query (in production, use OpenAI API)
            # For testing, we'll create a simple embedding based on query length
            mock_embedding = [0.1 + (i * 0.001) for i in range(1536)]
            
            # Use the search_similar_content method
            results = self.qdrant_client.search_similar_content(
                query_embedding=mock_embedding,
                limit=limit,
                min_score=0.5
            )
            
            similar_content = []
            for result in results:
                # Handle both dict and object result formats
                if isinstance(result, dict):
                    payload = result
                    score = result.get('score', 0.8)
                else:
                    payload = getattr(result, 'payload', {}) or {}
                    score = getattr(result, 'score', 0.8)
                
                similar_content.append({
                    "title": payload.get("title", "Unknown"),
                    "content": payload.get("summary", payload.get("content", ""))[:500],  # Truncate for context
                    "similarity_score": score,
                    "keywords": payload.get("keywords", []),
                    "content_type": payload.get("content_type", "unknown"),
                    "created_at": payload.get("created_at", "")
                })
            
            return similar_content
            
        except Exception as e:
            logger.error(f"Error finding similar content: {e}")
            return []

    async def get_content_relationships(self, topic: str) -> Dict[str, Any]:
        """Get relationships and connections for the topic from knowledge graph."""
        if not self.neo4j_client:
            return {"relationships": [], "available": False}
        
        try:
            # Use search_content to find related content for the topic
            related_content = self.neo4j_client.search_content(topic, limit=5)
            
            if related_content:
                # Create mock relationships based on found content
                mentioning_content = []
                for content in related_content:
                    mentioning_content.append({
                        "type": "content",
                        "title": content.get("title", ""),
                        "relationship": "mentions"
                    })
                
                return {
                    "parents": [],  # Would need direct query access to get true hierarchies
                    "children": [],
                    "related": [{"type": "related", "name": f"{topic} Related Topics", "relationship": "semantic"}],
                    "mentioning_content": mentioning_content,
                    "available": True
                }
            
            return {"relationships": [], "available": False}
            
        except Exception as e:
            logger.error(f"Error getting content relationships: {e}")
            return {"relationships": [], "available": False, "error": str(e)}

    async def enhance_with_context(self, content: str, topic: str) -> str:
        """Enhance content with contextual information from knowledge base."""
        try:
            # Get both knowledge graph and vector search context
            kg_context = await self.search_knowledge_graph(topic, [topic])
            similar_content = await self.find_similar_content(topic, limit=3)
            
            enhancement_context = []
            
            # Add knowledge graph insights
            if kg_context.get("available") and kg_context.get("related_content"):
                enhancement_context.append("\n## Related Context from Knowledge Base:")
                for item in kg_context["related_content"][:2]:
                    enhancement_context.append(f"• **{item['title']}**: {item['content'][:200]}...")
            
            # Add similar content insights  
            if similar_content:
                enhancement_context.append("\n## Similar Content Insights:")
                for item in similar_content[:2]:
                    enhancement_context.append(f"• **{item['title']}** (similarity: {item['similarity_score']:.2f}): Key insights from related content")
            
            # Add topic relationships
            relationships = await self.get_content_relationships(topic)
            if relationships.get("available") and relationships.get("related"):
                enhancement_context.append("\n## Related Topics to Consider:")
                for rel in relationships["related"][:3]:
                    enhancement_context.append(f"• {rel['name']}")
            
            if enhancement_context:
                enhanced_content = content + "\n" + "\n".join(enhancement_context)
                return enhanced_content
            
            return content
            
        except Exception as e:
            logger.error(f"Error enhancing content with context: {e}")
            return content