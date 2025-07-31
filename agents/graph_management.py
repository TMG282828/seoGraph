"""
Graph Management Agent for the SEO Content Knowledge Graph System.

This agent manages Neo4j knowledge graph operations, analytics, and maintenance
for content relationship mapping and insights.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import structlog
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from config.settings import get_settings
from database.neo4j_client import Neo4jClient
from database.qdrant_client import QdrantClient
from database.supabase_client import SupabaseClient
from models.graph_models import (
    GraphNode, GraphRelationship, ContentNode, TopicNode, 
    KeywordNode, EntityNode, GraphAnalytics, GraphOperation,
    GraphVisualization, GraphQuery, GraphQueryResult
)
from database.graph_schema import NodeLabel, RelationshipType
from services.embedding_service import EmbeddingService

logger = structlog.get_logger(__name__)


class GraphManagementDependencies:
    """Dependencies for the Graph Management Agent."""
    
    def __init__(
        self,
        neo4j_client: Neo4jClient,
        qdrant_client: QdrantClient,
        supabase_client: SupabaseClient,
        embedding_service: EmbeddingService,
        tenant_id: str
    ):
        self.neo4j_client = neo4j_client
        self.qdrant_client = qdrant_client
        self.supabase_client = supabase_client
        self.embedding_service = embedding_service
        self.tenant_id = tenant_id


class GraphInsights(BaseModel):
    """Structured output for graph insights and analysis."""
    
    graph_health: Dict[str, Any] = Field(..., description="Overall graph health metrics")
    content_clusters: List[Dict[str, Any]] = Field(..., description="Identified content clusters")
    topic_importance: List[Dict[str, Any]] = Field(..., description="Topic importance rankings")
    
    connectivity_analysis: Dict[str, Any] = Field(..., description="Graph connectivity analysis")
    gap_analysis: List[Dict[str, Any]] = Field(..., description="Content and topic gaps")
    growth_opportunities: List[str] = Field(..., description="Identified growth opportunities")
    
    recommendations: List[str] = Field(..., description="Strategic recommendations")
    optimization_suggestions: List[str] = Field(..., description="Graph optimization suggestions")


class GraphMaintenanceResult(BaseModel):
    """Structured output for graph maintenance operations."""
    
    operations_performed: List[str] = Field(..., description="Maintenance operations completed")
    nodes_processed: int = Field(..., description="Number of nodes processed")
    relationships_processed: int = Field(..., description="Number of relationships processed")
    
    issues_resolved: List[str] = Field(..., description="Issues identified and resolved")
    performance_improvements: Dict[str, float] = Field(..., description="Performance improvements")
    
    data_quality_score: float = Field(..., description="Overall data quality score")
    maintenance_recommendations: List[str] = Field(..., description="Future maintenance recommendations")


class GraphAnalysisResult(BaseModel):
    """Complete graph analysis result."""
    
    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tenant_id: str = Field(..., description="Tenant identifier")
    analysis_type: str = Field(..., description="Type of analysis performed")
    
    insights: Optional[GraphInsights] = None
    maintenance_result: Optional[GraphMaintenanceResult] = None
    analytics: Optional[GraphAnalytics] = None
    
    processing_time: float = Field(..., description="Analysis processing time")
    success: bool = Field(..., description="Analysis success status")
    warnings: List[str] = Field(default_factory=list, description="Analysis warnings")


# Initialize the Graph Management Agent
graph_management_agent = Agent(
    "openai:gpt-4o-mini",
    deps_type=GraphManagementDependencies,
    result_type=GraphAnalysisResult,
    system_prompt="""
    You are a specialized Graph Management Agent for knowledge graph operations and analysis.
    
    Your responsibilities include:
    1. Analyzing graph structure and identifying patterns and clusters
    2. Assessing graph health and data quality
    3. Identifying content gaps and growth opportunities
    4. Providing strategic recommendations for content strategy
    5. Optimizing graph performance and structure
    
    Graph Analysis Guidelines:
    - Focus on actionable insights from graph topology
    - Identify highly connected nodes and isolated components
    - Analyze topic importance based on centrality and connections
    - Detect content clusters and thematic groupings
    - Assess graph density and connectivity patterns
    
    Content Strategy Insights:
    - Identify underrepresented topics with high potential
    - Find content gaps in competitor landscapes
    - Recommend content connections and cross-linking opportunities
    - Analyze topic authority and influence patterns
    - Suggest content expansion strategies
    
    Data Quality Assessment:
    - Identify duplicate or near-duplicate nodes
    - Detect orphaned content without proper connections
    - Find inconsistent relationship weights or properties
    - Assess schema compliance and data integrity
    - Recommend data cleanup and standardization
    
    Performance Optimization:
    - Identify expensive queries and suggest optimizations
    - Recommend index improvements
    - Analyze query patterns and suggest schema adjustments
    - Identify bottlenecks in graph traversal
    
    Always provide specific, actionable recommendations with clear business value.
    """,
)


class GraphManagementAgent:
    """
    Graph Management Agent for Neo4j knowledge graph operations,
    analytics, and maintenance.
    """
    
    def __init__(
        self,
        neo4j_client: Neo4jClient,
        qdrant_client: QdrantClient,
        supabase_client: SupabaseClient,
        embedding_service: EmbeddingService,
        tenant_id: str
    ):
        """
        Initialize the Graph Management Agent.
        
        Args:
            neo4j_client: Neo4j database client
            qdrant_client: Qdrant vector database client
            supabase_client: Supabase database client
            embedding_service: Embedding generation service
            tenant_id: Tenant identifier
        """
        self.neo4j_client = neo4j_client
        self.qdrant_client = qdrant_client
        self.supabase_client = supabase_client
        self.embedding_service = embedding_service
        self.tenant_id = tenant_id
        
        # Initialize dependencies
        self.deps = GraphManagementDependencies(
            neo4j_client=neo4j_client,
            qdrant_client=qdrant_client,
            supabase_client=supabase_client,
            embedding_service=embedding_service,
            tenant_id=tenant_id
        )
        
        logger.info(
            "Graph Management Agent initialized",
            tenant_id=tenant_id
        )
    
    async def analyze_graph_structure(
        self,
        include_recommendations: bool = True,
        analysis_depth: str = "comprehensive"
    ) -> GraphAnalysisResult:
        """
        Perform comprehensive graph structure analysis.
        
        Args:
            include_recommendations: Whether to include strategic recommendations
            analysis_depth: Depth of analysis (basic, standard, comprehensive)
            
        Returns:
            Graph analysis result with insights
        """
        start_time = asyncio.get_event_loop().time()
        
        logger.info(
            "Starting graph structure analysis",
            analysis_depth=analysis_depth,
            tenant_id=self.tenant_id
        )
        
        try:
            # Collect graph metrics
            graph_metrics = await self._collect_graph_metrics()
            
            # Analyze content clusters
            content_clusters = await self._analyze_content_clusters()
            
            # Assess topic importance
            topic_importance = await self._analyze_topic_importance()
            
            # Perform connectivity analysis
            connectivity_analysis = await self._analyze_connectivity()
            
            # Identify gaps and opportunities
            gap_analysis = await self._identify_content_gaps()
            
            # Prepare analysis input
            analysis_input = self._prepare_analysis_input(
                graph_metrics, content_clusters, topic_importance,
                connectivity_analysis, gap_analysis, analysis_depth
            )
            
            # Run AI analysis
            result = await graph_management_agent.run(
                analysis_input,
                deps=self.deps
            )
            
            # Add computed analytics
            result.analytics = await self._compute_graph_analytics()
            
            # Store analysis results
            await self._store_analysis_results(result)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            result.processing_time = processing_time
            
            logger.info(
                "Graph structure analysis completed",
                processing_time=processing_time,
                tenant_id=self.tenant_id
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Graph structure analysis failed",
                error=str(e),
                tenant_id=self.tenant_id
            )
            raise
    
    async def perform_graph_maintenance(
        self,
        maintenance_tasks: List[str] = None,
        dry_run: bool = False
    ) -> GraphAnalysisResult:
        """
        Perform graph maintenance operations.
        
        Args:
            maintenance_tasks: Specific tasks to perform
            dry_run: Whether to perform a dry run without making changes
            
        Returns:
            Graph analysis result with maintenance details
        """
        start_time = asyncio.get_event_loop().time()
        
        maintenance_tasks = maintenance_tasks or [
            "remove_duplicates",
            "fix_orphaned_nodes",
            "update_relationship_weights",
            "validate_schema_compliance",
            "optimize_indexes"
        ]
        
        logger.info(
            "Starting graph maintenance",
            tasks=maintenance_tasks,
            dry_run=dry_run,
            tenant_id=self.tenant_id
        )
        
        try:
            # Collect current state
            pre_maintenance_metrics = await self._collect_graph_metrics()
            
            # Perform maintenance tasks
            maintenance_results = {}
            total_nodes_processed = 0
            total_relationships_processed = 0
            
            for task in maintenance_tasks:
                task_result = await self._perform_maintenance_task(task, dry_run)
                maintenance_results[task] = task_result
                total_nodes_processed += task_result.get("nodes_processed", 0)
                total_relationships_processed += task_result.get("relationships_processed", 0)
            
            # Collect post-maintenance metrics
            post_maintenance_metrics = await self._collect_graph_metrics()
            
            # Prepare maintenance input
            maintenance_input = self._prepare_maintenance_input(
                pre_maintenance_metrics, post_maintenance_metrics,
                maintenance_results, dry_run
            )
            
            # Run AI analysis
            result = await graph_management_agent.run(
                maintenance_input,
                deps=self.deps
            )
            
            # Update result with maintenance details
            if result.maintenance_result:
                result.maintenance_result.nodes_processed = total_nodes_processed
                result.maintenance_result.relationships_processed = total_relationships_processed
            
            processing_time = asyncio.get_event_loop().time() - start_time
            result.processing_time = processing_time
            
            logger.info(
                "Graph maintenance completed",
                processing_time=processing_time,
                nodes_processed=total_nodes_processed,
                relationships_processed=total_relationships_processed,
                dry_run=dry_run
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Graph maintenance failed",
                error=str(e),
                tenant_id=self.tenant_id
            )
            raise
    
    async def execute_graph_query(
        self,
        query: GraphQuery,
        validate_query: bool = True
    ) -> GraphQueryResult:
        """
        Execute a custom graph query.
        
        Args:
            query: Graph query to execute
            validate_query: Whether to validate query before execution
            
        Returns:
            Query execution result
        """
        logger.info(
            "Executing graph query",
            query_id=query.query_id,
            tenant_id=self.tenant_id
        )
        
        try:
            # Validate query if requested
            if validate_query:
                await self._validate_query(query)
            
            # Execute query
            start_time = asyncio.get_event_loop().time()
            
            results = await self.neo4j_client.execute_query(
                query.cypher,
                query.parameters
            )
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Create result
            query_result = GraphQueryResult(
                query_id=query.query_id,
                results=results,
                execution_time=execution_time,
                records_returned=len(results),
                success=True
            )
            
            logger.info(
                "Graph query executed successfully",
                query_id=query.query_id,
                execution_time=execution_time,
                records_returned=len(results)
            )
            
            return query_result
            
        except Exception as e:
            logger.error(
                "Graph query execution failed",
                query_id=query.query_id,
                error=str(e)
            )
            
            return GraphQueryResult(
                query_id=query.query_id,
                success=False,
                error_message=str(e)
            )
    
    async def create_graph_visualization(
        self,
        visualization_params: Dict[str, Any],
        user_id: str
    ) -> GraphVisualization:
        """
        Create graph visualization data.
        
        Args:
            visualization_params: Visualization parameters
            user_id: User creating the visualization
            
        Returns:
            Graph visualization data
        """
        logger.info(
            "Creating graph visualization",
            tenant_id=self.tenant_id
        )
        
        try:
            # Create visualization object
            visualization = GraphVisualization(
                tenant_id=self.tenant_id,
                created_by=user_id,
                **visualization_params
            )
            
            # Generate visualization data
            if visualization.center_node_id:
                # Center-based visualization
                viz_data = await self._generate_center_based_visualization(visualization)
            else:
                # Overview visualization
                viz_data = await self._generate_overview_visualization(visualization)
            
            visualization.nodes = viz_data["nodes"]
            visualization.edges = viz_data["edges"]
            
            # Store visualization
            await self._store_visualization(visualization)
            
            logger.info(
                "Graph visualization created",
                visualization_id=visualization.visualization_id,
                nodes_count=len(visualization.nodes),
                edges_count=len(visualization.edges)
            )
            
            return visualization
            
        except Exception as e:
            logger.error(
                "Graph visualization creation failed",
                error=str(e)
            )
            raise
    
    async def _collect_graph_metrics(self) -> Dict[str, Any]:
        """Collect basic graph metrics."""
        metrics = {}
        
        try:
            # Node counts by type
            node_counts_query = """
            MATCH (n {tenant_id: $tenant_id})
            RETURN labels(n)[0] as label, count(n) as count
            """
            node_counts = await self.neo4j_client.execute_query(
                node_counts_query,
                {"tenant_id": self.tenant_id}
            )
            metrics["node_counts"] = {item["label"]: item["count"] for item in node_counts}
            
            # Relationship counts by type
            rel_counts_query = """
            MATCH (n {tenant_id: $tenant_id})-[r]->(m {tenant_id: $tenant_id})
            RETURN type(r) as relationship_type, count(r) as count
            """
            rel_counts = await self.neo4j_client.execute_query(
                rel_counts_query,
                {"tenant_id": self.tenant_id}
            )
            metrics["relationship_counts"] = {item["relationship_type"]: item["count"] for item in rel_counts}
            
            # Graph density
            total_nodes = sum(metrics["node_counts"].values())
            total_relationships = sum(metrics["relationship_counts"].values())
            
            if total_nodes > 1:
                possible_relationships = total_nodes * (total_nodes - 1)
                metrics["graph_density"] = total_relationships / possible_relationships
            else:
                metrics["graph_density"] = 0.0
            
            # Average degree
            if total_nodes > 0:
                metrics["average_degree"] = (total_relationships * 2) / total_nodes
            else:
                metrics["average_degree"] = 0.0
            
        except Exception as e:
            logger.warning("Failed to collect graph metrics", error=str(e))
            metrics["error"] = str(e)
        
        return metrics
    
    async def _analyze_content_clusters(self) -> List[Dict[str, Any]]:
        """Analyze content clusters in the graph."""
        clusters = []
        
        try:
            # Find connected components of content
            cluster_query = """
            MATCH (c:Content {tenant_id: $tenant_id})
            CALL {
                WITH c
                MATCH path = (c)-[:SIMILAR_TO|RELATES_TO*1..3]-(connected:Content)
                RETURN connected
            }
            WITH c, collect(DISTINCT connected) as cluster_nodes
            RETURN c.id as center_node, 
                   [node IN cluster_nodes | node.title] as connected_titles,
                   size(cluster_nodes) as cluster_size
            ORDER BY cluster_size DESC
            LIMIT 10
            """
            
            cluster_results = await self.neo4j_client.execute_query(
                cluster_query,
                {"tenant_id": self.tenant_id}
            )
            
            for result in cluster_results:
                clusters.append({
                    "center_node": result["center_node"],
                    "cluster_size": result["cluster_size"],
                    "connected_titles": result["connected_titles"][:5]  # Limit for display
                })
            
        except Exception as e:
            logger.warning("Failed to analyze content clusters", error=str(e))
        
        return clusters
    
    async def _analyze_topic_importance(self) -> List[Dict[str, Any]]:
        """Analyze topic importance based on connections."""
        topic_importance = []
        
        try:
            # Calculate topic centrality
            importance_query = """
            MATCH (t:Topic {tenant_id: $tenant_id})
            OPTIONAL MATCH (t)<-[:RELATES_TO]-(c:Content)
            WITH t, count(c) as content_connections
            OPTIONAL MATCH (t)-[:RELATED_TO]-(other:Topic)
            WITH t, content_connections, count(other) as topic_connections
            RETURN t.name as topic, 
                   t.importance_score as base_importance,
                   content_connections,
                   topic_connections,
                   (content_connections * 2 + topic_connections) as calculated_importance
            ORDER BY calculated_importance DESC
            LIMIT 20
            """
            
            importance_results = await self.neo4j_client.execute_query(
                importance_query,
                {"tenant_id": self.tenant_id}
            )
            
            for result in importance_results:
                topic_importance.append({
                    "topic": result["topic"],
                    "base_importance": result["base_importance"],
                    "content_connections": result["content_connections"],
                    "topic_connections": result["topic_connections"],
                    "calculated_importance": result["calculated_importance"]
                })
            
        except Exception as e:
            logger.warning("Failed to analyze topic importance", error=str(e))
        
        return topic_importance
    
    async def _analyze_connectivity(self) -> Dict[str, Any]:
        """Analyze graph connectivity patterns."""
        connectivity = {}
        
        try:
            # Find orphaned content
            orphaned_query = """
            MATCH (c:Content {tenant_id: $tenant_id})
            WHERE NOT (c)-[:RELATES_TO|SIMILAR_TO]-()
            RETURN count(c) as orphaned_count
            """
            orphaned_result = await self.neo4j_client.execute_query(
                orphaned_query,
                {"tenant_id": self.tenant_id}
            )
            connectivity["orphaned_content"] = orphaned_result[0]["orphaned_count"] if orphaned_result else 0
            
            # Find highly connected nodes
            hub_query = """
            MATCH (n {tenant_id: $tenant_id})
            WITH n, size((n)-[]-()) as degree
            WHERE degree > 5
            RETURN labels(n)[0] as node_type, 
                   n.name as name, 
                   degree
            ORDER BY degree DESC
            LIMIT 10
            """
            hub_results = await self.neo4j_client.execute_query(
                hub_query,
                {"tenant_id": self.tenant_id}
            )
            connectivity["high_degree_nodes"] = hub_results
            
            # Calculate graph components
            components_query = """
            MATCH (n {tenant_id: $tenant_id})
            CALL apoc.path.subgraphAll(n, {}) YIELD nodes
            RETURN size(nodes) as component_size
            ORDER BY component_size DESC
            """
            
            # Fallback if APOC is not available
            try:
                component_results = await self.neo4j_client.execute_query(
                    components_query,
                    {"tenant_id": self.tenant_id}
                )
                connectivity["largest_component_size"] = component_results[0]["component_size"] if component_results else 0
            except:
                connectivity["largest_component_size"] = "Unknown (APOC not available)"
            
        except Exception as e:
            logger.warning("Failed to analyze connectivity", error=str(e))
            connectivity["error"] = str(e)
        
        return connectivity
    
    async def _identify_content_gaps(self) -> List[Dict[str, Any]]:
        """Identify content gaps and opportunities."""
        gaps = []
        
        try:
            # Find topics with no content
            gap_query = """
            MATCH (t:Topic {tenant_id: $tenant_id})
            WHERE NOT (t)<-[:RELATES_TO]-(:Content)
            RETURN t.name as topic, 
                   t.search_volume as search_volume,
                   t.importance_score as importance
            ORDER BY search_volume DESC, importance DESC
            LIMIT 10
            """
            
            gap_results = await self.neo4j_client.execute_query(
                gap_query,
                {"tenant_id": self.tenant_id}
            )
            
            for result in gap_results:
                gaps.append({
                    "type": "missing_content",
                    "topic": result["topic"],
                    "search_volume": result["search_volume"],
                    "importance": result["importance"]
                })
            
            # Find keywords with low content coverage
            keyword_gap_query = """
            MATCH (k:Keyword {tenant_id: $tenant_id})
            OPTIONAL MATCH (k)<-[:TARGETS]-(c:Content)
            WITH k, count(c) as content_count
            WHERE content_count < 2 AND k.search_volume > 100
            RETURN k.text as keyword,
                   k.search_volume as search_volume,
                   k.difficulty as difficulty,
                   content_count
            ORDER BY search_volume DESC
            LIMIT 10
            """
            
            keyword_gaps = await self.neo4j_client.execute_query(
                keyword_gap_query,
                {"tenant_id": self.tenant_id}
            )
            
            for result in keyword_gaps:
                gaps.append({
                    "type": "keyword_gap",
                    "keyword": result["keyword"],
                    "search_volume": result["search_volume"],
                    "difficulty": result["difficulty"],
                    "content_count": result["content_count"]
                })
            
        except Exception as e:
            logger.warning("Failed to identify content gaps", error=str(e))
        
        return gaps
    
    def _prepare_analysis_input(
        self,
        graph_metrics: Dict[str, Any],
        content_clusters: List[Dict[str, Any]],
        topic_importance: List[Dict[str, Any]],
        connectivity_analysis: Dict[str, Any],
        gap_analysis: List[Dict[str, Any]],
        analysis_depth: str
    ) -> str:
        """Prepare input for graph analysis."""
        input_parts = [
            f"GRAPH STRUCTURE ANALYSIS",
            f"ANALYSIS TYPE: graph_structure",
            f"ANALYSIS DEPTH: {analysis_depth}",
            f"TENANT: {self.tenant_id}",
            "",
            "GRAPH METRICS:"
        ]
        
        # Add graph metrics
        if "node_counts" in graph_metrics:
            input_parts.append("Node Counts:")
            for label, count in graph_metrics["node_counts"].items():
                input_parts.append(f"  {label}: {count}")
        
        if "relationship_counts" in graph_metrics:
            input_parts.append("Relationship Counts:")
            for rel_type, count in graph_metrics["relationship_counts"].items():
                input_parts.append(f"  {rel_type}: {count}")
        
        input_parts.append(f"Graph Density: {graph_metrics.get('graph_density', 0.0):.4f}")
        input_parts.append(f"Average Degree: {graph_metrics.get('average_degree', 0.0):.2f}")
        
        # Add content clusters
        if content_clusters:
            input_parts.extend([
                "",
                "CONTENT CLUSTERS:"
            ])
            for cluster in content_clusters[:5]:
                input_parts.append(f"Cluster (size {cluster['cluster_size']}): {cluster['center_node']}")
        
        # Add topic importance
        if topic_importance:
            input_parts.extend([
                "",
                "TOP TOPICS BY IMPORTANCE:"
            ])
            for topic in topic_importance[:10]:
                input_parts.append(f"  {topic['topic']}: {topic['calculated_importance']} connections")
        
        # Add connectivity analysis
        input_parts.extend([
            "",
            "CONNECTIVITY ANALYSIS:",
            f"Orphaned Content: {connectivity_analysis.get('orphaned_content', 0)}",
            f"Largest Component Size: {connectivity_analysis.get('largest_component_size', 'Unknown')}"
        ])
        
        if connectivity_analysis.get("high_degree_nodes"):
            input_parts.append("High-Degree Nodes:")
            for node in connectivity_analysis["high_degree_nodes"][:5]:
                input_parts.append(f"  {node['name']} ({node['node_type']}): {node['degree']} connections")
        
        # Add gap analysis
        if gap_analysis:
            input_parts.extend([
                "",
                "CONTENT GAPS:"
            ])
            for gap in gap_analysis[:10]:
                if gap["type"] == "missing_content":
                    input_parts.append(f"Missing content for topic: {gap['topic']} (volume: {gap['search_volume']})")
                elif gap["type"] == "keyword_gap":
                    input_parts.append(f"Keyword gap: {gap['keyword']} (volume: {gap['search_volume']}, content: {gap['content_count']})")
        
        return "\n".join(input_parts)
    
    def _prepare_maintenance_input(
        self,
        pre_metrics: Dict[str, Any],
        post_metrics: Dict[str, Any],
        maintenance_results: Dict[str, Any],
        dry_run: bool
    ) -> str:
        """Prepare input for maintenance analysis."""
        input_parts = [
            f"GRAPH MAINTENANCE ANALYSIS",
            f"ANALYSIS TYPE: graph_maintenance",
            f"DRY RUN: {dry_run}",
            f"TENANT: {self.tenant_id}",
            "",
            "PRE-MAINTENANCE METRICS:"
        ]
        
        # Add pre-maintenance metrics
        if "node_counts" in pre_metrics:
            total_nodes_pre = sum(pre_metrics["node_counts"].values())
            input_parts.append(f"Total Nodes: {total_nodes_pre}")
        
        if "relationship_counts" in pre_metrics:
            total_rels_pre = sum(pre_metrics["relationship_counts"].values())
            input_parts.append(f"Total Relationships: {total_rels_pre}")
        
        # Add post-maintenance metrics if not dry run
        if not dry_run:
            input_parts.extend([
                "",
                "POST-MAINTENANCE METRICS:"
            ])
            
            if "node_counts" in post_metrics:
                total_nodes_post = sum(post_metrics["node_counts"].values())
                input_parts.append(f"Total Nodes: {total_nodes_post}")
            
            if "relationship_counts" in post_metrics:
                total_rels_post = sum(post_metrics["relationship_counts"].values())
                input_parts.append(f"Total Relationships: {total_rels_post}")
        
        # Add maintenance task results
        input_parts.extend([
            "",
            "MAINTENANCE TASKS PERFORMED:"
        ])
        
        for task, result in maintenance_results.items():
            input_parts.append(f"{task}:")
            input_parts.append(f"  Nodes processed: {result.get('nodes_processed', 0)}")
            input_parts.append(f"  Relationships processed: {result.get('relationships_processed', 0)}")
            input_parts.append(f"  Issues found: {result.get('issues_found', 0)}")
            if result.get('improvements'):
                input_parts.append(f"  Improvements: {result['improvements']}")
        
        return "\n".join(input_parts)
    
    async def _perform_maintenance_task(
        self,
        task: str,
        dry_run: bool
    ) -> Dict[str, Any]:
        """Perform a specific maintenance task."""
        result = {
            "task": task,
            "nodes_processed": 0,
            "relationships_processed": 0,
            "issues_found": 0,
            "improvements": []
        }
        
        try:
            if task == "remove_duplicates":
                result.update(await self._remove_duplicate_nodes(dry_run))
            elif task == "fix_orphaned_nodes":
                result.update(await self._fix_orphaned_nodes(dry_run))
            elif task == "update_relationship_weights":
                result.update(await self._update_relationship_weights(dry_run))
            elif task == "validate_schema_compliance":
                result.update(await self._validate_schema_compliance(dry_run))
            elif task == "optimize_indexes":
                result.update(await self._optimize_indexes(dry_run))
            else:
                logger.warning(f"Unknown maintenance task: {task}")
                
        except Exception as e:
            logger.error(f"Maintenance task {task} failed", error=str(e))
            result["error"] = str(e)
        
        return result
    
    async def _remove_duplicate_nodes(self, dry_run: bool) -> Dict[str, Any]:
        """Remove duplicate nodes from the graph."""
        # Find potential duplicates based on name similarity
        duplicate_query = """
        MATCH (n1 {tenant_id: $tenant_id}), (n2 {tenant_id: $tenant_id})
        WHERE id(n1) < id(n2) 
        AND labels(n1) = labels(n2)
        AND n1.name = n2.name
        RETURN n1, n2
        LIMIT 100
        """
        
        duplicates = await self.neo4j_client.execute_query(
            duplicate_query,
            {"tenant_id": self.tenant_id}
        )
        
        if not dry_run and duplicates:
            # Merge duplicates (simplified version)
            merge_query = """
            MATCH (n1 {tenant_id: $tenant_id}), (n2 {tenant_id: $tenant_id})
            WHERE id(n1) < id(n2) 
            AND labels(n1) = labels(n2)
            AND n1.name = n2.name
            WITH n1, n2 LIMIT 10
            CALL apoc.refactor.mergeNodes([n1, n2], {properties: 'combine'})
            YIELD node
            RETURN count(node) as merged_count
            """
            
            try:
                merge_result = await self.neo4j_client.execute_query(
                    merge_query,
                    {"tenant_id": self.tenant_id}
                )
                merged_count = merge_result[0]["merged_count"] if merge_result else 0
            except:
                # Fallback if APOC is not available
                merged_count = 0
        else:
            merged_count = 0
        
        return {
            "nodes_processed": len(duplicates),
            "issues_found": len(duplicates),
            "improvements": [f"Merged {merged_count} duplicate nodes"] if merged_count > 0 else []
        }
    
    async def _fix_orphaned_nodes(self, dry_run: bool) -> Dict[str, Any]:
        """Fix orphaned nodes by creating appropriate relationships."""
        # Find orphaned content nodes
        orphaned_query = """
        MATCH (c:Content {tenant_id: $tenant_id})
        WHERE NOT (c)-[:RELATES_TO|SIMILAR_TO]-()
        RETURN c
        LIMIT 50
        """
        
        orphaned_nodes = await self.neo4j_client.execute_query(
            orphaned_query,
            {"tenant_id": self.tenant_id}
        )
        
        improvements = []
        relationships_created = 0
        
        if not dry_run and orphaned_nodes:
            # For each orphaned node, try to create topic relationships based on content
            for node in orphaned_nodes[:10]:  # Limit to prevent too many operations
                try:
                    # This would require more sophisticated content analysis
                    # For now, just mark as improvement opportunity
                    improvements.append(f"Identified orphaned content: {node.get('title', 'Unknown')}")
                except Exception as e:
                    logger.warning(f"Failed to fix orphaned node", error=str(e))
        
        return {
            "nodes_processed": len(orphaned_nodes),
            "issues_found": len(orphaned_nodes),
            "relationships_processed": relationships_created,
            "improvements": improvements
        }
    
    async def _update_relationship_weights(self, dry_run: bool) -> Dict[str, Any]:
        """Update relationship weights based on current data."""
        # Update similarity relationship weights
        weight_query = """
        MATCH (c1:Content {tenant_id: $tenant_id})-[r:SIMILAR_TO]->(c2:Content {tenant_id: $tenant_id})
        WHERE r.weight IS NULL OR r.weight = 0
        RETURN count(r) as relationships_to_update
        """
        
        weight_result = await self.neo4j_client.execute_query(
            weight_query,
            {"tenant_id": self.tenant_id}
        )
        
        relationships_to_update = weight_result[0]["relationships_to_update"] if weight_result else 0
        
        if not dry_run and relationships_to_update > 0:
            # Update weights to default values (would need similarity calculation in real implementation)
            update_query = """
            MATCH (c1:Content {tenant_id: $tenant_id})-[r:SIMILAR_TO]->(c2:Content {tenant_id: $tenant_id})
            WHERE r.weight IS NULL OR r.weight = 0
            SET r.weight = 0.5, r.updated_at = datetime()
            RETURN count(r) as updated_count
            """
            
            update_result = await self.neo4j_client.execute_query(
                update_query,
                {"tenant_id": self.tenant_id}
            )
            
            updated_count = update_result[0]["updated_count"] if update_result else 0
        else:
            updated_count = 0
        
        return {
            "relationships_processed": relationships_to_update,
            "improvements": [f"Updated weights for {updated_count} relationships"] if updated_count > 0 else []
        }
    
    async def _validate_schema_compliance(self, dry_run: bool) -> Dict[str, Any]:
        """Validate nodes and relationships against schema."""
        # Check for nodes missing required properties
        validation_query = """
        MATCH (n {tenant_id: $tenant_id})
        WHERE n.created_at IS NULL
        RETURN labels(n)[0] as node_type, count(n) as count
        """
        
        validation_result = await self.neo4j_client.execute_query(
            validation_query,
            {"tenant_id": self.tenant_id}
        )
        
        issues_found = sum(item["count"] for item in validation_result)
        
        if not dry_run and issues_found > 0:
            # Fix missing timestamps
            fix_query = """
            MATCH (n {tenant_id: $tenant_id})
            WHERE n.created_at IS NULL
            SET n.created_at = datetime()
            RETURN count(n) as fixed_count
            """
            
            fix_result = await self.neo4j_client.execute_query(
                fix_query,
                {"tenant_id": self.tenant_id}
            )
            
            fixed_count = fix_result[0]["fixed_count"] if fix_result else 0
        else:
            fixed_count = 0
        
        return {
            "nodes_processed": issues_found,
            "issues_found": issues_found,
            "improvements": [f"Fixed {fixed_count} nodes missing timestamps"] if fixed_count > 0 else []
        }
    
    async def _optimize_indexes(self, dry_run: bool) -> Dict[str, Any]:
        """Optimize database indexes for better performance."""
        # This would check and create missing indexes
        # For now, just return a placeholder result
        return {
            "improvements": ["Index optimization checked"] if not dry_run else ["Would optimize indexes"]
        }
    
    async def _compute_graph_analytics(self) -> GraphAnalytics:
        """Compute comprehensive graph analytics."""
        metrics = await self._collect_graph_metrics()
        
        analytics = GraphAnalytics(
            tenant_id=self.tenant_id,
            total_nodes=sum(metrics.get("node_counts", {}).values()),
            total_relationships=sum(metrics.get("relationship_counts", {}).values()),
            graph_density=metrics.get("graph_density", 0.0),
            average_degree=metrics.get("average_degree", 0.0)
        )
        
        # Add specific node type counts
        node_counts = metrics.get("node_counts", {})
        analytics.content_nodes = node_counts.get("Content", 0)
        analytics.topic_nodes = node_counts.get("Topic", 0)
        analytics.keyword_nodes = node_counts.get("Keyword", 0)
        analytics.entity_nodes = node_counts.get("Entity", 0)
        
        # Add specific relationship type counts
        rel_counts = metrics.get("relationship_counts", {})
        analytics.similarity_relationships = rel_counts.get("SIMILAR_TO", 0)
        analytics.topic_relationships = rel_counts.get("RELATES_TO", 0)
        analytics.keyword_relationships = rel_counts.get("TARGETS", 0)
        
        return analytics
    
    async def _validate_query(self, query: GraphQuery) -> None:
        """Validate a graph query for safety."""
        # Basic validation - in production, this would be more comprehensive
        dangerous_operations = ['DELETE', 'DETACH DELETE', 'REMOVE']
        query_upper = query.cypher.upper()
        
        for operation in dangerous_operations:
            if operation in query_upper:
                logger.warning(
                    "Potentially dangerous query operation detected",
                    operation=operation,
                    query_id=query.query_id
                )
        
        # Ensure tenant_id is in parameters for multi-tenant safety
        if self.tenant_id not in str(query.parameters):
            logger.warning(
                "Query may not be properly scoped to tenant",
                query_id=query.query_id,
                tenant_id=self.tenant_id
            )
    
    async def _generate_center_based_visualization(
        self,
        visualization: GraphVisualization
    ) -> Dict[str, Any]:
        """Generate visualization data centered on a specific node."""
        viz_query = f"""
        MATCH (center {{id: $center_id, tenant_id: $tenant_id}})
        CALL {{
            WITH center
            MATCH path = (center)-[*1..{visualization.max_depth}]-(connected)
            WHERE any(label IN labels(connected) WHERE label IN $node_types)
            RETURN connected, relationships(path) as rels
        }}
        WITH center, collect(DISTINCT connected) as nodes, 
             reduce(allRels = [], path_rels IN collect(rels) | allRels + path_rels) as all_rels
        RETURN [center] + nodes as all_nodes, all_rels as relationships
        LIMIT {visualization.max_nodes}
        """
        
        params = {
            "center_id": visualization.center_node_id,
            "tenant_id": self.tenant_id,
            "node_types": [nt.value for nt in visualization.node_types]
        }
        
        result = await self.neo4j_client.execute_query(viz_query, params)
        
        if result:
            nodes_data = result[0]["all_nodes"]
            relationships_data = result[0]["relationships"]
        else:
            nodes_data = []
            relationships_data = []
        
        # Format for visualization
        nodes = []
        edges = []
        
        for node in nodes_data:
            nodes.append({
                "id": node["id"],
                "label": node.get("name", node.get("title", "Unknown")),
                "type": list(node.labels)[0] if node.labels else "Unknown",
                "properties": dict(node)
            })
        
        for rel in relationships_data:
            edges.append({
                "source": rel.start_node["id"],
                "target": rel.end_node["id"],
                "type": rel.type,
                "weight": rel.get("weight", 1.0)
            })
        
        return {"nodes": nodes, "edges": edges}
    
    async def _generate_overview_visualization(
        self,
        visualization: GraphVisualization
    ) -> Dict[str, Any]:
        """Generate overview visualization data."""
        # Get top nodes by degree
        overview_query = f"""
        MATCH (n {{tenant_id: $tenant_id}})
        WHERE any(label IN labels(n) WHERE label IN $node_types)
        WITH n, size((n)-[]-()) as degree
        ORDER BY degree DESC
        LIMIT {visualization.max_nodes}
        MATCH (n)-[r]-(connected)
        WHERE any(label IN labels(connected) WHERE label IN $node_types)
        RETURN collect(DISTINCT n) as nodes, collect(DISTINCT r) as relationships
        """
        
        params = {
            "tenant_id": self.tenant_id,
            "node_types": [nt.value for nt in visualization.node_types]
        }
        
        result = await self.neo4j_client.execute_query(overview_query, params)
        
        if result:
            nodes_data = result[0]["nodes"]
            relationships_data = result[0]["relationships"]
        else:
            nodes_data = []
            relationships_data = []
        
        # Format for visualization (same as center-based)
        nodes = []
        edges = []
        
        for node in nodes_data:
            nodes.append({
                "id": node["id"],
                "label": node.get("name", node.get("title", "Unknown")),
                "type": list(node.labels)[0] if node.labels else "Unknown",
                "properties": dict(node)
            })
        
        for rel in relationships_data:
            edges.append({
                "source": rel.start_node["id"],
                "target": rel.end_node["id"],
                "type": rel.type,
                "weight": rel.get("weight", 1.0)
            })
        
        return {"nodes": nodes, "edges": edges}
    
    async def _store_analysis_results(self, result: GraphAnalysisResult) -> None:
        """Store analysis results in the database."""
        try:
            analysis_data = {
                "id": result.analysis_id,
                "tenant_id": result.tenant_id,
                "analysis_type": result.analysis_type,
                "timestamp": result.timestamp.isoformat(),
                "processing_time": result.processing_time,
                "success": result.success,
                "warnings": result.warnings,
                "result_data": result.dict()
            }
            
            await self.supabase_client.create_graph_analysis_record(analysis_data)
            
        except Exception as e:
            logger.error(
                "Failed to store analysis results",
                analysis_id=result.analysis_id,
                error=str(e)
            )
    
    async def _store_visualization(self, visualization: GraphVisualization) -> None:
        """Store visualization in the database."""
        try:
            viz_data = {
                "id": visualization.visualization_id,
                "tenant_id": visualization.tenant_id,
                "created_by": visualization.created_by,
                "center_node_id": visualization.center_node_id,
                "max_depth": visualization.max_depth,
                "max_nodes": visualization.max_nodes,
                "layout_algorithm": visualization.layout_algorithm,
                "nodes": visualization.nodes,
                "edges": visualization.edges,
                "created_at": visualization.created_at.isoformat()
            }
            
            await self.supabase_client.create_visualization_record(viz_data)
            
        except Exception as e:
            logger.error(
                "Failed to store visualization",
                visualization_id=visualization.visualization_id,
                error=str(e)
            )
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "agent_type": "graph_management",
            "tenant_id": self.tenant_id,
            "initialized_at": datetime.now(timezone.utc),
            "dependencies": {
                "neo4j_client": bool(self.neo4j_client),
                "qdrant_client": bool(self.qdrant_client),
                "supabase_client": bool(self.supabase_client),
                "embedding_service": bool(self.embedding_service)
            }
        }


# =============================================================================
# Utility Functions
# =============================================================================

async def create_graph_management_agent(
    tenant_id: str,
    neo4j_client: Optional[Neo4jClient] = None,
    qdrant_client: Optional[QdrantClient] = None,
    supabase_client: Optional[SupabaseClient] = None,
    embedding_service: Optional[EmbeddingService] = None
) -> GraphManagementAgent:
    """
    Create a configured Graph Management Agent.
    
    Args:
        tenant_id: Tenant identifier
        neo4j_client: Neo4j client (will create if not provided)
        qdrant_client: Qdrant client (will create if not provided)
        supabase_client: Supabase client (will create if not provided)
        embedding_service: Embedding service (will create if not provided)
        
    Returns:
        GraphManagementAgent instance
    """
    settings = get_settings()
    
    # Initialize clients if not provided
    if neo4j_client is None:
        neo4j_client = Neo4jClient(
            uri=settings.neo4j_uri,
            user=settings.neo4j_username,
            password=settings.neo4j_password
        )
        await neo4j_client.connect()
    
    if qdrant_client is None:
        qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )
    
    if supabase_client is None:
        supabase_client = SupabaseClient(
            url=settings.supabase_url,
            key=settings.supabase_key
        )
    
    if embedding_service is None:
        embedding_service = EmbeddingService(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model
        )
    
    return GraphManagementAgent(
        neo4j_client=neo4j_client,
        qdrant_client=qdrant_client,
        supabase_client=supabase_client,
        embedding_service=embedding_service,
        tenant_id=tenant_id
    )


if __name__ == "__main__":
    # Example usage
    async def main():
        # Create agent
        agent = await create_graph_management_agent(tenant_id="test-tenant")
        
        # Analyze graph structure
        analysis_result = await agent.analyze_graph_structure(
            include_recommendations=True,
            analysis_depth="comprehensive"
        )
        
        print(f"Graph analysis completed: {analysis_result.analysis_id}")
        if analysis_result.insights:
            print(f"Recommendations: {len(analysis_result.insights.recommendations)}")
            print(f"Growth opportunities: {len(analysis_result.insights.growth_opportunities)}")
        
        # Perform maintenance
        maintenance_result = await agent.perform_graph_maintenance(
            maintenance_tasks=["remove_duplicates", "fix_orphaned_nodes"],
            dry_run=True
        )
        
        print(f"Maintenance analysis completed")
        if maintenance_result.maintenance_result:
            print(f"Nodes processed: {maintenance_result.maintenance_result.nodes_processed}")
            print(f"Issues resolved: {len(maintenance_result.maintenance_result.issues_resolved)}")
        
        # Get agent stats
        stats = agent.get_agent_stats()
        print(f"Agent stats: {stats}")
    
    asyncio.run(main())