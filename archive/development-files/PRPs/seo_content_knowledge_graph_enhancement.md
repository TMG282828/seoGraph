name: "SEO Content Knowledge Graph System - Production Enhancement"
description: |

## Purpose
Complete and enhance the existing SEO Content Knowledge Graph System to production readiness by implementing missing features, improving existing components, and adding comprehensive monitoring and analytics capabilities.

## Core Principles
1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Global rules**: Be sure to follow all rules in CLAUDE.md

---

## Goal
Transform the existing SEO Content Knowledge Graph System from a functional prototype into a production-ready platform with comprehensive gap analysis, advanced content generation workflows, competitor monitoring, and sophisticated analytics dashboards.

## Why
- **Business value**: Provides enterprise-grade SEO content automation and strategy optimization
- **Integration**: Completes the vision of a comprehensive multi-agent content ecosystem
- **Problems solved**: Automates complex SEO workflows, provides actionable insights, and scales content production intelligently

## What
Enhanced system featuring:
- Production-ready gap analysis engine with competitor intelligence
- Advanced content brief management and workflows
- Advanced content generation workflows with human-in-the-loop
- Comprehensive monitoring with performance tracking
- Interactive knowledge graph visualization
- Sophisticated analytics and reporting dashboards
- Automated competitor content monitoring
- Brand voice consistency validation
- Advanced SEO rule engine

### Success Criteria
- [ ] Gap analysis engine identifies content opportunities with 85%+ accuracy
- [✅] Advanced content brief management with database persistence
- [ ] Content generation workflow produces brand-consistent content
- [ ] Knowledge graph visualization provides actionable insights
- [ ] Monitoring dashboard tracks all key performance metrics
- [ ] Competitor analysis identifies opportunities and threats
- [ ] Human-in-the-loop workflows enable quality control
- [ ] Advanced analytics provide strategic insights
- [ ] System handles enterprise-scale content volumes
- [ ] Brand voice validation ensures consistency across all content

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- url: https://ai.pydantic.dev/agents/
  why: Advanced agent patterns, tool registration, context management
  
- url: https://ai.pydantic.dev/multi-agent-applications/
  why: Agent orchestration, delegation patterns, usage tracking
  
- url: https://developers.google.com/drive/api/v3/reference/
  why: Google Drive API integration for content brief management
  
- url: https://docs.langfuse.com/
  why: Advanced observability, custom metrics, performance tracking
  
- url: https://neo4j.com/docs/cypher-manual/current/
  why: Advanced graph queries, optimization patterns, analytics
  
- url: https://qdrant.tech/documentation/concepts/search/
  why: Advanced vector search, filtering, hybrid search patterns
  
- url: https://docs.searxng.org/
  why: SearXNG API integration, rate limiting, search optimization
  
- url: https://github.com/unstructured-io/unstructured
  why: Advanced document processing, content extraction patterns
  
- url: https://pypi.org/project/pytrends/
  why: Google Trends integration for topic research
  
- url: https://developers.google.com/google-ads/api/docs/keyword-planning/overview
  why: Google Keyword Planner integration for search volume data
  
- file: /Users/kitan/Desktop/apps/Context-Engineering-Intro/agents/content_analysis.py
  why: Existing agent patterns, dependency injection, structured outputs
  
- file: /Users/kitan/Desktop/apps/Context-Engineering-Intro/database/neo4j_client.py
  why: Database connection patterns, async operations, error handling
  
- file: /Users/kitan/Desktop/apps/Context-Engineering-Intro/models/content_models.py
  why: Existing data model patterns, validation, serialization
  
- file: /Users/kitan/Desktop/apps/Context-Engineering-Intro/services/content_ingestion.py
  why: File processing patterns, async operations, error handling
  
- file: /Users/kitan/Desktop/apps/Context-Engineering-Intro/web/main.py
  why: FastAPI patterns, routing, middleware, authentication
  
- file: /Users/kitan/Desktop/apps/Context-Engineering-Intro/config/settings.py
  why: Configuration management, environment variables, validation
  
- file: /Users/kitan/Desktop/apps/Context-Engineering-Intro/CLAUDE.md
  why: Project coding standards, testing patterns, conventions
```

### Current Codebase Structure
```bash
Context-Engineering-Intro/
├── agents/
│   ├── content_analysis.py       # ✅ Implemented - Content Analysis Agent
│   ├── content_generation.py     # ✅ Implemented - Content Generation Agent
│   ├── graph_management.py       # ✅ Implemented - Graph Management Agent
│   ├── quality_assurance.py      # ✅ Implemented - Quality Assurance Agent
│   └── seo_research.py           # ✅ Implemented - SEO Research Agent
├── database/
│   ├── neo4j_client.py           # ✅ Implemented - Neo4j operations
│   ├── qdrant_client.py          # ✅ Implemented - Vector database
│   └── supabase_client.py        # ✅ Implemented - User management
├── models/
│   ├── content_models.py         # ✅ Implemented - Content data models
│   ├── seo_models.py             # ✅ Implemented - SEO data models
│   └── graph_models.py           # ✅ Implemented - Graph data models
├── services/
│   ├── content_ingestion.py      # ✅ Implemented - File processing
│   ├── embedding_service.py      # ✅ Implemented - Vector embeddings
│   └── searxng_service.py        # ✅ Implemented - Search integration
├── web/
│   ├── main.py                   # ✅ Implemented - FastAPI application
│   └── templates/                # ✅ Implemented - Web interface
├── config/
│   └── settings.py               # ✅ Implemented - Configuration
└── tests/                        # ✅ Partially implemented - Test structure
```

### Files to be Enhanced/Added
```bash
Context-Engineering-Intro/
├── services/
│   ├── gap_analysis.py           # ➕ NEW - Advanced gap analysis engine
│   ├── google_drive_service.py   # ➕ NEW - Google Drive integration
│   ├── competitor_monitoring.py  # ➕ NEW - Competitor analysis
│   ├── analytics_service.py      # ➕ NEW - Advanced analytics
│   └── workflow_orchestrator.py  # ➕ NEW - Content workflow management
├── agents/
│   ├── trend_analysis.py         # ➕ NEW - Trend analysis agent
│   └── competitor_analysis.py    # ➕ NEW - Competitor analysis agent
├── web/
│   ├── api/
│   │   ├── analytics.py          # ➕ NEW - Analytics API endpoints
│   │   ├── workflows.py          # ➕ NEW - Workflow management API
│   │   └── integrations.py       # ➕ NEW - Integration management API
│   ├── static/
│   │   ├── js/
│   │   │   ├── graph-viz.js      # ➕ NEW - Graph visualization
│   │   │   ├── analytics.js      # ➕ NEW - Analytics dashboard
│   │   │   └── workflow.js       # ➕ NEW - Workflow management
│   │   └── css/
│   │       └── dashboard.css     # ➕ NEW - Enhanced styling
│   └── templates/
│       ├── analytics.html        # ➕ NEW - Analytics dashboard
│       ├── workflows.html        # ➕ NEW - Workflow management
│       └── integrations.html     # ➕ NEW - Integration management
├── config/
│   ├── brand_voice.py           # ➕ NEW - Brand voice configuration
│   ├── seo_rules.py             # ➕ NEW - SEO rules engine
│   └── workflow_templates.py    # ➕ NEW - Workflow templates
├── cli/
│   ├── main.py                  # ➕ NEW - Enhanced CLI interface
│   └── commands/
│       ├── analytics.py         # ➕ NEW - Analytics commands
│       ├── workflows.py         # ➕ NEW - Workflow commands
│       └── integrations.py      # ➕ NEW - Integration commands
└── monitoring/
    ├── __init__.py              # ➕ NEW - Monitoring package
    ├── metrics.py               # ➕ NEW - Custom metrics
    ├── alerting.py              # ➕ NEW - Alert management
    └── dashboards.py            # ➕ NEW - Monitoring dashboards
```

### Known Gotchas & Library Quirks
```python
# CRITICAL: Google Drive API requires OAuth2 flow and proper scope management
# CRITICAL: Crawl4AI rate limits are aggressive - implement proper backoff
# CRITICAL: Neo4j APOC procedures needed for advanced graph algorithms
# CRITICAL: Qdrant hybrid search requires careful query construction
# CRITICAL: SearXNG instances can be unstable - implement health checks
# CRITICAL: Langfuse custom metrics require structured event logging
# CRITICAL: Brand voice validation needs consistent prompt engineering
# CRITICAL: Graph visualization performance degrades with >1000 nodes
# CRITICAL: Competitor monitoring must respect robots.txt and rate limits
# CRITICAL: Content workflow state management requires proper locking
# CRITICAL: Analytics queries can be expensive - implement caching
# CRITICAL: Human-in-the-loop workflows need proper session management
```

## Implementation Blueprint

### Data models and structure

```python
# models/workflow_models.py - Content workflow management
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

class WorkflowStatus(str, Enum):
    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    APPROVED = "approved"
    PUBLISHED = "published"
    ARCHIVED = "archived"

class ContentBrief(BaseModel):
    id: str = Field(..., description="Unique brief identifier")
    title: str = Field(..., min_length=1, max_length=200)
    target_keywords: List[str] = Field(..., min_items=1)
    content_type: str = Field(..., description="Type of content to create")
    target_audience: str = Field(..., description="Target audience description")
    tone: str = Field(..., description="Content tone and style")
    word_count: int = Field(..., ge=100, le=10000)
    deadline: Optional[datetime] = None
    google_drive_id: Optional[str] = None
    tenant_id: str = Field(..., description="Tenant identifier")
    created_by: str = Field(..., description="User who created brief")
    assigned_to: Optional[str] = None
    status: WorkflowStatus = WorkflowStatus.DRAFT
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ContentWorkflow(BaseModel):
    id: str = Field(..., description="Unique workflow identifier")
    brief_id: str = Field(..., description="Associated content brief")
    content_id: Optional[str] = None
    workflow_type: str = Field(..., description="Type of workflow")
    status: WorkflowStatus = WorkflowStatus.DRAFT
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    current_step: int = Field(0, ge=0)
    assigned_agents: List[str] = Field(default_factory=list)
    human_reviewers: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    tenant_id: str = Field(..., description="Tenant identifier")

# models/analytics_models.py - Analytics and reporting
class ContentPerformanceMetrics(BaseModel):
    content_id: str = Field(..., description="Content identifier")
    organic_traffic: int = Field(0, ge=0)
    keyword_rankings: Dict[str, int] = Field(default_factory=dict)
    backlinks: int = Field(0, ge=0)
    social_shares: int = Field(0, ge=0)
    engagement_rate: float = Field(0.0, ge=0.0, le=1.0)
    conversion_rate: float = Field(0.0, ge=0.0, le=1.0)
    bounce_rate: float = Field(0.0, ge=0.0, le=1.0)
    time_on_page: float = Field(0.0, ge=0.0)
    measured_at: datetime = Field(default_factory=datetime.now)
    tenant_id: str = Field(..., description="Tenant identifier")

class CompetitorAnalysis(BaseModel):
    competitor_domain: str = Field(..., description="Competitor domain")
    content_topics: List[str] = Field(default_factory=list)
    keyword_gaps: List[str] = Field(default_factory=list)
    content_frequency: float = Field(0.0, ge=0.0)
    average_word_count: int = Field(0, ge=0)
    backlink_profile: Dict[str, Any] = Field(default_factory=dict)
    social_presence: Dict[str, Any] = Field(default_factory=dict)
    analyzed_at: datetime = Field(default_factory=datetime.now)
    tenant_id: str = Field(..., description="Tenant identifier")

class ContentGapOpportunity(BaseModel):
    topic: str = Field(..., description="Gap topic")
    priority_score: float = Field(..., ge=0.0, le=1.0)
    search_volume: int = Field(0, ge=0)
    competition_level: str = Field(..., description="Competition level")
    trend_direction: str = Field(..., description="Trend direction")
    related_keywords: List[str] = Field(default_factory=list)
    competitor_coverage: Dict[str, int] = Field(default_factory=dict)
    content_suggestions: List[str] = Field(default_factory=list)
    estimated_traffic: int = Field(0, ge=0)
    difficulty_score: float = Field(0.0, ge=0.0, le=1.0)
    identified_at: datetime = Field(default_factory=datetime.now)
    tenant_id: str = Field(..., description="Tenant identifier")
```

### List of tasks to be completed to fulfill the PRP in order

```yaml
Task 1: Enhance Gap Analysis Engine
MODIFY services/gap_analysis.py:
  - ADD advanced competitor analysis integration
  - IMPLEMENT trend correlation algorithms
  - ADD content opportunity scoring
  - INTEGRATE Google Trends and keyword data
  - IMPLEMENT caching for expensive queries

Task 2: Enhanced Content Brief Management (COMPLETED ✅)
ENHANCED web/api/brief_routes.py:
  - ✅ IMPLEMENTED database persistence for content briefs
  - ✅ ADD comprehensive brief analysis and validation
  - ✅ IMPLEMENT real-time AI chat for brief development
  - ✅ ADD auto-save functionality for work preservation
  - ✅ IMPLEMENT brief versioning and history tracking

Task 3: Build Competitor Monitoring System
CREATE services/competitor_monitoring.py:
  - PATTERN: Scheduled crawling with rate limiting
  - IMPLEMENT content extraction and analysis
  - ADD competitor content change detection
  - INTEGRATE with Crawl4AI for robust scraping
  - IMPLEMENT competitor keyword tracking

Task 4: Create Advanced Analytics Service
CREATE services/analytics_service.py:
  - PATTERN: Time-series data collection and analysis
  - IMPLEMENT custom metrics calculation
  - ADD performance trend analysis
  - INTEGRATE with Google Analytics API
  - IMPLEMENT automated reporting

Task 5: Build Workflow Orchestrator
CREATE services/workflow_orchestrator.py:
  - PATTERN: State machine for content workflows
  - IMPLEMENT human-in-the-loop approvals
  - ADD workflow template management
  - INTEGRATE with notification systems
  - IMPLEMENT workflow analytics

Task 6: Create Trend Analysis Agent
CREATE agents/trend_analysis.py:
  - PATTERN: Pydantic AI agent with trend analysis tools
  - IMPLEMENT Google Trends integration
  - ADD social media trend analysis
  - INTEGRATE with SearXNG for trend validation
  - IMPLEMENT trend prediction algorithms

Task 7: Build Competitor Analysis Agent
CREATE agents/competitor_analysis.py:
  - PATTERN: Agent with competitor intelligence tools
  - IMPLEMENT competitor content analysis
  - ADD keyword gap identification
  - INTEGRATE with competitor monitoring service
  - IMPLEMENT competitive advantage scoring

Task 8: Enhance Brand Voice Configuration
CREATE config/brand_voice.py:
  - PATTERN: Structured brand voice validation
  - IMPLEMENT voice consistency scoring
  - ADD tone and style guidelines
  - INTEGRATE with content generation
  - IMPLEMENT brand voice learning

Task 9: Create SEO Rules Engine
CREATE config/seo_rules.py:
  - PATTERN: Rule-based SEO validation
  - IMPLEMENT content optimization rules
  - ADD keyword optimization guidelines
  - INTEGRATE with quality assurance
  - IMPLEMENT rule customization

Task 10: Build Advanced Analytics API
CREATE web/api/analytics.py:
  - PATTERN: FastAPI endpoints with complex queries
  - IMPLEMENT dashboard data endpoints
  - ADD real-time metrics streaming
  - INTEGRATE with caching layer
  - IMPLEMENT custom report generation

Task 11: Create Workflow Management API
CREATE web/api/workflows.py:
  - PATTERN: RESTful workflow management
  - IMPLEMENT workflow state management
  - ADD human approval endpoints
  - INTEGRATE with notification system
  - IMPLEMENT workflow templates

Task 12: Build Integration Management API
CREATE web/api/integrations.py:
  - PATTERN: Third-party integration management
  - IMPLEMENT OAuth flow management
  - ADD integration health monitoring
  - INTEGRATE with external APIs
  - IMPLEMENT integration analytics

Task 13: Create Interactive Graph Visualization
CREATE web/static/js/graph-viz.js:
  - PATTERN: D3.js or Cytoscape.js for graph rendering
  - IMPLEMENT interactive node exploration
  - ADD filtering and search capabilities
  - INTEGRATE with Neo4j data
  - IMPLEMENT performance optimization

Task 14: Build Analytics Dashboard
CREATE web/templates/analytics.html:
  - PATTERN: Responsive dashboard with charts
  - IMPLEMENT real-time data updates
  - ADD custom metric visualization
  - INTEGRATE with Chart.js or D3.js
  - IMPLEMENT export functionality

Task 15: Create Workflow Management Interface
CREATE web/templates/workflows.html:
  - PATTERN: Kanban-style workflow management
  - IMPLEMENT drag-and-drop interfaces
  - ADD approval workflow visualization
  - INTEGRATE with notification system
  - IMPLEMENT workflow templates

Task 16: Build Enhanced CLI Interface
CREATE cli/main.py:
  - PATTERN: Click-based CLI with rich formatting
  - IMPLEMENT batch processing commands
  - ADD interactive workflow management
  - INTEGRATE with all services
  - IMPLEMENT progress tracking

Task 17: Create Monitoring and Alerting
CREATE monitoring/metrics.py:
  - PATTERN: Custom metrics collection
  - IMPLEMENT performance monitoring
  - ADD alert rule management
  - INTEGRATE with Langfuse and Prometheus
  - IMPLEMENT anomaly detection

Task 18: Enhance Testing Infrastructure
MODIFY tests/ package:
  - ADD integration tests for new services
  - IMPLEMENT load testing for analytics
  - ADD end-to-end workflow testing
  - INTEGRATE with CI/CD pipeline
  - IMPLEMENT performance benchmarks

Task 19: Create Documentation and Guides
CREATE documentation:
  - PATTERN: Comprehensive API documentation
  - IMPLEMENT user guides and tutorials
  - ADD architecture diagrams
  - INTEGRATE with code examples
  - IMPLEMENT interactive documentation

Task 20: Implement Production Deployment
CREATE deployment configuration:
  - PATTERN: Docker containerization
  - IMPLEMENT Kubernetes manifests
  - ADD environment-specific configs
  - INTEGRATE with CI/CD pipeline
  - IMPLEMENT monitoring and logging
```

### Per task pseudocode for critical components

```python
# Task 1: Enhanced Gap Analysis Engine
class GapAnalysisEngine:
    def __init__(self, neo4j_client: Neo4jClient, qdrant_client: QdrantClient, 
                 competitor_service: CompetitorMonitoringService):
        self.neo4j_client = neo4j_client
        self.qdrant_client = qdrant_client
        self.competitor_service = competitor_service
        self.cache = TTLCache(maxsize=1000, ttl=3600)
    
    async def analyze_content_gaps(self, tenant_id: str, industry: str) -> List[ContentGapOpportunity]:
        """Analyze content gaps with competitor intelligence."""
        # PATTERN: Multi-source data aggregation with caching
        # CRITICAL: Cache expensive queries to avoid rate limits
        
        cache_key = f"gaps_{tenant_id}_{industry}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Get existing content topics
        existing_topics = await self.neo4j_client.get_tenant_topics(tenant_id)
        
        # Get trending topics
        trending_topics = await self._get_trending_topics(industry)
        
        # Get competitor coverage
        competitor_coverage = await self.competitor_service.get_topic_coverage(industry)
        
        # Identify gaps using vector similarity
        gaps = await self._identify_content_gaps(
            existing_topics, trending_topics, competitor_coverage
        )
        
        # Score opportunities
        scored_gaps = await self._score_opportunities(gaps)
        
        # Cache results
        self.cache[cache_key] = scored_gaps
        return scored_gaps
    
    async def _score_opportunities(self, gaps: List[ContentGapOpportunity]) -> List[ContentGapOpportunity]:
        """Score content opportunities based on multiple factors."""
        # PATTERN: Multi-factor scoring algorithm
        # CRITICAL: Normalize scores across different data sources
        
        for gap in gaps:
            # Base score from search volume
            volume_score = min(gap.search_volume / 10000, 1.0)
            
            # Competition penalty
            competition_penalty = 1.0 - gap.difficulty_score
            
            # Trend boost
            trend_boost = 1.2 if gap.trend_direction == "rising" else 1.0
            
            # Competitor gap boost
            competitor_gap_boost = 1.0 + (1.0 - len(gap.competitor_coverage) / 10)
            
            # Calculate final score
            gap.priority_score = (
                volume_score * competition_penalty * trend_boost * competitor_gap_boost
            )
        
        return sorted(gaps, key=lambda x: x.priority_score, reverse=True)

# Task 2: Google Drive Integration
class GoogleDriveService:
    def __init__(self, credentials_file: str):
        self.credentials_file = credentials_file
        self.service = None
        self.webhook_url = None
    
    async def initialize(self):
        """Initialize Google Drive service with OAuth2."""
        # PATTERN: OAuth2 flow with refresh token management
        # CRITICAL: Handle token refresh and re-authentication
        
        flow = InstalledAppFlow.from_client_secrets_file(
            self.credentials_file, 
            scopes=['https://www.googleapis.com/auth/drive']
        )
        
        credentials = flow.run_local_server(port=0)
        self.service = build('drive', 'v3', credentials=credentials)
    
    async def sync_content_briefs(self, tenant_id: str, folder_id: str) -> List[ContentBrief]:
        """Sync content briefs from Google Drive folder."""
        # PATTERN: Incremental sync with change detection
        # CRITICAL: Handle large folders with pagination
        
        try:
            # Get files from folder
            query = f"'{folder_id}' in parents and mimeType contains 'document'"
            files = self.service.files().list(q=query, pageSize=100).execute()
            
            briefs = []
            for file_metadata in files.get('files', []):
                # Download and parse content brief
                content = await self._download_file_content(file_metadata['id'])
                brief = await self._parse_content_brief(content, file_metadata, tenant_id)
                briefs.append(brief)
            
            return briefs
            
        except Exception as e:
            logger.error(f"Failed to sync content briefs: {e}")
            raise
    
    async def _parse_content_brief(self, content: str, metadata: Dict, tenant_id: str) -> ContentBrief:
        """Parse content brief from Google Doc."""
        # PATTERN: Natural language parsing with fallback
        # CRITICAL: Handle various document formats gracefully
        
        # Use AI to extract structured data from document
        agent = Agent('openai:gpt-4o-mini')
        result = await agent.run(
            f"Extract content brief data from this document: {content}",
            result_type=ContentBrief
        )
        
        # Add metadata
        brief = result.data
        brief.google_drive_id = metadata['id']
        brief.tenant_id = tenant_id
        brief.created_by = metadata.get('owners', [{}])[0].get('emailAddress', 'unknown')
        
        return brief

# Task 5: Workflow Orchestrator
class WorkflowOrchestrator:
    def __init__(self, neo4j_client: Neo4jClient, notification_service: NotificationService):
        self.neo4j_client = neo4j_client
        self.notification_service = notification_service
        self.workflows = {}
    
    async def create_workflow(self, brief: ContentBrief, workflow_type: str) -> ContentWorkflow:
        """Create new content workflow."""
        # PATTERN: State machine workflow management
        # CRITICAL: Handle concurrent workflow state changes
        
        workflow_id = str(uuid.uuid4())
        steps = await self._get_workflow_steps(workflow_type)
        
        workflow = ContentWorkflow(
            id=workflow_id,
            brief_id=brief.id,
            workflow_type=workflow_type,
            steps=steps,
            tenant_id=brief.tenant_id
        )
        
        # Store workflow
        await self.neo4j_client.create_workflow_node(workflow)
        self.workflows[workflow_id] = workflow
        
        # Start first step
        await self._execute_workflow_step(workflow, 0)
        
        return workflow
    
    async def _execute_workflow_step(self, workflow: ContentWorkflow, step_index: int):
        """Execute specific workflow step."""
        # PATTERN: Async task execution with error handling
        # CRITICAL: Handle failures gracefully with retry logic
        
        if step_index >= len(workflow.steps):
            await self._complete_workflow(workflow)
            return
        
        step = workflow.steps[step_index]
        step_type = step.get('type')
        
        try:
            if step_type == 'agent_execution':
                await self._execute_agent_step(workflow, step)
            elif step_type == 'human_review':
                await self._initiate_human_review(workflow, step)
            elif step_type == 'validation':
                await self._validate_content(workflow, step)
            elif step_type == 'publication':
                await self._publish_content(workflow, step)
            
            # Move to next step
            workflow.current_step += 1
            await self._save_workflow_state(workflow)
            
            # Execute next step
            await self._execute_workflow_step(workflow, workflow.current_step)
            
        except Exception as e:
            logger.error(f"Workflow step failed: {e}")
            await self._handle_workflow_error(workflow, step_index, e)

# Task 13: Interactive Graph Visualization
"""
// web/static/js/graph-viz.js
class GraphVisualization {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        this.options = {
            nodeLimit: 1000,
            edgeLimit: 2000,
            layoutAlgorithm: 'force-atlas2',
            ...options
        };
        this.cy = null;
        this.filterState = {
            nodeTypes: new Set(),
            edgeTypes: new Set(),
            searchTerm: ''
        };
    }
    
    async initialize() {
        // PATTERN: Progressive loading with performance optimization
        // CRITICAL: Limit node count to prevent performance issues
        
        this.cy = cytoscape({
            container: this.container,
            style: await this.loadStyles(),
            layout: {
                name: this.options.layoutAlgorithm,
                animate: true,
                animationDuration: 500
            },
            elements: []
        });
        
        this.setupEventHandlers();
        await this.loadInitialData();
    }
    
    async loadGraphData(tenantId, filters = {}) {
        // PATTERN: Paginated data loading with caching
        // CRITICAL: Cache graph data to reduce server load
        
        const cacheKey = `graph_${tenantId}_${JSON.stringify(filters)}`;
        let data = this.getFromCache(cacheKey);
        
        if (!data) {
            const response = await fetch(`/api/graph/data?tenant_id=${tenantId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(filters)
            });
            
            data = await response.json();
            this.setCache(cacheKey, data, 300); // 5 minute cache
        }
        
        // Limit nodes for performance
        if (data.nodes.length > this.options.nodeLimit) {
            data.nodes = data.nodes.slice(0, this.options.nodeLimit);
        }
        
        return data;
    }
    
    applyFilters(nodeTypes, edgeTypes, searchTerm) {
        // PATTERN: Client-side filtering with debouncing
        // CRITICAL: Debounce filter changes to prevent excessive updates
        
        clearTimeout(this.filterTimeout);
        this.filterTimeout = setTimeout(() => {
            this.filterState = { nodeTypes, edgeTypes, searchTerm };
            this.updateVisibleElements();
        }, 300);
    }
    
    updateVisibleElements() {
        // PATTERN: Efficient element visibility toggling
        // CRITICAL: Use CSS classes for performance
        
        const elements = this.cy.elements();
        
        elements.forEach(element => {
            const shouldShow = this.elementPassesFilters(element);
            element.style('display', shouldShow ? 'element' : 'none');
        });
        
        this.cy.fit();
    }
}
"""

# Task 14: Analytics Dashboard
class AnalyticsService:
    def __init__(self, neo4j_client: Neo4jClient, qdrant_client: QdrantClient):
        self.neo4j_client = neo4j_client
        self.qdrant_client = qdrant_client
        self.cache = TTLCache(maxsize=500, ttl=1800)  # 30 minute cache
    
    async def get_content_performance_dashboard(self, tenant_id: str, 
                                             date_range: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Get comprehensive content performance dashboard data."""
        # PATTERN: Multi-metric dashboard with real-time updates
        # CRITICAL: Cache expensive aggregations
        
        cache_key = f"dashboard_{tenant_id}_{date_range[0]}_{date_range[1]}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Aggregate metrics in parallel
        tasks = [
            self._get_content_metrics(tenant_id, date_range),
            self._get_keyword_performance(tenant_id, date_range),
            self._get_topic_trends(tenant_id, date_range),
            self._get_competitive_analysis(tenant_id, date_range),
            self._get_workflow_metrics(tenant_id, date_range)
        ]
        
        results = await asyncio.gather(*tasks)
        
        dashboard_data = {
            'content_metrics': results[0],
            'keyword_performance': results[1],
            'topic_trends': results[2],
            'competitive_analysis': results[3],
            'workflow_metrics': results[4],
            'generated_at': datetime.now(),
            'tenant_id': tenant_id
        }
        
        self.cache[cache_key] = dashboard_data
        return dashboard_data
    
    async def _get_content_metrics(self, tenant_id: str, date_range: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Get content performance metrics."""
        # PATTERN: Complex aggregation queries with optimization
        # CRITICAL: Use indexes for date range queries
        
        query = """
        MATCH (c:Content {tenant_id: $tenant_id})
        WHERE c.created_at >= $start_date AND c.created_at <= $end_date
        OPTIONAL MATCH (c)-[:PERFORMS]->(m:Metrics)
        RETURN 
            count(c) as total_content,
            avg(m.seo_score) as avg_seo_score,
            avg(m.organic_traffic) as avg_organic_traffic,
            sum(m.organic_traffic) as total_organic_traffic,
            avg(m.engagement_rate) as avg_engagement_rate,
            collect(DISTINCT c.content_type) as content_types
        """
        
        result = await self.neo4j_client.execute_query(query, {
            'tenant_id': tenant_id,
            'start_date': date_range[0].isoformat(),
            'end_date': date_range[1].isoformat()
        })
        
        return result[0] if result else {}
```

### Integration Points
```yaml
ENVIRONMENT_ADDITIONS:
  - add to: .env
  - vars: |
      # Google Drive Integration
      GOOGLE_DRIVE_CREDENTIALS_FILE=./config/google_drive_credentials.json
      GOOGLE_DRIVE_WEBHOOK_URL=https://your-domain.com/api/webhooks/google-drive
      
      # Competitor Monitoring
      CRAWL4AI_API_KEY=your-crawl4ai-api-key
      COMPETITOR_CRAWL_INTERVAL=24h
      COMPETITOR_DOMAINS=competitor1.com,competitor2.com
      
      # Analytics Integration
      GOOGLE_ANALYTICS_VIEW_ID=123456789
      GOOGLE_ANALYTICS_CREDENTIALS_FILE=./config/ga_credentials.json
      
      # Advanced Features
      CACHE_TTL_SECONDS=1800
      GRAPH_VISUALIZATION_NODE_LIMIT=1000
      WORKFLOW_NOTIFICATION_WEBHOOK=https://your-domain.com/api/webhooks/workflow
      
      # Monitoring and Alerting
      PROMETHEUS_ENDPOINT=http://localhost:9090
      ALERT_MANAGER_WEBHOOK=https://your-domain.com/api/webhooks/alerts
      
DOCKER_COMPOSE_ADDITIONS:
  - Services: Prometheus, Grafana, Redis (for caching)
  - Volumes: Persistent storage for analytics data
  - Networks: Monitoring network for metrics collection
  
DEPENDENCY_ADDITIONS:
  - Update requirements.txt with:
    - google-api-python-client
    - google-auth-oauthlib
    - crawl4ai
    - pytrends
    - prometheus-client
    - cachetools
    - asyncio-mqtt
    - websockets
    - celery[redis]
    - flower
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
ruff check . --fix                 # Auto-fix style issues
ruff format .                      # Format code
mypy . --ignore-missing-imports    # Type checking

# Expected: No errors. If errors, READ and fix.
```

### Level 2: Unit Tests
```python
# test_services/test_gap_analysis.py
async def test_gap_analysis_with_competitor_data():
    """Test gap analysis integrates competitor intelligence."""
    engine = GapAnalysisEngine(mock_neo4j, mock_qdrant, mock_competitor_service)
    
    # Mock competitor data
    mock_competitor_service.get_topic_coverage.return_value = {
        'ai': 50, 'machine learning': 30, 'deep learning': 10
    }
    
    gaps = await engine.analyze_content_gaps(
        tenant_id="test_tenant",
        industry="technology"
    )
    
    assert len(gaps) > 0
    assert all(isinstance(gap, ContentGapOpportunity) for gap in gaps)
    assert all(0.0 <= gap.priority_score <= 1.0 for gap in gaps)
    assert gaps[0].priority_score >= gaps[-1].priority_score  # Sorted by priority

# test_services/test_google_drive_service.py
async def test_google_drive_sync():
    """Test Google Drive content brief synchronization."""
    service = GoogleDriveService('test_credentials.json')
    
    # Mock Google Drive API responses
    mock_files = {
        'files': [
            {
                'id': 'file1',
                'name': 'Content Brief - SEO Guide',
                'mimeType': 'application/vnd.google-apps.document'
            }
        ]
    }
    
    with patch.object(service, 'service') as mock_service:
        mock_service.files().list().execute.return_value = mock_files
        
        briefs = await service.sync_content_briefs(
            tenant_id="test_tenant",
            folder_id="test_folder"
        )
        
        assert len(briefs) == 1
        assert briefs[0].google_drive_id == 'file1'
        assert briefs[0].tenant_id == "test_tenant"

# test_services/test_workflow_orchestrator.py
async def test_workflow_execution():
    """Test workflow orchestration with multiple steps."""
    orchestrator = WorkflowOrchestrator(mock_neo4j, mock_notification_service)
    
    brief = ContentBrief(
        id="test_brief",
        title="Test Content Brief",
        target_keywords=["seo", "content"],
        content_type="article",
        target_audience="developers",
        tone="professional",
        word_count=1000,
        tenant_id="test_tenant",
        created_by="test_user"
    )
    
    workflow = await orchestrator.create_workflow(brief, "standard_article")
    
    assert workflow.id is not None
    assert workflow.brief_id == brief.id
    assert workflow.status == WorkflowStatus.IN_PROGRESS
    assert len(workflow.steps) > 0
    assert workflow.current_step == 0

# test_monitoring/test_metrics.py
async def test_custom_metrics_collection():
    """Test custom metrics collection and aggregation."""
    metrics_service = MetricsService(mock_neo4j, mock_langfuse)
    
    # Test metric collection
    await metrics_service.record_agent_execution(
        agent_name="content_analysis",
        tenant_id="test_tenant",
        execution_time=1.5,
        token_usage=150,
        success=True
    )
    
    # Test metric aggregation
    metrics = await metrics_service.get_agent_metrics(
        agent_name="content_analysis",
        tenant_id="test_tenant",
        time_range="1d"
    )
    
    assert metrics['total_executions'] == 1
    assert metrics['avg_execution_time'] == 1.5
    assert metrics['total_tokens'] == 150
    assert metrics['success_rate'] == 1.0
```

```bash
# Run tests iteratively until passing:
pytest tests/ -v --cov=services --cov=agents --cov=monitoring --cov-report=term-missing

# Expected: All tests pass with >85% coverage
# If failing: Debug specific test, fix code, re-run
```

### Level 3: Integration Tests
```bash
# Start enhanced development environment
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d

# Wait for all services to be ready
sleep 60

# Test Google Drive integration
python -c "
import asyncio
from services.google_drive_service import GoogleDriveService

async def test_google_drive():
    service = GoogleDriveService('./config/google_drive_credentials.json')
    await service.initialize()
    
    # Test folder listing
    folders = await service.list_folders()
    print(f'Found {len(folders)} folders')
    
    # Test webhook setup
    webhook_url = await service.setup_webhook('test_folder_id')
    print(f'Webhook URL: {webhook_url}')

asyncio.run(test_google_drive())
"

# Test gap analysis with real data
python -c "
import asyncio
from services.gap_analysis import GapAnalysisEngine
from database.neo4j_client import get_neo4j_client
from database.qdrant_client import QdrantClient
from services.competitor_monitoring import CompetitorMonitoringService

async def test_gap_analysis():
    neo4j = await get_neo4j_client()
    qdrant = QdrantClient()
    competitor_service = CompetitorMonitoringService()
    
    engine = GapAnalysisEngine(neo4j, qdrant, competitor_service)
    
    gaps = await engine.analyze_content_gaps(
        tenant_id='test_tenant',
        industry='technology'
    )
    
    print(f'Found {len(gaps)} content gaps')
    for gap in gaps[:5]:
        print(f'- {gap.topic}: {gap.priority_score:.3f}')
    
    await neo4j.close()

asyncio.run(test_gap_analysis())
"

# Test workflow orchestration
python -c "
import asyncio
from services.workflow_orchestrator import WorkflowOrchestrator
from models.workflow_models import ContentBrief

async def test_workflow():
    orchestrator = WorkflowOrchestrator(mock_neo4j, mock_notification_service)
    
    brief = ContentBrief(
        id='test_brief',
        title='Test Content Brief',
        target_keywords=['seo', 'content'],
        content_type='article',
        target_audience='developers',
        tone='professional',
        word_count=1000,
        tenant_id='test_tenant',
        created_by='test_user'
    )
    
    workflow = await orchestrator.create_workflow(brief, 'standard_article')
    print(f'Created workflow: {workflow.id}')
    print(f'Status: {workflow.status}')
    print(f'Current step: {workflow.current_step}')

asyncio.run(test_workflow())
"

# Test analytics dashboard data
curl -X GET "http://localhost:8000/api/v1/analytics/dashboard?tenant_id=test_tenant&start_date=2024-01-01&end_date=2024-12-31" \
  -H "Authorization: Bearer test_token"

# Expected: JSON response with comprehensive dashboard data
# Check that all metrics are present and formatted correctly

# Test graph visualization endpoint
curl -X POST "http://localhost:8000/api/v1/graph/data" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test_token" \
  -d '{
    "tenant_id": "test_tenant",
    "node_types": ["content", "topic", "keyword"],
    "limit": 100
  }'

# Expected: Graph data with nodes and edges
# Check that data is properly formatted for visualization

# Test workflow API
curl -X POST "http://localhost:8000/api/v1/workflows" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test_token" \
  -d '{
    "brief_id": "test_brief",
    "workflow_type": "standard_article",
    "tenant_id": "test_tenant"
  }'

# Expected: Created workflow with ID and status
```

## Final Validation Checklist
- [ ] All tests pass: `pytest tests/ -v`
- [ ] No linting errors: `ruff check . && ruff format . --check`
- [ ] No type errors: `mypy . --ignore-missing-imports`
- [ ] Docker environment starts with all services
- [ ] Google Drive integration connects and syncs briefs
- [ ] Gap analysis identifies opportunities with competitor data
- [ ] Workflow orchestration executes multi-step processes
- [ ] Analytics dashboard displays comprehensive metrics
- [ ] Graph visualization renders interactive knowledge graph
- [ ] Competitor monitoring tracks content changes
- [ ] Brand voice validation maintains consistency
- [ ] Monitoring system tracks performance metrics
- [ ] Human-in-the-loop workflows enable approvals
- [ ] CLI interface supports all operations
- [ ] API endpoints handle all CRUD operations
- [ ] Authentication and authorization work correctly
- [ ] Caching improves performance for expensive operations
- [ ] Error handling provides meaningful feedback
- [ ] Notification system alerts users of important events
- [ ] Backup and recovery procedures are documented
- [ ] Performance meets enterprise-scale requirements
- [ ] Documentation covers all features and workflows

---

## Anti-Patterns to Avoid
- ❌ Don't bypass OAuth2 security for Google Drive integration
- ❌ Don't crawl competitor sites without respecting robots.txt
- ❌ Don't ignore rate limits on external API calls
- ❌ Don't store sensitive credentials in configuration files
- ❌ Don't create workflows without proper error handling
- ❌ Don't render large graphs without performance optimization
- ❌ Don't skip caching for expensive analytics queries
- ❌ Don't ignore brand voice consistency in generated content
- ❌ Don't create circular dependencies in workflow steps
- ❌ Don't skip monitoring for critical system components
- ❌ Don't hardcode workflow templates in application code
- ❌ Don't ignore data privacy regulations in analytics collection

## Confidence Score: 9/10

Very high confidence due to:
- Comprehensive research on all integration technologies
- Clear patterns from existing codebase implementation
- Well-established service architecture and patterns
- Detailed validation gates and testing strategies
- Experience with similar enterprise content systems
- Thorough documentation of all integration points

Minor uncertainty on:
- Google Drive API rate limits and quotas in production
- Competitor monitoring legal and ethical considerations
- Performance optimization for large-scale graph visualizations
- Complex workflow state management edge cases

The extensive context, progressive implementation approach, and comprehensive validation loops should enable successful implementation of all enhancement features.