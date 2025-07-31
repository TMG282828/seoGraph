## FEATURE:

Core Architecture:

Multi-Agent System using Pydantic AI:
- Content Analysis Agent: Stores and analyzes existing content to extract topics, keywords, and semantic relationships
- SEO Research Agent: Uses SearXNG API to identify trending topics, keyword opportunities, and competitor analysis
- Content Generation Agent: Generates new content based on identified gaps and opportunities
- Graph Management Agent: Maintains and updates the knowledge graph structure in Neo4j
- Quality Assurance Agent: Ensures content aligns with brand voice and quality standards

Data Infrastructure:

- Neo4j Graph Database: Stores content relationships, topic hierarchies, and SEO metadata
-- Nodes: Topics, Articles, Keywords, Trends
-- Edges: Semantic relationships, internal links, keyword associations

- Qdrant Vector Database: Stores content embeddings for semantic search and similarity matching
- Supabase: User management, content versioning, and API authentication

Key Functionalities:

- Content Ingestion Pipeline:
-- Parse existing content (markdown, HTML, PDFs)
-- Extract topics, entities, and keywords
-- Generate embeddings and store in Qdrant
-- Build knowledge graph in Neo4j

Gap Analysis Engine:

- Compare existing content coverage against trending topics
- Identify missing subtopics and related keywords
- Prioritize opportunities based on search volume and competition

Content Generation Workflow:

- Topic selection based on gap analysis
- Option of importing briefs from google drive
- Outline generation using graph relationships
- Content drafting with brand voice consistency
- Internal linking suggestions based on graph connections

Monitoring & Analytics:

- Track content performance metrics
- Monitor keyword rankings (integration with Google Ads API)
- Visualize content graph and coverage gaps
- Langfuse for LLM observability and cost tracking

User Interfaces:

- CLI Tool: For batch operations, content ingestion, and administrative tasks
- Web Dashboard (Ag-ui/OpenWebUI):
-- Interactive graph visualization
-- Content planning calendar
-- Human-in-the-loop content review and editing
-- SEO performance dashboards

## EXAMPLES:

Examples from our examples/ folder include:
- case studies of successful SEO strategies
- examples of how topics are interlinked for SEO purpose
- Agent UI/UX examples with "Human In the Loop" function
- examples of isual knowledge graphs
- example of ahrefs seo/keyword monitoring dashboard

Don’t copy any of these examples directly, it is for a different project entirely. But use this as inspiration and for best practices.

## DOCUMENTATION:

- Pydantic AI documentation: https://ai.pydantic.dev/
- SearXNG documentation: https://docs.searxng.org/
- Neo4j documentation: https://neo4j.com/docs/
- Qdrant documentation: https://qdrant.tech/documentation/
- Ag-ui documentation: https://docs.ag-ui.com/introduction
- OpenwebUi documentation: https://docs.openwebui.com/
- Supabase documentation: https://supabase.com/docs
- Langfuse documentation: https://langfuse.com/docs
- pytrends documentation: https://pypi.org/project/pytrends/
- Serpbear documentation: https://docs.serpbear.com/
- crawl4AI documentation: https://docs.crawl4ai.com/



## OTHER CONSIDERATIONS:

Technical Requirements:

Environment Setup:

- Include comprehensive .env with all required API keys and configurations
- Docker Compose setup for Neo4j, Qdrant, and Supabase local development
- Python virtual environment with dependencies managed via requirements.txt


Configuration:

- Brand voice configuration file (tone, style guidelines, prohibited terms)
- SEO rules configuration (keyword density, content length, meta descriptions)
- Graph schema definition for Neo4j


Data Pipeline:

- Scheduled crawling of competitor sites using Crawl4AI
- Regular updates from pytrends for trending topics
- Incremental graph updates without full rebuilds


Security & Performance:

- API rate limiting for external services
- Caching layer for frequently accessed graph queries
- User authentication via Supabase for multi-tenant support
 

Monitoring & Logging:

- Structured logging for all agent activities
- Cost tracking for LLM API calls via Langfuse
- Performance metrics for content generation speed and quality
