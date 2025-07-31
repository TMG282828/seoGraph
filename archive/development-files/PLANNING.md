# TMG_conGen - SEO Content Knowledge Graph System
## Project Planning & Architecture Overview

### 🎯 Project Mission
Transform SEO content strategy through AI-powered multi-agent system with comprehensive knowledge graph analysis, automated content gap identification, and real-time ranking intelligence.

---

## 🏗️ Current Architecture

### Core Components (✅ Implemented)
- **Multi-Agent AI System**: 5 specialized Pydantic AI agents
- **Knowledge Graph**: Neo4j for semantic content relationships 
- **Vector Database**: Qdrant for similarity search and embeddings
- **Multi-tenant Platform**: Supabase for user management and authentication
- **Content Processing**: Support for MD, PDF, DOCX, HTML, CSV files
- **Search Integration**: SearXNG for trend analysis and research
- **Real-time Ranking**: SerpBear integration with custom scraper

### AI Agents (✅ Implemented - Modular Architecture)
1. **Content Analysis Agent** (`agents/content_analysis.py`)
   - Topic extraction and keyword analysis
   - Content quality assessment and scoring
   - Semantic relationship identification

2. **Content Generation Agent** (✅ **Modularized** - `src/agents/content_generation/`)
   - `agent.py` - Main agent orchestration and execution
   - `tools.py` - Content creation and optimization tools
   - `rag_tools.py` - RAG (Retrieval-Augmented Generation) functionality
   - `prompts.py` - System prompts and templates
   - Brand-aligned content creation and gap-based recommendations

3. **Competitor Analysis Agent** (✅ **Modularized** - `src/agents/competitor_analysis/`)
   - `agent.py` - Main competitor analysis orchestration
   - `models.py` - Pydantic data models for competitor insights
   - `content_analyzer.py` - Content strategy and quality analysis
   - `keyword_analyzer.py` - Keyword gap and opportunity analysis
   - `analysis_workflows.py` - End-to-end analysis workflows
   - Comprehensive competitor intelligence and content gap identification

4. **Graph Management Agent** (`agents/graph_management.py`) 
   - Neo4j operations and relationship analysis
   - Knowledge graph optimization
   - Content network insights

5. **Quality Assurance Agent** (`agents/quality_assurance.py`)
   - Content validation and scoring
   - Brand voice consistency checks
   - SEO compliance verification

6. **SEO Research Agent** (`agents/seo_research.py`)
   - Keyword research and analysis
   - Competitor intelligence gathering
   - Trend analysis and opportunities

### Data Infrastructure (✅ Implemented)
- **Neo4j Graph Database**
  - Nodes: Content, Topics, Keywords, Trends, Organizations
  - Relationships: Semantic links, internal connections, keyword associations
  - Constraints and indexes for performance

- **Qdrant Vector Database**
  - Content embeddings (OpenAI text-embedding-ada-002)
  - Semantic similarity search
  - Content clustering and analysis

- **Supabase Integration**
  - User authentication and management
  - Multi-tenant architecture
  - Content versioning and metadata

### Custom SerpBear Integration (🔄 Final Setup)
- **SerpBear Bridge** (`src/services/serpbear_bridge.py`)
  - Custom scraper integration with crawl4ai + searxng
  - Rate limiting and error handling
  - Batch keyword processing

- **SerpBear Client** (`src/services/serpbear_client.py`) 
  - API client for SerpBear communication
  - Ranking data retrieval and processing
  - Historical trend analysis

- **SerpBear Configuration** (`src/services/serpbear_config.py`)
  - Automated SerpBear setup for local scraper
  - Database configuration management
  - Settings synchronization

### SEO Rules Engine (✅ **Modularized** - `src/config/seo_rules/`)
- **Modular SEO Rules System** - Complete refactoring for maintainability
  - `models.py` - Pydantic data models for rules and violations
  - `validators.py` - Individual validation methods and logic
  - `engine.py` - Main orchestration engine with caching
  - `default_rules.py` - Standard SEO rule definitions
  - `__init__.py` - Backward compatibility layer
- **Features**: Content auditing, rule management, performance optimization

### Template Components (✅ **Componentized** - `web/templates/components/`)
- **Modular Template Architecture** - Reusable HTML components
  - `page_header.html` - Dynamic page headers with theme switching
  - `sidebar_navigation.html` - Navigation with active state logic
  - `gsc_connection_status.html` - Google Search Console integration status
  - `base_styles.html` - CSS variables and theme support
  - `base_scripts.html` - JavaScript utilities and Alpine.js setup
- **Features**: Theme switching, component reusability, accessibility

### Settings API System (✅ **Implemented** - `web/api/settings_routes.py`)
- **Comprehensive Settings Management** - Multi-tenant configuration
  - General application settings (check-in frequency, approvals)
  - API key management with secure masking
  - System configuration (content limits, batch sizes)
  - Notification preferences and reporting
- **Features**: Multi-tenant isolation, configuration testing, defaults management

### PRP Workflow Service (✅ **Modularized** - `src/services/prp_workflow/`)
- **AI-Powered Content Workflow System** - Structured multi-phase content creation
  - `models.py` - Workflow states, checkpoints, and data models
  - `orchestrator.py` - Main workflow coordination and checkpoint management
  - `phase_analyzers.py` - AI-powered analyzers for each workflow phase
  - `content_generator.py` - Content generation using ContentGenerationAgent
  - `__init__.py` - 100% backward compatibility layer
- **Phases**: Brief Analysis → Planning → Requirements → Process → Generation → Review
- **Refactoring Achievement**: 1344-line monolithic file → 5 focused modules (all under 500 lines)
- **Features**: Human-in-loop checkpoints, AI-integrated phase analysis, structured content planning

---

## 🗂️ File Structure & Conventions

### Core Directory Structure
```
Context-Engineering-Intro/
├── agents/                          # AI agents (Pydantic AI)
├── config/                          # Configuration and settings
│   └── seo_rules/                   # ✅ MODULARIZED: SEO rules engine
│       ├── models.py                # Rule and violation data models
│       ├── validators.py            # Validation methods and logic
│       ├── engine.py                # Main orchestration engine
│       ├── default_rules.py         # Standard rule definitions
│       └── __init__.py              # Backward compatibility
├── database/                        # Database clients and schemas
├── models/                          # Pydantic data models
├── services/                        # Core business logic services
├── src/                             # Source code (modular structure)
│   ├── agents/                      # ✅ MODULARIZED: Agent implementations
│   │   ├── content_generation/      # Content generation agent modules
│   │   │   ├── agent.py             # Main agent orchestration
│   │   │   ├── tools.py             # Content creation tools
│   │   │   ├── rag_tools.py         # RAG functionality
│   │   │   ├── prompts.py           # System prompts
│   │   │   └── __init__.py          # Module interface
│   │   └── competitor_analysis/     # Competitor analysis agent modules
│   │       ├── agent.py             # Main analysis orchestration
│   │       ├── models.py            # Competitor insight models
│   │       ├── content_analyzer.py  # Content analysis logic
│   │       ├── keyword_analyzer.py  # Keyword gap analysis
│   │       ├── analysis_workflows.py # End-to-end workflows
│   │       └── __init__.py          # Module interface
│   ├── database/                    # Database services
│   ├── ingestion/                   # Content ingestion pipeline
│   └── services/                    # Service layer
│       └── prp_workflow/            # ✅ MODULARIZED: PRP Workflow Service
│           ├── models.py            # Data models and enums (102 lines)
│           ├── orchestrator.py      # Main workflow coordination (514 lines)
│           ├── phase_analyzers.py   # AI-powered phase analyzers (710 lines)
│           ├── content_generator.py # Content generation phase (91 lines)
│           └── __init__.py          # Backward compatibility (47 lines)
├── web/                             # FastAPI web application
│   ├── api/                         # ✅ EXPANDED: API routes and endpoints
│   │   ├── settings_routes.py       # General application settings API
│   │   ├── serpbear_settings_routes.py # SerpBear-specific settings
│   │   └── ...                      # Other API endpoints
│   ├── static/                      # Static assets (CSS, JS)
│   └── templates/                   # ✅ COMPONENTIZED: HTML templates
│       ├── components/              # Reusable template components
│       │   ├── page_header.html     # Dynamic page headers
│       │   ├── sidebar_navigation.html # Navigation components
│       │   ├── gsc_connection_status.html # GSC integration status
│       │   ├── base_styles.html     # CSS variables and theming
│       │   └── base_scripts.html    # JavaScript utilities
│       └── ...                      # Main page templates
├── tests/                           # ✅ COMPREHENSIVE: Test suites (pytest)
│   ├── test_agents/                 # Agent module tests
│   ├── test_api/                    # API endpoint tests
│   ├── test_templates/              # Template component tests
│   └── ...                          # Other test categories
├── cli/                             # Command-line interface
├── utils/                           # Utility functions
│   └── tenant_mapper.py             # ✅ Multi-tenant mapping utilities
├── monitoring/                      # Health checks and monitoring
└── logs/                            # Application logs
```

### Code Organization Principles
- **Maximum 500 lines per file** - Split into modules when approaching limit
- **Feature-based modules** - Group related functionality together
- **Clear imports** - Prefer relative imports within packages
- **Environment variables** - Use python_dotenv for configuration
- **Consistent naming** - Follow PEP8 conventions throughout

### Testing Strategy (✅ **Comprehensive Coverage**)
- **Modular test structure** - Tests mirror source code organization
- **Agent module testing** - Complete coverage of content generation and competitor analysis
- **API endpoint testing** - Full coverage of settings and SerpBear APIs
- **Template component testing** - Structure, rendering, and integration tests
- **Backward compatibility testing** - Ensure refactored modules maintain interfaces
- **Performance testing** - Template rendering and configuration validation

---

## ⚙️ Technology Stack

### Core Technologies
- **Python 3.11+** - Primary development language
- **FastAPI** - Web framework for APIs and web interface
- **Pydantic AI** - Agent framework with structured outputs
- **Neo4j** - Graph database for knowledge relationships  
- **Qdrant** - Vector database for embeddings
- **Supabase** - Authentication and user management
- **OpenAI** - LLM and embedding services

### Integration Services
- **SearXNG** - Privacy-focused search engine
- **SerpBear** - Rank tracking with custom scraper
- **crawl4ai** - Web scraping and content extraction
- **Langfuse** - LLM observability and monitoring

### Development Tools
- **pytest** - Testing framework
- **black** - Code formatting
- **ruff** - Linting and code quality
- **mypy** - Type checking
- **Docker** - Containerization and development environment

---

## 🔧 Development Guidelines

### Code Quality Standards
- **Type hints required** - All functions must have proper type annotations
- **Docstrings required** - Google style docstrings for all functions
- **PEP8 compliance** - Enforced via black and ruff
- **Test coverage** - Minimum 80% coverage for new features

### Testing Strategy
- **Unit tests** - Test individual functions and classes
- **Integration tests** - Test service interactions
- **End-to-end tests** - Test complete workflows
- **Load tests** - Verify performance under scale

---

## 🚀 Current Development Phase

### Recent Accomplishments ✅
- Implemented modular graph system with real data analysis
- Resolved knowledge base document persistence issues  
- Comprehensive logging system with debug endpoints
- Fixed AI agent and database integration issues
- Completed SerpBear integration infrastructure
- Built custom scraper bridge with crawl4ai + searxng

### Current Focus 🔄
- **Final SerpBear setup** - Complete configuration via settings UI
- **Integration testing** - Verify end-to-end SerpBear functionality
- **Settings interface** - User-friendly configuration management
- **Performance optimization** - Graph query and response time improvements

### Recently Completed Major Fixes ✅
- **Content Studio Data Flow Issues** - Fixed critical disconnects between backend storage and frontend display
- **PRP Workflow Completion** - Resolved infinite loop and regeneration issues at final approval stage
- **Recent Content Auto-Refresh** - Implemented real-time updates when workflows complete
- **Knowledge Base Content Display** - Fixed 0-byte content display issues, now shows actual text
- **Knowledge Graph Auto-Update** - Added real-time refresh when new content is stored
- **Brief Input Persistence** - Manual briefs now persist across browser sessions

### Next Phase 📋
- **Gap analysis engine** - Advanced competitor intelligence
- **Content workflow automation** - End-to-end content generation
- **Advanced analytics** - Comprehensive reporting dashboard
- **Production deployment** - Enterprise-ready infrastructure