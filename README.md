# SEO Content Knowledge Graph System

A comprehensive multi-agent AI system for SEO content analysis, optimization, and knowledge graph management. Built with Pydantic AI, Neo4j, Qdrant, and modern Python architecture.

**✅ Recently Enhanced**: The system has been completely refactored into a modular architecture with improved maintainability, comprehensive test coverage, and enhanced developer experience while maintaining 100% backward compatibility.

## 🚀 **Quick Deploy on Replit**

[![Run on Replit](https://replit.com/badge/github/your-username/your-repo)](https://replit.com/new/github/your-username/your-repo)

1. **Fork this repository** or import from GitHub to Replit
2. **Configure your API keys** in Replit Secrets (see [Environment Setup](#environment-setup))
3. **Click Run** - The system will start automatically!

The system includes a complete AI-powered content workflow with knowledge graph visualization and multi-tenant architecture.

## 🚀 Features

- **Multi-Agent AI System**: 5 specialized AI agents for content analysis, SEO research, content generation, graph management, and quality assurance
- **✅ Modular Architecture**: Clean, maintainable code structure with focused modules under 500 lines each
- **Knowledge Graph Storage**: Neo4j-powered semantic content relationships
- **Vector Search**: Qdrant integration for semantic similarity search
- **Multi-Tenant Architecture**: Supabase-based user and tenant management
- **Content Ingestion**: Support for multiple formats (MD, PDF, DOCX, HTML, CSV)
- **SEO Analysis**: Automated keyword extraction, content optimization, and quality scoring
- **Real-time Processing**: OpenAI embeddings with intelligent caching
- **Search Integration**: SearXNG for trend analysis and competitor research
- **✅ Comprehensive Testing**: Full test coverage for all modular components
- **✅ Template Components**: Reusable HTML components with theme switching

## 🏗️ Architecture

### Core Components

- **AI Agents**: Pydantic AI-powered agents for specialized content tasks
- **Knowledge Graph**: Neo4j for storing content relationships and semantic data
- **Vector Database**: Qdrant for embedding storage and similarity search
- **User Management**: Supabase for authentication and multi-tenancy
- **Search Engine**: SearXNG for external search and trend analysis
- **Caching Layer**: Redis for performance optimization

### AI Agents (Modular Architecture)

1. **Content Analysis Agent**: Topic extraction, keyword analysis, quality assessment
2. **SEO Research Agent**: Keyword research, trend analysis, competitor insights
3. **Content Generation Agent** (✅ **Modularized**): 
   - Modular structure in `src/agents/content_generation/`
   - Separate modules for tools, prompts, RAG functionality
   - Brand-aligned content creation and optimization
4. **Competitor Analysis Agent** (✅ **Modularized**):
   - Modular structure in `src/agents/competitor_analysis/`
   - Content analysis, keyword analysis, and workflow modules
   - Comprehensive competitor intelligence and gap analysis
5. **Graph Management Agent**: Neo4j operations, relationship analysis
6. **Quality Assurance Agent**: Content validation, scoring, recommendations

## 🚀 Quick Start

### Environment Setup

#### **For Replit Deployment (Recommended)**

Configure these secrets in your Replit project's **Secrets tab**:

**Essential (Required):**
- `OPENAI_API_KEY`: Your OpenAI API key for AI content generation
- `OPENAI_MODEL`: `gpt-4o-mini` (recommended for cost efficiency)
- `OPENAI_EMBEDDING_MODEL`: `text-embedding-ada-002`

**Database Services (Optional - for full functionality):**
- `NEO4J_URI`: Your Neo4j AuraDB connection string
- `NEO4J_USERNAME`: `neo4j`
- `NEO4J_PASSWORD`: Your Neo4j password
- `QDRANT_URL`: Your Qdrant Cloud URL
- `QDRANT_API_KEY`: Your Qdrant API key
- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_ANON_KEY`: Your Supabase anonymous key
- `SUPABASE_SERVICE_KEY`: Your Supabase service role key

**Optional Services:**
- `LANGFUSE_PUBLIC_KEY` & `LANGFUSE_SECRET_KEY`: For LLM observability
- `GOOGLE_CLIENT_ID` & `GOOGLE_CLIENT_SECRET`: For Google OAuth

#### **For Local Development**

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- OpenAI API key
- Git

### 1. Clone and Setup

```bash
git clone <repository-url>
cd Context-Engineering-Intro
```

### 2. Environment Configuration

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Quick Start with Docker

```bash
# Start essential services
./start_minimal.sh

# Or start all services
docker-compose up -d
```

### 4. Python Environment

```bash
# Using conda (recommended)
conda create -n seo_content python=3.11
conda activate seo_content

# Or using venv
python -m venv venv_linux
source venv_linux/bin/activate  # Linux/Mac
# venv_linux\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 5. Test the System

```bash
# Run basic tests
python test_system.py --test-level basic

# Run demos
python demo_simple.py
python process_your_content.py your_file.md
```

## 📖 Usage Examples

### Content Processing Pipeline

```python
from src.services.content_ingestion import ContentIngestionService
from src.services.embedding_service import EmbeddingService

# Ingest content
ingestion_service = ContentIngestionService()
content_item = await ingestion_service.ingest_file(
    "path/to/content.md", 
    tenant_id="your-tenant", 
    author_id="user-id"
)

# Generate embeddings
embedding_service = EmbeddingService()
embedding = await embedding_service.generate_embedding(content_item.content)
```

### AI Agent Usage

```python
from src.agents.content_analysis_agent import ContentAnalysisAgent
from models.content_models import ContentAnalysisRequest

# Create agent
agent = await create_content_analysis_agent(tenant_id="your-tenant")

# Analyze content
request = ContentAnalysisRequest(
    content="Your content here...",
    include_seo_analysis=True,
    include_topic_extraction=True
)

result = await agent.analyze_content(request)
print(f"SEO Score: {result.seo_analysis.content_quality_score}")
```

## 🐳 Docker Services

### Minimal Setup (Fastest)

```bash
docker-compose -f docker-compose.minimal.yml up -d
```

Services included:
- Neo4j (Graph Database)
- Qdrant (Vector Database)  
- SearXNG (Search Engine)
- Redis (Caching)
- PostgreSQL (Simple DB)

### Full Setup (Production)

```bash
docker-compose up -d
```

Additional services:
- Supabase (Complete auth platform)
- Grafana (Monitoring)
- Prometheus (Metrics)

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Core AI Services
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=gpt-4o-mini

# Database URLs (Docker defaults)
NEO4J_URI=bolt://localhost:7687
QDRANT_URL=http://localhost:6333
SUPABASE_URL=postgresql://postgres:postgres@localhost:5432/seo_content

# Optional Services
LANGFUSE_PUBLIC_KEY=your-key-here  # LLM observability
SEARXNG_URL=http://localhost:8080  # Search engine
```

### Service URLs (Default Docker Setup)

- Neo4j Browser: http://localhost:7474 (neo4j/password)
- SearXNG: http://localhost:8080
- Qdrant Dashboard: http://localhost:6333/dashboard
- Grafana: http://localhost:3000 (admin/admin)

## 🧪 Testing

### Test Levels

```bash
# Basic functionality
python test_system.py --test-level basic

# Full system test
python test_system.py --test-level full
```

### Demo Scripts

```bash
# Core services demo
python demo_simple.py

# Process your own content
python process_your_content.py path/to/your/file.md
```

## 📊 Monitoring & Observability

### Langfuse Integration

Monitor AI agent performance and costs:

1. Set up Langfuse account at https://cloud.langfuse.com
2. Add keys to `.env`
3. View traces and metrics in Langfuse dashboard

### Metrics

Track key metrics:
- Content processing throughput
- AI agent response times
- Embedding generation costs
- SEO score improvements

## 🛠️ Development

### Project Structure (Clean & Modular Architecture)

```
├── src/                           # ✅ Consolidated source code
│   ├── agents/                    # ✅ Modularized AI agent implementations
│   │   ├── content_generation/    # Content generation modules  
│   │   ├── competitor_analysis/   # Competitor analysis modules
│   │   └── *.py                   # Other specialized agents
│   ├── services/                  # ✅ Consolidated core services
│   │   ├── embedding_service.py   # AI embeddings with caching
│   │   ├── content_ingestion.py   # Multi-format content processing
│   │   ├── analytics_service.py   # Performance analytics
│   │   └── *.py                   # Additional services
│   ├── config/                    # Advanced configuration
│   │   └── seo_rules/             # ✅ Modularized SEO rules engine
│   ├── database/                  # Database service clients
│   └── ingestion/                 # Content ingestion modules
├── agents/                        # ⚠️ Legacy compatibility layer
├── services/                      # ⚠️ Legacy compatibility layer  
├── config/                        # Configuration management
├── database/                      # Database clients (Neo4j, Qdrant, Supabase)
├── models/                        # Pydantic data models
├── web/                           # FastAPI web interface
│   ├── api/                       # ✅ Expanded API routes
│   └── templates/                 # ✅ Componentized HTML templates
│       └── components/            # Reusable template components
├── tests/                         # ✅ Comprehensive test suites
├── cli/                           # Command-line interface
├── scripts/                       # ✅ Organized development scripts
│   ├── test/                      # Test and validation scripts
│   └── dev/                       # Development utility scripts
├── data/                          # ✅ Runtime data (gitignored)
│   ├── dev-databases/             # Development SQLite files
│   ├── test-results/              # Test output and results
│   └── dev-configs/               # Development configurations
├── utils/                         # ✅ Multi-tenant utilities
└── examples/                      # Example content and docs
```

**🔄 Migration Notes**: Legacy `agents/` and `services/` directories provide backward compatibility with deprecation warnings. New code should import from `src/agents/` and `src/services/`.

### Code Quality

```bash
# Format code
black .

# Lint code  
ruff check .

# Type checking
mypy .

# Run tests
pytest
```

## 🚀 Production Deployment

### Docker Production

```bash
# Build production image
docker build -t seo-content-system .

# Deploy with docker-compose
docker-compose -f docker-compose.prod.yml up -d
```

### Environment Setup

1. Set production environment variables
2. Configure SSL certificates
3. Set up monitoring and logging
4. Configure backup strategies

## 📚 Documentation

- **[Modular Architecture Guide](MODULAR_ARCHITECTURE.md)** - ✅ **NEW**: Comprehensive guide to the modular system architecture
- [Planning Document](PLANNING.md) - Project architecture and implementation status
- [Development Guide](CLAUDE.md) - Development standards and conventions
- [Task Management](TASK.md) - Current development tasks and priorities
- [Examples](examples/) - Sample content and use cases

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Pydantic AI](https://ai.pydantic.dev/)
- Powered by [OpenAI](https://openai.com/)
- Uses [Neo4j](https://neo4j.com/) for knowledge graphs
- Vector search by [Qdrant](https://qdrant.tech/)
- Search integration via [SearXNG](https://searxng.github.io/searxng/)

---

🎯 **Ready for production SEO content analysis and optimization at scale!**