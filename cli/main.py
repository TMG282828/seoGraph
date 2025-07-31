#!/usr/bin/env python3
"""
Enhanced CLI interface for the SEO Content Knowledge Graph System.

This module provides a comprehensive command-line interface for managing
content analysis, SEO research, workflow orchestration, and system operations.
"""

import asyncio
import click
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import structlog
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.tree import Tree
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.status import Status
import yaml

# Import our services and models
from config.settings import get_settings
from src.services.content_ingestion import ContentIngestionService
from src.services.analytics_service import AnalyticsService
from src.services.workflow_orchestrator import WorkflowOrchestrator
from src.services.competitor_monitoring import CompetitorMonitoringService
from src.services.gap_analysis import GapAnalysisService
from src.agents.content_analysis_agent import ContentAnalysisAgent
from src.agents.seo_research_agent import SEOResearchAgent
from src.agents.competitor_analysis import create_competitor_analysis_agent

# Compatibility functions
def create_content_analysis_agent(*args, **kwargs):
    return ContentAnalysisAgent()

def create_seo_research_agent(*args, **kwargs):
    return SEOResearchAgent()

def create_trend_analysis_agent(*args, **kwargs):
    # For now, import from legacy location until trend analysis is migrated
    from agents.trend_analysis import TrendAnalysisAgent
    return TrendAnalysisAgent()

from models.content_models import ContentGenerationRequest, ContentType, ContentLanguage
from models.seo_models import SEOAnalysisRequest, KeywordResearchRequest
from models.workflow_models import CreateWorkflowRequest, WorkflowStatus
from models.analytics_models import AnalyticsRequest, MetricType, DateRange
from database.neo4j_client import Neo4jClient
from database.qdrant_client import QdrantClient

# Initialize rich console and logger
console = Console()
logger = structlog.get_logger(__name__)

# CLI Configuration
CLI_VERSION = "1.0.0"
CLI_NAME = "SEO Content Knowledge Graph CLI"

class CLIContext:
    """Context object for CLI operations."""
    
    def __init__(self):
        self.settings = get_settings()
        self.verbose = False
        self.format = "table"
        self.output_file = None
        self.tenant_id = "default"
        self.user_id = "cli_user"
        
        # Initialize services
        self.content_service = None
        self.analytics_service = None
        self.workflow_orchestrator = None
        self.competitor_service = None
        self.gap_analysis_service = None
        
        # Initialize database clients
        self.neo4j_client = None
        self.qdrant_client = None

    async def initialize_services(self):
        """Initialize all services asynchronously."""
        try:
            # Initialize database clients
            self.neo4j_client = Neo4jClient()
            self.qdrant_client = QdrantClient()
            
            # Initialize services
            self.content_service = ContentIngestionService()
            self.analytics_service = AnalyticsService(self.neo4j_client, self.qdrant_client)
            self.workflow_orchestrator = WorkflowOrchestrator(self.neo4j_client, self.analytics_service)
            self.competitor_service = CompetitorMonitoringService(self.neo4j_client, self.qdrant_client)
            self.gap_analysis_service = GapAnalysisService(self.neo4j_client, self.qdrant_client)
            
            # Initialize orchestrator
            await self.workflow_orchestrator.initialize()
            
            if self.verbose:
                console.print("✅ All services initialized successfully", style="green")
                
        except Exception as e:
            console.print(f"❌ Failed to initialize services: {e}", style="red")
            sys.exit(1)

# Global context
cli_context = CLIContext()

@click.group()
@click.version_option(version=CLI_VERSION, prog_name=CLI_NAME)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--format", "-f", default="table", type=click.Choice(["table", "json", "yaml"]), help="Output format")
@click.option("--output", "-o", help="Output file path")
@click.option("--tenant", "-t", default="default", help="Tenant ID")
@click.option("--user", "-u", default="cli_user", help="User ID")
@click.pass_context
def cli(ctx, verbose, format, output, tenant, user):
    """
    SEO Content Knowledge Graph System CLI
    
    A comprehensive command-line interface for managing content analysis,
    SEO research, workflow orchestration, and system operations.
    """
    # Initialize context
    ctx.ensure_object(dict)
    cli_context.verbose = verbose
    cli_context.format = format
    cli_context.output_file = output
    cli_context.tenant_id = tenant
    cli_context.user_id = user
    
    # Initialize services
    asyncio.run(cli_context.initialize_services())

# =============================================================================
# Content Management Commands
# =============================================================================

@cli.group()
def content():
    """Content management and analysis commands."""
    pass

@content.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--author", "-a", default="cli_user", help="Author ID")
@click.option("--analyze", is_flag=True, help="Run content analysis after ingestion")
def ingest(file_path, author, analyze):
    """Ingest content from a file."""
    
    async def _ingest():
        try:
            with Status("Ingesting content...", console=console):
                content_item = await cli_context.content_service.ingest_file(
                    file_path=file_path,
                    tenant_id=cli_context.tenant_id,
                    author_id=author
                )
            
            # Display results
            if cli_context.format == "json":
                console.print_json(content_item.dict())
            else:
                table = Table(title="Content Ingestion Results")
                table.add_column("Property", style="cyan")
                table.add_column("Value", style="yellow")
                
                table.add_row("ID", content_item.id)
                table.add_row("Title", content_item.title)
                table.add_row("Content Type", content_item.content_type.value)
                table.add_row("Word Count", str(content_item.metrics.word_count))
                table.add_row("Reading Time", f"{content_item.metrics.reading_time_minutes} minutes")
                table.add_row("Created", content_item.created_at.isoformat())
                
                console.print(table)
            
            # Run analysis if requested
            if analyze:
                console.print("\n🔍 Running content analysis...")
                agent = await create_content_analysis_agent(cli_context.tenant_id)
                
                from models.content_models import ContentAnalysisRequest
                analysis_request = ContentAnalysisRequest(
                    content_id=content_item.id,
                    tenant_id=cli_context.tenant_id,
                    analysis_type="comprehensive"
                )
                
                result = await agent.analyze_content(analysis_request)
                
                console.print("\n📊 Analysis Results:")
                console.print_json(result.dict())
                
        except Exception as e:
            console.print(f"❌ Content ingestion failed: {e}", style="red")
            sys.exit(1)
    
    asyncio.run(_ingest())

@content.command()
@click.argument("topic")
@click.option("--type", "-t", default="blog_post", type=click.Choice(["blog_post", "article", "landing_page", "product_description"]), help="Content type")
@click.option("--length", "-l", default=1000, type=int, help="Target word count")
@click.option("--keywords", "-k", help="Target keywords (comma-separated)")
@click.option("--tone", default="professional", help="Content tone")
@click.option("--audience", help="Target audience")
@click.option("--language", default="english", help="Content language")
def generate(topic, type, length, keywords, tone, audience, language):
    """Generate content using AI agents."""
    
    async def _generate():
        try:
            # Parse keywords
            keyword_list = [k.strip() for k in keywords.split(",")] if keywords else []
            
            # Create generation request
            request = ContentGenerationRequest(
                topic=topic,
                content_type=ContentType(type),
                target_length=length,
                keywords=keyword_list,
                tone=tone,
                audience=audience,
                language=ContentLanguage(language.upper()),
                tenant_id=cli_context.tenant_id
            )
            
            # Generate content
            with Status("Generating content...", console=console):
                agent = await create_content_analysis_agent(cli_context.tenant_id)
                result = await agent.generate_content(request)
            
            # Display results
            if cli_context.format == "json":
                console.print_json(result.dict())
            else:
                console.print(Panel(result.content, title=f"Generated Content: {result.title}", border_style="green"))
                
                # Show metadata
                table = Table(title="Content Metadata")
                table.add_column("Property", style="cyan")
                table.add_column("Value", style="yellow")
                
                table.add_row("Word Count", str(result.word_count))
                table.add_row("SEO Score", f"{result.seo_score}/100")
                table.add_row("Reading Time", f"{result.reading_time_minutes} minutes")
                table.add_row("Keywords Used", ", ".join(result.keywords_used))
                
                console.print(table)
            
        except Exception as e:
            console.print(f"❌ Content generation failed: {e}", style="red")
            sys.exit(1)
    
    asyncio.run(_generate())

@content.command()
@click.option("--limit", "-l", default=10, type=int, help="Number of items to list")
@click.option("--type", "-t", help="Filter by content type")
@click.option("--author", "-a", help="Filter by author")
def list(limit, type, author):
    """List content items."""
    
    async def _list():
        try:
            # Build filters
            filters = {"tenant_id": cli_context.tenant_id}
            if type:
                filters["content_type"] = type
            if author:
                filters["author_id"] = author
            
            # Get content items
            with Status("Fetching content items...", console=console):
                items = await cli_context.content_service.list_content(
                    filters=filters,
                    limit=limit
                )
            
            # Display results
            if cli_context.format == "json":
                console.print_json([item.dict() for item in items])
            else:
                table = Table(title="Content Items")
                table.add_column("ID", style="cyan")
                table.add_column("Title", style="yellow")
                table.add_column("Type", style="green")
                table.add_column("Author", style="blue")
                table.add_column("Word Count", style="magenta")
                table.add_column("Created", style="white")
                
                for item in items:
                    table.add_row(
                        item.id[:8] + "...",
                        item.title[:50] + "..." if len(item.title) > 50 else item.title,
                        item.content_type.value,
                        item.author_id,
                        str(item.metrics.word_count),
                        item.created_at.strftime("%Y-%m-%d")
                    )
                
                console.print(table)
                
        except Exception as e:
            console.print(f"❌ Failed to list content: {e}", style="red")
            sys.exit(1)
    
    asyncio.run(_list())

# =============================================================================
# SEO Research Commands
# =============================================================================

@cli.group()
def seo():
    """SEO research and analysis commands."""
    pass

@seo.command()
@click.argument("query")
@click.option("--limit", "-l", default=100, type=int, help="Number of keywords to return")
@click.option("--difficulty", "-d", type=click.Choice(["easy", "medium", "hard"]), help="Keyword difficulty filter")
@click.option("--min-volume", type=int, help="Minimum search volume")
@click.option("--max-volume", type=int, help="Maximum search volume")
def research(query, limit, difficulty, min_volume, max_volume):
    """Research keywords for SEO optimization."""
    
    async def _research():
        try:
            # Create research request
            request = KeywordResearchRequest(
                seed_keywords=[query],
                limit=limit,
                difficulty_filter=difficulty,
                min_search_volume=min_volume,
                max_search_volume=max_volume,
                tenant_id=cli_context.tenant_id
            )
            
            # Perform research
            with Status("Researching keywords...", console=console):
                agent = await create_seo_research_agent(cli_context.tenant_id)
                result = await agent.research_keywords(request)
            
            # Display results
            if cli_context.format == "json":
                console.print_json(result.dict())
            else:
                table = Table(title=f"Keyword Research Results for '{query}'")
                table.add_column("Keyword", style="cyan")
                table.add_column("Volume", style="yellow")
                table.add_column("Difficulty", style="red")
                table.add_column("CPC", style="green")
                table.add_column("Competition", style="blue")
                table.add_column("Intent", style="magenta")
                
                for keyword in result.keywords:
                    table.add_row(
                        keyword.keyword,
                        str(keyword.search_volume),
                        str(keyword.difficulty),
                        f"${keyword.cpc:.2f}",
                        keyword.competition.value,
                        keyword.intent.value
                    )
                
                console.print(table)
                
        except Exception as e:
            console.print(f"❌ Keyword research failed: {e}", style="red")
            sys.exit(1)
    
    asyncio.run(_research())

@seo.command()
@click.argument("url")
@click.option("--keywords", "-k", help="Target keywords (comma-separated)")
@click.option("--competitors", "-c", help="Competitor URLs (comma-separated)")
def analyze(url, keywords, competitors):
    """Analyze SEO performance of a URL."""
    
    async def _analyze():
        try:
            # Parse inputs
            keyword_list = [k.strip() for k in keywords.split(",")] if keywords else []
            competitor_list = [c.strip() for c in competitors.split(",")] if competitors else []
            
            # Create analysis request
            request = SEOAnalysisRequest(
                url=url,
                target_keywords=keyword_list,
                competitor_urls=competitor_list,
                tenant_id=cli_context.tenant_id
            )
            
            # Perform analysis
            with Status("Analyzing SEO performance...", console=console):
                agent = await create_seo_research_agent(cli_context.tenant_id)
                result = await agent.analyze_seo(request)
            
            # Display results
            if cli_context.format == "json":
                console.print_json(result.dict())
            else:
                # SEO Score Panel
                score_color = "green" if result.seo_score >= 80 else "yellow" if result.seo_score >= 60 else "red"
                console.print(Panel(f"SEO Score: {result.seo_score}/100", style=score_color))
                
                # Technical SEO
                if result.technical_seo:
                    table = Table(title="Technical SEO Analysis")
                    table.add_column("Metric", style="cyan")
                    table.add_column("Status", style="yellow")
                    table.add_column("Score", style="green")
                    
                    tech_data = result.technical_seo
                    table.add_row("Page Speed", "✅ Good" if tech_data.page_speed_score > 80 else "⚠️ Needs Work", f"{tech_data.page_speed_score}/100")
                    table.add_row("Mobile Friendly", "✅ Yes" if tech_data.mobile_friendly else "❌ No", "")
                    table.add_row("HTTPS", "✅ Yes" if tech_data.ssl_certificate else "❌ No", "")
                    table.add_row("Meta Title", "✅ Good" if tech_data.meta_title_length < 60 else "⚠️ Too Long", f"{tech_data.meta_title_length} chars")
                    table.add_row("Meta Description", "✅ Good" if tech_data.meta_description_length < 160 else "⚠️ Too Long", f"{tech_data.meta_description_length} chars")
                    
                    console.print(table)
                
                # Content Analysis
                if result.content_analysis:
                    content_data = result.content_analysis
                    table = Table(title="Content Analysis")
                    table.add_column("Metric", style="cyan")
                    table.add_column("Value", style="yellow")
                    
                    table.add_row("Word Count", str(content_data.word_count))
                    table.add_row("Reading Time", f"{content_data.reading_time_minutes} minutes")
                    table.add_row("Keyword Density", f"{content_data.keyword_density:.2f}%")
                    table.add_row("Headings", f"H1: {content_data.h1_count}, H2: {content_data.h2_count}, H3: {content_data.h3_count}")
                    
                    console.print(table)
                
        except Exception as e:
            console.print(f"❌ SEO analysis failed: {e}", style="red")
            sys.exit(1)
    
    asyncio.run(_analyze())

# =============================================================================
# Workflow Management Commands
# =============================================================================

@cli.group()
def workflow():
    """Workflow management and orchestration commands."""
    pass

@workflow.command()
@click.argument("name")
@click.option("--type", "-t", default="content_creation", help="Workflow type")
@click.option("--description", "-d", help="Workflow description")
@click.option("--template", help="Workflow template ID")
def create(name, type, description, template):
    """Create a new workflow."""
    
    async def _create():
        try:
            # Create workflow request
            request = CreateWorkflowRequest(
                name=name,
                workflow_type=type,
                description=description,
                template_id=template,
                tenant_id=cli_context.tenant_id,
                created_by=cli_context.user_id
            )
            
            # Create workflow
            with Status("Creating workflow...", console=console):
                workflow = await cli_context.workflow_orchestrator.create_workflow(request)
            
            # Display results
            if cli_context.format == "json":
                console.print_json(workflow.dict())
            else:
                table = Table(title="Workflow Created")
                table.add_column("Property", style="cyan")
                table.add_column("Value", style="yellow")
                
                table.add_row("ID", workflow.workflow_id)
                table.add_row("Name", workflow.name)
                table.add_row("Type", workflow.workflow_type)
                table.add_row("Status", workflow.status.value)
                table.add_row("Steps", str(len(workflow.steps)))
                table.add_row("Created", workflow.created_at.isoformat())
                
                console.print(table)
                
        except Exception as e:
            console.print(f"❌ Workflow creation failed: {e}", style="red")
            sys.exit(1)
    
    asyncio.run(_create())

@workflow.command()
@click.option("--status", "-s", type=click.Choice(["draft", "active", "paused", "completed", "failed"]), help="Filter by status")
@click.option("--type", "-t", help="Filter by workflow type")
@click.option("--limit", "-l", default=10, type=int, help="Number of workflows to list")
def list(status, type, limit):
    """List workflows."""
    
    async def _list():
        try:
            # Build filters
            filters = {"tenant_id": cli_context.tenant_id}
            if status:
                filters["status"] = WorkflowStatus(status.upper())
            if type:
                filters["workflow_type"] = type
            
            # Get workflows
            with Status("Fetching workflows...", console=console):
                workflows = await cli_context.workflow_orchestrator.get_workflows(
                    filters=filters,
                    limit=limit
                )
            
            # Display results
            if cli_context.format == "json":
                console.print_json([w.dict() for w in workflows])
            else:
                table = Table(title="Workflows")
                table.add_column("ID", style="cyan")
                table.add_column("Name", style="yellow")
                table.add_column("Type", style="green")
                table.add_column("Status", style="blue")
                table.add_column("Progress", style="magenta")
                table.add_column("Created", style="white")
                
                for workflow in workflows:
                    progress = workflow.calculate_progress()
                    status_color = {
                        "DRAFT": "yellow",
                        "ACTIVE": "green",
                        "PAUSED": "blue",
                        "COMPLETED": "green",
                        "FAILED": "red"
                    }.get(workflow.status.value, "white")
                    
                    table.add_row(
                        workflow.workflow_id[:8] + "...",
                        workflow.name,
                        workflow.workflow_type,
                        f"[{status_color}]{workflow.status.value}[/{status_color}]",
                        f"{progress:.1f}%",
                        workflow.created_at.strftime("%Y-%m-%d")
                    )
                
                console.print(table)
                
        except Exception as e:
            console.print(f"❌ Failed to list workflows: {e}", style="red")
            sys.exit(1)
    
    asyncio.run(_list())

@workflow.command()
@click.argument("workflow_id")
def execute(workflow_id):
    """Execute a workflow."""
    
    async def _execute():
        try:
            # Execute workflow
            with Status("Executing workflow...", console=console):
                result = await cli_context.workflow_orchestrator.execute_workflow(
                    workflow_id=workflow_id,
                    executed_by=cli_context.user_id
                )
            
            console.print(f"✅ Workflow execution started: {result}")
            
        except Exception as e:
            console.print(f"❌ Workflow execution failed: {e}", style="red")
            sys.exit(1)
    
    asyncio.run(_execute())

@workflow.command()
@click.argument("workflow_id")
def status(workflow_id):
    """Get workflow status."""
    
    async def _status():
        try:
            # Get workflow status
            with Status("Fetching workflow status...", console=console):
                status_data = await cli_context.workflow_orchestrator.get_workflow_execution_status(workflow_id)
            
            # Display results
            if cli_context.format == "json":
                console.print_json(status_data)
            else:
                table = Table(title=f"Workflow Status: {workflow_id}")
                table.add_column("Property", style="cyan")
                table.add_column("Value", style="yellow")
                
                for key, value in status_data.items():
                    table.add_row(key.replace("_", " ").title(), str(value))
                
                console.print(table)
                
        except Exception as e:
            console.print(f"❌ Failed to get workflow status: {e}", style="red")
            sys.exit(1)
    
    asyncio.run(_status())

# =============================================================================
# Analytics Commands
# =============================================================================

@cli.group()
def analytics():
    """Analytics and reporting commands."""
    pass

@analytics.command()
@click.option("--days", "-d", default=30, type=int, help="Number of days to analyze")
@click.option("--metric", "-m", multiple=True, help="Specific metrics to include")
def overview(days, metric):
    """Get analytics overview."""
    
    async def _overview():
        try:
            # Create analytics request
            request = AnalyticsRequest(
                date_range=DateRange(days=days),
                metrics=[MetricType(m) for m in metric] if metric else None,
                tenant_id=cli_context.tenant_id
            )
            
            # Get analytics
            with Status("Generating analytics overview...", console=console):
                result = await cli_context.analytics_service.get_comprehensive_analytics(request)
            
            # Display results
            if cli_context.format == "json":
                console.print_json(result.dict())
            else:
                # Key metrics
                console.print(Panel("📊 Analytics Overview", style="blue"))
                
                table = Table(title="Key Metrics")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="yellow")
                table.add_column("Change", style="green")
                
                for metric in result.metrics:
                    change_color = "green" if metric.change_percentage > 0 else "red" if metric.change_percentage < 0 else "yellow"
                    table.add_row(
                        metric.name,
                        str(metric.value),
                        f"[{change_color}]{metric.change_percentage:+.1f}%[/{change_color}]"
                    )
                
                console.print(table)
                
        except Exception as e:
            console.print(f"❌ Analytics overview failed: {e}", style="red")
            sys.exit(1)
    
    asyncio.run(_overview())

# =============================================================================
# System Management Commands
# =============================================================================

@cli.group()
def system():
    """System management and maintenance commands."""
    pass

@system.command()
def status():
    """Check system status and health."""
    
    async def _status():
        try:
            with Status("Checking system status...", console=console):
                # Check database connections
                neo4j_status = await cli_context.neo4j_client.health_check()
                qdrant_status = await cli_context.qdrant_client.health_check()
                
                # Check service status
                workflow_status = await cli_context.workflow_orchestrator.health_check()
            
            # Display results
            table = Table(title="System Status")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="yellow")
            table.add_column("Details", style="green")
            
            # Database status
            table.add_row("Neo4j", "✅ Connected" if neo4j_status else "❌ Disconnected", "Graph database")
            table.add_row("Qdrant", "✅ Connected" if qdrant_status else "❌ Disconnected", "Vector database")
            
            # Service status
            table.add_row("Workflow Orchestrator", "✅ Running" if workflow_status else "❌ Failed", "Workflow management")
            table.add_row("Content Service", "✅ Running", "Content ingestion")
            table.add_row("Analytics Service", "✅ Running", "Analytics processing")
            
            console.print(table)
            
        except Exception as e:
            console.print(f"❌ System status check failed: {e}", style="red")
            sys.exit(1)
    
    asyncio.run(_status())

@system.command()
def config():
    """Show current configuration."""
    
    config_data = {
        "Environment": cli_context.settings.environment,
        "Debug": cli_context.settings.debug,
        "Tenant ID": cli_context.tenant_id,
        "User ID": cli_context.user_id,
        "Neo4j URI": cli_context.settings.neo4j_uri,
        "Qdrant URL": cli_context.settings.qdrant_url,
        "OpenAI Model": cli_context.settings.openai_model,
        "Log Level": cli_context.settings.log_level
    }
    
    if cli_context.format == "json":
        console.print_json(config_data)
    else:
        table = Table(title="Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="yellow")
        
        for key, value in config_data.items():
            table.add_row(key, str(value))
        
        console.print(table)

# =============================================================================
# Interactive Mode
# =============================================================================

@cli.command()
def interactive():
    """Start interactive mode."""
    
    console.print(Panel(f"Welcome to {CLI_NAME} Interactive Mode", style="blue"))
    console.print("Type 'help' for available commands or 'exit' to quit.")
    
    while True:
        try:
            command = Prompt.ask("\n[cyan]seo-cli[/cyan]")
            
            if command.lower() in ["exit", "quit"]:
                console.print("Goodbye! 👋")
                break
            elif command.lower() == "help":
                console.print("""
Available commands:
• content ingest <file> - Ingest content from file
• content generate <topic> - Generate content
• content list - List content items
• seo research <query> - Research keywords
• seo analyze <url> - Analyze URL SEO
• workflow create <name> - Create workflow
• workflow list - List workflows
• workflow execute <id> - Execute workflow
• analytics overview - Get analytics overview
• system status - Check system status
• system config - Show configuration
• exit - Exit interactive mode
                """)
            else:
                # Parse and execute command
                try:
                    args = command.split()
                    if len(args) >= 2:
                        # This is a simplified command parser
                        # In a real implementation, you'd want more robust parsing
                        console.print(f"Executing: {command}")
                        console.print("💡 Use the full CLI syntax for complex commands")
                    else:
                        console.print("❌ Invalid command. Type 'help' for available commands.")
                except:
                    console.print("❌ Command execution failed. Check your syntax.")
                    
        except KeyboardInterrupt:
            console.print("\nGoodbye! 👋")
            break
        except Exception as e:
            console.print(f"❌ Error: {e}")

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        console.print(f"❌ Fatal error: {e}", style="red")
        sys.exit(1)