"""
Content management CLI commands for the SEO Content Knowledge Graph System.

This module provides CLI commands for content ingestion, generation, and management.
"""

import asyncio
import click
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.status import Status

from src.services.content_ingestion import ContentIngestionService
from src.agents.content_analysis_agent import ContentAnalysisAgent

def create_content_analysis_agent(*args, **kwargs):
    return ContentAnalysisAgent()

from models.content_models import (
    ContentAnalysisRequest, 
    ContentGenerationRequest, 
    ContentType, 
    ContentLanguage
)

console = Console()

def content_commands(cli_context):
    """Return content management commands."""
    
    @click.group()
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
                raise
        
        asyncio.run(_ingest())
    
    @content.command()
    @click.argument("topic")
    @click.option("--type", "-t", default="blog_post", 
                  type=click.Choice(["blog_post", "article", "landing_page", "product_description"]), 
                  help="Content type")
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
                    content_type=ContentType(type.upper()),
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
                raise
        
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
                raise
        
        asyncio.run(_list())
    
    @content.command()
    @click.argument("content_id")
    def analyze(content_id):
        """Analyze existing content."""
        
        async def _analyze():
            try:
                with Status("Analyzing content...", console=console):
                    agent = await create_content_analysis_agent(cli_context.tenant_id)
                    
                    analysis_request = ContentAnalysisRequest(
                        content_id=content_id,
                        tenant_id=cli_context.tenant_id,
                        analysis_type="comprehensive"
                    )
                    
                    result = await agent.analyze_content(analysis_request)
                
                # Display results
                if cli_context.format == "json":
                    console.print_json(result.dict())
                else:
                    console.print(Panel(f"Analysis for Content ID: {content_id}", style="blue"))
                    
                    # Content metrics
                    table = Table(title="Content Analysis Results")
                    table.add_column("Metric", style="cyan")
                    table.add_column("Value", style="yellow")
                    table.add_column("Status", style="green")
                    
                    table.add_row("Word Count", str(result.word_count), "✅ Good" if result.word_count > 300 else "⚠️ Too Short")
                    table.add_row("Reading Time", f"{result.reading_time_minutes} minutes", "")
                    table.add_row("SEO Score", f"{result.seo_score}/100", "✅ Good" if result.seo_score > 70 else "⚠️ Needs Work")
                    table.add_row("Readability", f"{result.readability_score}/100", "✅ Good" if result.readability_score > 60 else "⚠️ Difficult")
                    
                    console.print(table)
                    
                    # Show recommendations if available
                    if hasattr(result, 'recommendations') and result.recommendations:
                        console.print("\n💡 Recommendations:")
                        for i, rec in enumerate(result.recommendations, 1):
                            console.print(f"{i}. {rec}")
                    
            except Exception as e:
                console.print(f"❌ Content analysis failed: {e}", style="red")
                raise
        
        asyncio.run(_analyze())
    
    return content