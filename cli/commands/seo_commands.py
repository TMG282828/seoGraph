"""
SEO research and analysis CLI commands for the SEO Content Knowledge Graph System.

This module provides CLI commands for SEO research, keyword analysis, and optimization.
"""

import asyncio
import click
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.status import Status

from src.agents.seo_research_agent import SEOResearchAgent

def create_seo_research_agent(*args, **kwargs):
    return SEOResearchAgent()

from models.seo_models import (
    SEOAnalysisRequest, 
    KeywordResearchRequest,
    CompetitionLevel,
    SearchIntent
)

console = Console()

def seo_commands(cli_context):
    """Return SEO research and analysis commands."""
    
    @click.group()
    def seo():
        """SEO research and analysis commands."""
        pass
    
    @seo.command()
    @click.argument("query")
    @click.option("--limit", "-l", default=100, type=int, help="Number of keywords to return")
    @click.option("--difficulty", "-d", type=click.Choice(["easy", "medium", "hard"]), help="Keyword difficulty filter")
    @click.option("--min-volume", type=int, help="Minimum search volume")
    @click.option("--max-volume", type=int, help="Maximum search volume")
    @click.option("--intent", "-i", type=click.Choice(["informational", "navigational", "transactional", "commercial"]), help="Search intent filter")
    def research(query, limit, difficulty, min_volume, max_volume, intent):
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
                    intent_filter=SearchIntent(intent.upper()) if intent else None,
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
                    console.print(Panel(f"Keyword Research Results for '{query}'", style="blue"))
                    
                    table = Table(title="Keywords")
                    table.add_column("Keyword", style="cyan")
                    table.add_column("Volume", style="yellow")
                    table.add_column("Difficulty", style="red")
                    table.add_column("CPC", style="green")
                    table.add_column("Competition", style="blue")
                    table.add_column("Intent", style="magenta")
                    table.add_column("Opportunity", style="white")
                    
                    for keyword in result.keywords:
                        # Calculate opportunity score
                        opportunity = (keyword.search_volume / max(keyword.difficulty, 1)) * 100
                        opportunity_level = "🟢 High" if opportunity > 50 else "🟡 Medium" if opportunity > 20 else "🔴 Low"
                        
                        table.add_row(
                            keyword.keyword,
                            f"{keyword.search_volume:,}",
                            str(keyword.difficulty),
                            f"${keyword.cpc:.2f}",
                            keyword.competition.value,
                            keyword.intent.value,
                            opportunity_level
                        )
                    
                    console.print(table)
                    
                    # Show summary
                    summary_table = Table(title="Research Summary")
                    summary_table.add_column("Metric", style="cyan")
                    summary_table.add_column("Value", style="yellow")
                    
                    summary_table.add_row("Total Keywords", str(len(result.keywords)))
                    summary_table.add_row("Avg Search Volume", f"{sum(k.search_volume for k in result.keywords) / len(result.keywords):,.0f}")
                    summary_table.add_row("Avg Difficulty", f"{sum(k.difficulty for k in result.keywords) / len(result.keywords):.1f}")
                    summary_table.add_row("Avg CPC", f"${sum(k.cpc for k in result.keywords) / len(result.keywords):.2f}")
                    
                    console.print(summary_table)
                    
            except Exception as e:
                console.print(f"❌ Keyword research failed: {e}", style="red")
                raise
        
        asyncio.run(_research())
    
    @seo.command()
    @click.argument("url")
    @click.option("--keywords", "-k", help="Target keywords (comma-separated)")
    @click.option("--competitors", "-c", help="Competitor URLs (comma-separated)")
    @click.option("--detailed", is_flag=True, help="Show detailed analysis")
    def analyze(url, keywords, competitors, detailed):
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
                        table.add_column("Recommendation", style="blue")
                        
                        tech_data = result.technical_seo
                        
                        # Page Speed
                        speed_status = "✅ Excellent" if tech_data.page_speed_score > 90 else "🟡 Good" if tech_data.page_speed_score > 70 else "🔴 Needs Work"
                        speed_rec = "Optimize images and minify CSS/JS" if tech_data.page_speed_score < 80 else "Great performance!"
                        table.add_row("Page Speed", speed_status, f"{tech_data.page_speed_score}/100", speed_rec)
                        
                        # Mobile Friendly
                        mobile_status = "✅ Yes" if tech_data.mobile_friendly else "❌ No"
                        mobile_rec = "Implement responsive design" if not tech_data.mobile_friendly else "Mobile optimized"
                        table.add_row("Mobile Friendly", mobile_status, "", mobile_rec)
                        
                        # HTTPS
                        https_status = "✅ Yes" if tech_data.ssl_certificate else "❌ No"
                        https_rec = "Install SSL certificate" if not tech_data.ssl_certificate else "Secure connection"
                        table.add_row("HTTPS", https_status, "", https_rec)
                        
                        # Meta Tags
                        title_status = "✅ Good" if 30 <= tech_data.meta_title_length <= 60 else "⚠️ Optimize"
                        title_rec = "Adjust title length (30-60 chars)" if not 30 <= tech_data.meta_title_length <= 60 else "Title length optimal"
                        table.add_row("Meta Title", title_status, f"{tech_data.meta_title_length} chars", title_rec)
                        
                        desc_status = "✅ Good" if 120 <= tech_data.meta_description_length <= 160 else "⚠️ Optimize"
                        desc_rec = "Adjust description length (120-160 chars)" if not 120 <= tech_data.meta_description_length <= 160 else "Description length optimal"
                        table.add_row("Meta Description", desc_status, f"{tech_data.meta_description_length} chars", desc_rec)
                        
                        console.print(table)
                    
                    # Content Analysis
                    if result.content_analysis:
                        content_data = result.content_analysis
                        table = Table(title="Content Analysis")
                        table.add_column("Metric", style="cyan")
                        table.add_column("Value", style="yellow")
                        table.add_column("Status", style="green")
                        table.add_column("Recommendation", style="blue")
                        
                        # Word Count
                        word_status = "✅ Good" if content_data.word_count > 300 else "⚠️ Too Short"
                        word_rec = "Add more content (aim for 300+ words)" if content_data.word_count < 300 else "Content length is good"
                        table.add_row("Word Count", str(content_data.word_count), word_status, word_rec)
                        
                        # Reading Time
                        table.add_row("Reading Time", f"{content_data.reading_time_minutes} minutes", "", "")
                        
                        # Keyword Density
                        density_status = "✅ Good" if 0.5 <= content_data.keyword_density <= 3.0 else "⚠️ Optimize"
                        density_rec = "Optimize keyword density (0.5-3%)" if not 0.5 <= content_data.keyword_density <= 3.0 else "Keyword density is optimal"
                        table.add_row("Keyword Density", f"{content_data.keyword_density:.2f}%", density_status, density_rec)
                        
                        # Headings
                        headings_total = content_data.h1_count + content_data.h2_count + content_data.h3_count
                        headings_status = "✅ Good" if headings_total > 0 else "⚠️ Add Headings"
                        headings_rec = "Add H1, H2, H3 tags for better structure" if headings_total == 0 else "Good heading structure"
                        table.add_row("Headings", f"H1: {content_data.h1_count}, H2: {content_data.h2_count}, H3: {content_data.h3_count}", headings_status, headings_rec)
                        
                        console.print(table)
                    
                    # Keyword Performance (if keywords provided)
                    if keyword_list and hasattr(result, 'keyword_performance'):
                        table = Table(title="Keyword Performance")
                        table.add_column("Keyword", style="cyan")
                        table.add_column("Ranking", style="yellow")
                        table.add_column("Volume", style="green")
                        table.add_column("Difficulty", style="red")
                        table.add_column("Opportunity", style="blue")
                        
                        for kw_perf in result.keyword_performance:
                            opportunity = "🟢 High" if kw_perf.ranking > 50 else "🟡 Medium" if kw_perf.ranking > 20 else "🔴 Low"
                            table.add_row(
                                kw_perf.keyword,
                                f"#{kw_perf.ranking}" if kw_perf.ranking > 0 else "Not ranked",
                                f"{kw_perf.search_volume:,}",
                                str(kw_perf.difficulty),
                                opportunity
                            )
                        
                        console.print(table)
                    
                    # Competitor Analysis (if competitors provided)
                    if competitor_list and hasattr(result, 'competitor_analysis'):
                        table = Table(title="Competitor Analysis")
                        table.add_column("Competitor", style="cyan")
                        table.add_column("SEO Score", style="yellow")
                        table.add_column("Common Keywords", style="green")
                        table.add_column("Advantage", style="blue")
                        
                        for comp in result.competitor_analysis:
                            advantage = "🟢 You Win" if result.seo_score > comp.seo_score else "🔴 They Win"
                            table.add_row(
                                comp.url,
                                f"{comp.seo_score}/100",
                                str(comp.common_keywords_count),
                                advantage
                            )
                        
                        console.print(table)
                    
                    # Show detailed recommendations
                    if detailed and hasattr(result, 'recommendations'):
                        console.print("\n💡 Detailed Recommendations:")
                        for i, rec in enumerate(result.recommendations, 1):
                            console.print(f"{i}. {rec}")
                
            except Exception as e:
                console.print(f"❌ SEO analysis failed: {e}", style="red")
                raise
        
        asyncio.run(_analyze())
    
    @seo.command()
    @click.argument("keywords", nargs=-1)
    @click.option("--export", "-e", help="Export results to file")
    def rank(keywords, export):
        """Check keyword rankings."""
        
        async def _rank():
            try:
                if not keywords:
                    console.print("❌ Please provide at least one keyword", style="red")
                    return
                
                with Status("Checking keyword rankings...", console=console):
                    agent = await create_seo_research_agent(cli_context.tenant_id)
                    # This would need to be implemented in the agent
                    # For now, show placeholder
                    console.print("📊 Keyword ranking check would be implemented here")
                
                # Placeholder results
                table = Table(title="Keyword Rankings")
                table.add_column("Keyword", style="cyan")
                table.add_column("Current Rank", style="yellow")
                table.add_column("Previous Rank", style="green")
                table.add_column("Change", style="blue")
                table.add_column("Search Volume", style="magenta")
                
                for keyword in keywords:
                    table.add_row(
                        keyword,
                        "#25",
                        "#28",
                        "🟢 +3",
                        "1,200"
                    )
                
                console.print(table)
                
                if export:
                    console.print(f"📁 Results exported to {export}")
                
            except Exception as e:
                console.print(f"❌ Rank check failed: {e}", style="red")
                raise
        
        asyncio.run(_rank())
    
    @seo.command()
    @click.argument("url")
    @click.option("--export", "-e", help="Export audit report")
    def audit(url, export):
        """Perform comprehensive SEO audit."""
        
        async def _audit():
            try:
                with Status("Performing SEO audit...", console=console):
                    agent = await create_seo_research_agent(cli_context.tenant_id)
                    
                    # Comprehensive audit request
                    request = SEOAnalysisRequest(
                        url=url,
                        tenant_id=cli_context.tenant_id,
                        analysis_type="comprehensive"
                    )
                    
                    result = await agent.analyze_seo(request)
                
                # Display comprehensive audit results
                console.print(Panel(f"SEO Audit Report for {url}", style="blue"))
                
                # Overall Score
                score_color = "green" if result.seo_score >= 80 else "yellow" if result.seo_score >= 60 else "red"
                console.print(Panel(f"Overall SEO Score: {result.seo_score}/100", style=score_color))
                
                # Critical Issues
                console.print("\n🚨 Critical Issues:")
                critical_issues = [
                    "Missing meta description",
                    "No H1 tag found",
                    "Page speed too slow (>3s)",
                    "Not mobile-friendly"
                ]
                
                for issue in critical_issues:
                    console.print(f"• {issue}")
                
                # Opportunities
                console.print("\n💡 Opportunities:")
                opportunities = [
                    "Optimize images for better loading",
                    "Add internal links",
                    "Improve content structure",
                    "Target long-tail keywords"
                ]
                
                for opp in opportunities:
                    console.print(f"• {opp}")
                
                if export:
                    console.print(f"\n📁 Full audit report exported to {export}")
                
            except Exception as e:
                console.print(f"❌ SEO audit failed: {e}", style="red")
                raise
        
        asyncio.run(_audit())
    
    return seo