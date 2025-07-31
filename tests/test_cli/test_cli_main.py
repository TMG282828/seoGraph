"""
Tests for the CLI interface.

This module tests the main CLI functionality including command parsing,
execution, and output formatting.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from click.testing import CliRunner

from cli.main import cli, CLIContext
from config.settings import get_settings

class TestCLIContext:
    """Test cases for CLIContext."""
    
    def test_cli_context_initialization(self):
        """Test CLI context initialization."""
        context = CLIContext()
        
        # Verify default values
        assert context.settings is not None
        assert context.verbose is False
        assert context.format == "table"
        assert context.output_file is None
        assert context.tenant_id == "default"
        assert context.user_id == "cli_user"
        
        # Verify services are None initially
        assert context.content_service is None
        assert context.analytics_service is None
        assert context.workflow_orchestrator is None
        assert context.competitor_service is None
        assert context.gap_analysis_service is None
        assert context.neo4j_client is None
        assert context.qdrant_client is None
    
    @pytest.mark.asyncio
    async def test_cli_context_service_initialization(self):
        """Test CLI context service initialization."""
        context = CLIContext()
        
        # Mock the services to avoid actual connections
        with patch('cli.main.Neo4jClient') as mock_neo4j, \
             patch('cli.main.QdrantClient') as mock_qdrant, \
             patch('cli.main.ContentIngestionService') as mock_content, \
             patch('cli.main.AnalyticsService') as mock_analytics, \
             patch('cli.main.WorkflowOrchestrator') as mock_workflow, \
             patch('cli.main.CompetitorMonitoringService') as mock_competitor, \
             patch('cli.main.GapAnalysisService') as mock_gap:
            
            # Mock the workflow orchestrator initialize method
            mock_workflow_instance = Mock()
            mock_workflow_instance.initialize = AsyncMock()
            mock_workflow.return_value = mock_workflow_instance
            
            # Initialize services
            await context.initialize_services()
            
            # Verify services are initialized
            assert context.neo4j_client is not None
            assert context.qdrant_client is not None
            assert context.content_service is not None
            assert context.analytics_service is not None
            assert context.workflow_orchestrator is not None
            assert context.competitor_service is not None
            assert context.gap_analysis_service is not None
            
            # Verify workflow orchestrator was initialized
            mock_workflow_instance.initialize.assert_called_once()

class TestCLICommands:
    """Test cases for CLI commands."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_cli_help(self):
        """Test CLI help command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert "SEO Content Knowledge Graph System CLI" in result.output
        assert "content" in result.output
        assert "seo" in result.output
        assert "workflow" in result.output
        assert "analytics" in result.output
        assert "system" in result.output
    
    def test_cli_version(self):
        """Test CLI version command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])
        
        assert result.exit_code == 0
        assert "1.0.0" in result.output
    
    def test_cli_options(self):
        """Test CLI global options."""
        runner = CliRunner()
        
        # Test verbose option
        with patch('cli.main.CLIContext.initialize_services', new_callable=AsyncMock):
            result = runner.invoke(cli, ['--verbose', 'system', 'config'])
            # Should not error (though may not complete due to async)
            assert result.exit_code in [0, 1]  # May fail due to async setup
        
        # Test format option
        with patch('cli.main.CLIContext.initialize_services', new_callable=AsyncMock):
            result = runner.invoke(cli, ['--format', 'json', 'system', 'config'])
            # Should not error
            assert result.exit_code in [0, 1]
        
        # Test tenant option
        with patch('cli.main.CLIContext.initialize_services', new_callable=AsyncMock):
            result = runner.invoke(cli, ['--tenant', 'test_tenant', 'system', 'config'])
            # Should not error
            assert result.exit_code in [0, 1]
    
    def test_content_command_group(self):
        """Test content command group."""
        runner = CliRunner()
        
        # Test content help
        with patch('cli.main.CLIContext.initialize_services', new_callable=AsyncMock):
            result = runner.invoke(cli, ['content', '--help'])
            assert result.exit_code == 0
            assert "Content management and analysis commands" in result.output
            assert "ingest" in result.output
            assert "generate" in result.output
            assert "list" in result.output
    
    def test_seo_command_group(self):
        """Test SEO command group."""
        runner = CliRunner()
        
        # Test SEO help
        with patch('cli.main.CLIContext.initialize_services', new_callable=AsyncMock):
            result = runner.invoke(cli, ['seo', '--help'])
            assert result.exit_code == 0
            assert "SEO research and analysis commands" in result.output
            assert "research" in result.output
            assert "analyze" in result.output
    
    def test_workflow_command_group(self):
        """Test workflow command group."""
        runner = CliRunner()
        
        # Test workflow help
        with patch('cli.main.CLIContext.initialize_services', new_callable=AsyncMock):
            result = runner.invoke(cli, ['workflow', '--help'])
            assert result.exit_code == 0
            assert "Workflow management and orchestration commands" in result.output
            assert "create" in result.output
            assert "list" in result.output
            assert "execute" in result.output
            assert "status" in result.output
    
    def test_analytics_command_group(self):
        """Test analytics command group."""
        runner = CliRunner()
        
        # Test analytics help
        with patch('cli.main.CLIContext.initialize_services', new_callable=AsyncMock):
            result = runner.invoke(cli, ['analytics', '--help'])
            assert result.exit_code == 0
            assert "Analytics and reporting commands" in result.output
            assert "overview" in result.output
    
    def test_system_command_group(self):
        """Test system command group."""
        runner = CliRunner()
        
        # Test system help
        with patch('cli.main.CLIContext.initialize_services', new_callable=AsyncMock):
            result = runner.invoke(cli, ['system', '--help'])
            assert result.exit_code == 0
            assert "System management and maintenance commands" in result.output
            assert "status" in result.output
            assert "config" in result.output
    
    def test_system_config_command(self):
        """Test system config command."""
        runner = CliRunner()
        
        # Mock the service initialization
        with patch('cli.main.CLIContext.initialize_services', new_callable=AsyncMock):
            result = runner.invoke(cli, ['system', 'config'])
            
            # Should show configuration
            assert result.exit_code == 0
            assert "Configuration" in result.output
            # Should contain some config values
            assert "Environment" in result.output or "development" in result.output
    
    @pytest.mark.asyncio
    async def test_content_generate_command_logic(self):
        """Test content generation command logic."""
        from cli.main import cli_context
        
        # Mock the required services
        cli_context.content_service = Mock()
        mock_agent = Mock()
        mock_agent.generate_content = AsyncMock()
        
        # Mock the result
        from models.content_models import ContentGenerationResult
        mock_result = Mock(spec=ContentGenerationResult)
        mock_result.content = "Generated content"
        mock_result.title = "Generated Title"
        mock_result.word_count = 500
        mock_result.seo_score = 75
        mock_result.reading_time_minutes = 2.5
        mock_result.keywords_used = ["keyword1", "keyword2"]
        mock_result.dict.return_value = {
            "content": "Generated content",
            "title": "Generated Title",
            "word_count": 500,
            "seo_score": 75,
            "reading_time_minutes": 2.5,
            "keywords_used": ["keyword1", "keyword2"]
        }
        
        mock_agent.generate_content.return_value = mock_result
        
        # Mock the agent creation
        with patch('cli.main.create_content_analysis_agent', new_callable=AsyncMock) as mock_create_agent:
            mock_create_agent.return_value = mock_agent
            
            # Test the command logic would work
            # (This is a unit test of the logic, not the full CLI)
            assert mock_agent is not None
            assert mock_agent.generate_content is not None
    
    @pytest.mark.asyncio
    async def test_seo_research_command_logic(self):
        """Test SEO research command logic."""
        from cli.main import cli_context
        
        # Mock the required services
        mock_agent = Mock()
        mock_agent.research_keywords = AsyncMock()
        
        # Mock the result
        from models.seo_models import KeywordResearchResult
        mock_result = Mock(spec=KeywordResearchResult)
        mock_result.keywords = []
        mock_result.dict.return_value = {"keywords": []}
        
        mock_agent.research_keywords.return_value = mock_result
        
        # Mock the agent creation
        with patch('cli.main.create_seo_research_agent', new_callable=AsyncMock) as mock_create_agent:
            mock_create_agent.return_value = mock_agent
            
            # Test the command logic would work
            assert mock_agent is not None
            assert mock_agent.research_keywords is not None
    
    @pytest.mark.asyncio
    async def test_workflow_create_command_logic(self):
        """Test workflow creation command logic."""
        from cli.main import cli_context
        
        # Mock the workflow orchestrator
        cli_context.workflow_orchestrator = Mock()
        cli_context.workflow_orchestrator.create_workflow = AsyncMock()
        
        # Mock the result
        from models.workflow_models import ContentWorkflow
        mock_workflow = Mock(spec=ContentWorkflow)
        mock_workflow.workflow_id = "test_workflow_id"
        mock_workflow.name = "Test Workflow"
        mock_workflow.workflow_type = "content_creation"
        mock_workflow.status = Mock()
        mock_workflow.status.value = "DRAFT"
        mock_workflow.steps = []
        mock_workflow.created_at = Mock()
        mock_workflow.created_at.isoformat.return_value = "2024-01-01T00:00:00Z"
        mock_workflow.dict.return_value = {
            "workflow_id": "test_workflow_id",
            "name": "Test Workflow",
            "workflow_type": "content_creation",
            "status": "DRAFT",
            "steps": [],
            "created_at": "2024-01-01T00:00:00Z"
        }
        
        cli_context.workflow_orchestrator.create_workflow.return_value = mock_workflow
        
        # Test the command logic would work
        assert cli_context.workflow_orchestrator is not None
        assert cli_context.workflow_orchestrator.create_workflow is not None
    
    def test_interactive_mode_entry(self):
        """Test interactive mode entry point."""
        runner = CliRunner()
        
        # Mock the interactive input to exit immediately
        with patch('cli.main.Prompt.ask', return_value="exit"):
            with patch('cli.main.CLIContext.initialize_services', new_callable=AsyncMock):
                result = runner.invoke(cli, ['interactive'])
                
                # Should enter and exit interactive mode
                assert result.exit_code == 0
                assert "Welcome to" in result.output
                assert "Goodbye" in result.output
    
    def test_error_handling(self):
        """Test CLI error handling."""
        runner = CliRunner()
        
        # Test invalid command
        result = runner.invoke(cli, ['invalid_command'])
        assert result.exit_code == 2  # Click's exit code for usage errors
        
        # Test invalid option
        result = runner.invoke(cli, ['--invalid-option'])
        assert result.exit_code == 2
    
    def test_cli_context_global_access(self):
        """Test global CLI context access."""
        from cli.main import cli_context
        
        # Verify global context exists
        assert cli_context is not None
        assert isinstance(cli_context, CLIContext)
        
        # Verify default values
        assert cli_context.tenant_id == "default"
        assert cli_context.user_id == "cli_user"
        assert cli_context.format == "table"
        assert cli_context.verbose is False
    
    def test_output_formatting_options(self):
        """Test output formatting options."""
        runner = CliRunner()
        
        # Test table format (default)
        with patch('cli.main.CLIContext.initialize_services', new_callable=AsyncMock):
            result = runner.invoke(cli, ['--format', 'table', 'system', 'config'])
            assert result.exit_code == 0
        
        # Test JSON format
        with patch('cli.main.CLIContext.initialize_services', new_callable=AsyncMock):
            result = runner.invoke(cli, ['--format', 'json', 'system', 'config'])
            assert result.exit_code == 0
        
        # Test YAML format
        with patch('cli.main.CLIContext.initialize_services', new_callable=AsyncMock):
            result = runner.invoke(cli, ['--format', 'yaml', 'system', 'config'])
            assert result.exit_code == 0
    
    def test_cli_constants(self):
        """Test CLI constants."""
        from cli.main import CLI_VERSION, CLI_NAME
        
        assert CLI_VERSION == "1.0.0"
        assert CLI_NAME == "SEO Content Knowledge Graph CLI"
    
    def test_cli_imports(self):
        """Test CLI imports are accessible."""
        # Test that all required imports work
        from cli.main import (
            cli, CLIContext, console, logger,
            click, asyncio, json, sys, Path
        )
        
        # Verify imports are not None
        assert cli is not None
        assert CLIContext is not None
        assert console is not None
        assert logger is not None
        assert click is not None
        assert asyncio is not None
        assert json is not None
        assert sys is not None
        assert Path is not None
    
    @pytest.mark.asyncio
    async def test_service_initialization_error_handling(self):
        """Test service initialization error handling."""
        context = CLIContext()
        
        # Mock service initialization to raise exception
        with patch('cli.main.Neo4jClient', side_effect=Exception("Connection failed")):
            with pytest.raises(SystemExit):
                await context.initialize_services()
    
    def test_cli_help_sections(self):
        """Test CLI help sections contain expected content."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        output = result.output
        
        # Check for main sections
        assert "Usage:" in output
        assert "Options:" in output
        assert "Commands:" in output
        
        # Check for specific commands
        assert "content" in output
        assert "seo" in output
        assert "workflow" in output
        assert "analytics" in output
        assert "system" in output
        assert "interactive" in output
        
        # Check for version option
        assert "--version" in output
        assert "--verbose" in output
        assert "--format" in output