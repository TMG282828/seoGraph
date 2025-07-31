"""
Tests for the Neo4j Client module.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio

from database.neo4j_client import Neo4jClient


class TestNeo4jClient:
    """Test suite for Neo4jClient."""

    def test_neo4j_client_initialization(self):
        """Test Neo4jClient initialization."""
        client = Neo4jClient()
        
        assert client is not None
        assert hasattr(client, '_driver')
        assert hasattr(client, 'uri')
        assert hasattr(client, 'user')
        assert hasattr(client, 'password')

    def test_neo4j_client_default_config(self):
        """Test Neo4jClient with default configuration."""
        client = Neo4jClient()
        
        # Check default values
        assert client.uri == "bolt://localhost:7687"
        assert client.user == "neo4j"
        assert client.password == "password"

    def test_neo4j_client_custom_config(self):
        """Test Neo4jClient with custom configuration."""
        client = Neo4jClient(
            uri="bolt://custom-host:7687",
            user="custom_user",
            password="custom_pass"
        )
        
        assert client.uri == "bolt://custom-host:7687"
        assert client.user == "custom_user"
        assert client.password == "custom_pass"

    @patch('database.neo4j_client.GraphDatabase')
    def test_neo4j_client_driver_creation(self, mock_graph_db):
        """Test Neo4j driver creation."""
        mock_driver = MagicMock()
        mock_graph_db.driver.return_value = mock_driver
        
        client = Neo4jClient()
        
        # Verify driver was created with correct parameters
        mock_graph_db.driver.assert_called_once_with(
            "bolt://localhost:7687",
            auth=("neo4j", "password")
        )

    @patch('database.neo4j_client.GraphDatabase')
    def test_neo4j_client_context_manager(self, mock_graph_db):
        """Test Neo4jClient as context manager."""
        mock_driver = MagicMock()
        mock_graph_db.driver.return_value = mock_driver
        
        with Neo4jClient() as client:
            assert client is not None
            assert hasattr(client, 'driver')
        
        # Verify driver was closed
        mock_driver.close.assert_called_once()

    @patch('database.neo4j_client.GraphDatabase')
    def test_neo4j_client_execute_query(self, mock_graph_db):
        """Test executing a query."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        
        mock_graph_db.driver.return_value = mock_driver
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = mock_result
        
        client = Neo4jClient()
        
        # Execute a test query
        query = "MATCH (n:Node) RETURN n"
        parameters = {"param1": "value1"}
        
        result = client.execute_query(query, parameters)
        
        # Verify session and query execution
        mock_driver.session.assert_called_once()
        mock_session.run.assert_called_once_with(query, parameters)
        assert result == mock_result

    @patch('database.neo4j_client.GraphDatabase')
    def test_neo4j_client_execute_query_without_parameters(self, mock_graph_db):
        """Test executing a query without parameters."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        
        mock_graph_db.driver.return_value = mock_driver
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = mock_result
        
        client = Neo4jClient()
        
        # Execute a test query without parameters
        query = "MATCH (n:Node) RETURN n"
        
        result = client.execute_query(query)
        
        # Verify session and query execution
        mock_driver.session.assert_called_once()
        mock_session.run.assert_called_once_with(query, {})
        assert result == mock_result

    @patch('database.neo4j_client.GraphDatabase')
    def test_neo4j_client_execute_write_transaction(self, mock_graph_db):
        """Test executing a write transaction."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        
        mock_graph_db.driver.return_value = mock_driver
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.write_transaction.return_value = mock_result
        
        client = Neo4jClient()
        
        # Define a transaction function
        def transaction_func(tx):
            return tx.run("CREATE (n:Node {name: $name})", name="test")
        
        result = client.execute_write_transaction(transaction_func)
        
        # Verify write transaction execution
        mock_driver.session.assert_called_once()
        mock_session.write_transaction.assert_called_once_with(transaction_func)
        assert result == mock_result

    @patch('database.neo4j_client.GraphDatabase')
    def test_neo4j_client_execute_read_transaction(self, mock_graph_db):
        """Test executing a read transaction."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        
        mock_graph_db.driver.return_value = mock_driver
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.read_transaction.return_value = mock_result
        
        client = Neo4jClient()
        
        # Define a transaction function
        def transaction_func(tx):
            return tx.run("MATCH (n:Node) RETURN n")
        
        result = client.execute_read_transaction(transaction_func)
        
        # Verify read transaction execution
        mock_driver.session.assert_called_once()
        mock_session.read_transaction.assert_called_once_with(transaction_func)
        assert result == mock_result

    @patch('database.neo4j_client.GraphDatabase')
    def test_neo4j_client_health_check(self, mock_graph_db):
        """Test Neo4j health check."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        
        mock_graph_db.driver.return_value = mock_driver
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = mock_result
        
        client = Neo4jClient()
        
        # Test health check
        health_status = client.health_check()
        
        # Verify health check query was executed
        mock_driver.session.assert_called_once()
        mock_session.run.assert_called_once_with("RETURN 1")
        assert health_status is True

    @patch('database.neo4j_client.GraphDatabase')
    def test_neo4j_client_health_check_failure(self, mock_graph_db):
        """Test Neo4j health check failure."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        
        mock_graph_db.driver.return_value = mock_driver
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.side_effect = Exception("Connection failed")
        
        client = Neo4jClient()
        
        # Test health check failure
        health_status = client.health_check()
        
        # Verify health check returned False on exception
        assert health_status is False

    @patch('database.neo4j_client.GraphDatabase')
    def test_neo4j_client_create_indexes(self, mock_graph_db):
        """Test creating indexes."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        
        mock_graph_db.driver.return_value = mock_driver
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        client = Neo4jClient()
        
        # Test creating indexes
        client.create_indexes()
        
        # Verify session was used
        mock_driver.session.assert_called_once()

    @patch('database.neo4j_client.GraphDatabase')
    def test_neo4j_client_create_constraints(self, mock_graph_db):
        """Test creating constraints."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        
        mock_graph_db.driver.return_value = mock_driver
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        client = Neo4jClient()
        
        # Test creating constraints
        client.create_constraints()
        
        # Verify session was used
        mock_driver.session.assert_called_once()

    @patch('database.neo4j_client.GraphDatabase')
    def test_neo4j_client_close(self, mock_graph_db):
        """Test closing the Neo4j client."""
        mock_driver = MagicMock()
        mock_graph_db.driver.return_value = mock_driver
        
        client = Neo4jClient()
        client.close()
        
        # Verify driver was closed
        mock_driver.close.assert_called_once()