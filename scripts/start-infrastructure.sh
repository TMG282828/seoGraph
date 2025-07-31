#!/bin/bash

# =============================================================================
# Infrastructure Startup Script for SEO Content Knowledge Graph System
# =============================================================================

set -e  # Exit on any error

echo "🚀 Starting SEO Content Knowledge Graph Infrastructure..."

# Function to check if a service is running
check_service() {
    local service_name=$1
    local port=$2
    local max_attempts=${3:-30}
    local attempt=1
    
    echo "⏳ Waiting for $service_name on port $port..."
    
    while [ $attempt -le $max_attempts ]; do
        if nc -z localhost $port 2>/dev/null; then
            echo "✅ $service_name is running on port $port"
            return 0
        fi
        
        echo "   Attempt $attempt/$max_attempts: $service_name not ready..."
        sleep 2
        ((attempt++))
    done
    
    echo "❌ $service_name failed to start on port $port after $max_attempts attempts"
    return 1
}

# Function to check Docker Compose status
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null && ! command -v docker &> /dev/null; then
        echo "❌ Docker is not installed or not running"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        echo "❌ Docker daemon is not running"
        exit 1
    fi
    
    echo "✅ Docker is running"
}

# Function to start specific services
start_core_services() {
    echo "🐳 Starting core infrastructure services..."
    
    # Start only the core services needed for basic functionality
    docker-compose up -d neo4j qdrant redis
    
    # Wait for services to be ready
    check_service "Neo4j" 7474 30
    check_service "Qdrant" 6333 30
    check_service "Redis" 6379 15
}

start_full_services() {
    echo "🐳 Starting full infrastructure stack..."
    
    # Start all services including monitoring
    docker-compose up -d
    
    # Wait for core services
    check_service "Neo4j" 7474 30
    check_service "Qdrant" 6333 30
    check_service "Redis" 6379 15
    check_service "SearXNG" 8080 45
    check_service "Prometheus" 9090 30
    check_service "Grafana" 3000 30
}

# Function to initialize databases
initialize_databases() {
    echo "🔧 Initializing databases..."
    
    # Initialize Neo4j schema
    echo "   Setting up Neo4j constraints and indexes..."
    python -c "
from src.database.neo4j_client import neo4j_client
success = neo4j_client.initialize_schema()
print('✅ Neo4j schema initialized' if success else '❌ Neo4j schema initialization failed')
"
    
    # Initialize Qdrant collections
    echo "   Setting up Qdrant collections..."
    python -c "
from src.database.qdrant_client import qdrant_client
success = qdrant_client.initialize_collections()
print('✅ Qdrant collections initialized' if success else '❌ Qdrant collections initialization failed')
"
}

# Function to run health checks
run_health_checks() {
    echo "🔍 Running health checks..."
    
    # Test Neo4j connection
    python -c "
from src.database.neo4j_client import neo4j_client
try:
    stats = neo4j_client.get_organization_stats('demo-org')
    print('✅ Neo4j connection test passed')
except Exception as e:
    print(f'❌ Neo4j connection test failed: {e}')
"
    
    # Test Qdrant connection
    python -c "
from src.database.qdrant_client import qdrant_client
try:
    analytics = qdrant_client.get_semantic_analytics()
    print('✅ Qdrant connection test passed')
except Exception as e:
    print(f'❌ Qdrant connection test failed: {e}')
"
    
    # Test Redis connection
    python -c "
import redis
try:
    r = redis.Redis(host='localhost', port=6379)
    r.ping()
    print('✅ Redis connection test passed')
except Exception as e:
    print(f'❌ Redis connection test failed: {e}')
"
}

# Function to show service URLs
show_service_urls() {
    echo ""
    echo "🌐 Service URLs:"
    echo "   Neo4j Browser:  http://localhost:7474"
    echo "   Qdrant API:     http://localhost:6333"
    echo "   SearXNG:        http://localhost:8080"
    echo "   Prometheus:     http://localhost:9090"
    echo "   Grafana:        http://localhost:3000 (admin/admin)"
    echo "   Application:    http://localhost:8000"
    echo ""
}

# Main execution
main() {
    case "${1:-core}" in
        "core")
            echo "Starting core services only..."
            check_docker_compose
            start_core_services
            initialize_databases
            run_health_checks
            show_service_urls
            ;;
        "full")
            echo "Starting full infrastructure stack..."
            check_docker_compose
            start_full_services
            initialize_databases
            run_health_checks
            show_service_urls
            ;;
        "health")
            echo "Running health checks only..."
            run_health_checks
            ;;
        "stop")
            echo "🛑 Stopping all services..."
            docker-compose down
            echo "✅ All services stopped"
            ;;
        "restart")
            echo "🔄 Restarting services..."
            docker-compose down
            sleep 3
            $0 ${2:-core}
            ;;
        "logs")
            echo "📋 Showing logs for ${2:-all services}..."
            if [ -n "$2" ]; then
                docker-compose logs -f "$2"
            else
                docker-compose logs -f
            fi
            ;;
        *)
            echo "Usage: $0 {core|full|health|stop|restart|logs} [service_name]"
            echo ""
            echo "Commands:"
            echo "  core     - Start core services (Neo4j, Qdrant, Redis)"
            echo "  full     - Start all services including monitoring"
            echo "  health   - Run health checks on running services"
            echo "  stop     - Stop all services"
            echo "  restart  - Restart services"
            echo "  logs     - Show logs (optionally for specific service)"
            exit 1
            ;;
    esac
}

# Check if required files exist
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ docker-compose.yml not found. Please run from project root."
    exit 1
fi

if [ ! -f ".env" ] && [ ! -f ".env.example" ]; then
    echo "⚠️  No .env file found. Please copy .env.example to .env and configure."
fi

# Run main function
main "$@"