#!/bin/bash

# SEO Content Knowledge Graph System - Production Deployment Script
# For VPS deployment with Docker Compose

set -e

echo "🚀 Starting SEO Content Knowledge Graph System deployment..."

# Configuration
DOMAIN="${DOMAIN:-localhost}"
EMAIL="${EMAIL:-admin@example.com}"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   log_error "This script should not be run as root for security reasons"
   exit 1
fi

# Check prerequisites
log_info "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    log_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if .env.production exists
if [[ ! -f ".env.production" ]]; then
    log_error ".env.production file not found. Please create it with your production environment variables."
    exit 1
fi

# Create necessary directories
log_info "Creating necessary directories..."
mkdir -p logs/nginx
mkdir -p nginx/ssl
mkdir -p static/uploads
mkdir -p backups

# Generate SSL certificates if they don't exist (self-signed for development)
if [[ ! -f "nginx/ssl/cert.pem" ]]; then
    log_warning "SSL certificates not found. Generating self-signed certificates..."
    log_warning "For production, replace these with proper SSL certificates from Let's Encrypt or your CA"
    
    openssl req -x509 -newkey rsa:4096 -keyout nginx/ssl/key.pem -out nginx/ssl/cert.pem \
        -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=${DOMAIN}"
    
    chmod 600 nginx/ssl/key.pem
    chmod 644 nginx/ssl/cert.pem
    
    log_success "Self-signed SSL certificates generated"
fi

# Build and start services
log_info "Building Docker images..."
docker-compose -f docker-compose.production.yml build

log_info "Starting services..."
docker-compose -f docker-compose.production.yml up -d

# Wait for services to be healthy
log_info "Waiting for services to be ready..."
sleep 30

# Check service health
log_info "Checking service health..."
if docker-compose -f docker-compose.production.yml ps | grep -q "Up (healthy)"; then
    log_success "Services are running and healthy"
else
    log_warning "Some services may not be healthy yet. Check logs with: docker-compose -f docker-compose.production.yml logs"
fi

# Display service status
log_info "Service status:"
docker-compose -f docker-compose.production.yml ps

# Display access URLs
echo ""
log_success "Deployment completed successfully!"
echo ""
echo "📍 Access URLs:"
echo "   🏠 Application: https://${DOMAIN}"
echo "   🔐 Login: https://${DOMAIN}/login"
echo "   📊 API Docs: https://${DOMAIN}/api/docs"
echo "   ❤️  Health Check: https://${DOMAIN}/api/health"
echo ""
echo "🔧 Management Commands:"
echo "   📋 View logs: docker-compose -f docker-compose.production.yml logs -f"
echo "   🔄 Restart: docker-compose -f docker-compose.production.yml restart"
echo "   🛑 Stop: docker-compose -f docker-compose.production.yml down"
echo "   🗑️  Clean up: docker-compose -f docker-compose.production.yml down -v --remove-orphans"
echo ""
echo "🔒 Security Notes:"
echo "   - Replace self-signed certificates with proper SSL certificates"
echo "   - Configure firewall to only allow necessary ports (80, 443, 22)"
echo "   - Set up regular backups of your data"
echo "   - Monitor logs regularly"
echo ""

# Optional: Set up log rotation
if command -v logrotate &> /dev/null; then
    log_info "Setting up log rotation..."
    sudo tee /etc/logrotate.d/seo-app > /dev/null <<EOF
$(pwd)/logs/*.log {
    weekly
    rotate 4
    compress
    delaycompress
    missingok
    notifempty
    create 644 $USER $USER
}
EOF
    log_success "Log rotation configured"
fi

log_success "🎉 SEO Content Knowledge Graph System is now running in production mode!"