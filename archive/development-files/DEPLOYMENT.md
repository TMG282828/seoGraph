# Production Deployment Guide

Complete guide for deploying the SEO Knowledge Graph system with SerpBear integration in production environments.

## 🚀 Production Ready Features

✅ **SerpBear Integration** - Automated rank tracking with fallback systems  
✅ **Docker Compose** - Multi-container orchestration  
✅ **Health Monitoring** - Comprehensive health checks and alerts  
✅ **Security Hardening** - Production security configurations  
✅ **Database Scaling** - Neo4j AuraDB, Supabase, Qdrant cloud  
✅ **Load Balancing** - Nginx reverse proxy configuration  
✅ **Automated Scheduling** - Background rank processing  

## 📋 Prerequisites

### Required Services
- **Docker & Docker Compose** (v20.10+)
- **Domain Name** with SSL certificate
- **Cloud Database Accounts**:
  - Neo4j AuraDB instance
  - Supabase project  
  - Qdrant cloud cluster
- **API Keys**:
  - OpenAI API key
  - Google OAuth credentials
  - SerpBear instance

### Server Requirements
- **CPU**: 2+ cores (4+ recommended)
- **RAM**: 4GB minimum (8GB+ recommended)  
- **Storage**: 20GB+ SSD
- **Network**: Public IP with ports 80, 443 open

## 🔧 Quick Production Setup

### 1. Server Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker --version
docker-compose --version
```

### 2. Application Deployment

```bash
# Clone repository
git clone <your-repo-url>
cd Context-Engineering-Intro

# Configure environment
cp .env.production .env
nano .env  # Update with your production values

# Deploy with Docker Compose
docker-compose -f docker-compose.production.yml up -d

# Verify deployment
docker-compose -f docker-compose.production.yml ps
```

### 3. SerpBear Setup

```bash
# SerpBear will start automatically via Docker Compose
# Access SerpBear at: http://your-domain:3001

# 1. Create admin account
# 2. Add your domain
# 3. Add initial keywords (10-20 recommended)
# 4. Generate API key in Settings > API
# 5. Update .env with your API key
```

### 4. SSL Certificate Setup

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d yourdomain.com -d serpbear.yourdomain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## 🎯 Success Checklist

After deployment, verify:

- [ ] **Application Health**: `/api/health` returns 200
- [ ] **SerpBear Integration**: Dashboard shows real ranking data  
- [ ] **Database Connections**: All health checks pass
- [ ] **SSL Certificate**: A+ grade on SSL Labs
- [ ] **Monitoring**: Logs are being generated and rotated
- [ ] **Backups**: Automated backup system working
- [ ] **Performance**: Response times < 2 seconds
- [ ] **Security**: All sensitive data encrypted
- [ ] **Automation**: Nightly rank updates running
- [ ] **Alerts**: Error notification system active

## 📞 Support

**Pre-deployment checklist**: Run `python test_serpbear_integration.py`  
**Health monitoring**: Monitor `/api/health/detailed` endpoint  
**Performance metrics**: Available at `/api/metrics` (if Prometheus enabled)  
**Documentation**: Full API docs at `/docs` when deployed  

---

🎉 **Your SEO Knowledge Graph system with SerpBear integration is now production-ready!**

Visit your domain to access the dashboard with real-time ranking data.

## SEO Content Knowledge Graph System - VPS Deployment

This guide covers deploying the SEO Content Knowledge Graph System to a VPS using Docker Compose.

## 📋 Prerequisites

### System Requirements
- **VPS**: 2+ CPU cores, 4+ GB RAM, 20+ GB storage
- **OS**: Ubuntu 20.04+ or similar Linux distribution
- **Docker**: Version 20.10+
- **Docker Compose**: Version 2.0+

### External Services
- **Supabase**: PostgreSQL database (managed)
- **Neo4j AuraDB**: Graph database (managed)
- **Qdrant Cloud**: Vector database (managed)
- **Google OAuth**: Authentication credentials
- **OpenAI API**: For AI agents

## 🛠️ Installation Steps

### 1. Server Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Install additional tools
sudo apt install -y nginx certbot python3-certbot-nginx ufw
```

### 2. Project Setup

```bash
# Clone repository
git clone <your-repo-url>
cd Context-Engineering-Intro

# Create production environment file
cp .env.example .env.production

# Edit environment variables
nano .env.production
```

### 3. Environment Configuration

Configure your `.env.production` file:

```bash
# Environment
ENVIRONMENT=production

# Database Connections
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_PASSWORD=your-password
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-api-key

# Authentication
JWT_SECRET_KEY=your-super-secret-jwt-key
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
GOOGLE_REDIRECT_URI=https://yourdomain.com/api/auth/google/callback

# AI Services
OPENAI_API_KEY=your-openai-api-key

# Application
DOMAIN=yourdomain.com
ADMIN_EMAIL=admin@yourdomain.com
```

### 4. SSL Certificates

For production, use Let's Encrypt:

```bash
# Install SSL certificate
sudo certbot --nginx -d yourdomain.com

# Copy certificates to nginx directory
sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem nginx/ssl/cert.pem
sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem nginx/ssl/key.pem
sudo chown $USER:$USER nginx/ssl/*.pem
```

### 5. Firewall Configuration

```bash
# Configure UFW firewall
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw status
```

### 6. Deploy Application

```bash
# Run deployment script
./deploy.sh

# Or manually:
docker-compose -f docker-compose.production.yml up -d
```

## 🔧 Management Commands

### Application Management
```bash
# View logs
docker-compose -f docker-compose.production.yml logs -f

# Restart services
docker-compose -f docker-compose.production.yml restart

# Stop services
docker-compose -f docker-compose.production.yml down

# Update application
git pull
docker-compose -f docker-compose.production.yml build
docker-compose -f docker-compose.production.yml up -d
```

### Database Management
```bash
# Backup database
docker exec seo-app python -m scripts.backup_data

# Check database connectivity
docker exec seo-app python -c "from src.database.supabase_client import supabase_client; print('✅ Connected')"
```

## 📊 Monitoring & Maintenance

### Health Checks
- **Application**: `https://yourdomain.com/api/health`
- **Prometheus**: `http://localhost:9090` (if enabled)
- **Logs**: `tail -f logs/*.log`

### Log Rotation
The deployment script automatically sets up log rotation. Manual configuration:

```bash
# Edit logrotate config
sudo nano /etc/logrotate.d/seo-app
```

### Backup Strategy
```bash
# Create backup script
cat > backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p backups/$DATE

# Backup logs
cp -r logs/ backups/$DATE/

# Backup uploads
cp -r static/uploads/ backups/$DATE/

# Compress backup
tar -czf backups/seo-backup-$DATE.tar.gz backups/$DATE/
rm -rf backups/$DATE/

echo "Backup completed: backups/seo-backup-$DATE.tar.gz"
EOF

chmod +x backup.sh

# Add to crontab for daily backups
(crontab -l 2>/dev/null; echo "0 2 * * * /path/to/backup.sh") | crontab -
```

## 🔒 Security Considerations

### 1. Server Hardening
```bash
# Disable root SSH
sudo sed -i 's/PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sudo systemctl restart ssh

# Install fail2ban
sudo apt install fail2ban
sudo systemctl enable fail2ban
```

### 2. Application Security
- ✅ JWT tokens with secure keys
- ✅ HTTPS/SSL encryption
- ✅ Rate limiting via Nginx
- ✅ Security headers configured
- ✅ Non-root Docker containers

### 3. Database Security
- ✅ Managed database services (Supabase, Neo4j AuraDB, Qdrant Cloud)
- ✅ Row-level security policies
- ✅ Encrypted connections
- ✅ Environment variable secrets

## 🚨 Troubleshooting

### Common Issues

#### 1. Docker Permission Issues
```bash
sudo chown -R $USER:$USER /var/run/docker.sock
```

#### 2. SSL Certificate Issues
```bash
# Renew Let's Encrypt certificates
sudo certbot renew --dry-run
```

#### 3. Database Connection Issues
```bash
# Test database connections
docker exec seo-app python -c "
from src.database.supabase_client import supabase_client
from src.database.neo4j_client import neo4j_client
from src.database.qdrant_client import qdrant_client
print('Databases:', 'OK' if all([supabase_client, neo4j_client, qdrant_client]) else 'FAILED')
"
```

#### 4. High Memory Usage
```bash
# Monitor resources
docker stats
htop

# Limit container memory
# Add to docker-compose.production.yml:
# mem_limit: 2g
```

### Logs Analysis
```bash
# Application logs
docker-compose -f docker-compose.production.yml logs seo-app

# Nginx logs
tail -f logs/nginx/access.log
tail -f logs/nginx/error.log

# System logs
sudo journalctl -u docker -f
```

## 📈 Performance Optimization

### 1. Redis Caching
Enable Redis in `docker-compose.production.yml` and update application to use caching.

### 2. Database Optimization
- **Supabase**: Monitor query performance in dashboard
- **Neo4j**: Use EXPLAIN for query optimization
- **Qdrant**: Configure collection parameters for your use case

### 3. CDN Integration
Consider using Cloudflare or AWS CloudFront for static assets.

## 🔄 CI/CD Pipeline (Optional)

### GitHub Actions Example
```yaml
name: Deploy to Production
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to VPS
        run: |
          ssh user@your-vps "cd /path/to/app && git pull && ./deploy.sh"
```

## 📞 Support

For issues or questions:
1. Check logs first: `docker-compose logs -f`
2. Review this documentation
3. Check application health endpoints
4. Contact your development team

---

**🎉 Your SEO Content Knowledge Graph System is now production-ready!**