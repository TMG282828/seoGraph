#!/usr/bin/env python3
"""
Streamlined FastAPI web application for SEO Content Knowledge Graph System.

This provides a modern REST API and serves the React frontend with modular route organization.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from starlette.responses import Response
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import json
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add the parent directory to the Python path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Setup comprehensive logging system
from config.logging_config import setup_logging, log_api_request, LoggerMixin
setup_logging(log_level="INFO", log_dir="logs")

logger = logging.getLogger(__name__)

# Import OpenTelemetry monitoring
try:
    from monitoring.otel_monitor import initialize_otel_monitoring, get_otel_monitor
    OTEL_MONITORING_AVAILABLE = True
except ImportError as e:
    OTEL_MONITORING_AVAILABLE = False
    logger.warning(f"OpenTelemetry monitoring not available: {e}")

# Helper functions
async def initialize_gsc_service(organization_id: str) -> bool:
    """
    Initialize Google Search Console service with stored credentials.
    Returns True if service was successfully initialized.
    """
    try:
        org_data = await supabase_client.get_organization(organization_id)
        if not org_data or not org_data.get("gsc_tokens"):
            logger.warning(f"No GSC tokens found for organization {organization_id}")
            return False
        
        import json
        gsc_tokens = org_data.get("gsc_tokens")
        if isinstance(gsc_tokens, str):
            try:
                gsc_tokens = json.loads(gsc_tokens)
            except json.JSONDecodeError:
                logger.error("Invalid GSC tokens format")
                return False
        
        access_token = gsc_tokens.get("access_token")
        refresh_token = gsc_tokens.get("refresh_token")
        
        if not access_token or not refresh_token:
            logger.error("Missing GSC access or refresh token")
            return False
        
        # Initialize GSC service with credentials
        gsc_service.initialize_service(access_token, refresh_token)
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize GSC service: {e}")
        return False

async def is_organization_configured(organization_id: str) -> bool:
    """
    Check if an organization is properly configured for production use.
    Returns True if the organization exists and has configuration_completed=True.
    """
    if not organization_id or organization_id.startswith("demo-"):
        return False
    
    try:
        org_data = await supabase_client.get_organization(organization_id)
        
        # Temporarily treat any non-demo organization as configured
        # This is a workaround while we fix the RLS/service role key issue
        if not org_data and organization_id and not organization_id.startswith("demo-"):
            logger.info(f"Organization {organization_id} not found in DB but treating as configured (temp workaround)")
            return True
        
        if not org_data:
            return False
        
        # Check direct configuration field first
        if org_data.get('configuration_completed'):
            return True
        
        # Check configuration_completed in settings field (may be nested or JSON string)
        settings = org_data.get('settings', {})
        if isinstance(settings, str):
            import json
            try:
                settings = json.loads(settings)
            except json.JSONDecodeError:
                settings = {}
        
        return settings.get('configuration_completed', False)
        
    except Exception as e:
        logger.error(f"Error checking organization configuration: {e}")
        # Temporarily treat as configured if we can't check (workaround)
        return True

# Import our services (adjust for testing)
try:
    from src.services.content_ingestion import ContentIngestionService
    from src.services.embedding_service import EmbeddingService
except ImportError:
    # Fallback for testing
    class ContentIngestionService:
        def __init__(self): pass
    class EmbeddingService:
        def __init__(self): pass

# Import new services for Google Search Console and OpenAI
try:
    from src.services.google_search_console import gsc_service
    from src.services.openai_seo import openai_seo_service
except ImportError as e:
    logger.warning(f"Service import error: {e} - using mock implementations")
    # Fallback classes for testing
    class MockGSCService:
        def add_site(self, site_url): return {"success": False, "error": "GSC service not available"}
        def get_performance_metrics(self, site_url): return {"success": False, "error": "GSC service not available"}
    class MockOpenAIService:
        def analyze_content_seo(self, content, keywords=None, page_type="blog_post", current_tags=None): return {"success": False, "error": "OpenAI service not available"}
    gsc_service = MockGSCService()
    openai_seo_service = MockOpenAIService()

# Import auth services
from src.auth.google_oauth import google_oauth_service
from src.database.supabase_client import supabase_client
from fastapi.responses import RedirectResponse as FastAPIRedirectResponse
try:
    from src.database.neo4j_client import neo4j_client
except ImportError:
    class MockClient:
        def initialize_schema(self): pass
    neo4j_client = MockClient()

# Import real Settings for configuration
try:
    from config.settings import Settings
    settings = Settings()
    logger.info("✅ Real Settings loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load Settings: {e}")
    # Fallback mock Settings
    class Settings:
        def __init__(self): pass
    settings = Settings()

# Import real database clients and services
try:
    from src.database.qdrant_client import qdrant_client
    from src.services.graph_vector_service import graph_vector_service
    from src.auth.auth_middleware import AuthMiddleware, get_current_user, get_current_organization, create_access_token
    logger.info("✅ Production database clients loaded successfully")
    PRODUCTION_MODE = True
except ImportError as e:
    logger.warning(f"Some production services not available: {e} - using fallback implementations")
    # Fallback classes for missing services
    class MockClient:
        def __init__(self): pass
        def create_collection(self, *args, **kwargs): pass
        def search(self, *args, **kwargs): return []
        def upsert(self, *args, **kwargs): pass
    class MockService:
        def __init__(self): pass
        def store_content(self, *args, **kwargs): return {"success": True}
        def search_similar(self, *args, **kwargs): return []
    class MockAuthMiddleware:
        def __init__(self, app, excluded_paths=None): pass
    async def get_current_user(): return {"id": "demo-user", "email": "demo@example.com"}
    async def get_current_organization(): return {"id": "demo-org", "name": "Demo Organization"}
    def create_access_token(data): return "demo-token"
    
    # Use mocks for missing services
    try:
        from src.database.qdrant_client import qdrant_client
    except ImportError:
        qdrant_client = MockClient()
    
    try:
        from src.services.graph_vector_service import graph_vector_service
    except ImportError:
        graph_vector_service = MockService()
        
    try:
        from src.auth.auth_middleware import AuthMiddleware
    except ImportError:
        AuthMiddleware = MockAuthMiddleware
    
    PRODUCTION_MODE = False

# Import API route modules
from web.api.seo_routes import router as seo_router
from web.api.research_routes import router as research_router  
from web.api.content_routes import router as content_router
from web.api.brief_routes import router as brief_router
from web.api.seo_monitor_routes import router as seo_monitor_router
from web.api.serpbear_integration_routes import router as serpbear_router
from web.api.serpbear_settings_routes import router as serpbear_settings_router
from web.api.serpbear_domain_routes import router as serpbear_domain_router
from web.api.settings_routes import router as settings_router
from web.api.health_routes import router as health_router
from web.api.graph import graph_router  # Import modular graph router
from web.api.debug_routes import debug_router
from web.api.graph_debug import debug_graph_router
from src.services.serpbear_bridge import router as serp_bridge_router

# SERP Content Integration
try:
    from web.api.serp_content_integration_routes import router as serp_content_router
    SERP_CONTENT_INTEGRATION_AVAILABLE = True
    logger.info("✅ SERP Content Integration routes loaded successfully")
except ImportError as e:
    SERP_CONTENT_INTEGRATION_AVAILABLE = False
    logger.warning(f"⚠️ SERP Content Integration routes not available: {e}")

# PRP Workflow routes
try:
    from web.api.prp_workflow_routes import router as prp_workflow_router
    PRP_WORKFLOW_AVAILABLE = True
    logger.info("✅ PRP Workflow routes loaded successfully")
except ImportError as e:
    PRP_WORKFLOW_AVAILABLE = False
    logger.warning(f"⚠️ PRP Workflow routes not available: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="SEO Content Knowledge Graph System",
    description="AI-powered SEO content research and optimization platform", 
    version="1.0.0"
)

# Add request logging middleware
import time
from starlette.middleware.base import BaseHTTPMiddleware

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all API requests with timing and status."""
    
    async def dispatch(self, request, call_next):
        start_time = time.time()
        
        # Log incoming request
        api_logger = logging.getLogger('api')
        api_logger.info(f"REQUEST: {request.method} {request.url.path}")
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log response with performance data
        log_api_request(
            endpoint=str(request.url.path),
            method=request.method,
            status_code=response.status_code,
            response_time=duration,
            client_ip=request.client.host if request.client else "unknown"
        )
        
        return response

app.add_middleware(RequestLoggingMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates - use correct path relative to web directory
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Mount static files
if Path("web/static").exists():
    app.mount("/static", StaticFiles(directory="web/static"), name="static")

# Global services (initialize once)
content_service = None
embedding_service = None
graph_service = None

# Authentication middleware - temporarily disabled for debugging until production
# logger.info(f"Adding AuthMiddleware: {AuthMiddleware}")
# app.add_middleware(
#     AuthMiddleware,
#     excluded_paths=['/login', '/onboarding', '/api/auth/', '/api/gsc/', '/api/organizations', '/api/health', '/static/', '/docs', '/openapi.json', '/health']
# )
# logger.info("AuthMiddleware added successfully")
logger.info("AuthMiddleware temporarily disabled - using session-based auth for Google Drive")

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global content_service, embedding_service, graph_service
    
    try:
        logger.info("🚀 Starting SEO Content Knowledge Graph System...")
        logger.info(f"🚀 Server starting on http://localhost:8000")
        
        # Initialize OpenTelemetry monitoring
        if OTEL_MONITORING_AVAILABLE:
            try:
                otel_monitor = await initialize_otel_monitoring()
                otel_monitor.instrument_fastapi(app)
                logger.info("✅ OpenTelemetry monitoring initialized")
            except Exception as e:
                logger.warning(f"⚠️ OpenTelemetry initialization failed: {e}")
        
        # Initialize content ingestion service
        content_service = ContentIngestionService(max_file_size_mb=50.0)
        logger.info(f"Content ingestion service initialized - " +
                   f"Extensions: {content_service.allowed_extensions}, " +
                   f"Max size: 50.0MB, Processors: {len(content_service.processors)}")
        
        # Initialize embedding service
        embedding_service = EmbeddingService()
        logger.info(f"Embedding service initialized - " +
                   f"Model: {embedding_service.model}, " +
                   f"Dimensions: {embedding_service.get_model_dimensions()}, " +
                   f"Max batch: {embedding_service.max_batch_size}")
        
        # Initialize database connections and schemas
        if PRODUCTION_MODE:
            logger.info("🔧 Initializing production database connections...")
            
            # Initialize Neo4j schema
            try:
                neo4j_client.initialize_schema()
                logger.info("✅ Neo4j schema initialized")
            except Exception as e:
                logger.warning(f"⚠️ Neo4j initialization failed: {e}")
            
            # Initialize vector database collections
            try:
                if hasattr(qdrant_client, 'ensure_collection'):
                    await qdrant_client.ensure_collection(
                        "content_embeddings", 
                        vector_size=1536,
                        distance="Cosine"
                    )
                    logger.info("✅ Qdrant collection initialized")
            except Exception as e:
                logger.warning(f"⚠️ Qdrant initialization failed: {e}")
                
            # Initialize graph-vector service with real connections
            try:
                graph_service = graph_vector_service
                if hasattr(graph_service, 'initialize'):
                    await graph_service.initialize()
                logger.info("✅ Graph-vector service initialized")
            except Exception as e:
                logger.warning(f"⚠️ Graph-vector service initialization failed: {e}")
                
        else:
            logger.info("🧪 Running in development mode with mock services")
            # Initialize Neo4j schema (fallback)
            try:
                neo4j_client.initialize_schema()
            except Exception as e:
                logger.warning(f"⚠️ Neo4j mock initialization: {e}")
            
            # Mock graph service
            graph_service = graph_vector_service
        
        logger.info("✅ Database schemas initialized")
        logger.info("🚀 SEO Content Knowledge Graph System started!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on application shutdown."""
    try:
        # Shutdown OpenTelemetry monitoring
        if OTEL_MONITORING_AVAILABLE:
            try:
                otel_monitor = get_otel_monitor()
                if otel_monitor:
                    await otel_monitor.shutdown()
                    logger.info("✅ OpenTelemetry monitoring shutdown complete")
            except Exception as e:
                logger.warning(f"⚠️ OpenTelemetry shutdown error: {e}")
        
        # Close SerpBear client sessions
        from src.services.serpbear_client import serpbear_client
        await serpbear_client.close_session()
        logger.info("✅ SerpBear client session closed")
        
        # Close Neo4j connections if needed
        if hasattr(neo4j_client, 'close'):
            neo4j_client.close()
            logger.info("✅ Neo4j client closed")
            
        logger.info("🔚 Application shutdown complete")
        
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")


# Include API routers
app.include_router(health_router)  # Health checks first
app.include_router(serp_bridge_router)  # Custom SERP bridge
app.include_router(serpbear_domain_router)  # SerpBear domain management
app.include_router(seo_router)
app.include_router(research_router)
app.include_router(content_router)
app.include_router(brief_router, prefix="/api/briefs")
app.include_router(seo_monitor_router)
app.include_router(serpbear_router)
app.include_router(serpbear_settings_router)
app.include_router(settings_router)
app.include_router(graph_router, prefix="/api")
app.include_router(debug_router, prefix="/api")
app.include_router(debug_graph_router, prefix="/api")

# Import and include workspace routes
try:
    from web.api.workspace_routes import router as workspace_router
    app.include_router(workspace_router)
    logger.info("✅ Workspace routes included in application")
    
    # Debug: List actual routes being registered
    workspace_routes = []
    for route in app.routes:
        if hasattr(route, 'path') and '/workspace' in route.path:
            workspace_routes.append(f"{list(route.methods)} -> {route.path}")
    logger.info(f"🔍 Registered {len(workspace_routes)} workspace routes: {workspace_routes[:3]}")
    
except ImportError as e:
    logger.warning(f"⚠️ Workspace routes not available: {e}")
except Exception as e:
    logger.error(f"❌ Workspace routes registration failed: {e}")

# Import and include workspace settings routes
try:
    from web.api.workspace_settings_routes import router as workspace_settings_router
    app.include_router(workspace_settings_router)
    logger.info("✅ Workspace settings routes included in application")
except ImportError as e:
    logger.warning(f"⚠️ Workspace settings routes not available: {e}")

# Include PRP Workflow router if available
if PRP_WORKFLOW_AVAILABLE:
    app.include_router(prp_workflow_router)
    logger.info("✅ PRP Workflow routes included in application")
else:
    logger.info("ℹ️ PRP Workflow routes skipped - service not available")

# Include SERP Content Integration router if available
if SERP_CONTENT_INTEGRATION_AVAILABLE:
    app.include_router(serp_content_router)
    logger.info("✅ SERP Content Integration routes included in application")
else:
    logger.info("ℹ️ SERP Content Integration routes skipped - service not available")

# Page routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Redirect to dashboard."""
    return RedirectResponse(url="/dashboard")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/seo-research", response_class=HTMLResponse) 
async def seo_research(request: Request):
    """SEO Research - Keyword analysis and competitor research."""
    return templates.TemplateResponse("seo_research.html", {"request": request})

@app.get("/content-studio", response_class=HTMLResponse)
async def content_studio(request: Request):
    """Content Studio - Analyze existing or generate new content."""
    return templates.TemplateResponse("content.html", {"request": request})

@app.get("/knowledge-base", response_class=HTMLResponse)
async def knowledge_base(request: Request):
    """Knowledge Base - Upload documents for RAG corpus."""
    return templates.TemplateResponse("knowledge_base.html", {"request": request})

@app.get("/seo-monitor", response_class=HTMLResponse)
async def seo_monitor(request: Request):
    """SEO Monitor - Performance tracking and rankings."""  
    return templates.TemplateResponse("seo.html", {"request": request})

@app.get("/graph", response_class=HTMLResponse)
async def graph_visualization(request: Request):
    """Graph Visualization - Knowledge graph and content relationships."""
    return templates.TemplateResponse("graph.html", {"request": request})


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """Settings - Application configuration and preferences."""
    return templates.TemplateResponse("settings.html", {"request": request})

# Legacy route redirects
@app.get("/seo", response_class=HTMLResponse)
async def seo_dashboard(request: Request):
    """Legacy redirect to SEO Monitor."""
    return RedirectResponse(url="/seo-monitor", status_code=301)

@app.get("/content", response_class=HTMLResponse)
async def content_analysis(request: Request):
    """Legacy redirect to Content Studio."""
    return RedirectResponse(url="/content-studio", status_code=301)

# Auth routes  
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page."""
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/onboarding", response_class=HTMLResponse)
async def onboarding_page(request: Request):
    """Onboarding page."""
    return templates.TemplateResponse("onboarding.html", {"request": request})

# Additional API routes (keeping essential ones from original main.py)
@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}

# Test workspace endpoint without auth dependencies
@app.get("/api/workspaces-test")
async def test_workspace_endpoint():
    """Test endpoint to verify workspace routing works."""
    return {"status": "test", "message": "Workspace routing works!", "timestamp": datetime.now().isoformat()}

# Google OAuth routes
@app.get("/api/auth/google")
async def google_auth_redirect():
    """Redirect to Google OAuth authorization."""
    try:
        auth_url = google_oauth_service.get_authorization_url()
        return FastAPIRedirectResponse(url=auth_url)
    except Exception as e:
        logger.error(f"Failed to get Google auth URL: {e}")
        raise HTTPException(status_code=500, detail="Google authentication not available")

@app.get("/api/auth/google/url")
async def get_google_auth_url():
    """Get Google OAuth authorization URL."""
    try:
        auth_url = google_oauth_service.get_authorization_url()
        return {"auth_url": auth_url}
    except Exception as e:
        logger.error(f"Failed to get Google auth URL: {e}")
        raise HTTPException(status_code=500, detail="Google authentication not available")

@app.get("/api/auth/google/callback")
async def google_auth_callback(code: str = None, error: str = None):
    """Handle Google OAuth callback."""
    if error:
        logger.error(f"Google OAuth error: {error}")
        return FastAPIRedirectResponse(url=f"/login?error=oauth_error")
    
    if not code:
        logger.error("No authorization code received")
        return FastAPIRedirectResponse(url=f"/login?error=no_code")
    
    try:
        # Exchange code for tokens and user info
        auth_result = await google_oauth_service.authenticate_user(code)
        
        # Set cookies and redirect to dashboard
        response = FastAPIRedirectResponse(url="/dashboard")
        response.set_cookie(
            key="access_token",
            value=auth_result["access_token"],
            max_age=86400,  # 24 hours
            httponly=True,
            secure=False,  # Set to True in production with HTTPS
            samesite="lax"
        )
        response.set_cookie(
            key="refresh_token",
            value=auth_result["refresh_token"],
            max_age=2592000,  # 30 days
            httponly=True,
            secure=False,  # Set to True in production with HTTPS
            samesite="lax"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Google authentication failed: {e}")
        return FastAPIRedirectResponse(url=f"/login?error=auth_failed")

@app.post("/api/auth/logout")
async def logout():
    """Logout user."""
    response = JSONResponse({"success": True, "message": "Logged out successfully"})
    response.delete_cookie("access_token")
    response.delete_cookie("refresh_token")
    return response

@app.get("/api/dashboard/metrics") 
async def get_dashboard_metrics(user: Dict[str, Any] = Depends(get_current_user)):
    """Get dashboard metrics and KPIs."""
    try:
        org_id = user.get("organization_id", "demo-org")  # Use real organization ID from authenticated user
        
        # Get real content counts from database
        total_content = 0
        total_keywords = 0
        avg_position = 0
        avg_position_change = 0
        
        try:
            # Query actual content count
            content_count = await supabase_client.query_content_count(org_id)  
            total_content = content_count if content_count else 0
        except:
            total_content = 0
            
        # Get real keyword data from SerpBear focused on primary domain
        try:
            from src.services.serpbear_client import serpbear_client
            from utils.tenant_mapper import TenantOrgMapper
            
            # Get primary domain from SerpBear settings
            serpbear_settings = await TenantOrgMapper.get_serpbear_settings(org_id)
            primary_domain = serpbear_settings.get("connection", {}).get("primary_domain")
            
            if not primary_domain:
                # No primary domain configured - provide helpful message
                logger.info(f"No primary domain configured for organization {org_id}")
                total_keywords = 0
                avg_position = 0
                # Will show message in metrics response
            else:
                # Get keywords only for the primary domain
                keywords = await serpbear_client.get_keywords(primary_domain)
                if keywords:
                    total_keywords = len(keywords)
                    # Calculate average position from current rankings
                    positions = [k.position for k in keywords if k.position and k.position > 0]
                    if positions:
                        avg_position = sum(positions) / len(positions)
                        # Mock position change for now - would need historical data
                        avg_position_change = -0.8
                else:
                    total_keywords = 0
                    avg_position = 0
                            
        except Exception as e:
            logger.warning(f"Failed to get SerpBear data: {e}")
            primary_domain = None
            total_keywords = 0
            avg_position = 0
        
        # Real metrics from SerpBear + mock data for other sources
        metrics = {
            "total_content": total_content,
            "total_keywords": total_keywords,  # Real data from SerpBear
            "organic_traffic": 47200,  # Mock - connect to GSC
            "organic_traffic_change": 23.0,
            "avg_position": round(avg_position, 1) if avg_position > 0 else 0,  # Real data from SerpBear
            "avg_position_change": avg_position_change,
            "content_score": 78,
            "content_score_change": 5.2,
            "active_campaigns": 12,
            "campaigns_change": 2,
            # Period comparison data for metrics cards
            "content_change": 0,  # No content uploaded yet
            "keywords_change": 0,  # No keywords tracked yet  
            "processed_change": 0,  # No processing activity yet
            "avg_seo_score": 0  # No content to score yet
        }
        
        # Get organization details
        org_info = await supabase_client.get_organization(org_id)
        org_name = org_info.get('name', 'Your Organization') if org_info else 'Your Organization'
        
        # Add primary domain context and messaging
        response_data = {
            "success": True,
            "metrics": metrics,
            "organization": org_name,
            "primary_domain": primary_domain if 'primary_domain' in locals() else None,
            "last_updated": datetime.now().isoformat()
        }
        
        # Add helpful messaging based on configuration state
        if not primary_domain:
            response_data["message"] = "Please configure your primary domain in SerpBear settings to see focused SEO metrics"
            response_data["action_needed"] = "set_primary_domain"
        else:
            response_data["message"] = f"Showing SEO metrics for primary domain: {primary_domain}"
            response_data["action_needed"] = None
            
        return response_data
        
    except Exception as e:
        logger.error(f"Dashboard metrics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load metrics: {str(e)}")

# Graph API adapters for frontend compatibility removed - using graph router instead

# Keyword network endpoint now handled by modular graph router

# Competitor landscape endpoint now handled by modular graph router

# Semantic clusters endpoint now handled by modular graph router

if __name__ == "__main__":
    import uvicorn
    
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
        access_log=True
    )
    server = uvicorn.Server(config)
    server.run()