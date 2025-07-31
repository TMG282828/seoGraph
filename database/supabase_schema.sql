-- SEO Content Knowledge Graph System - Supabase Schema
-- Multi-tenant architecture with organizations and users

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Organizations table
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    subscription_plan VARCHAR(50) DEFAULT 'free',
    subscription_status VARCHAR(50) DEFAULT 'active',
    settings JSONB DEFAULT '{}',
    brand_voice_config JSONB DEFAULT '{
        "tone": "professional",
        "formality": "semi-formal",
        "industry_context": "",
        "prohibited_terms": [],
        "preferred_phrases": [],
        "content_guidelines": {},
        "seo_preferences": {
            "target_keyword_density": 1.5,
            "content_length_preference": "medium",
            "internal_linking_style": "contextual"
        }
    }'::jsonb
);

-- Users table with organization relationship
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    role VARCHAR(50) DEFAULT 'member', -- admin, member, viewer
    display_name VARCHAR(255),
    avatar_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_seen TIMESTAMPTZ DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true,
    user_preferences JSONB DEFAULT '{
        "dashboard_layout": "default",
        "notification_settings": {
            "email": true,
            "browser": false
        },
        "theme": "dark"
    }'::jsonb
);

-- Content Sources table for ingestion configuration
CREATE TABLE content_sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL, -- 'website', 'gdrive', 'upload', 'cms', 'rss'
    config JSONB NOT NULL, -- URLs, API keys, folder paths, crawl settings
    status VARCHAR(50) DEFAULT 'pending', -- pending, active, paused, error
    last_sync TIMESTAMPTZ,
    sync_frequency VARCHAR(50) DEFAULT 'daily', -- hourly, daily, weekly, manual
    total_content_items INTEGER DEFAULT 0,
    last_error TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by UUID REFERENCES users(id)
);

-- Knowledge Base table for graph database references
CREATE TABLE knowledge_base (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    neo4j_database VARCHAR(100) NOT NULL, -- dedicated Neo4j database per org
    qdrant_collection VARCHAR(100) NOT NULL, -- dedicated collection per org
    graph_stats JSONB DEFAULT '{
        "total_nodes": 0,
        "total_relationships": 0,
        "content_nodes": 0,
        "topic_nodes": 0,
        "keyword_nodes": 0,
        "last_analysis": null
    }'::jsonb,
    embedding_model VARCHAR(100) DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Content Items table for tracking processed content
CREATE TABLE content_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    source_id UUID REFERENCES content_sources(id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    content_type VARCHAR(100), -- article, blog_post, page, guide, case_study
    url TEXT,
    content_hash VARCHAR(64) UNIQUE, -- SHA-256 hash for deduplication
    word_count INTEGER,
    seo_score DECIMAL(5,2),
    readability_score DECIMAL(5,2),
    keyword_density JSONB DEFAULT '{}',
    meta_data JSONB DEFAULT '{}', -- meta title, description, tags, etc.
    processing_status VARCHAR(50) DEFAULT 'pending', -- pending, processing, completed, error
    neo4j_node_id VARCHAR(100), -- reference to Neo4j node
    qdrant_point_id VARCHAR(100), -- reference to Qdrant point
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    processed_at TIMESTAMPTZ
);

-- SEO Keywords table for tracking keyword performance
CREATE TABLE seo_keywords (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    keyword VARCHAR(255) NOT NULL,
    search_volume INTEGER,
    competition_score DECIMAL(5,2),
    current_ranking INTEGER,
    target_ranking INTEGER,
    tracking_status VARCHAR(50) DEFAULT 'active', -- active, paused, archived
    last_checked TIMESTAMPTZ,
    ranking_history JSONB DEFAULT '[]',
    related_content_items UUID[] DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Content Generation Tasks table for agent workflow
CREATE TABLE content_generation_tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    created_by UUID REFERENCES users(id),
    task_type VARCHAR(100) NOT NULL, -- gap_analysis, content_generation, seo_optimization
    status VARCHAR(50) DEFAULT 'pending', -- pending, processing, completed, failed, cancelled
    priority VARCHAR(50) DEFAULT 'medium', -- low, medium, high, urgent
    input_data JSONB NOT NULL,
    output_data JSONB DEFAULT '{}',
    agent_assignments JSONB DEFAULT '{}', -- which agents are working on this
    progress_percentage INTEGER DEFAULT 0,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_organizations_slug ON organizations(slug);
CREATE INDEX idx_users_organization_id ON users(organization_id);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_content_sources_organization_id ON content_sources(organization_id);
CREATE INDEX idx_content_sources_status ON content_sources(status);
CREATE INDEX idx_content_items_organization_id ON content_items(organization_id);
CREATE INDEX idx_content_items_source_id ON content_items(source_id);
CREATE INDEX idx_content_items_content_hash ON content_items(content_hash);
CREATE INDEX idx_seo_keywords_organization_id ON seo_keywords(organization_id);
CREATE INDEX idx_seo_keywords_keyword ON seo_keywords(keyword);
CREATE INDEX idx_content_generation_tasks_organization_id ON content_generation_tasks(organization_id);
CREATE INDEX idx_content_generation_tasks_status ON content_generation_tasks(status);

-- Row Level Security (RLS) Policies
ALTER TABLE organizations ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE content_sources ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge_base ENABLE ROW LEVEL SECURITY;
ALTER TABLE content_items ENABLE ROW LEVEL SECURITY;
ALTER TABLE seo_keywords ENABLE ROW LEVEL SECURITY;
ALTER TABLE content_generation_tasks ENABLE ROW LEVEL SECURITY;

-- RLS Policies for multi-tenant security
-- Organizations: Users can only see their own organization
CREATE POLICY "Users can view own organization" ON organizations
    FOR SELECT USING (id IN (
        SELECT organization_id FROM users WHERE auth.uid() = users.id
    ));

CREATE POLICY "Organization admins can update organization" ON organizations
    FOR UPDATE USING (id IN (
        SELECT organization_id FROM users 
        WHERE auth.uid() = users.id AND users.role = 'admin'
    ));

-- Users: Can view users in same organization
CREATE POLICY "Users can view organization members" ON users
    FOR SELECT USING (organization_id IN (
        SELECT organization_id FROM users WHERE auth.uid() = users.id
    ));

CREATE POLICY "Users can update own profile" ON users
    FOR UPDATE USING (auth.uid() = id);

-- Content Sources: Organization-scoped access
CREATE POLICY "Organization members can view content sources" ON content_sources
    FOR SELECT USING (organization_id IN (
        SELECT organization_id FROM users WHERE auth.uid() = users.id
    ));

CREATE POLICY "Organization admins can manage content sources" ON content_sources
    FOR ALL USING (organization_id IN (
        SELECT organization_id FROM users 
        WHERE auth.uid() = users.id AND users.role IN ('admin', 'member')
    ));

-- Knowledge Base: Organization-scoped access
CREATE POLICY "Organization members can view knowledge base" ON knowledge_base
    FOR SELECT USING (organization_id IN (
        SELECT organization_id FROM users WHERE auth.uid() = users.id
    ));

-- Content Items: Organization-scoped access
CREATE POLICY "Organization members can view content items" ON content_items
    FOR SELECT USING (organization_id IN (
        SELECT organization_id FROM users WHERE auth.uid() = users.id
    ));

-- SEO Keywords: Organization-scoped access
CREATE POLICY "Organization members can view seo keywords" ON seo_keywords
    FOR SELECT USING (organization_id IN (
        SELECT organization_id FROM users WHERE auth.uid() = users.id
    ));

-- Content Generation Tasks: Organization-scoped access
CREATE POLICY "Organization members can view content tasks" ON content_generation_tasks
    FOR SELECT USING (organization_id IN (
        SELECT organization_id FROM users WHERE auth.uid() = users.id
    ));

CREATE POLICY "Organization members can create content tasks" ON content_generation_tasks
    FOR INSERT WITH CHECK (organization_id IN (
        SELECT organization_id FROM users WHERE auth.uid() = users.id
    ));

-- Functions for common operations
CREATE OR REPLACE FUNCTION get_user_organization()
RETURNS UUID AS $$
BEGIN
    RETURN (
        SELECT organization_id 
        FROM users 
        WHERE auth.uid() = users.id
        LIMIT 1
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to create organization with admin user
CREATE OR REPLACE FUNCTION create_organization_with_admin(
    org_name TEXT,
    org_slug TEXT,
    admin_email TEXT,
    admin_name TEXT
) RETURNS UUID AS $$
DECLARE
    new_org_id UUID;
    new_user_id UUID;
BEGIN
    -- Create organization
    INSERT INTO organizations (name, slug)
    VALUES (org_name, org_slug)
    RETURNING id INTO new_org_id;
    
    -- Create knowledge base
    INSERT INTO knowledge_base (organization_id, neo4j_database, qdrant_collection)
    VALUES (new_org_id, 'org_' || replace(new_org_id::text, '-', '_'), 'org_' || replace(new_org_id::text, '-', '_'));
    
    -- Create admin user
    INSERT INTO users (email, organization_id, role, display_name)
    VALUES (admin_email, new_org_id, 'admin', admin_name)
    RETURNING id INTO new_user_id;
    
    RETURN new_org_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_organizations_updated_at BEFORE UPDATE ON organizations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_content_sources_updated_at BEFORE UPDATE ON content_sources
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_knowledge_base_updated_at BEFORE UPDATE ON knowledge_base
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_content_items_updated_at BEFORE UPDATE ON content_items
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_seo_keywords_updated_at BEFORE UPDATE ON seo_keywords
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_content_generation_tasks_updated_at BEFORE UPDATE ON content_generation_tasks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();