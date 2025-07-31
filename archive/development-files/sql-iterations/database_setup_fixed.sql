-- Supabase Database Setup for Workspace Functionality (Type-Safe Version)
-- Run these commands in your Supabase SQL Editor

-- ================================================================
-- 1. CREATE WORKSPACE TABLES (CLEAN VERSION)
-- ================================================================

-- Workspaces table - stores workspace metadata
CREATE TABLE IF NOT EXISTS workspaces (
    id TEXT PRIMARY KEY,  -- Using TEXT instead of VARCHAR for consistency
    name TEXT NOT NULL,
    description TEXT,
    avatar_url TEXT,
    organization_id UUID,  -- References organizations.id (UUID)
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Workspace settings table - stores configuration
CREATE TABLE IF NOT EXISTS workspace_settings (
    workspace_id TEXT PRIMARY KEY,  -- References workspaces.id (TEXT)
    workspace_name TEXT,
    description TEXT,
    avatar_url TEXT,
    seat_limit INTEGER DEFAULT 5 CHECK (seat_limit > 0),
    default_member_role TEXT DEFAULT 'member',
    auto_approve_invites BOOLEAN DEFAULT false,
    current_seats_used INTEGER DEFAULT 1 CHECK (current_seats_used >= 0),
    storage_used_gb DECIMAL(10,2) DEFAULT 0.0 CHECK (storage_used_gb >= 0),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
);

-- User workspace associations - many-to-many relationship
CREATE TABLE IF NOT EXISTS user_workspaces (
    id SERIAL PRIMARY KEY,
    user_id UUID NOT NULL,        -- References users.id (UUID)
    workspace_id TEXT NOT NULL,   -- References workspaces.id (TEXT)
    role TEXT DEFAULT 'member',
    is_current BOOLEAN DEFAULT false,
    joined_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, workspace_id),
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
);

-- Workspace invite codes for team management
CREATE TABLE IF NOT EXISTS workspace_invite_codes (
    code TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,   -- References workspaces.id (TEXT)
    created_by UUID,             -- References users.id (UUID)
    max_uses INTEGER DEFAULT 10 CHECK (max_uses > 0),
    current_uses INTEGER DEFAULT 0 CHECK (current_uses >= 0),
    expires_at TIMESTAMPTZ NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
    CHECK (current_uses <= max_uses)
);

-- ================================================================
-- 2. CREATE INDEXES FOR PERFORMANCE
-- ================================================================

CREATE INDEX IF NOT EXISTS idx_workspaces_organization_id ON workspaces(organization_id);
CREATE INDEX IF NOT EXISTS idx_user_workspaces_user_id ON user_workspaces(user_id);
CREATE INDEX IF NOT EXISTS idx_user_workspaces_workspace_id ON user_workspaces(workspace_id);
CREATE INDEX IF NOT EXISTS idx_user_workspaces_is_current ON user_workspaces(user_id, is_current);
CREATE INDEX IF NOT EXISTS idx_invite_codes_workspace_id ON workspace_invite_codes(workspace_id);
CREATE INDEX IF NOT EXISTS idx_invite_codes_active ON workspace_invite_codes(workspace_id, is_active);
CREATE INDEX IF NOT EXISTS idx_invite_codes_expires ON workspace_invite_codes(expires_at);

-- ================================================================
-- 3. SET UP ROW LEVEL SECURITY (RLS) - TYPE SAFE
-- ================================================================

-- Enable RLS on all tables
ALTER TABLE workspaces ENABLE ROW LEVEL SECURITY;
ALTER TABLE workspace_settings ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_workspaces ENABLE ROW LEVEL SECURITY;
ALTER TABLE workspace_invite_codes ENABLE ROW LEVEL SECURITY;

-- RLS Policies for workspaces table
CREATE POLICY "Users can view workspaces they belong to" ON workspaces
    FOR SELECT USING (
        id IN (
            SELECT workspace_id FROM user_workspaces 
            WHERE user_id = auth.uid()
        )
    );

CREATE POLICY "Users can insert workspaces" ON workspaces
    FOR INSERT WITH CHECK (true);

CREATE POLICY "Workspace owners can update" ON workspaces
    FOR UPDATE USING (
        id IN (
            SELECT workspace_id FROM user_workspaces 
            WHERE user_id = auth.uid() AND role IN ('owner', 'admin')
        )
    );

-- RLS Policies for workspace_settings table
CREATE POLICY "Users can view settings for their workspaces" ON workspace_settings
    FOR SELECT USING (
        workspace_id IN (
            SELECT workspace_id FROM user_workspaces 
            WHERE user_id = auth.uid()
        )
    );

CREATE POLICY "Workspace admins can update settings" ON workspace_settings
    FOR ALL USING (
        workspace_id IN (
            SELECT workspace_id FROM user_workspaces 
            WHERE user_id = auth.uid() AND role IN ('owner', 'admin')
        )
    );

-- RLS Policies for user_workspaces table
CREATE POLICY "Users can view their workspace memberships" ON user_workspaces
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY "Users can insert their own memberships" ON user_workspaces
    FOR INSERT WITH CHECK (user_id = auth.uid());

CREATE POLICY "Workspace admins can manage memberships" ON user_workspaces
    FOR ALL USING (
        workspace_id IN (
            SELECT workspace_id FROM user_workspaces 
            WHERE user_id = auth.uid() AND role IN ('owner', 'admin')
        )
    );

-- RLS Policies for workspace_invite_codes table
CREATE POLICY "Users can view invite codes for their workspaces" ON workspace_invite_codes
    FOR SELECT USING (
        workspace_id IN (
            SELECT workspace_id FROM user_workspaces 
            WHERE user_id = auth.uid()
        )
    );

CREATE POLICY "Workspace admins can manage invite codes" ON workspace_invite_codes
    FOR ALL USING (
        workspace_id IN (
            SELECT workspace_id FROM user_workspaces 
            WHERE user_id = auth.uid() AND role IN ('owner', 'admin')
        )
    );

-- ================================================================
-- 4. CREATE INITIAL DATA FOR TWISTWORLD ORGANIZATION
-- ================================================================

-- Insert default workspace for TwistWorld organization
INSERT INTO workspaces (id, name, description, organization_id) 
VALUES (
    'twistworld-org',
    'TwistWorld',
    'Main workspace for TwistWorld SEO Content AI',
    (SELECT id FROM organizations WHERE name ILIKE '%twistworld%' OR name ILIKE '%twist%' LIMIT 1)
) ON CONFLICT (id) DO NOTHING;

-- Create default workspace settings
INSERT INTO workspace_settings (
    workspace_id,
    workspace_name,
    description,
    seat_limit,
    default_member_role,
    auto_approve_invites,
    current_seats_used,
    storage_used_gb
) VALUES (
    'twistworld-org',
    'TwistWorld',
    'Main workspace for TwistWorld SEO Content AI',
    25,  -- Higher limit for main workspace
    'member',
    false,
    1,   -- Admin user counts as 1
    0.0
) ON CONFLICT (workspace_id) DO NOTHING;

-- Associate admin@twistworld.co.uk with the workspace as owner
INSERT INTO user_workspaces (user_id, workspace_id, role, is_current)
VALUES (
    (SELECT id FROM users WHERE email = 'admin@twistworld.co.uk' LIMIT 1),
    'twistworld-org',
    'owner',
    true
) ON CONFLICT (user_id, workspace_id) DO NOTHING;

-- ================================================================
-- 5. CREATE HELPER FUNCTIONS (OPTIONAL)
-- ================================================================

-- Function to automatically set updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at columns
DROP TRIGGER IF EXISTS update_workspaces_updated_at ON workspaces;
CREATE TRIGGER update_workspaces_updated_at 
    BEFORE UPDATE ON workspaces 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_workspace_settings_updated_at ON workspace_settings;
CREATE TRIGGER update_workspace_settings_updated_at 
    BEFORE UPDATE ON workspace_settings 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ================================================================
-- 6. VERIFICATION QUERIES (TYPE SAFE)
-- ================================================================

-- Verify tables were created
SELECT schemaname, tablename 
FROM pg_tables 
WHERE tablename IN ('workspaces', 'workspace_settings', 'user_workspaces', 'workspace_invite_codes')
ORDER BY tablename;

-- Verify admin user has workspace access (with proper type handling)
SELECT 
    u.email,
    w.name as workspace_name,
    uw.role,
    uw.is_current
FROM users u
JOIN user_workspaces uw ON u.id = uw.user_id
JOIN workspaces w ON uw.workspace_id = w.id
WHERE u.email = 'admin@twistworld.co.uk';

-- Show workspace settings
SELECT * FROM workspace_settings WHERE workspace_id = 'twistworld-org';

-- Verify data types are correct
SELECT 
    table_name,
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns 
WHERE table_name IN ('workspaces', 'workspace_settings', 'user_workspaces', 'workspace_invite_codes')
AND column_name IN ('id', 'user_id', 'workspace_id', 'organization_id')
ORDER BY table_name, column_name;

-- ================================================================
-- SETUP COMPLETE!
-- ================================================================

-- After running this script:
-- 1. Your workspace tables will be created with consistent TEXT/UUID types
-- 2. RLS policies will secure access to workspace data  
-- 3. admin@twistworld.co.uk will be set up as owner of 'twistworld-org' workspace
-- 4. Automatic triggers will handle timestamps
-- 5. Your workspace functionality should work without type errors