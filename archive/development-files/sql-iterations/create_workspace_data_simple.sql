-- Create Initial Workspace Data (Simple Version)
-- Run this in Supabase SQL Editor to create the workspace data

-- ================================================================
-- 1. CREATE ORGANIZATION (SIMPLE INSERT)
-- ================================================================

-- Create TwistWorld organization (simple insert without conflict handling)
INSERT INTO organizations (id, name, created_at) 
VALUES (
    gen_random_uuid(),
    'TwistWorld',
    NOW()
);

-- ================================================================
-- 2. CREATE WORKSPACE DATA
-- ================================================================

-- Create the workspace
INSERT INTO workspaces (id, name, description, organization_id) 
VALUES (
    'twistworld-org',
    'TwistWorld',
    'Main workspace for TwistWorld SEO Content AI',
    (SELECT id FROM organizations WHERE name = 'TwistWorld' LIMIT 1)
);

-- Create workspace settings
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
    25,
    'member',
    false,
    1,
    0.0
);

-- Associate admin user with workspace
INSERT INTO user_workspaces (user_id, workspace_id, role, is_current)
VALUES (
    (SELECT id FROM users WHERE email = 'admin@twistworld.co.uk' LIMIT 1),
    'twistworld-org',
    'owner',
    true
);

-- ================================================================
-- 3. VERIFICATION
-- ================================================================

-- Verify all data was created
SELECT 'Organizations' as table_name, COUNT(*) as count FROM organizations
UNION ALL
SELECT 'Workspaces', COUNT(*) FROM workspaces
UNION ALL  
SELECT 'Workspace Settings', COUNT(*) FROM workspace_settings
UNION ALL
SELECT 'User Workspaces', COUNT(*) FROM user_workspaces;

-- Show the created data
SELECT 
    w.id as workspace_id,
    w.name as workspace_name,
    o.name as organization_name,
    uw.role as user_role,
    u.email as user_email
FROM workspaces w
LEFT JOIN organizations o ON w.organization_id = o.id  
LEFT JOIN user_workspaces uw ON w.id = uw.workspace_id
LEFT JOIN users u ON uw.user_id = u.id
WHERE w.id = 'twistworld-org';