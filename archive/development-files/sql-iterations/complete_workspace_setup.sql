-- Complete Workspace Setup (Handle Existing Data)
-- Run this in Supabase SQL Editor to complete the workspace setup

-- ================================================================
-- 1. CHECK AND CREATE ORGANIZATION IF NEEDED
-- ================================================================

-- Only create TwistWorld organization if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM organizations WHERE name = 'TwistWorld') THEN
        INSERT INTO organizations (id, name, slug, created_at) 
        VALUES (
            gen_random_uuid(),
            'TwistWorld',
            'twistworld',
            NOW()
        );
    END IF;
END $$;

-- ================================================================
-- 2. CREATE MISSING WORKSPACE DATA
-- ================================================================

-- Create workspace settings if missing
INSERT INTO workspace_settings (
    workspace_id,
    workspace_name,
    description,
    seat_limit,
    default_member_role,
    auto_approve_invites,
    current_seats_used,
    storage_used_gb
) 
SELECT 
    'twistworld-org',
    'TwistWorld',
    'Main workspace for TwistWorld SEO Content AI',
    25,
    'member',
    false,
    1,
    0.0
WHERE NOT EXISTS (
    SELECT 1 FROM workspace_settings WHERE workspace_id = 'twistworld-org'
);

-- Associate admin user with workspace if not already associated
INSERT INTO user_workspaces (user_id, workspace_id, role, is_current)
SELECT 
    u.id,
    'twistworld-org',
    'owner',
    true
FROM users u
WHERE u.email = 'admin@twistworld.co.uk'
AND NOT EXISTS (
    SELECT 1 FROM user_workspaces uw 
    WHERE uw.user_id = u.id AND uw.workspace_id = 'twistworld-org'
);

-- Update workspace organization_id if needed
UPDATE workspaces 
SET organization_id = (SELECT id FROM organizations WHERE name = 'TwistWorld' LIMIT 1)
WHERE id = 'twistworld-org' 
AND organization_id IS NULL;

-- ================================================================
-- 3. VERIFICATION
-- ================================================================

-- Show current state
SELECT 'FINAL STATUS' as section, '=================' as details;

-- Verify all data exists
SELECT 'Organizations' as table_name, COUNT(*) as count FROM organizations
UNION ALL
SELECT 'Workspaces', COUNT(*) FROM workspaces
UNION ALL  
SELECT 'Workspace Settings', COUNT(*) FROM workspace_settings
UNION ALL
SELECT 'User Workspaces', COUNT(*) FROM user_workspaces;

-- Show the complete workspace setup
SELECT 
    'WORKSPACE DETAILS' as section,
    w.id as workspace_id,
    w.name as workspace_name,
    o.name as organization_name,
    o.slug as organization_slug,
    uw.role as user_role,
    u.email as user_email,
    ws.seat_limit,
    ws.current_seats_used
FROM workspaces w
LEFT JOIN organizations o ON w.organization_id = o.id  
LEFT JOIN user_workspaces uw ON w.id = uw.workspace_id
LEFT JOIN users u ON uw.user_id = u.id
LEFT JOIN workspace_settings ws ON w.id = ws.workspace_id
WHERE w.id = 'twistworld-org';

-- Final status check
SELECT 
    CASE 
        WHEN COUNT(*) > 0 THEN '✅ SUCCESS: Workspace setup complete!'
        ELSE '❌ ERROR: Workspace setup failed'
    END as status
FROM workspaces w
JOIN workspace_settings ws ON w.id = ws.workspace_id
JOIN user_workspaces uw ON w.id = uw.workspace_id
WHERE w.id = 'twistworld-org';