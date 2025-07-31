-- Temporary fix: Disable RLS for workspace testing
-- Run this if you want to test workspace functionality immediately
-- (You can re-enable RLS later when we fix the policies properly)

-- Disable RLS temporarily
ALTER TABLE workspaces DISABLE ROW LEVEL SECURITY;
ALTER TABLE workspace_settings DISABLE ROW LEVEL SECURITY;
ALTER TABLE user_workspaces DISABLE ROW LEVEL SECURITY;
ALTER TABLE workspace_invite_codes DISABLE ROW LEVEL SECURITY;

-- Test your workspace functionality now - should work without 500 errors

-- TO RE-ENABLE LATER (after fixing policies):
-- ALTER TABLE workspaces ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE workspace_settings ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE user_workspaces ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE workspace_invite_codes ENABLE ROW LEVEL SECURITY;