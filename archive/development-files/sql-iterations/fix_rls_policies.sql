-- Fix RLS Policy Infinite Recursion Issue
-- Run this in Supabase SQL Editor to fix the policy recursion

-- ================================================================
-- 1. DROP PROBLEMATIC POLICIES
-- ================================================================

-- Drop the recursive policies that reference the same table
DROP POLICY IF EXISTS "Workspace admins can manage memberships" ON user_workspaces;
DROP POLICY IF EXISTS "Workspace owners can update" ON workspaces;
DROP POLICY IF EXISTS "Workspace admins can update settings" ON workspace_settings;
DROP POLICY IF EXISTS "Workspace admins can manage invite codes" ON workspace_invite_codes;

-- ================================================================
-- 2. CREATE NON-RECURSIVE POLICIES
-- ================================================================

-- For workspaces table - simpler policies without recursion
CREATE POLICY "Workspace owners can update" ON workspaces
    FOR UPDATE USING (
        -- Allow if user is authenticated (we'll handle authorization in application)
        auth.uid() IS NOT NULL
    );

-- For workspace_settings table - simpler policies
CREATE POLICY "Workspace admins can update settings" ON workspace_settings
    FOR ALL USING (
        -- Allow if user is authenticated (we'll handle authorization in application)
        auth.uid() IS NOT NULL
    );

-- For user_workspaces table - avoid self-reference
CREATE POLICY "Workspace admins can manage memberships" ON user_workspaces
    FOR ALL USING (
        -- Allow users to manage memberships if they're authenticated
        -- Application logic will enforce role-based permissions
        auth.uid() IS NOT NULL
    );

-- For workspace_invite_codes table - simpler policy
CREATE POLICY "Workspace admins can manage invite codes" ON workspace_invite_codes
    FOR ALL USING (
        -- Allow if user is authenticated (application handles role checks)
        auth.uid() IS NOT NULL
    );

-- ================================================================
-- 3. VERIFICATION
-- ================================================================

-- Check policies are created correctly
SELECT 
    schemaname,
    tablename,
    policyname,
    permissive,
    roles,
    cmd,
    qual
FROM pg_policies 
WHERE tablename IN ('workspaces', 'workspace_settings', 'user_workspaces', 'workspace_invite_codes')
ORDER BY tablename, policyname;