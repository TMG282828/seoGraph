#!/usr/bin/env python3
"""
Quick test script to verify Supabase service role key is working.
Run this after updating your .env file with the correct service role key.
"""

import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_service_key():
    print("🔑 Testing Supabase Service Role Key")
    print("=" * 50)
    
    # Check environment variables
    url = os.getenv('SUPABASE_URL')
    service_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    
    print(f"📍 Supabase URL: {url}")
    print(f"🔐 Service Key: {'✅ SET' if service_key else '❌ NOT SET'} ({len(service_key or '')} chars)")
    
    if not url or not service_key:
        print("❌ Missing required environment variables")
        return False
    
    try:
        # Test service client
        from supabase import create_client
        
        print("\n🧪 Testing service client connection...")
        service_client = create_client(url, service_key)
        
        # Test basic query
        result = service_client.table('users').select('id, email').limit(1).execute()
        print(f"✅ Basic query: {len(result.data)} user(s) found")
        
        # Test workspace_settings access
        result = service_client.table('workspace_settings').select('workspace_id, workspace_name').execute()
        print(f"✅ Workspace settings: {len(result.data)} record(s) found")
        
        # Test workspace_settings update
        if result.data:
            test_workspace_id = result.data[0]['workspace_id']
            print(f"\n🔧 Testing workspace settings update for: {test_workspace_id}")
            
            update_result = service_client.table('workspace_settings').update({
                'description': 'Test update from service client'
            }).eq('workspace_id', test_workspace_id).execute()
            
            if update_result.data:
                print("✅ Workspace settings update: SUCCESS")
                print(f"   Updated description: {update_result.data[0].get('description', 'N/A')}")
                return True
            else:
                print("❌ Workspace settings update: FAILED")
                return False
        else:
            print("⚠️  No workspace settings found to test update")
            return True
            
    except Exception as e:
        print(f"❌ Service client test failed: {e}")
        return False

async def test_application_client():
    print("\n🏢 Testing application workspace client...")
    
    try:
        from src.database.supabase_client import supabase_client
        
        # Test workspace settings update through application
        success = await supabase_client.update_workspace_settings('twistworld-org', {
            'description': 'Test update through application client',
            'seat_limit': 35
        })
        
        if success:
            print("✅ Application workspace update: SUCCESS")
            
            # Verify the update
            settings = await supabase_client.get_workspace_settings('twistworld-org')
            if settings:
                print(f"   Updated description: {settings.get('description', 'N/A')}")
                print(f"   Updated seat limit: {settings.get('seat_limit', 'N/A')}")
            
            return True
        else:
            print("❌ Application workspace update: FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Application client test failed: {e}")
        return False

async def main():
    print("🚀 Starting Supabase Service Key Test")
    print("=" * 50)
    
    # Test service key directly
    service_ok = await test_service_key()
    
    if service_ok:
        # Test through application
        app_ok = await test_application_client()
        
        if app_ok:
            print("\n🎉 ALL TESTS PASSED!")
            print("✅ Service role key is working correctly")
            print("✅ Workspace settings can be updated")
            print("✅ Your workspace functionality should work in the browser")
        else:
            print("\n⚠️  Service key works, but application update failed")
    else:
        print("\n❌ Service key test failed - please check your key")
        print("👆 Follow the instructions above to get the correct service_role key")

if __name__ == "__main__":
    asyncio.run(main())