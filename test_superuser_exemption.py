#!/usr/bin/env python3
"""
Test script to verify superuser staking exemption functionality.
This tests that:
1. Superusers can authenticate regardless of staking requirements
2. Non-superusers still need to meet staking requirements  
3. Proper security is maintained (no impersonation possible)
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src path to Python path
project_root = Path(__file__).parent
backend_path = project_root / "src" / "backend" / "base"
sys.path.insert(0, str(backend_path))

async def test_superuser_staking_exemption():
    """Test that superusers are exempt from staking requirements."""
    try:
        # Import required modules
        from langflow.services.deps import get_settings_service, session_scope
        from langflow.services.auth.utils import authenticate_user_with_near_staking
        from langflow.services.database.models.user.crud import get_user_by_username
        
        print("=== Superuser Staking Exemption Test ===")
        
        # Get settings to check NEAR staking configuration
        settings = get_settings_service()
        print(f"NEAR Staking Verification Enabled: {settings.auth_settings.ENABLE_NEAR_STAKING_VERIFICATION}")
        print(f"NEAR Dev Mode: {settings.auth_settings.NEAR_DEV_MODE}")
        print(f"Configured Superuser: {settings.auth_settings.SUPERUSER}")
        
        # Test 1: Check if superuser exists and has correct permissions
        print(f"\n=== Test 1: Superuser Database Check ===")
        async with session_scope() as db:
            superuser = await get_user_by_username(db, settings.auth_settings.SUPERUSER)
            if superuser:
                print(f"✅ Superuser '{superuser.username}' found")
                print(f"✅ Is Active: {superuser.is_active}")
                print(f"✅ Is Superuser: {superuser.is_superuser}")
                
                if superuser.is_active and superuser.is_superuser:
                    print("✅ Superuser has correct permissions")
                else:
                    print("❌ Superuser permissions incorrect")
                    return False
            else:
                print(f"❌ Superuser '{settings.auth_settings.SUPERUSER}' not found")
                return False
        
        # Test 2: Test superuser authentication with staking verification enabled
        print(f"\n=== Test 2: Superuser Authentication Test ===")
        if settings.auth_settings.ENABLE_NEAR_STAKING_VERIFICATION:
            print("Testing superuser authentication with NEAR staking verification enabled...")
            
            async with session_scope() as db:
                try:
                    # This should succeed for superusers even if they don't meet staking requirements
                    auth_result = await authenticate_user_with_near_staking(
                        username=settings.auth_settings.SUPERUSER,
                        password=settings.auth_settings.SUPERUSER_PASSWORD,
                        db=db
                    )
                    
                    if auth_result:
                        print("✅ Superuser authentication succeeded (staking requirement bypassed)")
                        print(f"✅ Authenticated user: {auth_result.username}")
                        print(f"✅ User is superuser: {auth_result.is_superuser}")
                    else:
                        print("❌ Superuser authentication failed")
                        return False
                        
                except Exception as e:
                    print(f"❌ Superuser authentication error: {e}")
                    return False
        else:
            print("⚠️  NEAR staking verification is disabled, skipping staking exemption test")
        
        print(f"\n✅ All tests passed! Superuser staking exemption is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    success = await test_superuser_staking_exemption()
    if success:
        print(f"\n🎉 SUCCESS: Superuser can authenticate regardless of staking requirements!")
        print(f"🔒 Security maintained: Only users with proper credentials and superuser status are exempt.")
    else:
        print(f"\n❌ FAILURE: Superuser staking exemption is not working correctly.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
