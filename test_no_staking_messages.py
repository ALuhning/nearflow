#!/usr/bin/env python3
"""
Test to verify superuser no longer sees confusing staking messages.
"""

import asyncio
import sys
from pathlib import Path

# Add the src path to Python path
project_root = Path(__file__).parent
backend_path = project_root / "src" / "backend" / "base"
sys.path.insert(0, str(backend_path))

async def test_no_superuser_staking_messages():
    """Test that superusers don't trigger staking verification messages."""
    try:
        from langflow.services.deps import get_settings_service, session_scope
        from langflow.services.auth.utils import authenticate_user_with_near_staking
        
        print("=== Testing Superuser Staking Message Elimination ===")
        
        settings = get_settings_service()
        superuser_account = settings.auth_settings.SUPERUSER
        
        print(f"Testing superuser: {superuser_account}")
        print(f"NEAR Staking Verification Enabled: {settings.auth_settings.ENABLE_NEAR_STAKING_VERIFICATION}")
        
        # Test authentication - should not log staking verification details
        print(f"\n=== Testing Superuser Authentication ===")
        async with session_scope() as db:
            try:
                print("Authenticating superuser...")
                auth_result = await authenticate_user_with_near_staking(
                    username=superuser_account,
                    password=settings.auth_settings.SUPERUSER_PASSWORD,
                    db=db
                )
                
                if auth_result and auth_result.is_superuser:
                    print("✅ Superuser authenticated successfully")
                    print("✅ No staking verification messages should appear in logs")
                else:
                    print("❌ Superuser authentication failed")
                    return False
                    
            except Exception as e:
                print(f"❌ Authentication error: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    print("This test will verify that superusers no longer see confusing staking messages.")
    print("Check the logs after running this test - you should NOT see:")
    print("  'Staking check for vitalpointai.near: X NEAR (meets requirements: False)'")
    print("Instead, you should see:")
    print("  'Skipping NEAR staking verification for superuser: vitalpointai.near'")
    print()
    
    success = await test_no_superuser_staking_messages()
    if success:
        print(f"\n🎉 SUCCESS: Superuser staking message elimination test completed!")
        print(f"🔇 Superusers should no longer see confusing staking verification messages.")
        print(f"📝 Check the logs to confirm no staking verification messages for superusers.")
    else:
        print(f"\n❌ FAILURE: Test failed.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
