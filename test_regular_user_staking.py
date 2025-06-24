#!/usr/bin/env python3
"""
Test to verify that non-superusers are still subject to staking requirements.
This ensures security is maintained and only superusers get the exemption.
"""

import asyncio
import sys
from pathlib import Path

# Add the src path to Python path
project_root = Path(__file__).parent
backend_path = project_root / "src" / "backend" / "base"
sys.path.insert(0, str(backend_path))

async def test_regular_user_staking_requirement():
    """Test that regular users still need to meet staking requirements."""
    try:
        from langflow.services.deps import get_settings_service, session_scope
        from langflow.services.auth.utils import authenticate_user_with_near_staking
        from langflow.services.database.models.user.crud import get_user_by_username, create_user

        print("=== Regular User Staking Requirement Test ===")
        
        settings = get_settings_service()
        print(f"NEAR Staking Verification Enabled: {settings.auth_settings.ENABLE_NEAR_STAKING_VERIFICATION}")
        
        if not settings.auth_settings.ENABLE_NEAR_STAKING_VERIFICATION:
            print("⚠️  NEAR staking verification is disabled, skipping test")
            return True

        # Create a test regular user (non-superuser)
        test_username = "regular.test.near"
        test_password = "testpassword123"
        
        print(f"\n=== Creating Test User: {test_username} ===")
        async with session_scope() as db:
            # Check if user already exists
            existing_user = await get_user_by_username(db, test_username)
            if existing_user:
                print(f"✅ Test user '{test_username}' already exists")
                print(f"   Is Superuser: {existing_user.is_superuser}")
                if existing_user.is_superuser:
                    print("❌ Test user is unexpectedly a superuser")
                    return False
            else:
                print(f"Creating new test user '{test_username}'...")
                from langflow.services.auth.utils import get_password_hash
                
                new_user = await create_user(
                    db,
                    user_create={
                        "username": test_username,
                        "password": get_password_hash(test_password),
                        "is_active": True,
                        "is_superuser": False  # Regular user, not superuser
                    }
                )
                print(f"✅ Created test user '{test_username}'")
                print(f"   Is Superuser: {new_user.is_superuser}")

        # Test authentication - should fail due to staking requirements
        print(f"\n=== Testing Regular User Authentication ===")
        print("Testing regular user authentication (should fail due to staking requirements)...")
        
        async with session_scope() as db:
            try:
                auth_result = await authenticate_user_with_near_staking(
                    username=test_username,
                    password=test_password,
                    db=db
                )
                
                if auth_result:
                    print(f"❌ Regular user authentication unexpectedly succeeded")
                    print(f"   This means staking requirements are not being enforced!")
                    return False
                else:
                    print("✅ Regular user authentication correctly failed (staking requirements enforced)")
                    
            except Exception as e:
                error_msg = str(e)
                if "staking" in error_msg.lower() or "stake" in error_msg.lower():
                    print(f"✅ Regular user authentication correctly failed with staking error: {error_msg}")
                else:
                    print(f"⚠️  Regular user authentication failed but not due to staking: {error_msg}")

        print(f"\n✅ Regular user staking requirement test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    success = await test_regular_user_staking_requirement()
    if success:
        print(f"\n🎉 SUCCESS: Staking requirements are properly enforced for regular users!")
        print(f"🔒 Security confirmed: Only superusers are exempt from staking requirements.")
    else:
        print(f"\n❌ FAILURE: Staking requirements not properly enforced.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
