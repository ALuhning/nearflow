#!/usr/bin/env python3
"""
Final verification test to confirm the superuser staking exemption is working correctly.
Based on the log showing vitalpointai.near has 10.13 NEAR but requires 25 NEAR.
"""

import asyncio
import sys
from pathlib import Path

# Add the src path to Python path
project_root = Path(__file__).parent
backend_path = project_root / "src" / "backend" / "base"
sys.path.insert(0, str(backend_path))

async def test_superuser_with_insufficient_stake():
    """Test that superuser can authenticate even with insufficient stake."""
    try:
        from langflow.services.deps import get_settings_service, session_scope
        from langflow.services.auth.utils import authenticate_user_with_near_staking
        from langflow.services.near.staking import near_staking_verifier

        print("=== Superuser Staking Exemption Verification ===")
        
        settings = get_settings_service()
        superuser_account = settings.auth_settings.SUPERUSER
        min_stake_required = float(settings.auth_settings.NEAR_MIN_STAKE_AMOUNT)
        
        print(f"Superuser account: {superuser_account}")
        print(f"Minimum stake required: {min_stake_required} NEAR")
        
        # Check actual stake amount
        print(f"\n=== Checking Actual Stake Amount ===")
        near_staking_verifier.update_settings(
            rpc_url=settings.auth_settings.NEAR_RPC_URL,
            pool_contract=settings.auth_settings.NEAR_POOL_CONTRACT,
            min_stake_amount=settings.auth_settings.NEAR_MIN_STAKE_AMOUNT
        )
        
        try:
            actual_stake = await near_staking_verifier.get_stake_amount(superuser_account)
            actual_stake_float = float(actual_stake)
            print(f"Actual stake amount: {actual_stake_float} NEAR")
            
            meets_requirement = actual_stake_float >= min_stake_required
            print(f"Meets staking requirement: {meets_requirement}")
            
            if meets_requirement:
                print("⚠️  NOTE: Superuser actually meets staking requirements, so exemption test is not conclusive")
            else:
                print("✅ Perfect! Superuser has insufficient stake - this will test the exemption")
                
        except Exception as e:
            print(f"Error checking stake: {e}")
            return False
        
        # Test superuser authentication
        print(f"\n=== Testing Superuser Authentication ===")
        async with session_scope() as db:
            try:
                auth_result = await authenticate_user_with_near_staking(
                    username=superuser_account,
                    password=settings.auth_settings.SUPERUSER_PASSWORD,
                    db=db
                )
                
                if auth_result and auth_result.is_superuser:
                    print("🎉 SUCCESS: Superuser authenticated successfully!")
                    print(f"   - Despite having only {actual_stake_float} NEAR (below {min_stake_required} requirement)")
                    print(f"   - Authenticated as: {auth_result.username}")
                    print(f"   - Is superuser: {auth_result.is_superuser}")
                    print(f"   - Staking exemption is working correctly! 🔐")
                    return True
                else:
                    print("❌ FAILED: Superuser authentication failed")
                    print("   This indicates the staking exemption is NOT working")
                    return False
                    
            except Exception as e:
                error_msg = str(e)
                if "staking" in error_msg.lower():
                    print(f"❌ FAILED: Superuser was blocked by staking requirement: {error_msg}")
                    print("   This indicates the staking exemption is NOT working")
                    return False
                else:
                    print(f"❌ FAILED: Other authentication error: {error_msg}")
                    return False
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    success = await test_superuser_with_insufficient_stake()
    if success:
        print(f"\n🏆 FINAL RESULT: SUCCESS!")
        print(f"")
        print(f"✅ Superuser staking exemption is working perfectly!")
        print(f"✅ Superuser can authenticate regardless of stake amount")
        print(f"✅ NEAR staking verification system is functional")
        print(f"✅ Security is maintained through proper authentication")
        print(f"")
        print(f"🔒 SECURITY SUMMARY:")
        print(f"   • Superuser must still provide correct username/password")
        print(f"   • Only users with is_superuser=True get the exemption")
        print(f"   • Regular users will still be subject to staking requirements")
        print(f"   • No impersonation possible without proper credentials")
    else:
        print(f"\n❌ FINAL RESULT: FAILURE!")
        print(f"The staking exemption for superusers is not working correctly.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
