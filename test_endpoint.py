#!/usr/bin/env python3
"""
Test the /near-stake-check endpoint specifically for superuser exemption.
"""

import asyncio
import aiohttp
import sys

async def test_stake_check_endpoint():
    """Test the stake checking endpoint for superuser exemption."""
    try:
        print("=== Testing /near-stake-check endpoint ===")
        
        # Test the superuser account
        superuser_account = "vitalpointai.near"
        regular_account = "nonexistent.test.near"
        
        base_url = "http://localhost:7860"
        
        async with aiohttp.ClientSession() as session:
            # Test 1: Check superuser account
            print(f"\n=== Test 1: Checking superuser account: {superuser_account} ===")
            async with session.get(f"{base_url}/api/v1/login/near-stake-check/{superuser_account}") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Response received: {data}")
                    
                    if data.get("meets_requirements") == True and data.get("superuser") == True:
                        print("✅ Superuser correctly identified and exempt from staking")
                    else:
                        print("❌ Superuser exemption not working correctly")
                        return False
                else:
                    print(f"❌ Request failed with status: {response.status}")
                    error_text = await response.text()
                    print(f"Error: {error_text}")
                    return False
            
            # Test 2: Check regular account
            print(f"\n=== Test 2: Checking regular account: {regular_account} ===")
            async with session.get(f"{base_url}/api/v1/login/near-stake-check/{regular_account}") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Response received: {data}")
                    
                    if data.get("meets_requirements") == False and not data.get("superuser"):
                        print("✅ Regular account correctly subject to staking requirements")
                    else:
                        print("❌ Regular account staking check not working correctly")
                        return False
                else:
                    print(f"❌ Request failed with status: {response.status}")
                    error_text = await response.text()
                    print(f"Error: {error_text}")
                    return False
        
        print(f"\n✅ All endpoint tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    success = await test_stake_check_endpoint()
    if success:
        print(f"\n🎉 SUCCESS: /near-stake-check endpoint working correctly!")
        print(f"🔒 Superuser exemption is properly implemented in the API endpoint.")
    else:
        print(f"\n❌ FAILURE: Endpoint test failed.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
