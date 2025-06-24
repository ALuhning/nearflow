"""Test script to verify NEAR staking verification with real accounts."""

import asyncio
import sys
import os
sys.path.insert(0, '/home/vitalpointai/projects/nearflow/src/backend/base')

from langflow.services.near.staking import NEARStakingVerifier


async def test_real_near_account():
    """Test with real NEAR accounts to verify the staking verification works."""
    
    # Configure with your settings
    verifier = NEARStakingVerifier(
        rpc_url="https://rpc.mainnet.fastnear.com?apiKey=3c1b8c4dfab4e640040dd3009e1ccec93fcb84409f88ce220aa398750e20edac",
        pool_contract="vitalpoint.pool.near",
        min_stake_amount="25"
    )
    
    # Test accounts - you can replace these with real NEAR account IDs
    test_accounts = [
        "vitalpoint.near",
        "aaron.near", 
        "nonexistent-account-12345.near"  # This should fail
    ]
    
    for account_id in test_accounts:
        print(f"\n--- Testing account: {account_id} ---")
        try:
            result = await verifier.verify_staker(account_id)
            
            print(f"Is staker: {result['is_staker']}")
            print(f"Stake amount: {result['stake_amount']} NEAR")
            print(f"Meets minimum: {result['meets_minimum']}")
            if result['error']:
                print(f"Error: {result['error']}")
                
        except Exception as e:
            print(f"Exception occurred: {e}")


if __name__ == "__main__":
    asyncio.run(test_real_near_account())
