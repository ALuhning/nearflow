#!/usr/bin/env python3

import asyncio
import httpx
import json

async def test_key_detection(account_id: str):
    """Test that our key detection works for both FullAccess and FunctionCall keys"""
    
    rpc_url = "https://rpc.mainnet.near.org"
    
    print(f"Testing key detection for account: {account_id}")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            rpc_url,
            json={
                "jsonrpc": "2.0",
                "method": "query",
                "params": {
                    "request_type": "view_access_key_list",
                    "finality": "final",
                    "account_id": account_id
                },
                "id": 1
            },
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            print(f"RPC request failed with status {response.status_code}")
            return
        
        data = response.json()
        
        if "result" not in data or "keys" not in data["result"]:
            print(f"No keys found for account {account_id}")
            return
        
        print(f"\nFound {len(data['result']['keys'])} keys for {account_id}:")
        
        full_access_keys = []
        function_call_keys = []
        
        for key_info in data["result"]["keys"]:
            public_key = key_info["public_key"]
            permission = key_info["access_key"]["permission"]
            
            if permission == "FullAccess":
                full_access_keys.append(public_key)
                print(f"  ✅ FullAccess: {public_key}")
            elif isinstance(permission, dict) and "FunctionCall" in permission:
                function_call_keys.append(public_key)
                print(f"  🔧 FunctionCall: {public_key}")
                print(f"      Allowed methods: {permission['FunctionCall'].get('method_names', 'Any')}")
                print(f"      Receiver: {permission['FunctionCall'].get('receiver_id', 'Any')}")
            else:
                print(f"  ❓ Unknown: {public_key} (permission: {permission})")
        
        print(f"\nSummary for {account_id}:")
        print(f"  FullAccess keys: {len(full_access_keys)}")
        print(f"  FunctionCall keys: {len(function_call_keys)}")
        print(f"  Total usable keys for hardware wallet auth: {len(full_access_keys) + len(function_call_keys)}")
        
        return {
            "full_access": full_access_keys,
            "function_call": function_call_keys
        }

async def main():
    # Test with a few different account types
    test_accounts = [
        "vitalpoint.near",  # Regular account
        "wrap.near",        # Contract account
        "aurora",           # Top-level account
    ]
    
    for account in test_accounts:
        try:
            await test_key_detection(account)
            print("\n" + "="*80 + "\n")
        except Exception as e:
            print(f"Error testing {account}: {e}")
            print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
