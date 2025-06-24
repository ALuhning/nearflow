#!/usr/bin/env python3
"""
Test script to verify NEAR RPC key queries are working correctly
"""
import asyncio
import httpx
import json

async def test_near_key_query(account_id: str = "aaron.near"):
    """Test querying NEAR account keys"""
    rpc_url = "https://rpc.mainnet.near.org"
    
    async with httpx.AsyncClient() as client:
        # Test the query format we're using
        rpc_payload = {
            "jsonrpc": "2.0",
            "method": "query",
            "params": {
                "request_type": "view_access_key_list",
                "finality": "final",
                "account_id": account_id
            },
            "id": 1
        }
        
        print(f"Testing RPC query for account: {account_id}")
        print(f"RPC URL: {rpc_url}")
        print(f"Payload: {json.dumps(rpc_payload, indent=2)}")
        
        try:
            response = await client.post(
                rpc_url,
                json=rpc_payload,
                headers={"Content-Type": "application/json"},
                timeout=10.0
            )
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Response: {json.dumps(data, indent=2)}")
                
                if "result" in data and "keys" in data["result"]:
                    keys = data["result"]["keys"]
                    print(f"\nFound {len(keys)} keys for {account_id}:")
                    for i, key_info in enumerate(keys):
                        public_key = key_info["public_key"]
                        permission = key_info["access_key"]["permission"]
                        print(f"  {i+1}. {public_key} - {permission}")
                else:
                    print("No keys found in response")
            else:
                print(f"Error response: {response.text}")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_near_key_query())
