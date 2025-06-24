#!/usr/bin/env python3
"""
Test script to verify superuser frontend behavior
"""
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_superuser_frontend_flow():
    """Test the complete superuser frontend authentication flow"""
    
    base_url = "http://localhost:7860"
    superuser = os.getenv("LANGFLOW_SUPERUSER", "vitalpointai.near")
    password = os.getenv("LANGFLOW_SUPERUSER_PASSWORD", "supersecure")
    
    print(f"Testing superuser frontend flow for: {superuser}")
    print("=" * 60)
    
    # Step 1: Check if we can create a superuser session
    print("1. Testing superuser login...")
    login_data = {
        "username": superuser,
        "password": password
    }
    
    try:
        response = requests.post(f"{base_url}/api/v1/login", data=login_data)
        print(f"   Login response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            access_token = data.get("access_token")
            print(f"   ✅ Login successful - Got access token")
            
            # Step 2: Get user info with token
            print("\n2. Getting user info...")
            headers = {"Authorization": f"Bearer {access_token}"}
            user_response = requests.get(f"{base_url}/api/v1/users/whoami", headers=headers)
            
            if user_response.status_code == 200:
                user_data = user_response.json()
                print(f"   User ID: {user_data.get('id')}")
                print(f"   Username: {user_data.get('username')}")
                print(f"   Is Superuser: {user_data.get('is_superuser')}")
                print(f"   Is Active: {user_data.get('is_active')}")
                
                if user_data.get('is_superuser'):
                    print("   ✅ User is confirmed as superuser")
                    
                    # Step 3: Test NEAR stake check with superuser token
                    print("\n3. Testing NEAR stake check with superuser token...")
                    stake_response = requests.get(f"{base_url}/api/v1/near-stake-check/{superuser}", headers=headers)
                    print(f"   Stake check status: {stake_response.status_code}")
                    
                    if stake_response.status_code == 200:
                        stake_data = stake_response.json()
                        print(f"   Meets requirements: {stake_data.get('meets_requirements')}")
                        print(f"   Is superuser (from API): {stake_data.get('is_superuser')}")
                        print(f"   Current stake: {stake_data.get('current_stake', 'N/A')}")
                        
                        if stake_data.get('is_superuser') and stake_data.get('meets_requirements'):
                            print("   ✅ Superuser correctly bypasses staking requirements")
                        else:
                            print("   ❌ Superuser is NOT bypassing staking requirements")
                            print(f"      Debug info: {stake_data}")
                    else:
                        print(f"   ❌ Stake check failed: {stake_response.text}")
                else:
                    print("   ❌ User is NOT marked as superuser")
            else:
                print(f"   ❌ Failed to get user info: {user_response.status_code} - {user_response.text}")
                
        else:
            print(f"   ❌ Login failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"   ❌ Error during login test: {e}")
    
    # Step 4: Test unauthenticated NEAR stake check 
    print("\n4. Testing unauthenticated NEAR stake check...")
    try:
        response = requests.get(f"{base_url}/api/v1/near-stake-check/{superuser}")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Meets requirements: {data.get('meets_requirements')}")
            print(f"   Is superuser: {data.get('is_superuser')}")
            if data.get('is_superuser'):
                print("   ✅ Superuser detected even without authentication")
            else:
                print("   ❌ Superuser NOT detected without authentication")
        else:
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Step 5: Check NEAR auth enabled endpoint
    print("\n5. Testing NEAR auth enabled endpoint...")
    try:
        response = requests.get(f"{base_url}/api/v1/near-auth-enabled")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   NEAR auth enabled: {data.get('enabled')}")
            print(f"   Dev mode: {data.get('dev_mode')}")
        else:
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    test_superuser_frontend_flow()
