#!/usr/bin/env python3

import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_frontend_superuser_flow():
    """Test the frontend superuser flow with NEAR staking exemption."""
    base_url = "http://localhost:7860"
    
    print("Testing superuser frontend flow for: vitalpointai.near")
    print("=" * 60)
    
    # Step 1: Login as superuser
    print("1. Testing superuser login...")
    login_response = requests.post(f"{base_url}/api/v1/login/login", data={
        "username": "vitalpointai.near", 
        "password": "supersecure"
    })
    print(f"   Login response status: {login_response.status_code}")
    
    if login_response.status_code == 200:
        access_token = login_response.json().get("access_token")
        print("   ✅ Login successful - Got access token")
        
        # Step 2: Get user info
        print("\n2. Getting user info...")
        headers = {"Authorization": f"Bearer {access_token}"}
        user_response = requests.get(f"{base_url}/api/v1/auto_login", headers=headers)
        
        if user_response.status_code == 200:
            user_data = user_response.json()
            print(f"   User ID: {user_data.get('id')}")
            print(f"   Username: {user_data.get('username')}")
            print(f"   Is Superuser: {user_data.get('is_superuser')}")
            print(f"   Is Active: {user_data.get('is_active')}")
            print("   ✅ User is confirmed as superuser")
            
            # Step 3: Test NEAR stake check with superuser token
            print("\n3. Testing NEAR stake check with superuser token...")
            stake_response = requests.get(
                f"{base_url}/api/v1/login/near-stake-check/vitalpointai.near", 
                headers=headers
            )
            print(f"   Stake check status: {stake_response.status_code}")
            
            if stake_response.status_code == 200:
                stake_data = stake_response.json()
                print(f"   Meets requirements: {stake_data.get('meets_requirements')}")
                print(f"   Is superuser (from API): {stake_data.get('superuser')}")
                print(f"   Current stake: {stake_data.get('current_stake')}")
                
                if stake_data.get('meets_requirements') and stake_data.get('superuser'):
                    print("   ✅ Superuser is properly bypassing staking requirements")
                    print(f"      Superuser flag: {stake_data.get('superuser')}")
                    print(f"      User ID: {stake_data.get('user_id')}")
                else:
                    print("   ❌ Superuser is NOT bypassing staking requirements")
                    print(f"      Debug info: {stake_data}")
            else:
                print(f"   ❌ Stake check failed: {stake_response.text}")
        else:
            print(f"   ❌ Failed to get user info: {user_response.text}")
    else:
        print(f"   ❌ Login failed: {login_response.text}")
        
    # Step 4: Test unauthenticated NEAR stake check
    print("\n4. Testing unauthenticated NEAR stake check...")
    unauth_response = requests.get(f"{base_url}/api/v1/login/near-stake-check/vitalpointai.near")
    print(f"   Status: {unauth_response.status_code}")
    if unauth_response.status_code == 200:
        unauth_data = unauth_response.json()
        print(f"   Meets requirements: {unauth_data.get('meets_requirements')}")
        print(f"   Is superuser: {unauth_data.get('superuser')}")
        
        # This should use the fallback check (account_id == SUPERUSER)
        if unauth_data.get('meets_requirements') and unauth_data.get('superuser'):
            print("   ✅ Superuser detected via fallback mechanism")
        else:
            print("   ❌ Superuser NOT detected without authentication")
    else:
        print(f"   ❌ Unauthenticated stake check failed: {unauth_response.text}")
        
    # Step 5: Test NEAR auth enabled endpoint
    print("\n5. Testing NEAR auth enabled endpoint...")
    enabled_response = requests.get(f"{base_url}/api/v1/login/near-auth-enabled")
    print(f"   Status: {enabled_response.status_code}")
    if enabled_response.status_code == 200:
        enabled_data = enabled_response.json()
        print(f"   NEAR auth enabled: {enabled_data.get('enabled')}")
        print(f"   Dev mode: {enabled_data.get('dev_mode')}")

if __name__ == "__main__":
    test_frontend_superuser_flow()
