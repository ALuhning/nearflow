#!/usr/bin/env python3
"""
Test script to verify that superuser frontend skips staking checks.
"""
import os
import sys
import requests
import json
from typing import Dict, Any

# Add the backend directory to Python path
sys.path.insert(0, '/home/vitalpointai/projects/nearflow/src/backend/base')

from langflow.services.database.models.user.model import User
from langflow.services.settings import settings
from langflow.initial_setup.setup import initialize_superuser
from langflow.services.deps import get_session
from sqlalchemy.orm import Session

def test_superuser_setup():
    """Test that superuser is properly set up"""
    print("=== Testing Superuser Setup ===")
    
    # Get a database session
    session = next(get_session())
    
    try:
        # Check if superuser exists
        superuser = session.query(User).filter(User.username == settings.auth_settings.SUPERUSER).first()
        
        if superuser:
            print(f"✓ Superuser '{superuser.username}' exists")
            print(f"  - Active: {superuser.is_active}")
            print(f"  - Superuser: {superuser.is_superuser}")
            print(f"  - ID: {superuser.id}")
            
            if superuser.is_active and superuser.is_superuser:
                print("✓ Superuser is properly configured")
                return True
            else:
                print("✗ Superuser is not properly configured")
                return False
        else:
            print(f"✗ Superuser '{settings.auth_settings.SUPERUSER}' not found")
            return False
            
    except Exception as e:
        print(f"✗ Error checking superuser: {e}")
        return False
    finally:
        session.close()

def test_backend_running():
    """Test if backend is running"""
    print("\n=== Testing Backend Connection ===")
    
    try:
        response = requests.get("http://localhost:7860/health", timeout=5)
        if response.status_code == 200:
            print("✓ Backend is running")
            return True
        else:
            print(f"✗ Backend responded with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Cannot connect to backend: {e}")
        return False

def test_login_for_superuser():
    """Test login process for superuser account"""
    print("\n=== Testing Superuser Login Process ===")
    
    superuser_account = settings.auth_settings.SUPERUSER
    print(f"Testing with superuser account: {superuser_account}")
    
    # Test NEAR auth challenge
    try:
        challenge_response = requests.post("http://localhost:7860/api/v1/auth/near-challenge", 
                                         json={"near_account_id": superuser_account},
                                         timeout=10)
        
        if challenge_response.status_code == 200:
            print("✓ NEAR challenge endpoint accessible")
            challenge_data = challenge_response.json()
            
            # Check if staking check would be called
            stake_response = requests.get(f"http://localhost:7860/api/v1/near-stake-check/{superuser_account}",
                                        timeout=10)
            
            if stake_response.status_code == 200:
                stake_data = stake_response.json()
                print(f"✓ Stake check endpoint accessible")
                print(f"  - Meets requirements: {stake_data.get('meets_requirements', 'Unknown')}")
                print(f"  - Message: {stake_data.get('message', 'No message')}")
                
                if "superuser" in stake_data.get('message', '').lower():
                    print("✓ Superuser bypass is working in backend")
                    return True
                else:
                    print("? Superuser bypass not detected in message")
                    return stake_data.get('meets_requirements', False)
            else:
                print(f"✗ Stake check failed with status {stake_response.status_code}")
                return False
                
        else:
            print(f"✗ NEAR challenge failed with status {challenge_response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Network error during login test: {e}")
        return False

def main():
    print("Testing Superuser Frontend Implementation")
    print("=" * 50)
    
    # Check environment
    print(f"LANGFLOW_SUPERUSER: {os.getenv('LANGFLOW_SUPERUSER')}")
    print(f"Settings superuser: {settings.auth_settings.SUPERUSER}")
    
    success = True
    
    # Test superuser setup
    if not test_superuser_setup():
        success = False
    
    # Test backend connection
    if not test_backend_running():
        success = False
        print("\n⚠️  Backend is not running. Start it with: make backend")
        return
    
    # Test login process
    if not test_login_for_superuser():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed! Superuser frontend implementation is working.")
        print("\nNow test the frontend:")
        print("1. Start frontend: make frontend")
        print("2. Go to http://localhost:3000")
        print("3. Try to login with vitalpointai.near")
        print("4. Staking requirement messages should be hidden")
    else:
        print("✗ Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    main()
