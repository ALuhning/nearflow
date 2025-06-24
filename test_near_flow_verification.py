#!/usr/bin/env python3
"""
Verification script to check that both signature verification AND staking 
verification are enforced in the NEAR authentication flow.
"""

import asyncio
import base64
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

from langflow.services.auth.utils import (
    authenticate_near_account_with_signature,
    verify_near_signature,
    NEARPayload,
    serialize_near_payload
)
from langflow.services.near.staking import near_staking_verifier
from langflow.services.settings.service import SettingsService


async def test_signature_and_staking_enforcement():
    """Test that both signature verification and staking are required."""
    
    # Mock account data
    account_id = "test.near"
    public_key = "ed25519:8r6VqJmzMcYjzT2TfJHrHGJZXZMpGYQg7E8JbLZzKnZz"
    signature = "valid_signature_base64"
    message = "Login with NEAR"
    recipient = "nearflow"
    nonce = b'test_challenge_32_bytes_long!!'
    
    print("Testing NEAR authentication flow...")
    
    # Test 1: Valid signature but no staking - should fail
    print("\n1. Testing valid signature but no staking...")
    with patch('langflow.services.auth.utils.verify_signature_only', return_value=True), \
         patch('langflow.services.auth.utils.verify_full_key_belongs_to_user', return_value=True), \
         patch('langflow.services.near.staking.near_staking_verifier.is_staker_with_minimum_stake', return_value=False), \
         patch('langflow.services.auth.utils.get_settings_service') as mock_get_settings:
        
        # Mock settings to enable staking verification
        mock_settings = MagicMock()
        mock_settings.auth_settings.ENABLE_NEAR_STAKING_VERIFICATION = True
        mock_get_settings.return_value = mock_settings
        
        mock_session = MagicMock()
        
        user = await authenticate_near_account_with_signature(
            account_id=account_id,
            public_key=public_key,
            signature=signature,
            message=message,
            recipient=recipient,
            nonce=nonce,
            session=mock_session
        )
        
        assert user is None, "Authentication should fail when staking requirement is not met"
        print("✓ Authentication correctly failed when staking requirement is not met")
    
    # Test 2: Invalid signature but valid staking - should fail
    print("\n2. Testing invalid signature but valid staking...")
    with patch('langflow.services.auth.utils.verify_signature_only', return_value=False), \
         patch('langflow.services.auth.utils.verify_full_key_belongs_to_user', return_value=True), \
         patch('langflow.services.near.staking.near_staking_verifier.is_staker_with_minimum_stake', return_value=True), \
         patch('langflow.services.auth.utils.get_settings_service') as mock_get_settings:
        
        # Mock settings to enable staking verification
        mock_settings = MagicMock()
        mock_settings.auth_settings.ENABLE_NEAR_STAKING_VERIFICATION = True
        mock_get_settings.return_value = mock_settings
        
        mock_session = MagicMock()
        
        user = await authenticate_near_account_with_signature(
            account_id=account_id,
            public_key=public_key,
            signature=signature,
            message=message,
            recipient=recipient,
            nonce=nonce,
            session=mock_session
        )
        
        assert user is None, "Authentication should fail when signature is invalid"
        print("✓ Authentication correctly failed when signature is invalid")
    
    # Test 3: Valid signature AND valid staking - should succeed
    print("\n3. Testing valid signature AND valid staking...")
    with patch('langflow.services.auth.utils.verify_signature_only', return_value=True), \
         patch('langflow.services.auth.utils.verify_full_key_belongs_to_user', return_value=True), \
         patch('langflow.services.near.staking.near_staking_verifier.is_staker_with_minimum_stake', return_value=True), \
         patch('langflow.services.auth.utils.get_settings_service') as mock_get_settings, \
         patch('langflow.services.auth.utils.get_user_by_username', return_value=None), \
         patch('langflow.services.auth.utils.create_user_from_near_account') as mock_create_user, \
         patch('langflow.services.auth.utils.update_user_last_login_at'):
        
        # Mock settings to enable staking verification
        mock_settings = MagicMock()
        mock_settings.auth_settings.ENABLE_NEAR_STAKING_VERIFICATION = True
        mock_get_settings.return_value = mock_settings
        
        # Mock created user
        mock_user = MagicMock()
        mock_user.id = "user_id_123"
        mock_create_user.return_value = mock_user
        
        mock_session = MagicMock()
        
        user = await authenticate_near_account_with_signature(
            account_id=account_id,
            public_key=public_key,
            signature=signature,
            message=message,
            recipient=recipient,
            nonce=nonce,
            session=mock_session
        )
        
        assert user is not None, "Authentication should succeed when both signature and staking are valid"
        print("✓ Authentication correctly succeeded when both signature and staking are valid")
    
    # Test 4: Staking verification disabled - only signature required
    print("\n4. Testing staking verification disabled...")
    with patch('langflow.services.auth.utils.verify_signature_only', return_value=True), \
         patch('langflow.services.auth.utils.verify_full_key_belongs_to_user', return_value=True), \
         patch('langflow.services.auth.utils.get_settings_service') as mock_get_settings, \
         patch('langflow.services.auth.utils.get_user_by_username', return_value=None), \
         patch('langflow.services.auth.utils.create_user_from_near_account') as mock_create_user, \
         patch('langflow.services.auth.utils.update_user_last_login_at'):
        
        # Mock settings to disable staking verification
        mock_settings = MagicMock()
        mock_settings.auth_settings.ENABLE_NEAR_STAKING_VERIFICATION = False
        mock_get_settings.return_value = mock_settings
        
        # Mock created user
        mock_user = MagicMock()
        mock_user.id = "user_id_123"
        mock_create_user.return_value = mock_user
        
        mock_session = MagicMock()
        
        user = await authenticate_near_account_with_signature(
            account_id=account_id,
            public_key=public_key,
            signature=signature,
            message=message,
            recipient=recipient,
            nonce=nonce,
            session=mock_session
        )
        
        assert user is not None, "Authentication should succeed when staking is disabled and signature is valid"
        print("✓ Authentication correctly succeeded when staking verification is disabled")
    
    print("\n🎉 All tests passed! Both signature verification AND staking verification are properly enforced.")


if __name__ == "__main__":
    asyncio.run(test_signature_and_staking_enforcement())
