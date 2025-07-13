"""
Integration tests for NEAR signature-based authentication.
"""
import base64
import os
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from langflow.services.auth.utils import generate_near_challenge, serialize_near_payload, NEARPayload


class TestNEARSignatureAuthIntegration:
    """Integration tests for NEAR signature authentication flow."""

    @pytest.mark.asyncio
    async def test_complete_signature_auth_flow(self):
        """Test the complete NEAR signature authentication flow."""
        # This test would require actual NEAR wallet integration
        # For now, we'll test the flow with mocked components
        
        # Step 1: Generate challenge
        challenge_bytes = generate_near_challenge()
        challenge_b64 = base64.b64encode(challenge_bytes).decode('utf-8')
        
        assert len(challenge_bytes) == 32
        assert len(challenge_b64) > 0
        
        # Step 2: Create payload that would be signed
        payload = NEARPayload(
            message="Login with NEAR",
            nonce=challenge_bytes,
            recipient="nearflow"
        )
        
        serialized = serialize_near_payload(payload)
        assert len(serialized) > 0
        
        # The actual signing would happen in the frontend with NEAR wallet
        # Here we can test the payload structure

    @pytest.mark.asyncio 
    async def test_challenge_endpoint_integration(self, async_client: AsyncClient):
        """Test challenge generation endpoint integration."""
        response = await async_client.post(
            "/api/v1/login/near-challenge",
            json={"near_account_id": "test.near"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert "challenge" in data
        assert "message" in data
        assert "recipient" in data
        
        # Validate challenge can be decoded
        challenge_bytes = base64.b64decode(data["challenge"])
        assert len(challenge_bytes) == 32

    @pytest.mark.asyncio
    @patch('langflow.services.auth.utils.verify_near_signature')
    @patch('langflow.services.auth.utils.get_user_by_username')
    @patch('langflow.services.auth.utils.create_user_from_near_account')
    @patch('langflow.services.auth.utils.update_user_last_login_at')
    @patch('langflow.services.near.staking.near_staking_verifier.is_staker_with_minimum_stake')
    async def test_signature_auth_endpoint_integration(
        self, 
        mock_is_staker,
        mock_update_login,
        mock_create_user,
        mock_get_user,
        mock_verify_signature,
        async_client: AsyncClient,
        mock_user
    ):
        """Test signature authentication endpoint integration."""
        # Setup mocks
        mock_verify_signature.return_value = True
        mock_get_user.return_value = None  # User doesn't exist
        mock_create_user.return_value = mock_user
        mock_is_staker.return_value = True
        mock_update_login.return_value = AsyncMock()

        signature_data = {
            "near_account_id": "test.near",
            "public_key": "ed25519:8r6VqJmzMcYjzT2TfJHrHGJZXZMpGYQg7E8JbLZzKnZz",
            "signature": "base64_encoded_signature",
            "challenge": base64.b64encode(b"test_challenge_32_bytes_long!!").decode('utf-8'),
            "message": "Login with NEAR",
            "recipient": "nearflow"
        }

        with patch('langflow.services.auth.utils.create_user_tokens') as mock_tokens:
            mock_tokens.return_value = {
                "access_token": "test_access_token",
                "refresh_token": "test_refresh_token"
            }
            
            with patch('langflow.services.deps.get_variable_service') as mock_var:
                mock_var.return_value.initialize_user_variables = AsyncMock()
                
                with patch('langflow.initial_setup.setup.get_or_create_default_folder'):
                    response = await async_client.post(
                        "/api/v1/login/near-auth",
                        json=signature_data
                    )

        assert response.status_code == 200
        data = response.json()
        
        assert data["access_token"] == "test_access_token"
        assert data["refresh_token"] == "test_refresh_token"
        assert data["token_type"] == "bearer"

    @pytest.mark.asyncio
    @patch('langflow.services.auth.utils.verify_near_signature')
    async def test_signature_auth_invalid_signature(
        self, 
        mock_verify_signature,
        async_client: AsyncClient
    ):
        """Test authentication with invalid signature."""
        mock_verify_signature.return_value = False

        signature_data = {
            "near_account_id": "test.near",
            "public_key": "ed25519:invalid_key",
            "signature": "invalid_signature",
            "challenge": base64.b64encode(b"test_challenge_32_bytes_long!!").decode('utf-8'),
            "message": "Login with NEAR",
            "recipient": "nearflow"
        }

        response = await async_client.post(
            "/api/v1/login/near-auth",
            json=signature_data
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    @patch('langflow.services.auth.utils.verify_near_signature')
    @patch('langflow.services.near.staking.near_staking_verifier.is_staker_with_minimum_stake')
    async def test_signature_auth_staking_requirement_not_met(
        self, 
        mock_is_staker,
        mock_verify_signature,
        async_client: AsyncClient
    ):
        """Test authentication when staking requirements are not met."""
        mock_verify_signature.return_value = True
        mock_is_staker.return_value = False  # User doesn't meet staking requirements

        signature_data = {
            "near_account_id": "test.near",
            "public_key": "ed25519:valid_key",
            "signature": "valid_signature", 
            "challenge": base64.b64encode(b"test_challenge_32_bytes_long!!").decode('utf-8'),
            "message": "Login with NEAR",
            "recipient": "nearflow"
        }

        with patch('langflow.services.deps.get_settings_service') as mock_settings:
            mock_settings.return_value.auth_settings.ENABLE_NEAR_STAKING_VERIFICATION = True
            
            response = await async_client.post(
                "/api/v1/login/near-auth",
                json=signature_data
            )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_near_auth_status_endpoint(self, async_client: AsyncClient):
        """Test NEAR authentication status endpoint."""
        response = await async_client.get("/api/v1/login/near-auth-enabled")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "enabled" in data
        assert "pool_contract" in data
        assert "min_stake_amount" in data
        
        # Validate types
        assert isinstance(data["enabled"], bool)
        assert isinstance(data["pool_contract"], str)
        assert isinstance(data["min_stake_amount"], str)


class TestNEARRPCIntegration:
    """Test NEAR RPC integration for key verification."""

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.post')
    async def test_verify_full_key_belongs_to_user_success(self, mock_post):
        """Test successful key verification against NEAR RPC."""
        from langflow.services.auth.utils import verify_full_key_belongs_to_user
        
        # Mock RPC response
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {
                "keys": [
                    {
                        "public_key": "ed25519:test_key",
                        "access_key": {
                            "permission": "FullAccess"
                        }
                    }
                ]
            }
        }
        mock_post.return_value = mock_response

        result = await verify_full_key_belongs_to_user("test.near", "ed25519:test_key")
        
        assert result is True
        mock_post.assert_called_once()

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.post')
    async def test_verify_full_key_belongs_to_user_not_full_access(self, mock_post):
        """Test key verification when key is not full access."""
        from langflow.services.auth.utils import verify_full_key_belongs_to_user
        
        # Mock RPC response with function call key
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {
                "keys": [
                    {
                        "public_key": "ed25519:test_key",
                        "access_key": {
                            "permission": {
                                "FunctionCall": {
                                    "allowance": "1000000000000000000000000",
                                    "receiver_id": "some.contract"
                                }
                            }
                        }
                    }
                ]
            }
        }
        mock_post.return_value = mock_response

        result = await verify_full_key_belongs_to_user("test.near", "ed25519:test_key")
        
        assert result is False

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.post')
    async def test_verify_full_key_belongs_to_user_key_not_found(self, mock_post):
        """Test key verification when key is not found."""
        from langflow.services.auth.utils import verify_full_key_belongs_to_user
        
        # Mock RPC response without the target key
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {
                "keys": [
                    {
                        "public_key": "ed25519:different_key",
                        "access_key": {
                            "permission": "FullAccess"
                        }
                    }
                ]
            }
        }
        mock_post.return_value = mock_response

        result = await verify_full_key_belongs_to_user("test.near", "ed25519:test_key")
        
        assert result is False

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.post')
    async def test_verify_full_key_belongs_to_user_rpc_error(self, mock_post):
        """Test key verification when RPC returns error."""
        from langflow.services.auth.utils import verify_full_key_belongs_to_user
        
        # Mock RPC error response
        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        result = await verify_full_key_belongs_to_user("test.near", "ed25519:test_key")
        
        assert result is False


class TestNEARPayloadSerialization:
    """Test NEAR payload serialization."""

    def test_serialize_near_payload_basic(self):
        """Test basic payload serialization."""
        nonce = b"test_nonce_32_bytes_long!!!!!!"
        payload = NEARPayload(
            message="Login with NEAR",
            nonce=nonce,
            recipient="nearflow"
        )
        
        serialized = serialize_near_payload(payload)
        
        assert len(serialized) > 0
        assert isinstance(serialized, bytes)

    def test_serialize_near_payload_with_callback(self):
        """Test payload serialization with callback URL."""
        nonce = b"test_nonce_32_bytes_long!!!!!!"
        payload = NEARPayload(
            message="Login with NEAR",
            nonce=nonce,
            recipient="nearflow",
            callback_url="https://example.com/callback"
        )
        
        serialized = serialize_near_payload(payload)
        
        assert len(serialized) > 0
        assert isinstance(serialized, bytes)

    def test_near_payload_creation(self):
        """Test NEAR payload object creation."""
        nonce = b"a" * 32
        payload = NEARPayload(
            message="Test message",
            nonce=nonce,
            recipient="test-app"
        )
        
        assert payload.tag == 2147484061
        assert payload.message == "Test message"
        assert payload.nonce == nonce
        assert payload.recipient == "test-app"
        assert payload.callback_url is None

    def test_challenge_generation_consistency(self):
        """Test that challenge generation produces consistent results."""
        challenge1 = generate_near_challenge()
        challenge2 = generate_near_challenge()
        
        # Should be different (random)
        assert challenge1 != challenge2
        
        # Should be correct length  
        assert len(challenge1) == 32
        assert len(challenge2) == 32
        
        # Should be bytes
        assert isinstance(challenge1, bytes)
        assert isinstance(challenge2, bytes)
