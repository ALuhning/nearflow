"""
Unit tests for NEAR signature-based authentication endpoints.
"""
import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from langflow.services.auth.utils import generate_near_challenge


@pytest.fixture
def mock_near_challenge():
    """Mock NEAR challenge for testing."""
    challenge_bytes = b'test_challenge_32_bytes_long!!'
    return base64.b64encode(challenge_bytes).decode('utf-8')


@pytest.fixture
def mock_signature_data():
    """Mock signature data for testing."""
    return {
        "near_account_id": "test.near",
        "public_key": "ed25519:8r6VqJmzMcYjzT2TfJHrHGJZXZMpGYQg7E8JbLZzKnZz",
        "signature": "base64_encoded_signature_here",
        "challenge": "dGVzdF9jaGFsbGVuZ2VfMzJfYnl0ZXNfbG9uZyEh",  # base64 encoded
        "message": "Login with NEAR",
        "recipient": "nearflow"
    }


class TestNEARChallengeEndpoint:
    """Test the NEAR challenge generation endpoint."""

    def test_near_challenge_generation(self, client: TestClient):
        """Test successful challenge generation."""
        response = client.post(
            "/api/v1/login/near-challenge",
            json={"near_account_id": "test.near"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "challenge" in data
        assert "message" in data
        assert "recipient" in data
        assert data["message"] == "Login with NEAR"
        assert data["recipient"] == "nearflow"
        
        # Verify challenge is base64 encoded and can be decoded
        challenge_bytes = base64.b64decode(data["challenge"])
        assert len(challenge_bytes) == 32  # Should be 32 bytes

    def test_near_challenge_invalid_request(self, client: TestClient):
        """Test challenge generation with invalid request."""
        response = client.post(
            "/api/v1/login/near-challenge",
            json={}  # Missing near_account_id
        )
        
        assert response.status_code == 422  # Validation error

    @patch('langflow.api.v1.login.generate_near_challenge')
    def test_near_challenge_generation_error(self, mock_generate, client: TestClient):
        """Test error handling in challenge generation."""
        mock_generate.side_effect = Exception("Challenge generation failed")
        
        response = client.post(
            "/api/v1/login/near-challenge",
            json={"near_account_id": "test.near"}
        )
        
        assert response.status_code == 500
        assert "Challenge generation failed" in response.json()["detail"]


class TestNEARSignatureAuthentication:
    """Test the NEAR signature-based authentication endpoint."""

    @patch('langflow.services.auth.utils.authenticate_near_account_with_signature')
    @patch('langflow.services.auth.utils.create_user_tokens')
    @patch('langflow.services.deps.get_variable_service')
    @patch('langflow.initial_setup.setup.get_or_create_default_folder')
    async def test_successful_signature_login(
        self, 
        mock_folder, 
        mock_var_service, 
        mock_create_tokens, 
        mock_auth, 
        async_client: AsyncClient,
        mock_signature_data,
        mock_user
    ):
        """Test successful NEAR signature authentication."""
        # Setup mocks
        mock_auth.return_value = mock_user
        mock_create_tokens.return_value = {
            "access_token": "test_access_token",
            "refresh_token": "test_refresh_token"
        }
        mock_var_service.return_value.initialize_user_variables = AsyncMock()
        mock_folder.return_value = AsyncMock()

        response = await async_client.post(
            "/api/v1/login/near-auth",
            json=mock_signature_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["access_token"] == "test_access_token"
        assert data["refresh_token"] == "test_refresh_token"
        assert data["token_type"] == "bearer"
        assert "user_created" in data
        
        # Verify authentication was called with correct parameters
        mock_auth.assert_called_once()
        args = mock_auth.call_args[1]
        assert args["account_id"] == "test.near"
        assert args["public_key"] == mock_signature_data["public_key"]
        assert args["signature"] == mock_signature_data["signature"]

    @patch('langflow.services.auth.utils.authenticate_near_account_with_signature')
    async def test_invalid_signature_login(
        self, 
        mock_auth, 
        async_client: AsyncClient,
        mock_signature_data
    ):
        """Test authentication with invalid signature."""
        mock_auth.return_value = None  # Authentication failed
        
        response = await async_client.post(
            "/api/v1/login/near-auth",
            json=mock_signature_data
        )
        
        assert response.status_code == 401
        assert "authentication failed" in response.json()["detail"].lower()

    async def test_signature_login_missing_fields(self, async_client: AsyncClient):
        """Test authentication with missing required fields."""
        incomplete_data = {
            "near_account_id": "test.near",
            # Missing other required fields
        }
        
        response = await async_client.post(
            "/api/v1/login/near-auth",
            json=incomplete_data
        )
        
        assert response.status_code == 422  # Validation error

    async def test_signature_login_invalid_challenge(self, async_client: AsyncClient):
        """Test authentication with invalid base64 challenge."""
        invalid_data = {
            "near_account_id": "test.near",
            "public_key": "ed25519:8r6VqJmzMcYjzT2TfJHrHGJZXZMpGYQg7E8JbLZzKnZz",
            "signature": "valid_signature",
            "challenge": "invalid_base64!!!",
            "message": "Login with NEAR",
            "recipient": "nearflow"
        }
        
        response = await async_client.post(
            "/api/v1/login/near-auth",
            json=invalid_data
        )
        
        assert response.status_code == 500  # Base64 decode error

    @patch('langflow.services.auth.utils.authenticate_near_account_with_signature')
    @patch('langflow.services.settings.service.SettingsService')
    async def test_signature_login_with_staking_check(
        self, 
        mock_settings_service,
        mock_auth, 
        async_client: AsyncClient,
        mock_signature_data,
        mock_user
    ):
        """Test authentication with staking verification enabled."""
        # Setup mocks
        mock_auth.return_value = mock_user
        mock_settings = MagicMock()
        mock_settings.auth_settings.ENABLE_NEAR_STAKING_VERIFICATION = True
        mock_settings_service.return_value = mock_settings
        
        with patch('langflow.services.near.staking.near_staking_verifier.get_stake_amount') as mock_stake:
            mock_stake.return_value = "100.0"
            
            with patch('langflow.services.auth.utils.create_user_tokens') as mock_tokens:
                mock_tokens.return_value = {
                    "access_token": "test_token",
                    "refresh_token": "test_refresh"
                }
                
                with patch('langflow.services.deps.get_variable_service') as mock_var:
                    mock_var.return_value.initialize_user_variables = AsyncMock()
                    
                    with patch('langflow.initial_setup.setup.get_or_create_default_folder'):
                        response = await async_client.post(
                            "/api/v1/login/near-auth",
                            json=mock_signature_data
                        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["stake_amount"] == "100.0"


class TestNEARUtilityFunctions:
    """Test NEAR authentication utility functions."""

    def test_generate_near_challenge(self):
        """Test challenge generation."""
        challenge1 = generate_near_challenge()
        challenge2 = generate_near_challenge()
        
        assert len(challenge1) == 32
        assert len(challenge2) == 32
        assert challenge1 != challenge2  # Should be random

    @patch('langflow.services.auth.utils.verify_full_key_belongs_to_user')
    @patch('langflow.services.auth.utils.verify_signature_only')
    async def test_verify_near_signature(self, mock_verify_sig, mock_verify_key):
        """Test NEAR signature verification."""
        from langflow.services.auth.utils import verify_near_signature
        
        mock_verify_sig.return_value = True
        mock_verify_key.return_value = True
        
        result = await verify_near_signature(
            account_id="test.near",
            public_key="ed25519:test_key",
            signature="test_signature",
            message="Login with NEAR",
            recipient="nearflow",
            nonce=b"test_nonce_32_bytes_long!!!!!!"
        )
        
        assert result is True
        mock_verify_sig.assert_called_once()
        mock_verify_key.assert_called_once()

    @patch('langflow.services.auth.utils.verify_full_key_belongs_to_user')
    @patch('langflow.services.auth.utils.verify_signature_only')
    async def test_verify_near_signature_invalid(self, mock_verify_sig, mock_verify_key):
        """Test NEAR signature verification with invalid signature."""
        from langflow.services.auth.utils import verify_near_signature
        
        mock_verify_sig.return_value = False
        mock_verify_key.return_value = True
        
        result = await verify_near_signature(
            account_id="test.near",
            public_key="ed25519:test_key",
            signature="invalid_signature",
            message="Login with NEAR",
            recipient="nearflow",
            nonce=b"test_nonce_32_bytes_long!!!!!!"
        )
        
        assert result is False


class TestLegacyNEAREndpoints:
    """Test legacy NEAR endpoints for backward compatibility."""

    @patch('langflow.services.auth.utils.authenticate_near_account')
    @patch('langflow.services.auth.utils.create_user_tokens')
    @patch('langflow.services.deps.get_variable_service')
    @patch('langflow.initial_setup.setup.get_or_create_default_folder')
    async def test_legacy_near_login_still_works(
        self, 
        mock_folder, 
        mock_var_service, 
        mock_create_tokens, 
        mock_auth, 
        async_client: AsyncClient,
        mock_user
    ):
        """Test that legacy NEAR login endpoints still work for backward compatibility."""
        # Setup mocks
        mock_auth.return_value = (mock_user, True, "100.0")  # user, user_created, stake_amount
        mock_create_tokens.return_value = {
            "access_token": "test_access_token",
            "refresh_token": "test_refresh_token"
        }
        mock_var_service.return_value.initialize_user_variables = AsyncMock()
        mock_folder.return_value = AsyncMock()

        response = await async_client.post(
            "/api/v1/login/near-login",
            json={"near_account_id": "test.near"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["access_token"] == "test_access_token"
        assert data["refresh_token"] == "test_refresh_token"
        assert data["user_created"] is True
        assert data["stake_amount"] == "100.0"

    async def test_near_auth_enabled_endpoint(self, async_client: AsyncClient):
        """Test the NEAR auth status endpoint."""
        response = await async_client.get("/api/v1/login/near-auth-enabled")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "enabled" in data
        assert "pool_contract" in data
        assert "min_stake_amount" in data
