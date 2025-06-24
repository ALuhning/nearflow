"""Test NEAR authentication endpoints."""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from httpx import AsyncClient
from decimal import Decimal

from langflow.api.v1.schemas import NEARAccountLogin


@pytest.mark.asyncio
async def test_near_auth_enabled_endpoint(client: AsyncClient):
    """Test the NEAR auth enabled check endpoint."""
    response = await client.get("/api/v1/login/near-auth-enabled")
    assert response.status_code == 200
    data = response.json()
    assert "enabled" in data
    assert "pool_contract" in data
    assert "min_stake_amount" in data


@pytest.mark.asyncio
async def test_near_login_endpoint_success(client: AsyncClient):
    """Test successful NEAR account login."""
    
    with patch('langflow.api.v1.login.authenticate_near_account') as mock_auth, \
         patch('langflow.api.v1.login.create_user_tokens') as mock_tokens, \
         patch('langflow.api.v1.login.get_variable_service') as mock_var_service, \
         patch('langflow.api.v1.login.get_or_create_default_folder') as mock_folder:
        
        # Mock user object
        mock_user = AsyncMock()
        mock_user.id = "test-user-id"
        mock_user.store_api_key = "test-api-key"
        
        # Mock authentication result
        mock_auth.return_value = (mock_user, True, "150.5")  # user, user_created, stake_amount
        
        # Mock token creation
        mock_tokens.return_value = {
            "access_token": "test-access-token",
            "refresh_token": "test-refresh-token"
        }
        
        # Mock other services
        mock_var_service.return_value.initialize_user_variables = AsyncMock()
        mock_folder.return_value = AsyncMock()
        
        # Make request
        login_data = {
            "near_account_id": "testuser.near"
        }
        
        response = await client.post("/api/v1/login/near-login", json=login_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["access_token"] == "test-access-token"
        assert data["refresh_token"] == "test-refresh-token"
        assert data["token_type"] == "bearer"
        assert data["user_created"] is True
        assert data["stake_amount"] == "150.5"
        
        # Verify mocks were called
        mock_auth.assert_called_once_with("testuser.near", pytest.any)
        mock_tokens.assert_called_once()


@pytest.mark.asyncio
async def test_near_login_endpoint_insufficient_stake(client: AsyncClient):
    """Test NEAR login with insufficient stake."""
    
    with patch('langflow.api.v1.login.authenticate_near_account') as mock_auth:
        from fastapi import HTTPException
        
        # Mock authentication failure
        mock_auth.side_effect = HTTPException(
            status_code=403,
            detail="Access denied: Minimum stake of 25 NEAR required. Your current stake: 10 NEAR"
        )
        
        login_data = {
            "near_account_id": "lowstake.near"
        }
        
        response = await client.post("/api/v1/login/near-login", json=login_data)
        
        assert response.status_code == 403
        assert "Minimum stake" in response.json()["detail"]


@pytest.mark.asyncio
async def test_near_login_endpoint_not_staker(client: AsyncClient):
    """Test NEAR login when user is not a staker."""
    
    with patch('langflow.api.v1.login.authenticate_near_account') as mock_auth:
        from fastapi import HTTPException
        
        # Mock authentication failure
        mock_auth.side_effect = HTTPException(
            status_code=403,
            detail="Access denied: You must be a staker in vitalpoint.pool.near to access this service"
        )
        
        login_data = {
            "near_account_id": "nonstaker.near"
        }
        
        response = await client.post("/api/v1/login/near-login", json=login_data)
        
        assert response.status_code == 403
        assert "must be a staker" in response.json()["detail"]
