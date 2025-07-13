"""Tests for NEAR account-based login endpoints."""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi import status
from httpx import AsyncClient

from langflow.services.database.models.user.model import User


class TestNEARLoginEndpoints:
    """Test NEAR account login endpoints."""
    
    @pytest.mark.asyncio
    async def test_near_login_new_user_success(self, client: AsyncClient):
        """Test successful NEAR login with new user creation."""
        with patch('langflow.api.v1.login.get_settings_service') as mock_get_settings, \
             patch('langflow.api.v1.login.authenticate_near_account') as mock_auth_near, \
             patch('langflow.api.v1.login.create_user_tokens') as mock_create_tokens, \
             patch('langflow.api.v1.login.get_variable_service') as mock_var_service, \
             patch('langflow.api.v1.login.get_or_create_default_folder') as mock_create_folder:
            
            # Mock settings
            mock_auth_settings = AsyncMock()
            mock_auth_settings.REFRESH_HTTPONLY = True
            mock_auth_settings.REFRESH_SAME_SITE = "lax"
            mock_auth_settings.REFRESH_SECURE = False
            mock_auth_settings.REFRESH_TOKEN_EXPIRE_SECONDS = 3600
            mock_auth_settings.ACCESS_HTTPONLY = True
            mock_auth_settings.ACCESS_SAME_SITE = "lax"
            mock_auth_settings.ACCESS_SECURE = False
            mock_auth_settings.ACCESS_TOKEN_EXPIRE_SECONDS = 1800
            mock_auth_settings.COOKIE_DOMAIN = None
            mock_settings_service = AsyncMock()
            mock_settings_service.auth_settings = mock_auth_settings
            mock_get_settings.return_value = mock_settings_service
            
            # Mock user and authentication
            mock_user = AsyncMock()
            mock_user.id = "user-123"
            mock_user.store_api_key = "api-key-123"
            mock_auth_near.return_value = (mock_user, True, "150.5")  # user, user_created, stake_amount
            
            # Mock tokens
            mock_tokens = {
                "access_token": "access-token-123",
                "refresh_token": "refresh-token-123",
                "token_type": "bearer"
            }
            mock_create_tokens.return_value = mock_tokens
            
            # Mock services
            mock_var_service.return_value.initialize_user_variables = AsyncMock()
            mock_create_folder.return_value = AsyncMock()
            
            # Test the endpoint
            response = await client.post(
                "/api/v1/login/near",
                json={"near_account_id": "testuser.near"}
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["access_token"] == "access-token-123"
            assert data["refresh_token"] == "refresh-token-123"
            assert data["token_type"] == "bearer"
            assert data["user_created"] is True
            assert data["stake_amount"] == "150.5"
            
            # Verify calls
            mock_auth_near.assert_called_once_with("testuser.near", mock_get_settings.return_value.db)
            mock_create_tokens.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_near_login_existing_user_success(self, client: AsyncClient):
        """Test successful NEAR login with existing user."""
        with patch('langflow.api.v1.login.get_settings_service') as mock_get_settings, \
             patch('langflow.api.v1.login.authenticate_near_account') as mock_auth_near, \
             patch('langflow.api.v1.login.create_user_tokens') as mock_create_tokens, \
             patch('langflow.api.v1.login.get_variable_service') as mock_var_service, \
             patch('langflow.api.v1.login.get_or_create_default_folder') as mock_create_folder:
            
            # Mock settings
            mock_auth_settings = AsyncMock()
            mock_auth_settings.REFRESH_HTTPONLY = True
            mock_auth_settings.REFRESH_SAME_SITE = "lax"
            mock_auth_settings.REFRESH_SECURE = False
            mock_auth_settings.REFRESH_TOKEN_EXPIRE_SECONDS = 3600
            mock_auth_settings.ACCESS_HTTPONLY = True
            mock_auth_settings.ACCESS_SAME_SITE = "lax"
            mock_auth_settings.ACCESS_SECURE = False
            mock_auth_settings.ACCESS_TOKEN_EXPIRE_SECONDS = 1800
            mock_auth_settings.COOKIE_DOMAIN = None
            mock_settings_service = AsyncMock()
            mock_settings_service.auth_settings = mock_auth_settings
            mock_get_settings.return_value = mock_settings_service
            
            # Mock user and authentication (existing user)
            mock_user = AsyncMock()
            mock_user.id = "user-123"
            mock_user.store_api_key = "api-key-123"
            mock_auth_near.return_value = (mock_user, False, "200.0")  # user, user_created, stake_amount
            
            # Mock tokens
            mock_tokens = {
                "access_token": "access-token-123",
                "refresh_token": "refresh-token-123",
                "token_type": "bearer"
            }
            mock_create_tokens.return_value = mock_tokens
            
            # Mock services
            mock_var_service.return_value.initialize_user_variables = AsyncMock()
            mock_create_folder.return_value = AsyncMock()
            
            # Test the endpoint
            response = await client.post(
                "/api/v1/login/near",
                json={"near_account_id": "existinguser.near"}
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["access_token"] == "access-token-123"
            assert data["refresh_token"] == "refresh-token-123"
            assert data["token_type"] == "bearer"
            assert data["user_created"] is False
            assert data["stake_amount"] == "200.0"
    
    @pytest.mark.asyncio
    async def test_near_login_insufficient_stake(self, client: AsyncClient):
        """Test NEAR login failure due to insufficient stake."""
        with patch('langflow.api.v1.login.authenticate_near_account') as mock_auth_near:
            from fastapi import HTTPException
            
            # Mock authentication failure due to insufficient stake
            mock_auth_near.side_effect = HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: Minimum stake of 100 NEAR required. Current stake: 50 NEAR"
            )
            
            # Test the endpoint
            response = await client.post(
                "/api/v1/login/near",
                json={"near_account_id": "lowstaker.near"}
            )
            
            assert response.status_code == status.HTTP_403_FORBIDDEN
            data = response.json()
            assert "Minimum stake of 100 NEAR required" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_near_login_not_staker(self, client: AsyncClient):
        """Test NEAR login failure when user is not a staker."""
        with patch('langflow.api.v1.login.authenticate_near_account') as mock_auth_near:
            from fastapi import HTTPException
            
            # Mock authentication failure due to not being a staker
            mock_auth_near.side_effect = HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: nonstaker.near must be a staker in vitalpoint.pool.near"
            )
            
            # Test the endpoint
            response = await client.post(
                "/api/v1/login/near",
                json={"near_account_id": "nonstaker.near"}
            )
            
            assert response.status_code == status.HTTP_403_FORBIDDEN
            data = response.json()
            assert "must be a staker" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_near_login_invalid_account_id(self, client: AsyncClient):
        """Test NEAR login with invalid account ID format."""
        response = await client.post(
            "/api/v1/login/near",
            json={"near_account_id": ""}
        )
        
        # Should fail validation
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @pytest.mark.asyncio
    async def test_near_login_missing_account_id(self, client: AsyncClient):
        """Test NEAR login with missing account ID."""
        response = await client.post(
            "/api/v1/login/near",
            json={}
        )
        
        # Should fail validation
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
