"""Integration tests for NEAR staking authentication."""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi import HTTPException
from decimal import Decimal

from langflow.services.auth.utils import authenticate_user_with_near_staking
from langflow.services.settings.manager import SettingsService


class TestNEARStakingAuthentication:
    """Test NEAR staking authentication integration."""
    
    @pytest.mark.asyncio
    async def test_authenticate_user_with_near_staking_disabled(self, active_user, async_session):
        """Test authentication when NEAR staking verification is disabled."""
        with patch('langflow.services.auth.utils.get_settings_service') as mock_get_settings:
            # Mock settings with NEAR staking disabled
            mock_auth_settings = AsyncMock()
            mock_auth_settings.ENABLE_NEAR_STAKING_VERIFICATION = False
            mock_settings_service = AsyncMock()
            mock_settings_service.auth_settings = mock_auth_settings
            mock_get_settings.return_value = mock_settings_service
            
            with patch('langflow.services.auth.utils.authenticate_user', return_value=active_user):
                result = await authenticate_user_with_near_staking(
                    "testuser", "password", async_session
                )
                
                assert result == active_user
    
    @pytest.mark.asyncio
    async def test_authenticate_user_with_near_staking_success(self, active_user, async_session):
        """Test successful authentication with NEAR staking verification."""
        with patch('langflow.services.auth.utils.get_settings_service') as mock_get_settings:
            # Mock settings with NEAR staking enabled
            mock_auth_settings = AsyncMock()
            mock_auth_settings.ENABLE_NEAR_STAKING_VERIFICATION = True
            mock_auth_settings.NEAR_RPC_URL = "https://rpc.testnet.near.org"
            mock_auth_settings.NEAR_POOL_CONTRACT = "test.pool.near"
            mock_auth_settings.NEAR_MIN_STAKE_AMOUNT = "100"
            mock_settings_service = AsyncMock()
            mock_settings_service.auth_settings = mock_auth_settings
            mock_get_settings.return_value = mock_settings_service
            
            # Mock successful staking verification
            mock_staking_result = {
                "is_staker": True,
                "stake_amount": Decimal("150"),
                "meets_minimum": True,
                "error": None
            }
            
            with patch('langflow.services.auth.utils.authenticate_user', return_value=active_user), \
                 patch('langflow.services.auth.utils.near_staking_verifier') as mock_verifier:
                
                mock_verifier.verify_staker.return_value = mock_staking_result
                
                result = await authenticate_user_with_near_staking(
                    "testuser.near", "password", async_session
                )
                
                assert result == active_user
                mock_verifier.update_settings.assert_called_once()
                mock_verifier.verify_staker.assert_called_once_with("testuser.near")
    
    @pytest.mark.asyncio
    async def test_authenticate_user_not_staker(self, active_user, async_session):
        """Test authentication failure when user is not a staker."""
        with patch('langflow.services.auth.utils.get_settings_service') as mock_get_settings:
            # Mock settings with NEAR staking enabled
            mock_auth_settings = AsyncMock()
            mock_auth_settings.ENABLE_NEAR_STAKING_VERIFICATION = True
            mock_auth_settings.NEAR_POOL_CONTRACT = "test.pool.near"
            mock_settings_service = AsyncMock()
            mock_settings_service.auth_settings = mock_auth_settings
            mock_get_settings.return_value = mock_settings_service
            
            # Mock staking verification - not a staker
            mock_staking_result = {
                "is_staker": False,
                "stake_amount": Decimal("0"),
                "meets_minimum": False,
                "error": None
            }
            
            with patch('langflow.services.auth.utils.authenticate_user', return_value=active_user), \
                 patch('langflow.services.auth.utils.near_staking_verifier') as mock_verifier:
                
                mock_verifier.verify_staker.return_value = mock_staking_result
                
                with pytest.raises(HTTPException) as exc_info:
                    await authenticate_user_with_near_staking(
                        "testuser.near", "password", async_session
                    )
                
                assert exc_info.value.status_code == 403
                assert "must be a staker" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_authenticate_user_insufficient_stake(self, active_user, async_session):
        """Test authentication failure when user has insufficient stake."""
        with patch('langflow.services.auth.utils.get_settings_service') as mock_get_settings:
            # Mock settings with NEAR staking enabled
            mock_auth_settings = AsyncMock()
            mock_auth_settings.ENABLE_NEAR_STAKING_VERIFICATION = True
            mock_auth_settings.NEAR_MIN_STAKE_AMOUNT = "100"
            mock_settings_service = AsyncMock()
            mock_settings_service.auth_settings = mock_auth_settings
            mock_get_settings.return_value = mock_settings_service
            
            # Mock staking verification - insufficient stake
            mock_staking_result = {
                "is_staker": True,
                "stake_amount": Decimal("50"),
                "meets_minimum": False,
                "error": None
            }
            
            with patch('langflow.services.auth.utils.authenticate_user', return_value=active_user), \
                 patch('langflow.services.auth.utils.near_staking_verifier') as mock_verifier:
                
                mock_verifier.verify_staker.return_value = mock_staking_result
                
                with pytest.raises(HTTPException) as exc_info:
                    await authenticate_user_with_near_staking(
                        "testuser.near", "password", async_session
                    )
                
                assert exc_info.value.status_code == 403
                assert "Minimum stake of 100 NEAR required" in exc_info.value.detail
                assert "Your current stake: 50 NEAR" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_authenticate_user_staking_service_error(self, active_user, async_session):
        """Test authentication failure when staking service has an error."""
        with patch('langflow.services.auth.utils.get_settings_service') as mock_get_settings:
            # Mock settings with NEAR staking enabled
            mock_auth_settings = AsyncMock()
            mock_auth_settings.ENABLE_NEAR_STAKING_VERIFICATION = True
            mock_settings_service = AsyncMock()
            mock_settings_service.auth_settings = mock_auth_settings
            mock_get_settings.return_value = mock_settings_service
            
            with patch('langflow.services.auth.utils.authenticate_user', return_value=active_user), \
                 patch('langflow.services.auth.utils.near_staking_verifier') as mock_verifier:
                
                # Mock staking verifier to raise an exception
                mock_verifier.verify_staker.side_effect = Exception("Network error")
                
                with pytest.raises(HTTPException) as exc_info:
                    await authenticate_user_with_near_staking(
                        "testuser.near", "password", async_session
                    )
                
                assert exc_info.value.status_code == 500
                assert "temporarily unavailable" in exc_info.value.detail
