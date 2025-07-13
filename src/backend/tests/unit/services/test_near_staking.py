"""Tests for NEAR blockchain staking verification."""

import pytest
from unittest.mock import AsyncMock, patch
from decimal import Decimal

from langflow.services.near.staking import NEARStakingVerifier


class TestNEARStakingVerifier:
    """Test the NEARStakingVerifier class."""
    
    @pytest.fixture
    def verifier(self):
        """Create a NEARStakingVerifier instance for testing."""
        return NEARStakingVerifier(
            rpc_url="https://rpc.testnet.near.org",
            pool_contract="test.pool.near",
            min_stake_amount="50"
        )
    
    @pytest.mark.asyncio
    async def test_verify_staker_success(self, verifier):
        """Test successful staker verification."""
        # Mock the staker info response
        mock_staker_info = {
            "staked_balance": "100000000000000000000000000",  # 100 NEAR in yoctoNEAR
            "unstaked_balance": "0"
        }
        
        with patch.object(verifier, '_get_staker_info', return_value=mock_staker_info):
            result = await verifier.verify_staker("test.near")
            
            assert result["is_staker"] is True
            assert result["stake_amount"] == Decimal("100")
            assert result["meets_minimum"] is True
            assert result["error"] is None
    
    @pytest.mark.asyncio
    async def test_verify_staker_insufficient_stake(self, verifier):
        """Test staker with insufficient stake amount."""
        # Mock the staker info response with low stake
        mock_staker_info = {
            "staked_balance": "25000000000000000000000000",  # 25 NEAR in yoctoNEAR
            "unstaked_balance": "0"
        }
        
        with patch.object(verifier, '_get_staker_info', return_value=mock_staker_info):
            result = await verifier.verify_staker("test.near")
            
            assert result["is_staker"] is True
            assert result["stake_amount"] == Decimal("25")
            assert result["meets_minimum"] is False
            assert result["error"] is None
    
    @pytest.mark.asyncio
    async def test_verify_staker_not_found(self, verifier):
        """Test account that is not a staker."""
        with patch.object(verifier, '_get_staker_info', return_value=None):
            result = await verifier.verify_staker("nonexistent.near")
            
            assert result["is_staker"] is False
            assert result["stake_amount"] == Decimal("0")
            assert result["meets_minimum"] is False
            assert result["error"] is None
    
    @pytest.mark.asyncio
    async def test_verify_staker_error(self, verifier):
        """Test error handling during staker verification."""
        with patch.object(verifier, '_get_staker_info', side_effect=Exception("Network error")):
            result = await verifier.verify_staker("test.near")
            
            assert result["is_staker"] is False
            assert result["stake_amount"] == Decimal("0")
            assert result["meets_minimum"] is False
            assert result["error"] == "Network error"
    
    def test_update_settings(self, verifier):
        """Test updating verifier settings."""
        verifier.update_settings(
            rpc_url="https://rpc.mainnet.near.org",
            pool_contract="new.pool.near",
            min_stake_amount="200"
        )
        
        assert verifier.rpc_url == "https://rpc.mainnet.near.org"
        assert verifier.pool_contract == "new.pool.near"
        assert verifier.min_stake_amount == Decimal("200")
    
    def test_encode_args(self, verifier):
        """Test argument encoding for contract calls."""
        args = {"account_id": "test.near"}
        encoded = verifier._encode_args(args)
        
        # Verify the result is base64 encoded JSON
        import base64
        import json
        decoded = json.loads(base64.b64decode(encoded).decode())
        assert decoded == args
