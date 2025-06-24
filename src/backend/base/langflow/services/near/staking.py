"""NEAR blockchain integration services for Nearflow."""

import asyncio
import json
import os
from typing import Dict, Any, Optional
from decimal import Decimal

import aiohttp
from loguru import logger


class NEARStakingVerifier:
    """Verifies NEAR blockchain staking pool participation and stake amounts."""
    
    def __init__(self, rpc_url: str = None, pool_contract: str = None, min_stake_amount: str = None):
        self.rpc_url = rpc_url or "https://rpc.mainnet.near.org"
        self.pool_contract = pool_contract or "vitalpoint.pool.near"
        self.min_stake_amount = Decimal(min_stake_amount or "100")  # Default 100 NEAR
        
    def update_settings(self, rpc_url: str, pool_contract: str, min_stake_amount: str):
        """Update settings from auth configuration."""
        self.rpc_url = rpc_url
        self.pool_contract = pool_contract
        self.min_stake_amount = Decimal(min_stake_amount)
        
    async def verify_staker(self, account_id: str) -> Dict[str, Any]:
        """
        Verify if an account is a staker in the configured pool contract
        and check if their stake meets the minimum requirement.
        
        Args:
            account_id: NEAR account ID to check
            
        Returns:
            Dict containing verification results:
            {
                "is_staker": bool,
                "stake_amount": Decimal,
                "meets_minimum": bool,
                "error": str | None
            }
        """
        try:
            # Get the staker information from the pool contract
            stake_info = await self._get_staker_info(account_id)
            
            if stake_info is None:
                return {
                    "is_staker": False,
                    "stake_amount": Decimal("0"),
                    "meets_minimum": False,
                    "error": None
                }
            
            stake_amount = Decimal(stake_info.get("staked_balance", "0")) / Decimal("1e24")  # Convert from yoctoNEAR
            meets_minimum = stake_amount >= self.min_stake_amount
            
            logger.info(f"NEAR staking verification for {account_id}: stake={stake_amount} NEAR, minimum={self.min_stake_amount} NEAR")
            
            return {
                "is_staker": True,
                "stake_amount": stake_amount,
                "meets_minimum": meets_minimum,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error verifying NEAR staker {account_id}: {e}")
            return {
                "is_staker": False,
                "stake_amount": Decimal("0"),
                "meets_minimum": False,
                "error": str(e)
            }
    
    async def _get_staker_info(self, account_id: str) -> Optional[Dict[str, Any]]:
        """Get staker information from the pool contract."""
        try:
            # Call the pool contract to get staker info
            response = await self._call_contract_method(
                contract_id=self.pool_contract,
                method_name="get_account",
                args={"account_id": account_id}
            )
            
            if response and "result" in response:
                return response["result"]
            return None
            
        except Exception as e:
            logger.debug(f"Failed to get staker info for {account_id}: {e}")
            return None
    
    async def _call_contract_method(self, contract_id: str, method_name: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call a NEAR contract method via RPC."""
        payload = {
            "jsonrpc": "2.0",
            "id": "dontcare",
            "method": "query",
            "params": {
                "request_type": "call_function",
                "finality": "final",
                "account_id": contract_id,
                "method_name": method_name,
                "args_base64": self._encode_args(args)
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.rpc_url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    if "result" in data and "result" in data["result"]:
                        # Decode the result from base64
                        result_bytes = bytes(data["result"]["result"])
                        result_str = result_bytes.decode('utf-8')
                        return {"result": json.loads(result_str)}
                    elif "error" in data:
                        logger.error(f"NEAR RPC error: {data['error']}")
                        return None
                    return data
                else:
                    logger.error(f"NEAR RPC request failed with status {response.status}")
                    return None
    
    def _encode_args(self, args: Dict[str, Any]) -> str:
        """Encode arguments for NEAR contract call."""
        import base64
        return base64.b64encode(json.dumps(args).encode()).decode()

    async def get_stake_amount(self, account_id: str) -> Decimal:
        """
        Get the stake amount for a specific account.
        
        Args:
            account_id: NEAR account ID to check
            
        Returns:
            Decimal: The stake amount in NEAR tokens
        """
        try:
            result = await self.verify_staker(account_id)
            return result["stake_amount"]
        except Exception as e:
            logger.error(f"Error getting stake amount for {account_id}: {e}")
            return Decimal("0")

    async def is_staker_with_minimum_stake(self, account_id: str) -> bool:
        """
        Check if an account is a staker with sufficient minimum stake.
        
        Args:
            account_id: NEAR account ID to check
            
        Returns:
            bool: True if account is a staker with sufficient stake, False otherwise
        """
        try:
            result = await self.verify_staker(account_id)
            return result["is_staker"] and result["meets_minimum"]
        except Exception as e:
            logger.error(f"Error checking staker status for {account_id}: {e}")
            return False


# Global instance
near_staking_verifier = NEARStakingVerifier()
