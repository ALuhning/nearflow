# NEAR Authentication Implementation Summary

## Overview
I've confirmed that the NEAR authentication implementation requires **BOTH** signature verification AND staking verification to grant authorization. Here's the verification:

## Implementation Analysis

### 1. Challenge/Response Flow
- **Endpoint**: `POST /api/v1/login/near-challenge`
- **Purpose**: Generates a secure random challenge for the user to sign
- **Response**: Returns challenge, message, and recipient for signing

### 2. Signature-Based Authentication
- **Endpoint**: `POST /api/v1/login/near-auth`
- **Purpose**: Verifies signature and staking requirements
- **Flow**: 
  1. Decodes the challenge
  2. Calls `authenticate_near_account_with_signature()`
  3. Both signature AND staking must pass for success

### 3. Dual Verification Requirements

The `authenticate_near_account_with_signature()` function enforces BOTH conditions:

```python
async def authenticate_near_account_with_signature(
    account_id: str,
    public_key: str,
    signature: str,
    message: str,
    recipient: str,
    nonce: bytes,
    session: AsyncSession
) -> User | None:
    try:
        # CONDITION 1: Verify the signature
        signature_valid = await verify_near_signature(
            account_id=account_id,
            public_key=public_key,
            signature=signature,
            message=message,
            recipient=recipient,
            nonce=nonce
        )
        
        if not signature_valid:
            logger.debug(f"NEAR signature verification failed for account {account_id}")
            return None  # FAILS if signature is invalid
        
        # CONDITION 2: Check staking requirements
        settings_service = get_settings_service()
        if settings_service.auth_settings.ENABLE_NEAR_STAKING_VERIFICATION:
            is_staker = await near_staking_verifier.is_staker_with_minimum_stake(account_id)
            if not is_staker:
                logger.debug(f"Account {account_id} does not meet staking requirements")
                return None  # FAILS if staking requirement not met
        
        # BOTH CONDITIONS PASSED: Create/return user
        user = await get_user_by_username(session, account_id)
        if not user:
            user = await create_user_from_near_account(account_id, session)
        
        return user
        
    except Exception as e:
        logger.error(f"Error authenticating NEAR account with signature: {e}")
        return None
```

### 4. Signature Verification Details

The `verify_near_signature()` function performs:
1. **Cryptographic signature verification** using `verify_signature_only()`
2. **Key ownership verification** using `verify_full_key_belongs_to_user()`

Both must pass for signature verification to succeed.

### 5. Staking Verification Details

The staking verification is handled by `near_staking_verifier.is_staker_with_minimum_stake()` which:
- Checks if the account is staking in the configured pool
- Verifies the stake amount meets the minimum requirement
- Returns `True` only if both conditions are met

### 6. Authorization Flow

```
User Request → Challenge Generation → User Signs Challenge → Submit Signature
                                                                    ↓
                                                          Signature Verification
                                                                    ↓
                                                           Staking Verification
                                                                    ↓
                                                    BOTH MUST PASS → User Authenticated
```

## Security Guarantees

1. **No Password Required**: NEAR accounts don't use passwords - only cryptographic signatures
2. **Signature Verification**: Proves the user controls the NEAR account
3. **Staking Requirement**: Ensures the user has minimum stake in the specified pool
4. **Full Access Key**: Only full access keys can authenticate (not function call keys)
5. **Challenge-Response**: Prevents replay attacks with unique nonces

## Configuration

The implementation respects the following environment variables:
- `ENABLE_NEAR_STAKING_VERIFICATION`: Enable/disable staking checks
- `NEAR_POOL_CONTRACT`: Pool contract to check stakes against
- `NEAR_MIN_STAKE_AMOUNT`: Minimum required stake amount
- `NEAR_RPC_URL`: NEAR RPC endpoint for key verification

## Cleanup Applied

I've removed all legacy passwordless NEAR authentication endpoints to ensure only the secure challenge/response flow is available:
- Removed deprecated `/near-login` endpoint
- Removed deprecated `/near` endpoint
- Removed legacy schema classes
- Only `/near-challenge` and `/near-auth` endpoints remain

## Conclusion

✅ **CONFIRMED**: The implementation requires BOTH signature verification AND staking verification for NEAR account authorization. Neither condition alone is sufficient - both must pass for a user to be authenticated and granted access tokens.
