from __future__ import annotations

import base64
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from loguru import logger

from langflow.api.utils import DbSession
from langflow.api.v1.schemas import (
    Token,
    NEARChallengeRequest,
    NEARChallengeResponse,
    NEARSignatureLogin,
    NEARLoginResponse,
    UserRead,
)
from langflow.initial_setup.setup import get_or_create_default_folder
from langflow.services.auth.utils import (
    authenticate_user,
    authenticate_user_with_near_staking,
    authenticate_near_account_with_signature,
    create_refresh_token,
    create_user_longterm_token,
    create_user_tokens,
    generate_near_challenge,
    get_current_active_user,
)
from langflow.services.database.models.user.crud import get_user_by_id, get_user_by_username
from langflow.services.deps import get_settings_service, get_variable_service

router = APIRouter(tags=["Login"])


async def get_current_user_optional(request: Request) -> Optional[UserRead]:
    """Get current user if authenticated, otherwise return None."""
    try:
        # Try to get the user from the request
        from langflow.services.auth.utils import get_current_user
        user = await get_current_user(request)
        if user and user.is_active:
            return UserRead.model_validate(user, from_attributes=True)
        return None
    except Exception:
        # If authentication fails, return None
        return None


@router.post("/login", response_model=Token)
async def login_to_get_access_token(
    response: Response,
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: DbSession,
):
    auth_settings = get_settings_service().auth_settings
    try:
        # Use NEAR staking verification if enabled, otherwise use standard authentication
        if auth_settings.ENABLE_NEAR_STAKING_VERIFICATION:
            user = await authenticate_user_with_near_staking(form_data.username, form_data.password, db)
        else:
            user = await authenticate_user(form_data.username, form_data.password, db)
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    if user:
        tokens = await create_user_tokens(user_id=user.id, db=db, update_last_login=True)
        response.set_cookie(
            "refresh_token_lf",
            tokens["refresh_token"],
            httponly=auth_settings.REFRESH_HTTPONLY,
            samesite=auth_settings.REFRESH_SAME_SITE,
            secure=auth_settings.REFRESH_SECURE,
            expires=auth_settings.REFRESH_TOKEN_EXPIRE_SECONDS,
            domain=auth_settings.COOKIE_DOMAIN,
        )
        response.set_cookie(
            "access_token_lf",
            tokens["access_token"],
            httponly=auth_settings.ACCESS_HTTPONLY,
            samesite=auth_settings.ACCESS_SAME_SITE,
            secure=auth_settings.ACCESS_SECURE,
            expires=auth_settings.ACCESS_TOKEN_EXPIRE_SECONDS,
            domain=auth_settings.COOKIE_DOMAIN,
        )
        response.set_cookie(
            "apikey_tkn_lflw",
            str(user.store_api_key),
            httponly=auth_settings.ACCESS_HTTPONLY,
            samesite=auth_settings.ACCESS_SAME_SITE,
            secure=auth_settings.ACCESS_SECURE,
            expires=None,  # Set to None to make it a session cookie
            domain=auth_settings.COOKIE_DOMAIN,
        )
        await get_variable_service().initialize_user_variables(user.id, db)
        # Create default project for user if it doesn't exist
        _ = await get_or_create_default_folder(db, user.id)
        return tokens
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Bearer"},
    )


@router.get("/auto_login")
async def auto_login(response: Response, db: DbSession):
    auth_settings = get_settings_service().auth_settings

    if auth_settings.AUTO_LOGIN:
        user_id, tokens = await create_user_longterm_token(db)
        response.set_cookie(
            "access_token_lf",
            tokens["access_token"],
            httponly=auth_settings.ACCESS_HTTPONLY,
            samesite=auth_settings.ACCESS_SAME_SITE,
            secure=auth_settings.ACCESS_SECURE,
            expires=None,  # Set to None to make it a session cookie
            domain=auth_settings.COOKIE_DOMAIN,
        )

        user = await get_user_by_id(db, user_id)

        if user:
            if user.store_api_key is None:
                user.store_api_key = ""

            response.set_cookie(
                "apikey_tkn_lflw",
                str(user.store_api_key),  # Ensure it's a string
                httponly=auth_settings.ACCESS_HTTPONLY,
                samesite=auth_settings.ACCESS_SAME_SITE,
                secure=auth_settings.ACCESS_SECURE,
                expires=None,  # Set to None to make it a session cookie
                domain=auth_settings.COOKIE_DOMAIN,
            )

        return tokens

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={
            "message": "Auto login is disabled. Please enable it in the settings",
            "auto_login": False,
        },
    )


@router.post("/refresh")
async def refresh_token(
    request: Request,
    response: Response,
    db: DbSession,
):
    auth_settings = get_settings_service().auth_settings

    token = request.cookies.get("refresh_token_lf")

    if token:
        tokens = await create_refresh_token(token, db)
        response.set_cookie(
            "refresh_token_lf",
            tokens["refresh_token"],
            httponly=auth_settings.REFRESH_HTTPONLY,
            samesite=auth_settings.REFRESH_SAME_SITE,
            secure=auth_settings.REFRESH_SECURE,
            expires=auth_settings.REFRESH_TOKEN_EXPIRE_SECONDS,
            domain=auth_settings.COOKIE_DOMAIN,
        )
        response.set_cookie(
            "access_token_lf",
            tokens["access_token"],
            httponly=auth_settings.ACCESS_HTTPONLY,
            samesite=auth_settings.ACCESS_SAME_SITE,
            secure=auth_settings.ACCESS_SECURE,
            expires=auth_settings.ACCESS_TOKEN_EXPIRE_SECONDS,
            domain=auth_settings.COOKIE_DOMAIN,
        )
        return tokens
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid refresh token",
        headers={"WWW-Authenticate": "Bearer"},
    )


@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie("refresh_token_lf")
    response.delete_cookie("access_token_lf")
    response.delete_cookie("apikey_tkn_lflw")
    return {"message": "Logout successful"}


# NEAR Authentication Endpoints

@router.post("/near-challenge", response_model=NEARChallengeResponse)
async def get_near_challenge(
    challenge_request: NEARChallengeRequest,
):
    """
    Generate a challenge for NEAR authentication.
    The frontend will use this challenge to have the user sign with their NEAR wallet.
    """
    try:
        # Generate a secure random challenge
        challenge_bytes = generate_near_challenge()
        challenge_b64 = base64.b64encode(challenge_bytes).decode('utf-8')
        
        # Standard NEAR authentication message
        message = "Login with NEAR"
        recipient = "nearflow"  # Application name
        
        return NEARChallengeResponse(
            challenge=challenge_b64,
            message=message,
            recipient=recipient
        )
        
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating NEAR challenge: {str(exc)}",
        ) from exc


@router.post("/near-auth", response_model=NEARLoginResponse)
async def near_signature_login(
    response: Response,
    login_data: NEARSignatureLogin,
    db: DbSession,
):
    """
    Authenticate using NEAR signature verification and staking requirements.
    
    This is the only supported NEAR authentication method. It requires:
    1. Valid signature verification (cryptographic proof)
    2. Staking verification (if enabled in settings)
    
    Both conditions must be met for authentication to succeed.
    """
    auth_settings = get_settings_service().auth_settings

    try:
        # Decode the challenge
        challenge_bytes = base64.b64decode(login_data.challenge)
        
        # Authenticate using signature verification AND staking verification
        auth_result = await authenticate_near_account_with_signature(
            account_id=login_data.near_account_id,
            public_key=login_data.public_key,
            signature=login_data.signature,
            message=login_data.message,
            recipient=login_data.recipient,
            nonce=challenge_bytes,
            session=db
        )

        if auth_result:
            user, user_was_created = auth_result
            # Create tokens
            tokens = await create_user_tokens(user_id=user.id, db=db, update_last_login=True)

            # Set cookies
            response.set_cookie(
                "refresh_token_lf",
                tokens["refresh_token"],
                httponly=auth_settings.REFRESH_HTTPONLY,
                samesite=auth_settings.REFRESH_SAME_SITE,
                secure=auth_settings.REFRESH_SECURE,
                expires=auth_settings.REFRESH_TOKEN_EXPIRE_SECONDS,
                domain=auth_settings.COOKIE_DOMAIN,
            )
            response.set_cookie(
                "access_token_lf",
                tokens["access_token"],
                httponly=auth_settings.ACCESS_HTTPONLY,
                samesite=auth_settings.ACCESS_SAME_SITE,
                secure=auth_settings.ACCESS_SECURE,
                expires=auth_settings.ACCESS_TOKEN_EXPIRE_SECONDS,
                domain=auth_settings.COOKIE_DOMAIN,
            )
            response.set_cookie(
                "apikey_tkn_lflw",
                str(user.store_api_key or ""),
                httponly=auth_settings.ACCESS_HTTPONLY,
                samesite=auth_settings.ACCESS_SAME_SITE,
                secure=auth_settings.ACCESS_SECURE,
                expires=None,  # Session cookie
                domain=auth_settings.COOKIE_DOMAIN,
            )

            await get_variable_service().initialize_user_variables(user.id, db)
            # Create default project for user if it doesn't exist
            _ = await get_or_create_default_folder(db, user.id)

            # Get stake amount if staking verification is enabled
            stake_amount = None
            if auth_settings.ENABLE_NEAR_STAKING_VERIFICATION:
                # Check if this is the superuser - don't query actual stake for superusers
                if login_data.near_account_id == auth_settings.SUPERUSER:
                    stake_amount = "Superuser (exempt)"
                else:
                    from langflow.services.near.staking import near_staking_verifier
                    stake_decimal = await near_staking_verifier.get_stake_amount(login_data.near_account_id)
                    stake_amount = str(stake_decimal) if stake_decimal is not None else None

            return NEARLoginResponse(
                access_token=tokens["access_token"],
                refresh_token=tokens["refresh_token"],
                token_type="bearer",
                user_created=user_was_created,
                stake_amount=stake_amount
            )

    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"NEAR signature authentication failed: {str(exc)}",
        ) from exc

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="NEAR signature authentication failed",
    )


# NOTE: Legacy NEAR endpoints have been removed. Use /near-auth with proper signature verification.


@router.get("/near-stake-check/{account_id}")
async def check_near_staking(account_id: str, request: Request):
    """Check if a NEAR account meets staking requirements."""
    auth_settings = get_settings_service().auth_settings
    
    if not auth_settings.ENABLE_NEAR_STAKING_VERIFICATION:
        raise HTTPException(status_code=404, detail="NEAR authentication not enabled")
    
    try:
        # Check if the current authenticated user is a superuser
        current_user = await get_current_user_optional(request)
        if current_user and current_user.is_superuser:
            logger.info(f"Authenticated superuser ({current_user.username}) - bypassing staking check for {account_id}")
            return {
                "meets_requirements": True,
                "current_stake": "Superuser",
                "required_stake": auth_settings.NEAR_MIN_STAKE_AMOUNT,
                "superuser": True,
                "user_id": current_user.id
            }
        
        # Check if we're in dev mode
        if auth_settings.NEAR_DEV_MODE:
            logger.info(f"NEAR dev mode enabled - bypassing staking check for {account_id}")
            return {
                "meets_requirements": True,
                "current_stake": "Dev Mode",
                "required_stake": auth_settings.NEAR_MIN_STAKE_AMOUNT,
                "dev_mode": True
            }
        
        # Check if this is the designated superuser account (fallback check)
        if account_id == auth_settings.SUPERUSER:
            logger.info(f"Superuser account detected - bypassing staking check for {account_id}")
            return {
                "meets_requirements": True,
                "current_stake": "Superuser",
                "required_stake": auth_settings.NEAR_MIN_STAKE_AMOUNT,
                "superuser": True
            }
        
        # Initialize staking verifier
        from langflow.services.near.staking import NEARStakingVerifier
        
        staking_verifier = NEARStakingVerifier(
            pool_contract=auth_settings.NEAR_POOL_CONTRACT,
            min_stake_amount=auth_settings.NEAR_MIN_STAKE_AMOUNT,
        )
        
        # Get current stake amount for the account
        current_stake = await staking_verifier.get_stake_amount(account_id)
        meets_requirements = await staking_verifier.is_staker_with_minimum_stake(account_id)
        
        logger.info(f"Staking check for {account_id}: {current_stake} NEAR (meets requirements: {meets_requirements})")
        
        return {
            "meets_requirements": meets_requirements,
            "current_stake": str(current_stake),
            "required_stake": auth_settings.NEAR_MIN_STAKE_AMOUNT,
            "dev_mode": False
        }
        
    except Exception as e:
        logger.error(f"Error checking staking for {account_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check staking requirements: {str(e)}")


@router.get("/near-auth-enabled")
async def check_near_auth_enabled():
    """Check if NEAR authentication is enabled."""
    auth_settings = get_settings_service().auth_settings
    return {
        "enabled": auth_settings.ENABLE_NEAR_STAKING_VERIFICATION,
        "pool_contract": auth_settings.NEAR_POOL_CONTRACT,
        "min_stake_amount": auth_settings.NEAR_MIN_STAKE_AMOUNT,
        "dev_mode": auth_settings.NEAR_DEV_MODE,
        "superuser": auth_settings.SUPERUSER
    }


@router.get("/check-user-exists/{near_account_id}")
async def check_user_exists(near_account_id: str, db: DbSession):
    """Check if a NEAR account already has a NearFlow user account."""
    try:
        user = await get_user_by_username(db, near_account_id)
        
        if user:
            return {
                "exists": True,
                "user_id": user.id,
                "is_active": user.is_active,
                "is_superuser": user.is_superuser,
                "created_at": user.created_at.isoformat() if hasattr(user, 'created_at') and user.created_at else None
            }
        else:
            return {
                "exists": False
            }
    except Exception as e:
        logger.error(f"Error checking if user exists for {near_account_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check user existence: {str(e)}")

# Force reload comment
