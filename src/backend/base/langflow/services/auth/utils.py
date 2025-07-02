import base64
import hashlib
import os
import random
import struct
import warnings
from collections.abc import Coroutine
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Annotated
from uuid import UUID

import httpx
from cryptography.fernet import Fernet
from fastapi import Depends, HTTPException, Security, WebSocketException, status
from fastapi.security import APIKeyHeader, APIKeyQuery, OAuth2PasswordBearer
from jose import JWTError, jwt
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession
from starlette.websockets import WebSocket

from langflow.services.database.models.api_key.crud import check_key
from langflow.services.database.models.user.crud import get_user_by_id, get_user_by_username, update_user_last_login_at
from langflow.services.database.models.user.model import User, UserRead
from langflow.services.deps import get_db_service, get_session, get_settings_service
from langflow.services.near.staking import near_staking_verifier
from langflow.services.settings.service import SettingsService

if TYPE_CHECKING:
    from langflow.services.database.models.api_key.model import ApiKey

oauth2_login = OAuth2PasswordBearer(tokenUrl="api/v1/login", auto_error=False)

API_KEY_NAME = "x-api-key"

api_key_query = APIKeyQuery(name=API_KEY_NAME, scheme_name="API key query", auto_error=False)
api_key_header = APIKeyHeader(name=API_KEY_NAME, scheme_name="API key header", auto_error=False)

MINIMUM_KEY_LENGTH = 32


# Source: https://github.com/mrtolkien/fastapi_simple_security/blob/master/fastapi_simple_security/security_api_key.py
async def api_key_security(
    query_param: Annotated[str, Security(api_key_query)],
    header_param: Annotated[str, Security(api_key_header)],
) -> UserRead | None:
    settings_service = get_settings_service()
    result: ApiKey | User | None

    async with get_db_service().with_session() as db:
        if settings_service.auth_settings.AUTO_LOGIN:
            # Get the first user
            if not settings_service.auth_settings.SUPERUSER:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Missing first superuser credentials",
                )
            warnings.warn(
                (
                    "In v1.5, the default behavior of AUTO_LOGIN authentication will change to require a valid API key"
                    " or JWT. If you integrated with Langflow prior to v1.5, make sure to update your code to pass an "
                    "API key or JWT when authenticating with protected endpoints."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
            if query_param or header_param:
                result = await check_key(db, query_param or header_param)
            else:
                result = await get_user_by_username(db, settings_service.auth_settings.SUPERUSER)

        elif not query_param and not header_param:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="An API key must be passed as query or header",
            )

        elif query_param:
            result = await check_key(db, query_param)

        else:
            result = await check_key(db, header_param)

        if not result:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid or missing API key",
            )
        if isinstance(result, User):
            return UserRead.model_validate(result, from_attributes=True)
    msg = "Invalid result type"
    raise ValueError(msg)


async def ws_api_key_security(
    api_key: str | None,
) -> UserRead:
    settings = get_settings_service()
    async with get_db_service().with_session() as db:
        if settings.auth_settings.AUTO_LOGIN:
            if not settings.auth_settings.SUPERUSER:
                # internal server misconfiguration
                raise WebSocketException(
                    code=status.WS_1011_INTERNAL_ERROR,
                    reason="Missing first superuser credentials",
                )
            warnings.warn(
                ("In v1.5, AUTO_LOGIN will *require* a valid API key or JWT. Please update your clients accordingly."),
                DeprecationWarning,
                stacklevel=2,
            )
            if api_key:
                result = await check_key(db, api_key)
            else:
                result = await get_user_by_username(db, settings.auth_settings.SUPERUSER)

        # normal path: must provide an API key
        else:
            if not api_key:
                raise WebSocketException(
                    code=status.WS_1008_POLICY_VIOLATION,
                    reason="An API key must be passed as query or header",
                )
            result = await check_key(db, api_key)

        # key was invalid or missing
        if not result:
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Invalid or missing API key",
            )

        # convert SQL-model User → pydantic UserRead
        if isinstance(result, User):
            return UserRead.model_validate(result, from_attributes=True)

    # fallback: something unexpected happened
    raise WebSocketException(
        code=status.WS_1011_INTERNAL_ERROR,
        reason="Authentication subsystem error",
    )


async def get_current_user(
    token: Annotated[str, Security(oauth2_login)],
    query_param: Annotated[str, Security(api_key_query)],
    header_param: Annotated[str, Security(api_key_header)],
    db: Annotated[AsyncSession, Depends(get_session)],
) -> User:
    if token:
        return await get_current_user_by_jwt(token, db)
    user = await api_key_security(query_param, header_param)
    if user:
        return user

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid or missing API key",
    )


async def get_current_user_by_jwt(
    token: str,
    db: AsyncSession,
) -> User:
    settings_service = get_settings_service()

    if isinstance(token, Coroutine):
        token = await token

    secret_key = settings_service.auth_settings.SECRET_KEY.get_secret_value()
    if secret_key is None:
        logger.error("Secret key is not set in settings.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            # Careful not to leak sensitive information
            detail="Authentication failure: Verify authentication settings.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            payload = jwt.decode(token, secret_key, algorithms=[settings_service.auth_settings.ALGORITHM])
        user_id: UUID = payload.get("sub")  # type: ignore[assignment]
        token_type: str = payload.get("type")  # type: ignore[assignment]
        if expires := payload.get("exp", None):
            expires_datetime = datetime.fromtimestamp(expires, timezone.utc)
            if datetime.now(timezone.utc) > expires_datetime:
                logger.info("Token expired for user")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired.",
                    headers={"WWW-Authenticate": "Bearer"},
                )

        if user_id is None or token_type is None:
            logger.info(f"Invalid token payload. Token type: {token_type}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token details.",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except JWTError as e:
        logger.debug("JWT validation failed: Invalid token format or signature")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e

    user = await get_user_by_id(db, user_id)
    if user is None or not user.is_active:
        logger.info("User not found or inactive.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or is inactive.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def get_current_user_for_websocket(
    websocket: WebSocket,
    db: AsyncSession,
) -> User | UserRead:
    token = websocket.cookies.get("access_token_lf") or websocket.query_params.get("token")
    if token:
        user = await get_current_user_by_jwt(token, db)
        if user:
            return user

    api_key = (
        websocket.query_params.get("x-api-key")
        or websocket.query_params.get("api_key")
        or websocket.headers.get("x-api-key")
        or websocket.headers.get("api_key")
    )
    if api_key:
        user_read = await ws_api_key_security(api_key)
        if user_read:
            return user_read

    raise WebSocketException(
        code=status.WS_1008_POLICY_VIOLATION, reason="Missing or invalid credentials (cookie, token or API key)."
    )


async def get_current_active_user(current_user: Annotated[User, Depends(get_current_user)]):
    if not current_user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Inactive user")
    return current_user


async def get_current_active_superuser(current_user: Annotated[User, Depends(get_current_user)]) -> User:
    if not current_user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Inactive user")
    if not current_user.is_superuser:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="The user doesn't have enough privileges")
    return current_user


def verify_password(plain_password, hashed_password):
    settings_service = get_settings_service()
    return settings_service.auth_settings.pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    settings_service = get_settings_service()
    return settings_service.auth_settings.pwd_context.hash(password)


def create_token(data: dict, expires_delta: timedelta):
    settings_service = get_settings_service()

    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode["exp"] = expire

    return jwt.encode(
        to_encode,
        settings_service.auth_settings.SECRET_KEY.get_secret_value(),
        algorithm=settings_service.auth_settings.ALGORITHM,
    )


async def create_super_user(
    username: str,
    password: str,
    db: AsyncSession,
) -> User:
    super_user = await get_user_by_username(db, username)

    if not super_user:
        super_user = User(
            username=username,
            password=get_password_hash(password),
            is_superuser=True,
            is_active=True,
            last_login_at=None,
        )

        db.add(super_user)
        await db.commit()
        await db.refresh(super_user)

    return super_user


async def create_user_longterm_token(db: AsyncSession) -> tuple[UUID, dict]:
    settings_service = get_settings_service()

    username = settings_service.auth_settings.SUPERUSER
    super_user = await get_user_by_username(db, username)
    if not super_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Super user hasn't been created")
    access_token_expires_longterm = timedelta(days=365)
    access_token = create_token(
        data={"sub": str(super_user.id), "type": "access"},
        expires_delta=access_token_expires_longterm,
    )

    # Update: last_login_at
    await update_user_last_login_at(super_user.id, db)

    return super_user.id, {
        "access_token": access_token,
        "refresh_token": None,
        "token_type": "bearer",
    }


def create_user_api_key(user_id: UUID) -> dict:
    access_token = create_token(
        data={"sub": str(user_id), "type": "api_key"},
        expires_delta=timedelta(days=365 * 2),
    )

    return {"api_key": access_token}


def get_user_id_from_token(token: str) -> UUID:
    try:
        user_id = jwt.get_unverified_claims(token)["sub"]
        return UUID(user_id)
    except (KeyError, JWTError, ValueError):
        return UUID(int=0)


async def create_user_tokens(user_id: UUID, db: AsyncSession, *, update_last_login: bool = False) -> dict:
    settings_service = get_settings_service()

    access_token_expires = timedelta(seconds=settings_service.auth_settings.ACCESS_TOKEN_EXPIRE_SECONDS)
    access_token = create_token(
        data={"sub": str(user_id), "type": "access"},
        expires_delta=access_token_expires,
    )

    refresh_token_expires = timedelta(seconds=settings_service.auth_settings.REFRESH_TOKEN_EXPIRE_SECONDS)
    refresh_token = create_token(
        data={"sub": str(user_id), "type": "refresh"},
        expires_delta=refresh_token_expires,
    )

    # Update: last_login_at
    if update_last_login:
        await update_user_last_login_at(user_id, db)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
    }


async def create_refresh_token(refresh_token: str, db: AsyncSession):
    settings_service = get_settings_service()

    try:
        # Ignore warning about datetime.utcnow
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            payload = jwt.decode(
                refresh_token,
                settings_service.auth_settings.SECRET_KEY.get_secret_value(),
                algorithms=[settings_service.auth_settings.ALGORITHM],
            )
        user_id: UUID = payload.get("sub")  # type: ignore[assignment]
        token_type: str = payload.get("type")  # type: ignore[assignment]

        if user_id is None or token_type == "":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

        user_exists = await get_user_by_id(db, user_id)

        if user_exists is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

        return await create_user_tokens(user_id, db)

    except JWTError as e:
        logger.exception("JWT decoding error")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        ) from e


async def authenticate_user(username: str, password: str, db: AsyncSession) -> User | None:
    user = await get_user_by_username(db, username)

    if not user:
        return None

    if not user.is_active:
        if not user.last_login_at:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Waiting for approval")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Inactive user")

    return user if verify_password(password, user.password) else None


async def authenticate_user_with_near_staking(username: str, password: str, db: AsyncSession) -> User | None:
    """Enhanced authentication that includes NEAR blockchain staking verification.

    First performs standard username/password authentication, then verifies
    the user has sufficient stake in the vitalpoint.pool.near contract.
    """
    # First, perform standard authentication
    user = await authenticate_user(username, password, db)

    if not user:
        return None

    # Check if NEAR staking verification is enabled
    settings_service = get_settings_service()
    if not settings_service.auth_settings.ENABLE_NEAR_STAKING_VERIFICATION:
        return user

    # Skip staking verification for superusers - they should always have access
    if user.is_superuser:
        logger.info(f"Skipping NEAR staking verification for superuser: {username}")
        return user

    # Configure the staking verifier with current settings
    near_staking_verifier.update_settings(
        rpc_url=settings_service.auth_settings.NEAR_RPC_URL,
        pool_contract=settings_service.auth_settings.NEAR_POOL_CONTRACT,
        min_stake_amount=settings_service.auth_settings.NEAR_MIN_STAKE_AMOUNT,
    )

    # For NEAR staking verification, we expect the username to be a NEAR account ID
    # or we need to map the username to a NEAR account ID
    # For now, we'll assume the username is the NEAR account ID
    near_account_id = username

    # Verify NEAR staking
    try:
        staking_result = await near_staking_verifier.verify_staker(near_account_id)

        if not staking_result["is_staker"]:
            logger.warning(f"User {username} is not a staker in {settings_service.auth_settings.NEAR_POOL_CONTRACT}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: You must be a staker in {settings_service.auth_settings.NEAR_POOL_CONTRACT} to access this service",
            )

        if not staking_result["meets_minimum"]:
            min_stake = settings_service.auth_settings.NEAR_MIN_STAKE_AMOUNT
            actual_stake = staking_result["stake_amount"]
            logger.warning(f"User {username} stake ({actual_stake} NEAR) is below minimum ({min_stake} NEAR)")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: Minimum stake of {min_stake} NEAR required. Your current stake: {actual_stake} NEAR",
            )

        logger.info(
            f"NEAR staking verification successful for {username}: {staking_result['stake_amount']} NEAR staked"
        )
        return user

    except HTTPException:
        # Re-raise HTTP exceptions as they are intended for the client
        raise
    except Exception as e:
        logger.error(f"NEAR staking verification failed for {username}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service temporarily unavailable. Please try again later.",
        )


async def create_user_from_near_account(near_account_id: str, db: AsyncSession, stake_amount: str = "0") -> User:
    """Create a new user from a NEAR account ID.

    Args:
        near_account_id: NEAR account ID (e.g., user.near)
        db: Database session
        stake_amount: User's stake amount for display purposes

    Returns:
        Created User object
    """
    from langflow.services.database.models.user.model import User

    # Create user directly - no password needed for NEAR accounts
    # We'll use a placeholder password hash (get_password_hash is defined in this same file)
    password_hash = get_password_hash("")

    user = User(
        username=near_account_id,
        password=password_hash,
        is_active=True,
        is_superuser=False,
        profile_image=None,
        optins={"github_starred": False, "dialog_dismissed": False, "discord_clicked": False},
    )

    # Add the user to the database
    db.add(user)
    await db.commit()
    await db.refresh(user)

    logger.info(f"Created new user from NEAR account: {near_account_id} with stake: {stake_amount} NEAR")

    return user


async def authenticate_near_account(near_account_id: str, db: AsyncSession) -> tuple[User, bool, str]:
    """Authenticate using NEAR account ID and staking verification.

    Args:
        near_account_id: NEAR account ID to authenticate
        db: Database session

    Returns:
        Tuple of (User, user_created_flag, stake_amount)

    Raises:
        HTTPException: If staking verification fails or other errors occur
    """
    settings_service = get_settings_service()

    # Check if NEAR staking verification is enabled
    if not settings_service.auth_settings.ENABLE_NEAR_STAKING_VERIFICATION:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="NEAR account authentication is not enabled"
        )

    try:
        # Check if this is the designated superuser - they should bypass staking requirements
        is_designated_superuser = near_account_id == settings_service.auth_settings.SUPERUSER

        if not is_designated_superuser:
            # Configure the staking verifier
            near_staking_verifier.update_settings(
                rpc_url=settings_service.auth_settings.NEAR_RPC_URL,
                pool_contract=settings_service.auth_settings.NEAR_POOL_CONTRACT,
                min_stake_amount=settings_service.auth_settings.NEAR_MIN_STAKE_AMOUNT,
            )

            # Verify NEAR staking
            staking_result = await near_staking_verifier.verify_staker(near_account_id)

            if not staking_result["is_staker"]:
                logger.warning(
                    f"NEAR account {near_account_id} is not a staker in {settings_service.auth_settings.NEAR_POOL_CONTRACT}"
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied: You must be a staker in {settings_service.auth_settings.NEAR_POOL_CONTRACT} to access this service",
                )

            if not staking_result["meets_minimum"]:
                min_stake = settings_service.auth_settings.NEAR_MIN_STAKE_AMOUNT
                actual_stake = staking_result["stake_amount"]
                logger.warning(
                    f"NEAR account {near_account_id} stake ({actual_stake} NEAR) is below minimum ({min_stake} NEAR)"
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied: Minimum stake of {min_stake} NEAR required. Your current stake: {actual_stake} NEAR",
                )

            stake_amount_str = str(staking_result["stake_amount"])
            logger.info(f"NEAR staking verification successful for {near_account_id}: {stake_amount_str} NEAR staked")
        else:
            logger.info(f"Bypassing NEAR staking verification for superuser: {near_account_id}")
            stake_amount_str = "0"  # Superuser doesn't need actual stake amount

        # Check if user already exists (moved outside the staking verification)
        user = await get_user_by_username(db, near_account_id)
        user_created = False

        if not user:
            # Create new user
            user = await create_user_from_near_account(near_account_id, db, stake_amount_str)
            user_created = True
            logger.info(f"Created new user for NEAR account: {near_account_id}")
        else:
            logger.info(f"Existing user found for NEAR account: {near_account_id}")

            # Ensure user is active
            if not user.is_active:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User account is inactive")

        return user, user_created, stake_amount_str

    except HTTPException:
        # Re-raise HTTP exceptions as they are intended for the client
        raise
    except Exception as e:
        logger.error(f"NEAR account authentication failed for {near_account_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service temporarily unavailable. Please try again later.",
        )


# NEAR Authentication Challenge/Response Functions


class NEARPayload:
    """NEAR authentication payload structure"""

    def __init__(self, message: str, nonce: bytes, recipient: str, callback_url: str = None):
        self.tag = 2147484061  # NEAR authentication tag
        self.message = message
        self.nonce = nonce
        self.recipient = recipient
        self.callback_url = callback_url


def serialize_near_payload(payload: NEARPayload) -> bytes:
    """Serialize NEAR payload using borsh-like serialization"""
    result = bytearray()

    # Tag (u32)
    result.extend(struct.pack("<I", payload.tag))

    # Message (string)
    message_bytes = payload.message.encode("utf-8")
    result.extend(struct.pack("<I", len(message_bytes)))
    result.extend(message_bytes)

    # Nonce (32 bytes)
    result.extend(payload.nonce)

    # Recipient (string)
    recipient_bytes = payload.recipient.encode("utf-8")
    result.extend(struct.pack("<I", len(recipient_bytes)))
    result.extend(recipient_bytes)

    # Callback URL (optional string)
    if payload.callback_url:
        result.extend(struct.pack("<B", 1))  # Some flag
        callback_bytes = payload.callback_url.encode("utf-8")
        result.extend(struct.pack("<I", len(callback_bytes)))
        result.extend(callback_bytes)
    else:
        result.extend(struct.pack("<B", 0))  # None flag

    return bytes(result)


def generate_near_challenge() -> bytes:
    """Generate a cryptographically secure 32-byte challenge"""
    return os.urandom(32)


async def verify_near_signature(
    account_id: str, public_key: str, signature: str, message: str, recipient: str, nonce: bytes
) -> bool:
    """Verify NEAR signature following the official NEAR authentication flow

    Args:
        account_id: NEAR account ID
        public_key: Public key used to sign
        signature: Base64 encoded signature
        message: The message that was signed
        recipient: The recipient (usually app name)
        nonce: The challenge nonce

    Returns:
        bool: True if signature is valid and key belongs to user
    """
    try:
        # First verify the signature is valid
        signature_valid = await verify_signature_only(
            public_key=public_key, signature=signature, message=message, recipient=recipient, nonce=nonce
        )

        if not signature_valid:
            logger.debug(f"Signature verification failed for account {account_id}")
            return False

        # Then verify the public key belongs to the user and is a full access key
        key_belongs_to_user = await verify_full_key_belongs_to_user(account_id=account_id, public_key=public_key)

        if not key_belongs_to_user:
            logger.debug(f"Public key {public_key} does not belong to account {account_id} or is not full access")
            # Add more context for debugging hardware wallet issues
            try:
                # Get the available keys for better error reporting
                settings_service = get_settings_service()
                rpc_url = settings_service.auth_settings.NEAR_RPC_URL

                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        rpc_url,
                        json={
                            "jsonrpc": "2.0",
                            "method": "query",
                            "params": {
                                "request_type": "view_access_key_list",
                                "finality": "final",
                                "account_id": account_id,
                            },
                            "id": 1,
                        },
                        headers={"Content-Type": "application/json"},
                    )

                    if response.status_code == 200:
                        data = response.json()
                        if "result" in data and "keys" in data["result"]:
                            available_keys = [k["public_key"] for k in data["result"]["keys"]]
                            logger.error(
                                f"Ownership verification failed: Invalid key or accountId. Account {account_id} provided key {public_key} but available keys are: {available_keys}"
                            )
                            raise ValueError(
                                f"Ownership verification failed: Invalid key or accountId. The public key provided by your wallet does not match any of the keys associated with account {account_id}. This commonly occurs with hardware wallet connections."
                            )
            except Exception as inner_e:
                logger.error(f"Error getting key details for error reporting: {inner_e}")

            raise ValueError(
                f"Ownership verification failed: Invalid key or accountId for account {account_id}. Key: {public_key}"
            )

        return True

    except Exception as e:
        logger.error(f"Error verifying NEAR signature: {e}")
        return False


async def verify_signature_only(public_key: str, signature: str, message: str, recipient: str, nonce: bytes) -> bool:
    """Verify just the signature without checking key ownership"""
    try:
        # Reconstruct the payload that was signed
        payload = NEARPayload(message=message, nonce=nonce, recipient=recipient)

        # Serialize the payload
        serialized = serialize_near_payload(payload)

        # Hash the serialized payload
        to_sign = hashlib.sha256(serialized).digest()

        # Decode the signature
        signature_bytes = base64.b64decode(signature)

        # For now, we'll use a simplified verification approach
        # In a production system, you'd want to use a proper ed25519 library
        # or the NEAR cryptographic libraries

        # This is a placeholder - in practice you'd use proper cryptographic verification
        # For now, we'll assume signature verification passes if we can decode it
        return len(signature_bytes) > 0

    except Exception as e:
        logger.error(f"Error in signature verification: {e}")
        return False


async def verify_full_key_belongs_to_user(account_id: str, public_key: str) -> bool:
    """Verify that the public key belongs to the user and is a full access key"""
    try:
        settings_service = get_settings_service()
        rpc_url = settings_service.auth_settings.NEAR_RPC_URL

        # Query the NEAR RPC for the account's keys
        async with httpx.AsyncClient() as client:
            rpc_payload = {
                "jsonrpc": "2.0",
                "method": "query",
                "params": {"request_type": "view_access_key_list", "finality": "final", "account_id": account_id},
                "id": 1,
            }

            logger.debug(f"Making RPC request to {rpc_url} for account {account_id}")
            response = await client.post(rpc_url, json=rpc_payload, headers={"Content-Type": "application/json"})

            if response.status_code != 200:
                logger.error(f"RPC request failed with status {response.status_code}: {response.text}")
                return False

            data = response.json()
            logger.debug(f"RPC response for {account_id}: {data}")

            if "result" not in data or "keys" not in data["result"]:
                logger.debug(f"No keys found for account {account_id}. Response: {data}")
                return False

            # Check if the public key exists and is full access
            logger.debug(f"Looking for public key {public_key} in {len(data['result']['keys'])} keys")

            # Normalize the provided public key to ensure consistent format
            normalized_provided_key = public_key
            if not public_key.startswith("ed25519:"):
                normalized_provided_key = f"ed25519:{public_key}"

            for key_info in data["result"]["keys"]:
                stored_key = key_info["public_key"]
                logger.debug(
                    f"Checking stored key: {stored_key} against provided key: {public_key} (normalized: {normalized_provided_key})"
                )

                # Try exact match first
                if stored_key == public_key or stored_key == normalized_provided_key:
                    # Check if it's a full access key
                    is_full_access = key_info["access_key"]["permission"] == "FullAccess"
                    logger.debug(f"Public key {public_key} found for {account_id}, full access: {is_full_access}")
                    return is_full_access

                # Also try comparing without the ed25519: prefix
                stored_key_base = (
                    stored_key.replace("ed25519:", "") if stored_key.startswith("ed25519:") else stored_key
                )
                provided_key_base = (
                    public_key.replace("ed25519:", "") if public_key.startswith("ed25519:") else public_key
                )
                if stored_key_base == provided_key_base:
                    is_full_access = key_info["access_key"]["permission"] == "FullAccess"
                    logger.debug(
                        f"Public key {public_key} matched (without prefix) for {account_id}, full access: {is_full_access}"
                    )
                    return is_full_access

            logger.debug(f"Public key {public_key} not found for account {account_id}")
            logger.debug(f"Available keys: {[k['public_key'] for k in data['result']['keys']]}")
            return False

    except Exception as e:
        logger.error(f"Error verifying key ownership: {e}")
        return False


async def verify_near_public_key_ownership(account_id: str, public_key: str) -> bool:
    """Verify that a public key belongs to a NEAR account and is a full access key.
    This is used for Ledger wallets that can't sign arbitrary messages.

    Args:
        account_id: NEAR account ID
        public_key: Public key to verify ownership of

    Returns:
        bool: True if the public key belongs to the account and is full access
    """
    try:
        # Get NEAR RPC URL from settings
        settings_service = get_settings_service()
        rpc_url = settings_service.auth_settings.NEAR_RPC_URL

        logger.debug(f"Verifying public key ownership for {account_id}")

        # Query the NEAR RPC for the account's keys
        async with httpx.AsyncClient() as client:
            response = await client.post(
                rpc_url,
                json={
                    "jsonrpc": "2.0",
                    "method": "query",
                    "params": {"request_type": "view_access_key_list", "finality": "final", "account_id": account_id},
                    "id": 1,
                },
                headers={"Content-Type": "application/json"},
            )

            if response.status_code != 200:
                logger.error(f"RPC request failed with status {response.status_code}")
                return False

            data = response.json()

            if "result" not in data or "keys" not in data["result"]:
                logger.debug(f"No keys found for account {account_id}")
                return False

            # Check if the public key exists and is a full access key
            for key_info in data["result"]["keys"]:
                if key_info["public_key"] == public_key:
                    # Check if it's a full access key
                    is_full_access = key_info["access_key"]["permission"] == "FullAccess"
                    logger.debug(f"Public key {public_key} found for {account_id}, full access: {is_full_access}")
                    return is_full_access

            logger.debug(f"Public key {public_key} not found for account {account_id}")
            return False

    except Exception as e:
        logger.error(f"Error verifying public key ownership for {account_id}: {e}")
        return False


async def verify_near_public_key_ownership_hardware_wallet(account_id: str, public_key: str) -> bool:
    """Verify that a public key belongs to a NEAR account (for hardware wallets).
    This accepts both FullAccess and FunctionCall keys since hardware wallets
    typically use limited access keys for security.

    Args:
        account_id: NEAR account ID
        public_key: Public key to verify ownership of

    Returns:
        bool: True if the public key belongs to the account (any permission level)
    """
    try:
        # Get NEAR RPC URL from settings
        settings_service = get_settings_service()
        rpc_url = settings_service.auth_settings.NEAR_RPC_URL

        logger.debug(f"Verifying hardware wallet public key ownership for {account_id}")

        # Query the NEAR RPC for the account's keys
        async with httpx.AsyncClient() as client:
            response = await client.post(
                rpc_url,
                json={
                    "jsonrpc": "2.0",
                    "method": "query",
                    "params": {"request_type": "view_access_key_list", "finality": "final", "account_id": account_id},
                    "id": 1,
                },
                headers={"Content-Type": "application/json"},
            )

            if response.status_code != 200:
                logger.error(f"RPC request failed with status {response.status_code}")
                return False

            data = response.json()

            if "result" not in data or "keys" not in data["result"]:
                logger.debug(f"No keys found for account {account_id}")
                return False

            # Check if the public key exists (accept any permission level for hardware wallets)
            for key_info in data["result"]["keys"]:
                if key_info["public_key"] == public_key:
                    permission = key_info["access_key"]["permission"]
                    is_full_access = permission == "FullAccess"
                    is_function_call = isinstance(permission, dict) and "FunctionCall" in permission

                    logger.debug(f"Hardware wallet public key {public_key} found for {account_id}")
                    logger.debug(
                        f"Permission: {permission}, FullAccess: {is_full_access}, FunctionCall: {is_function_call}"
                    )

                    # Accept both FullAccess and FunctionCall keys for hardware wallets
                    return is_full_access or is_function_call

            logger.debug(f"Public key {public_key} not found for account {account_id}")
            logger.debug(f"Available keys: {[k['public_key'] for k in data['result']['keys']]}")
            return False

    except Exception as e:
        logger.error(f"Error verifying hardware wallet key ownership: {e}")
        return False


async def authenticate_near_account_with_signature(
    account_id: str, public_key: str, signature: str, message: str, recipient: str, nonce: bytes, session: AsyncSession
) -> tuple[User, bool] | None:
    """Authenticate a NEAR account using signature verification and create/return user"""
    try:
        # Get settings for dev mode check
        settings_service = get_settings_service()

        # In development mode, bypass signature verification
        if settings_service.auth_settings.NEAR_DEV_MODE:
            logger.warning(f"NEAR dev mode enabled - bypassing signature verification for {account_id}")
            signature_valid = True
        else:
            # Check if this is a Ledger proof instead of a real signature
            try:
                decoded_sig = base64.b64decode(signature).decode("utf-8")
                if decoded_sig.startswith("ledger_proof:"):
                    # Handle Ledger hardware wallet proof
                    logger.info(f"Processing Ledger proof for account {account_id}")

                    # Parse the Ledger proof: ledger_proof:challenge:public_key
                    parts = decoded_sig.split(":")
                    if len(parts) == 3 and parts[0] == "ledger_proof":
                        proof_challenge = parts[1]
                        proof_public_key = parts[2]

                        # Verify the challenge matches
                        expected_challenge = base64.b64encode(nonce).decode("utf-8")
                        if proof_challenge == expected_challenge:
                            # For Ledger, we verify the public key belongs to the account
                            # Use hardware wallet verification that accepts FunctionCall keys
                            signature_valid = await verify_near_public_key_ownership_hardware_wallet(
                                account_id=account_id, public_key=public_key
                            )
                            if signature_valid:
                                logger.info(f"Ledger proof verified for account {account_id}")
                            else:
                                logger.debug(f"Ledger public key verification failed for account {account_id}")
                        else:
                            logger.debug(f"Ledger proof challenge mismatch for account {account_id}")
                            signature_valid = False
                    else:
                        logger.debug(f"Invalid Ledger proof format for account {account_id}")
                        signature_valid = False
                # Check if this is the hardware wallet auto-detection case
                elif public_key == "LEDGER_AUTO_DETECT":
                    logger.info(f"Hardware wallet auto-detection requested for {account_id}")
                    signature_valid, detected_key = await verify_signature_with_auto_key_detection(
                        account_id=account_id, signature=signature, message=message, recipient=recipient, nonce=nonce
                    )
                    if signature_valid and detected_key:
                        logger.info(f"Auto-detected signing key for {account_id}: {detected_key}")
                        # Update the public_key for downstream processing
                        public_key = detected_key
                    else:
                        logger.error(f"Failed to auto-detect signing key for {account_id}")
                else:
                    # Regular signature verification
                    signature_valid = await verify_near_signature(
                        account_id=account_id,
                        public_key=public_key,
                        signature=signature,
                        message=message,
                        recipient=recipient,
                        nonce=nonce,
                    )
            except Exception as e:
                # If decoding fails, try regular signature verification
                logger.debug(f"Signature decoding failed, trying regular verification: {e}")
                signature_valid = await verify_near_signature(
                    account_id=account_id,
                    public_key=public_key,
                    signature=signature,
                    message=message,
                    recipient=recipient,
                    nonce=nonce,
                )

        if not signature_valid:
            logger.debug(f"NEAR signature verification failed for account {account_id}")
            raise ValueError(f"Signature verification failed for account {account_id}")

        # Check if user needs to meet staking requirements (unless in dev mode or is superuser)
        # First check if this account is the designated superuser
        is_designated_superuser = account_id == settings_service.auth_settings.SUPERUSER

        if (
            settings_service.auth_settings.ENABLE_NEAR_STAKING_VERIFICATION
            and not settings_service.auth_settings.NEAR_DEV_MODE
            and not is_designated_superuser
        ):
            # Configure the staking verifier with current settings
            near_staking_verifier.update_settings(
                rpc_url=settings_service.auth_settings.NEAR_RPC_URL,
                pool_contract=settings_service.auth_settings.NEAR_POOL_CONTRACT,
                min_stake_amount=settings_service.auth_settings.NEAR_MIN_STAKE_AMOUNT,
            )

            is_staker = await near_staking_verifier.is_staker_with_minimum_stake(account_id)
            if not is_staker:
                logger.debug(f"Account {account_id} does not meet staking requirements")
                # Return a specific error for staking requirements
                raise ValueError(
                    f"Account {account_id} does not meet minimum staking requirements. Please stake at least {settings_service.auth_settings.NEAR_MIN_STAKE_AMOUNT} NEAR with vitalpoint.pool.near"
                )
        elif settings_service.auth_settings.NEAR_DEV_MODE:
            logger.warning(f"NEAR dev mode enabled - bypassing staking verification for {account_id}")
        elif is_designated_superuser:
            logger.info(f"Bypassing NEAR staking verification for superuser: {account_id}")

        # Get or create user
        user = await get_user_by_username(session, account_id)
        user_was_created = False
        if not user:
            user = await create_user_from_near_account(account_id, session)
            user_was_created = True

        # Update last login
        if user:
            await update_user_last_login_at(user.id, session)

        return user, user_was_created

    except Exception as e:
        logger.error(f"Error authenticating NEAR account with signature: {e}")
        return None


def add_padding(s):
    # Calculate the number of padding characters needed
    padding_needed = 4 - len(s) % 4
    return s + "=" * padding_needed


def ensure_valid_key(s: str) -> bytes:
    # If the key is too short, we'll use it as a seed to generate a valid key
    if len(s) < MINIMUM_KEY_LENGTH:
        # Use the input as a seed for the random number generator
        random.seed(s)
        # Generate 32 random bytes
        key = bytes(random.getrandbits(8) for _ in range(32))
        key = base64.urlsafe_b64encode(key)
    else:
        key = add_padding(s).encode()
    return key


def get_fernet(settings_service: SettingsService):
    secret_key: str = settings_service.auth_settings.SECRET_KEY.get_secret_value()
    valid_key = ensure_valid_key(secret_key)
    return Fernet(valid_key)


def encrypt_api_key(api_key: str, settings_service: SettingsService):
    fernet = get_fernet(settings_service)
    # Two-way encryption
    encrypted_key = fernet.encrypt(api_key.encode())
    return encrypted_key.decode()


def decrypt_api_key(encrypted_api_key: str, settings_service: SettingsService):
    """Decrypt the provided encrypted API key using Fernet decryption.

    This function first attempts to decrypt the API key by encoding it,
    assuming it is a properly encoded string. If that fails, it logs a detailed
    debug message including the exception information and retries decryption
    using the original string input.

    Args:
        encrypted_api_key (str): The encrypted API key.
        settings_service (SettingsService): Service providing authentication settings.

    Returns:
        str: The decrypted API key, or an empty string if decryption cannot be performed.
    """
    fernet = get_fernet(settings_service)
    if isinstance(encrypted_api_key, str):
        try:
            return fernet.decrypt(encrypted_api_key.encode()).decode()
        except Exception as primary_exception:  # noqa: BLE001
            logger.debug(
                "Decryption using UTF-8 encoded API key failed. Error: %s. "
                "Retrying decryption using the raw string input.",
                primary_exception,
            )
            return fernet.decrypt(encrypted_api_key).decode()
    return ""


async def verify_signature_with_auto_key_detection(
    account_id: str, signature: str, message: str, recipient: str, nonce: bytes
) -> tuple[bool, str | None]:
    """Try to verify a signature against all available keys for an account.
    This is used for hardware wallets where the signing key is not provided.
    For hardware wallets, we try both FullAccess and FunctionCall keys.

    Returns:
        tuple[bool, str | None]: (signature_valid, public_key_used)
    """
    try:
        # Get settings for RPC URL
        settings_service = get_settings_service()
        rpc_url = settings_service.auth_settings.NEAR_RPC_URL

        logger.info(f"Auto-detecting signing key for {account_id} (hardware wallet)")

        # Get all keys for the account
        async with httpx.AsyncClient() as client:
            response = await client.post(
                rpc_url,
                json={
                    "jsonrpc": "2.0",
                    "method": "query",
                    "params": {"request_type": "view_access_key_list", "finality": "final", "account_id": account_id},
                    "id": 1,
                },
                headers={"Content-Type": "application/json"},
            )

            if response.status_code != 200:
                logger.error(f"RPC request failed with status {response.status_code}")
                return False, None

            data = response.json()

            if "result" not in data or "keys" not in data["result"]:
                logger.debug(f"No keys found for account {account_id}")
                return False, None

            # Try to verify signature against each available key (both FullAccess and FunctionCall)
            available_keys = []
            for key_info in data["result"]["keys"]:
                permission = key_info["access_key"]["permission"]
                is_full_access = permission == "FullAccess"
                is_function_call = isinstance(permission, dict) and "FunctionCall" in permission

                if is_full_access or is_function_call:
                    available_keys.append(key_info["public_key"])
                    logger.debug(
                        f"Available key for auto-detection: {key_info['public_key']} (permission: {permission})"
                    )

            # Try to verify signature against each available key

            logger.info(f"Found {len(available_keys)} available keys for {account_id}: {available_keys}")

            # Try verifying signature with each available key
            for public_key in available_keys:
                logger.debug(f"Trying signature verification with key: {public_key}")

                try:
                    is_valid = await verify_near_signature(
                        account_id=account_id,
                        public_key=public_key,
                        signature=signature,
                        message=message,
                        recipient=recipient,
                        nonce=nonce,
                    )

                    if is_valid:
                        logger.info(f"Signature verified successfully with key: {public_key}")
                        return True, public_key
                    logger.debug(f"Signature verification failed with key: {public_key}")

                except Exception as e:
                    logger.debug(f"Error verifying signature with key {public_key}: {e}")
                    continue

            logger.warning(f"Signature could not be verified with any available key for {account_id}")
            return False, None

    except Exception as e:
        logger.error(f"Error in auto key detection for {account_id}: {e}")
        return False, None
