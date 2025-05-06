from fastapi import APIRouter, Request, Response
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import base64
from urllib.parse import quote, unquote
import json
from langflow.services.deps import get_settings_service
import os

# Constants
AUTH_COOKIE_NAME = "auth"

# FastAPI router
router = APIRouter(tags=["Auth"])

class SignedMessageInput(BaseModel):
    accountId: str
    publicKey: str
    signature: str  # base64-encoded
    message: str
    callbackUrl: str
    nonce: str
    recipient: str

environment = os.getenv("LANGFLOW_ENV", "development").lower()
print(f"Setting cookie for environment: {environment}")

environ = os.getenv("ENV", "development").lower()
print(f"Setting cookie for environment: {environ}")

domain = os.getenv("LANGFLOW_DOMAIN", "vitalpoint.ai").lower()

is_production = environment == "production" and environ == "production"

# Auth cookie parser
def parse_auth_cookie(request: Request):
    cookie = request.cookies.get(AUTH_COOKIE_NAME)
    if not cookie:
        return {"error": True, "public_key": None, "signature": None}

    try:
        decoded = base64.b64decode(unquote(cookie)).decode("utf-8")
        data = json.loads(decoded)
        return {
            "error": False,
            "callback_url": data.get("callback_url"),
            "nonce": data.get("nonce"),
            "recipient": data.get("recipient"),
            "message": data.get("message"),
            "on_behalf_of": data.get("on_behalf_of"),
            "public_key": data.get("public_key"),
            "signature": data.get("signature"),
            "account_id": data.get("account_id"),
        }
    except Exception:
        return {"error": True, "public_key": None, "signature": None}

@router.post("/auth/sign-message")
async def sign_message_auth(input_data: SignedMessageInput, response: Response):
    
    cookie_dict = {
        "account_id": input_data.accountId,
        "signature": input_data.signature,
        "public_key": input_data.publicKey,
        "callback_url": input_data.callbackUrl,
        "nonce": input_data.nonce.lstrip("0") or "0",
        "recipient": input_data.recipient,
        "message": input_data.message,
        "on_behalf_of": None,
    }
    # Encode as JSON then base64
    encoded_cookie = quote(
        base64.b64encode(json.dumps(cookie_dict).encode("utf-8")).decode("utf-8")
    )

    cookie_config = {
        "key": AUTH_COOKIE_NAME,
        "value": encoded_cookie,
        "httponly": True,
        "samesite": "None" if is_production else "Lax",
        "secure": is_production,
        "expires": None,
        "domain": domain if is_production else None,
    }

    try:
        response.set_cookie(**cookie_config)
    except Exception as e:
        # Log or print the error for debugging
        import logging
        logging.exception("Failed to set auth cookie")
        return {
            "ok": False,
            "error": "Failed to set auth cookie",
            "details": str(e),
        }

    return {"ok": True, "accountId": input_data.accountId}

@router.post("/auth/sign-out")
def sign_out(response: Response):
    response.delete_cookie("auth", path="/", domain=domain if is_production else None)
    return {"ok": True}


@router.get("/auth/session")
async def get_session(request: Request):
    auth = parse_auth_cookie(request)
    if auth["error"]:
        # Gracefully return empty session instead of 401
        return {
            "account_id": None,
            "public_key": None,
            "signature": None,
            "callback_url": None,
            "nonce": None,
            "recipient": None,
            "message": None,
            "on_behalf_of": None,
        }
    return {
        "account_id": auth["account_id"],
        "public_key": auth["public_key"],
        "signature": auth["signature"],
        "callback_url": auth["callback_url"],
        "nonce": auth["nonce"],
        "recipient": auth["recipient"],
        "message": auth["message"],
        "on_behalf_of": auth["on_behalf_of"],
    }

# Custom error handler for validation (mimics ZodError flattening)
def include_custom_handlers(app):
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "code": "BAD_REQUEST",
                "message": "Validation error",
                "data": {
                    "zodError": exc.errors(),
                },
            },
        )
