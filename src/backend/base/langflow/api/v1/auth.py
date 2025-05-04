from fastapi import APIRouter, Request, Response
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import base64
from urllib.parse import quote, unquote
import json
from langflow.services.deps import get_settings_service

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

# Auth cookie parser
def parse_auth_cookie(request: Request):
    cookie = request.cookies.get(AUTH_COOKIE_NAME)
    if not cookie:
        return {"error": True, "authorization": None, "signature": None}

    try:
        decoded = base64.b64decode(unquote(cookie)).decode("utf-8")
        data = json.loads(decoded)
        return {
            "error": False,
            "authorization": data.get("public_key"),
            "signature": data.get("signature"),
            "account_id": data.get("account_id"),
        }
    except Exception:
        return {"error": True, "authorization": None, "signature": None}

@router.post("/auth/sign-message")
async def sign_message_auth(input_data: SignedMessageInput, response: Response):
    auth_settings = get_settings_service().auth_settings
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
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=encoded_cookie,
        httponly=auth_settings.ACCESS_HTTPONLY,
        samesite=auth_settings.ACCESS_SAME_SITE,
        secure=auth_settings.ACCESS_SECURE,
        expires=None,
        domain=auth_settings.COOKIE_DOMAIN,
    )

    return {"ok": True, "accountId": input_data.accountId}

@router.get("/auth/session")
async def get_session(request: Request):
    auth = parse_auth_cookie(request)
    if auth["error"]:
        # Gracefully return empty session instead of 401
        return {
            "accountId": None,
            "publicKey": None,
            "signature": None,
        }
    return {
        "accountId": auth["account_id"],
        "publicKey": auth["authorization"],
        "signature": auth["signature"],
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
