"""Schemas for NEAR account-based authentication."""

from pydantic import BaseModel, Field


class NEARAccountLogin(BaseModel):
    """Schema for NEAR account login request."""

    near_account_id: str = Field(..., description="NEAR account ID (e.g., user.near)")


class NEARAccountCreate(BaseModel):
    """Schema for creating a user with NEAR account."""

    near_account_id: str = Field(..., description="NEAR account ID (e.g., user.near)")
    profile_image: str | None = Field(None, description="Optional profile image URL")


class NEARLoginResponse(BaseModel):
    """Response schema for NEAR account login."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user_created: bool = Field(description="Whether a new user was created")
    stake_amount: str = Field(description="User's stake amount in NEAR tokens")
