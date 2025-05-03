import os

class Settings:
    ENV: str = os.getenv("ENV", "local")  # "development" or "production"

    # Cookie settings
    COOKIE_SECURE: bool = ENV == "production"
    COOKIE_HTTPONLY: bool = ENV == "production"
    COOKIE_SAMESITE: str = "Lax"  # Could also be "Strict" or "None"

settings = Settings()
