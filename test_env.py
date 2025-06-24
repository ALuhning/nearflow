#!/usr/bin/env python3
import os
from dotenv import load_dotenv

print("=== Environment Variable Test ===")
print("Before loading .env:")
print(f"LANGFLOW_SUPERUSER: {os.getenv('LANGFLOW_SUPERUSER', 'NOT SET')}")
print(f"LANGFLOW_AUTO_LOGIN: {os.getenv('LANGFLOW_AUTO_LOGIN', 'NOT SET')}")

print("\nLoading .env file...")
load_dotenv(".env")

print("After loading .env:")
print(f"LANGFLOW_SUPERUSER: {os.getenv('LANGFLOW_SUPERUSER', 'NOT SET')}")
print(f"LANGFLOW_AUTO_LOGIN: {os.getenv('LANGFLOW_AUTO_LOGIN', 'NOT SET')}")

# Try to import langflow settings
try:
    from langflow.services.deps import get_settings_service
    settings = get_settings_service()
    print(f"\nSettings service auth:")
    print(f"AUTO_LOGIN: {settings.auth_settings.AUTO_LOGIN}")
    print(f"SUPERUSER: {settings.auth_settings.SUPERUSER}")
    print(f"SUPERUSER_PASSWORD: {settings.auth_settings.SUPERUSER_PASSWORD}")
except Exception as e:
    print(f"\nError importing settings: {e}")
