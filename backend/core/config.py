"""
Config loader for environment variables.

- Centralizes configuration so secrets never live in code.
- Uses pydantic-settings to load from .env.
"""

from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Either GEMINI_API_KEY or GEMINI_API_KEYS (comma-separated) must be provided
    gemini_api_key: Optional[str] = None
    gemini_api_keys: Optional[str] = None  # e.g., "key1,key2,key3"
    gemini_model: str = "gemini-1.5-flash"

    # JSON persistence controls
    data_dir: str = "data"         # where JSON files live (e.g., "data/sessions/<id>.json")
    persist_history: bool = True   # set False to keep only in memory

    class Config:
        env_file = ".env"


settings = Settings()
