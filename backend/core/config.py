"""
Centralized settings loaded from environment (.env).
"""

from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Gemini keys: either a single key or a comma-separated pool
    gemini_api_key: Optional[str] = None
    gemini_api_keys: Optional[str] = None
    gemini_model: str = "gemini-1.5-flash"

    # Persistence
    data_dir: str = "data"
    persist_history: bool = True

    class Config:
        env_file = ".env"


settings = Settings()
