from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API keys (existing)
    gemini_api_key: Optional[str] = None
    gemini_api_keys: Optional[str] = None
    gemini_model: str = "gemini-1.5-flash"

    # NEW: JSON persistence controls
    data_dir: str = "data"
    persist_history: bool = True  # set False to disable file writes

    class Config:
        env_file = ".env"

settings = Settings()
