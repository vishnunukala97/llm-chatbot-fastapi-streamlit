from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List

class Settings(BaseSettings):
    GEMINI_API_KEY: str = Field(..., description="Google AI Studio API key")
    GEMINI_MODEL: str = Field("gemini-1.5-flash", description="Gemini model name")

    REQUEST_TIMEOUT: int = Field(20, description="Upstream timeout seconds")
    MAX_RETRIES: int = Field(5, description="Transient error retry attempts")

    ALLOWED_ORIGINS: List[str] = Field(default_factory=lambda: ["http://localhost:8501"])
    LOG_LEVEL: str = Field("INFO")

    # 👇 add extra="ignore"
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

settings = Settings()
