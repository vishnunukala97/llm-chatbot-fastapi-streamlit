from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal

class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables or .env file.
    """
    provider: Literal["openai", "gemini"] = Field("openai", env="PROVIDER")
    openai_api_key: str = Field("", env="OPENAI_API_KEY")
    gemini_api_key: str = Field("", env="GEMINI_API_KEY")
    backend_host: str = Field("127.0.0.1", env="BACKEND_HOST")
    backend_port: int = Field(8000, env="BACKEND_PORT")
    openai_model: str = Field("gpt-4o-mini", env="OPENAI_MODEL")
    timeout_seconds: int = Field(30, env="TIMEOUT_SECONDS")
    rate_limit_qps: int = Field(2, env="RATE_LIMIT_QPS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
