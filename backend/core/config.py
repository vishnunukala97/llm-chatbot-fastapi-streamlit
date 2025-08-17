"""
Purpose:
- Central place to read configuration from environment variables.
- Uses pydantic-settings so we get validation + defaults.
- Keeps secrets OUT of code. You store them in .env (which is .gitignored).

How it works:
- When `Settings()` is created, it loads values from `.env` automatically.
- We expose a singleton `settings` you can import anywhere.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Your Google Gemini API key (required). If missing, pydantic will raise an error
    gemini_api_key: str

    # Which Gemini model to use; safe default is flash for speed
    gemini_model: str = "gemini-1.5-flash"

    class Config:
        # Tell pydantic to read from this file by default
        env_file = ".env"


# Create a single shared instance you can import
settings = Settings()
