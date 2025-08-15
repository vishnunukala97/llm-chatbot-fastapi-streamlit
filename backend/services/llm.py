import logging
import httpx
import asyncio
from typing import Any, Callable, Dict, Type
from backend.core.config import settings
from backend.core.logging import setup_logging

log = logging.getLogger("llm")

class LLMError(Exception):
    """Base exception for LLM client errors."""
    pass

class LLMAuthError(LLMError):
    """Raised for authentication errors (401)."""
    pass

class LLMRateLimitError(LLMError):
    """Raised for rate limiting (429)."""
    def __init__(self, retry_after: float = 1.0):
        super().__init__(f"Rate limited, retry after {retry_after}s")
        self.retry_after = retry_after

class LLMTimeoutError(LLMError):
    """Raised for timeouts."""
    pass

class LLMResponseError(LLMError):
    """Raised for malformed or unexpected responses."""
    pass

class LLMServerError(LLMError):
    """Raised for 5xx server errors."""
    pass

class LLMUnknownError(LLMError):
    """Raised for unknown/unexpected errors."""
    pass

ERROR_MAP: Dict[int, Type[LLMError]] = {
    401: LLMAuthError,
    429: LLMRateLimitError,
    500: LLMServerError,
    502: LLMServerError,
    503: LLMServerError,
    504: LLMServerError,
}

def raise_llm_error(status_code: int, **kwargs) -> LLMError:
    exc_class = ERROR_MAP.get(status_code, LLMUnknownError)
    if exc_class is LLMRateLimitError:
        return exc_class(kwargs.get("retry_after", 1.0))
    return exc_class(f"LLM error: status {status_code}")


class LLMClient:
    """
    Abstraction for LLM providers (OpenAI, Gemini).
    """
    def __init__(self, http_client: httpx.AsyncClient):
        self.http_client = http_client
        self.openai_key = settings.openai_api_key
        self.gemini_key = settings.gemini_api_key
        self.model = settings.openai_model
        self.timeout = settings.timeout_seconds

    async def generate_reply(self, user_message: str, provider: str = None) -> str:
        if provider is None:
            provider = settings.provider
        if provider == "openai":
            return await self._openai_reply(user_message)
        elif provider == "gemini":
            return await self._gemini_reply(user_message)
        else:
            raise LLMUnknownError(f"Unknown provider: {provider}")

    async def _openai_reply(self, user_message: str) -> str:
        if not self.openai_key:
            raise LLMAuthError("Missing OpenAI API key")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": user_message}],
            "max_tokens": 1024,
            "temperature": 0.7,
        }
        return await self._request_with_retries(
            url, headers, body, self._parse_openai_response
        )

    async def _gemini_reply(self, user_message: str) -> str:
        if not self.gemini_key:
            raise LLMAuthError("Missing Gemini API key")
        # Use the latest Gemini 1.5 Pro endpoint and model name
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key={self.gemini_key}"
        headers = {"Content-Type": "application/json"}
        body = {
            "contents": [{"parts": [{"text": user_message}]}]
        }
        return await self._request_with_retries(
            url, headers, body, self._parse_gemini_response
        )

    async def _request_with_retries(
        self, url: str, headers: dict, body: dict, parser: Callable[[httpx.Response], str]
    ) -> str:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = await self.http_client.post(
                    url, headers=headers, json=body, timeout=self.timeout
                )
                if resp.status_code in ERROR_MAP:
                    if resp.status_code == 429:
                        retry_after = float(resp.headers.get("retry-after", "1"))
                        log.warning({"event": "rate_limit", "retry_after": retry_after, "attempt": attempt+1})
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_after * (2 ** attempt))
                            continue
                        raise raise_llm_error(429, retry_after=retry_after)
                    log.error({"event": "llm_error", "status": resp.status_code, "body": resp.text, "provider": self.provider})
                    raise raise_llm_error(resp.status_code)
                resp.raise_for_status()
                return parser(resp)
            except httpx.TimeoutException as e:
                log.error({"event": "timeout", "error": str(e)})
                if attempt < max_retries - 1:
                    await asyncio.sleep(1.5 ** attempt)
                    continue
                raise LLMTimeoutError("Request timed out")
            except httpx.HTTPError as e:
                log.error({"event": "network_error", "error": str(e)})
                raise LLMUnknownError("Network error")
            except Exception as e:
                log.error({"event": "unexpected_error", "error": str(e)})
                raise LLMUnknownError("Unexpected error")
        raise LLMUnknownError("Failed after retries")

    @staticmethod
    def _parse_openai_response(resp: httpx.Response) -> str:
        try:
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            log.error({"event": "parse_error", "provider": "openai", "error": str(e), "body": resp.text})
            raise LLMResponseError("Malformed OpenAI response")

    @staticmethod
    def _parse_gemini_response(resp: httpx.Response) -> str:
        try:
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            log.error({"event": "parse_error", "provider": "gemini", "error": str(e), "body": resp.text})
            raise LLMResponseError("Malformed Gemini response")

# Example for unit testing:
# import pytest, httpx, respx
# @pytest.mark.asyncio
# async def test_openai_success():
#     async with httpx.AsyncClient() as client:
#         llm = LLMClient(client)
#         # Use respx to mock OpenAI endpoint and test llm.generate_reply(...)
