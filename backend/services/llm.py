"""
Gemini LLM service (production-ready, M1–M3):
- Clean SDK integration, timeouts, retries, key rotation
- Validation/sanitization, per-session history, JSON persistence
- Token-bucket rate limiting
- Telemetry (latency, usage), structured logging
"""

from __future__ import annotations

import logging
import random
import re
import time
from collections import deque
from dataclasses import dataclass
from threading import RLock
from typing import Deque, Dict, List, Optional, Tuple

import google.generativeai as genai

from backend.core.config import settings
from backend.storage.json_store import JSONStore

# ---------------- logging ----------------
logger = logging.getLogger("LLMService")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | LLMService | %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# ------------- result type --------------
@dataclass
class ChatResult:
    reply: str
    model: str
    total_history_turns: int
    input_chars: int
    usage: Optional[dict] = None
    latency_ms: Optional[int] = None
    from_cache: bool = False

# -------------- exceptions --------------
class LLMServiceError(Exception): ...
class InvalidAPIKeyError(LLMServiceError): ...
class RateLimitError(LLMServiceError): ...
class TimeoutError(LLMServiceError): ...
class MalformedResponseError(LLMServiceError): ...
class ValidationError(LLMServiceError): ...

# --------- token bucket limiter ---------
class TokenBucket:
    def __init__(self, capacity: int, refill_rate: float) -> None:
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.time()

    def _refill(self) -> None:
        now = time.time()
        elapsed = now - self.last_refill
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now

    def consume(self, n: float = 1.0) -> bool:
        self._refill()
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False

# --------------- service ----------------
class LLMService:
    DEFAULT_MODEL = settings.gemini_model
    TIMEOUT_S = 20
    MAX_OUTPUT_TOKENS = 512
    TEMPERATURE = 0.4
    TOP_P = 0.95

    MAX_MESSAGE_CHARS = 4000
    MAX_HISTORY_TURNS = 16
    MAX_HISTORY_CHARS = 16000

    RL_CAPACITY = 30            # ~30/min
    RL_REFILL_RATE = 0.5        # tokens/sec

    RETRIES = 3
    BACKOFF_BASE = 0.4
    BACKOFF_JITTER = 0.25

    def __init__(self,
                 model: Optional[str] = None,
                 timeout_s: Optional[int] = None,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 max_output_tokens: Optional[int] = None) -> None:

        # generation config
        self.model_name = (model or self.DEFAULT_MODEL).strip()
        self.timeout_s = timeout_s or self.TIMEOUT_S
        self.temperature = self.TEMPERATURE if temperature is None else temperature
        self.top_p = self.TOP_P if top_p is None else top_p
        self.max_output_tokens = max_output_tokens or self.MAX_OUTPUT_TOKENS

        # key pool
        single = (settings.gemini_api_key or "").strip()
        pool_raw = settings.gemini_api_keys
        if pool_raw:
            pool = [k.strip() for k in str(pool_raw).split(",") if k.strip()]
        else:
            pool = [k for k in [single] if k]
        if not pool:
            raise InvalidAPIKeyError("No Gemini API key(s) provided in environment.")

        self._keys: List[str] = pool
        self._key_index = 0
        self._lock = RLock()

        # in-memory state
        self._history: Dict[str, Deque[Dict[str, str]]] = {}
        self._buckets: Dict[str, TokenBucket] = {}

        # JSON persistence
        self._store: JSONStore | None = JSONStore(settings.data_dir) if settings.persist_history else None

        # SDK init
        with self._lock:
            genai.configure(api_key=self._current_key())
            self._model = self._make_model()

        logger.info("LLMService ready (model=%s, keys=%d, persist=%s, data_dir=%s)",
                    self.model_name, len(self._keys), bool(self._store), getattr(settings, "data_dir", "data"))

    # -------- public API (simple) --------
    def get_response(self, message: str, *, session_id: Optional[str] = None,
                     system_instruction: Optional[str] = None) -> str:
        try:
            res = self.chat(message, session_id=session_id, system_instruction=system_instruction)
            return res.reply
        except ValidationError as e:
            logger.warning("Validation error: %s", e)
            return str(e)
        except InvalidAPIKeyError:
            return "Authentication failed: invalid API key."
        except RateLimitError:
            return "Too many requests. Please wait and try again."
        except TimeoutError:
            return "The model request timed out. Please retry."
        except MalformedResponseError:
            return "I couldn’t understand the model response."
        except Exception as e:
            logger.exception("Unexpected error in get_response: %s", e)
            return "Something went wrong while generating a response."

    # -------- public API (rich) ----------
    def chat(self, message: str, *, session_id: Optional[str] = None,
             system_instruction: Optional[str] = None,
             temperature: Optional[float] = None,
             top_p: Optional[float] = None,
             max_output_tokens: Optional[int] = None) -> ChatResult:

        user_text = self._sanitize(self._validate_message(message))

        if session_id is not None and not self._check_rate_limit(session_id):
            raise RateLimitError("Too many requests. Slow down and try again.")

        contents = self._build_contents(user_text, session_id=session_id)

        gen_config = {
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": self.top_p if top_p is None else top_p,
            "max_output_tokens": self.max_output_tokens if max_output_tokens is None else max_output_tokens,
        }

        start = time.time()
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.RETRIES + 1):
            try:
                model = self._model
                if system_instruction:
                    model = genai.GenerativeModel(self.model_name, system_instruction=system_instruction)

                resp = model.generate_content(
                    contents=contents,
                    generation_config=gen_config,
                    request_options={"timeout": self.timeout_s},
                )

                reply = (getattr(resp, "text", None) or "").strip()
                if not reply:
                    raise MalformedResponseError("No text returned by the model.")

                latency_ms = int((time.time() - start) * 1000)

                total_turns = self._update_history_after_success(
                    session_id=session_id, user_text=user_text, reply_text=reply
                )

                usage = None
                try:
                    um = getattr(resp, "usage_metadata", None)
                    if um:
                        usage = {
                            "input_characters": getattr(um, "input_characters", None),
                            "output_characters": getattr(um, "output_characters", None),
                            "total_characters": getattr(um, "total_characters", None),
                        }
                except Exception:
                    usage = None

                logger.info("Gemini success (session=%s, latency_ms=%s, attempt=%d)",
                            session_id, latency_ms, attempt)

                return ChatResult(
                    reply=reply,
                    model=self.model_name,
                    total_history_turns=total_turns,
                    input_chars=len(user_text),
                    usage=usage,
                    latency_ms=latency_ms,
                )

            except Exception as e:
                last_exc = e
                action, classification = self._classify_error(e)
                logger.warning("Gemini error (attempt=%d/%d, class=%s, action=%s, session=%s): %s",
                               attempt, self.RETRIES, classification, action, session_id, e)

                if action == "rotate_key":
                    if self._rotate_key():
                        with self._lock:
                            genai.configure(api_key=self._current_key())
                            self._model = self._make_model()

                if action in ("retry", "rotate_key_retry"):
                    time.sleep(self._backoff_time(attempt))
                    continue

                break  # non-retryable

        assert last_exc is not None
        _, classification = self._classify_error(last_exc)

        if classification == "invalid_key":
            raise InvalidAPIKeyError("Invalid or unauthorized API key.")
        if classification == "timeout":
            raise TimeoutError("The model request timed out.")
        if classification == "rate_limit":
            raise RateLimitError("Rate limited by API.")
        if classification == "malformed":
            raise MalformedResponseError("Malformed response from model.")
        if classification == "validation":
            raise ValidationError(str(last_exc))

        raise LLMServiceError(str(last_exc))

    # -------- session utilities ----------
    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._history.pop(session_id, None)
            self._buckets.pop(session_id, None)
        if self._store is not None:
            self._store.clear(session_id)

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        with self._lock:
            dq = self._history.get(session_id)
            if dq is None and self._store is not None:
                disk_msgs = self._store.load(session_id)
                dq = deque(disk_msgs)
                self._history[session_id] = dq
            return list(dq or [])

    # ------------- internals -------------
    def _validate_message(self, message: str) -> str:
        if message is None:
            raise ValidationError("Message cannot be null.")
        msg = str(message).strip()
        if not msg:
            raise ValidationError("Please enter a message.")
        if len(msg) > self.MAX_MESSAGE_CHARS:
            raise ValidationError(f"Message exceeds {self.MAX_MESSAGE_CHARS} characters.")
        return msg

    @staticmethod
    def _sanitize(text: str) -> str:
        return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", text)

    def _check_rate_limit(self, session_id: str) -> bool:
        with self._lock:
            bucket = self._buckets.get(session_id)
            if bucket is None:
                bucket = TokenBucket(self.RL_CAPACITY, self.RL_REFILL_RATE)
                self._buckets[session_id] = bucket
        return bucket.consume(1.0)

    def _build_contents(self, user_text: str, *, session_id: Optional[str]) -> List[Dict[str, object]]:
        contents: List[Dict[str, object]] = []

        if session_id is not None:
            with self._lock:
                dq = self._history.get(session_id)
                if dq is None and self._store is not None:
                    disk_msgs = self._store.load(session_id)
                    dq = deque(disk_msgs)
                    self._history[session_id] = dq

                if dq:
                    for m in dq:
                        contents.append({"role": m["role"], "parts": [m["content"]]})

        contents.append({"role": "user", "parts": [user_text]})

        total_chars = sum(len("".join(msg.get("parts", []))) for msg in contents)
        if total_chars > self.MAX_HISTORY_CHARS:
            i = 0
            while i < len(contents) - 1 and total_chars > self.MAX_HISTORY_CHARS:
                msg_len = len("".join(contents[i].get("parts", [])))
                total_chars -= msg_len
                i += 1
            contents = contents[i:]

        return contents

    def _update_history_after_success(self, *, session_id: Optional[str], user_text: str, reply_text: str) -> int:
        if session_id is None:
            return 0

        now = time.time()
        with self._lock:
            dq = self._history.get(session_id)
            if dq is None:
                dq = deque()
                self._history[session_id] = dq

            dq.append({"role": "user", "content": user_text, "ts": now})
            dq.append({"role": "model", "content": reply_text, "ts": time.time()})

            while len(dq) > self.MAX_HISTORY_TURNS:
                dq.popleft()

            def total_chars() -> int:
                return sum(len(m["content"]) for m in dq)

            while total_chars() > self.MAX_HISTORY_CHARS and dq:
                dq.popleft()

            total = len(dq)

            if self._store is not None:
                try:
                    self._store.save(session_id, list(dq))
                except Exception as e:
                    logger.warning("Failed to persist session '%s': %s", session_id, e)

            return total

    def _current_key(self) -> str:
        return self._keys[self._key_index]

    def _rotate_key(self) -> bool:
        with self._lock:
            if len(self._keys) <= 1:
                return False
            self._key_index = (self._key_index + 1) % len(self._keys)
            logger.info("Rotated Gemini API key (index=%d)", self._key_index)
            return True

    def _make_model(self):
        return genai.GenerativeModel(self.model_name)

    def _backoff_time(self, attempt: int) -> float:
        base = self.BACKOFF_BASE * (2 ** (attempt - 1))
        jitter = random.uniform(0, self.BACKOFF_JITTER)
        return base + jitter

    def _classify_error(self, exc: Exception) -> Tuple[str, str]:
        """
        Return (action, classification)
        action: 'retry' | 'rotate_key_retry' | 'rotate_key' | 'fail'
        classification: 'invalid_key'|'rate_limit'|'timeout'|'malformed'|'validation'|'other'
        """
        msg = str(exc).lower()

        if ("unauthorized" in msg or "401" in msg or
            "invalid api key" in msg or "api key not valid" in msg or
            "api_key_invalid" in msg or "permission denied" in msg):
            return "rotate_key_retry", "invalid_key"

        if "429" in msg or "rate limit" in msg or "resource exhausted" in msg or "quota" in msg:
            return "rotate_key_retry", "rate_limit"

        if "deadline exceeded" in msg or "timeout" in msg:
            return "retry", "timeout"

        if "invalid argument" in msg or "bad request" in msg:
            return "fail", "validation"

        if isinstance(exc, MalformedResponseError):
            return "retry", "malformed"

        if "temporarily unavailable" in msg or "try again" in msg or "reset" in msg:
            return "retry", "other"

        return "fail", "other"
