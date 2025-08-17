"""
JSONStore: simple, production-friendly JSON persistence for chat sessions.

Design goals:
- One JSON file per session_id:  data/sessions/{safe_id}.json
- Atomic writes (temp file + rename) to avoid partial/corrupt files
- Filename sanitization to prevent path traversal
- Thread-safe via an internal lock (sufficient for single-process FastAPI)
- Easy to swap later for Redis/DB with the same interface

Schema (per file):
[
  {"role": "user",  "content": "...", "ts": 1712345678.123},
  {"role": "model", "content": "...", "ts": 1712345679.456}
]
"""

from __future__ import annotations
import json
import os
import re
import tempfile
import time
from pathlib import Path
from threading import RLock
from typing import List, Dict


class JSONStore:
    """
    Persist session histories as JSON files under data_dir/sessions.
    """

    def __init__(self, data_dir: str = "data") -> None:
        self.base = Path(data_dir).resolve()
        self.sessions_dir = self.base / "sessions"
        self._lock = RLock()
        # Ensure directories exist
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------- public API --------------------------

    def load(self, session_id: str) -> List[Dict[str, str]]:
        """
        Load a session history. Returns [] if missing.
        """
        path = self._session_path(session_id)
        if not path.exists():
            return []
        with self._lock:
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                # Basic shape check
                if not isinstance(data, list):
                    return []
                return [
                    {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
                    for m in data
                    if isinstance(m, dict)
                ]
            except Exception:
                # Corrupt or unreadable file — treat as empty
                return []

    def save(self, session_id: str, messages: List[Dict[str, str]]) -> None:
        """
        Atomically save the full history for a session.
        """
        path = self._session_path(session_id)
        with self._lock:
            # Enrich with timestamps (optional, helps later debugging/analytics)
            payload = [
                {"role": m["role"], "content": m["content"], "ts": time.time()}
                for m in messages
            ]
            # Write to a temp file, then replace — atomic on the same filesystem
            tmp_fd, tmp_path = tempfile.mkstemp(dir=str(self.sessions_dir), prefix="sess_", suffix=".json")
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, path)
            finally:
                # Ensure temp file is gone if os.replace failed early
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

    def clear(self, session_id: str) -> None:
        """
        Delete the session file if present.
        """
        path = self._session_path(session_id)
        with self._lock:
            if path.exists():
                try:
                    path.unlink()
                except Exception:
                    # Best-effort: ignore deletion failures
                    pass

    # ------------------------ internal helpers ----------------------

    _SAFE_RE = re.compile(r"[^a-zA-Z0-9_\-]")

    def _safe_name(self, session_id: str) -> str:
        """
        Sanitize session_id for filename use (defense-in-depth).
        """
        sid = session_id.strip()
        if not sid:
            sid = "anonymous"
        # Replace unsafe chars with underscore
        return self._SAFE_RE.sub("_", sid)[:128]  # guard excessive length

    def _session_path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{self._safe_name(session_id)}.json"
