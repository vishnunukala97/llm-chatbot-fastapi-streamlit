"""
JSONStore: JSON file persistence for chat sessions.

- One file per session: data/sessions/{safe_session_id}.json
- Atomic writes (tmp + replace)
- Filename sanitization
- Thread-safe with a lock
- Each entry stores {"role","content","ts"}
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from threading import RLock
from typing import Dict, List


class JSONStore:
    def __init__(self, data_dir: str = "data") -> None:
        self.base = Path(data_dir).resolve()
        self.sessions_dir = self.base / "sessions"
        self._lock = RLock()
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def load(self, session_id: str) -> List[Dict[str, str]]:
        """Load session history (list of dicts). Returns [] if missing/corrupt."""
        path = self._session_path(session_id)
        if not path.exists():
            return []
        with self._lock:
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    return []
                out: List[Dict[str, str]] = []
                for m in data:
                    if not isinstance(m, dict):
                        continue
                    out.append({
                        "role": str(m.get("role", "")),
                        "content": str(m.get("content", "")),
                        "ts": float(m.get("ts") or 0.0),
                    })
                return out
            except Exception:
                return []

    def save(self, session_id: str, messages: List[Dict[str, str]]) -> None:
        """Atomically write the full session history."""
        path = self._session_path(session_id)
        with self._lock:
            payload = []
            for m in messages:
                ts = float(m.get("ts") or time.time())
                payload.append({"role": m["role"], "content": m["content"], "ts": ts})

            fd, tmp_path = tempfile.mkstemp(
                dir=str(self.sessions_dir), prefix="sess_", suffix=".json"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, path)
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

    def clear(self, session_id: str) -> None:
        """Delete the session file if present."""
        path = self._session_path(session_id)
        with self._lock:
            if path.exists():
                try:
                    path.unlink()
                except Exception:
                    pass

    # ---- helpers ----
    _SAFE_RE = re.compile(r"[^a-zA-Z0-9_\-]")

    def _safe_name(self, session_id: str) -> str:
        sid = session_id.strip() or "anonymous"
        return self._SAFE_RE.sub("_", sid)[:128]

    def _session_path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{self._safe_name(session_id)}.json"
