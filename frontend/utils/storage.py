from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time, uuid, json, re
from config import get_chat_data_dir

def ensure_data_dir() -> Path:
    """Ensure chat data directory exists and return its Path."""
    return get_chat_data_dir()

def safe_filename(title: str) -> str:
    """Sanitize title for safe filename (Windows/Unix)."""
    return re.sub(r"[^a-zA-Z0-9_\-.]", "_", title.strip())[:40] or "chat"

def list_chats() -> Tuple[List[Dict], List[str]]:
    """
    List all chat dicts sorted by updated_at desc.
    Returns (list of chats, list of warning strings).
    """
    chats = []
    warnings = []
    for f in ensure_data_dir().glob("*.json"):
        try:
            with f.open("r", encoding="utf-8") as fp:
                chat = json.load(fp)
            if not all(k in chat for k in ("id", "title", "created_at", "updated_at", "messages")):
                warnings.append(f"Skipping corrupt chat: {f.name}")
                continue
            chats.append(chat)
        except Exception:
            warnings.append(f"Could not load chat: {f.name}")
    chats.sort(key=lambda c: c.get("updated_at", ""), reverse=True)
    return chats, warnings

def load_chat(chat_id: str) -> Optional[Dict]:
    """Load chat by id, or None if not found/corrupt."""
    f = ensure_data_dir() / f"{chat_id}.json"
    try:
        with f.open("r", encoding="utf-8") as fp:
            chat = json.load(fp)
        if not all(k in chat for k in ("id", "title", "created_at", "updated_at", "messages")):
            return None
        return chat
    except Exception:
        return None

def save_chat(chat: Dict) -> None:
    """Write chat dict to disk as pretty JSON. Update updated_at."""
    chat["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    f = ensure_data_dir() / f"{chat['id']}.json"
    with f.open("w", encoding="utf-8") as fp:
        json.dump(chat, fp, indent=2, ensure_ascii=False)

def new_chat(title: Optional[str] = None) -> Dict:
    """Create a new chat dict with unique id and timestamps."""
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    chat_id = uuid.uuid4().hex
    return {
        "id": chat_id,
        "title": title or "Untitled",
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }

def sanitize_input(text: str, max_len: int) -> str:
    """Trim and cap input text to max_len."""
    return text.strip()[:max_len]