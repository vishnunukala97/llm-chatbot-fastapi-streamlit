"""
Streamlit UI (production-grade, aligns with M1/M2 requirements):
- Uses st.chat_input + st.chat_message + st.spinner (M1)
- Maintains a stable session_id and persists conversation by fetching /history on load (M2)
- Clear chat button that also clears server history via /clear-session (M2)
- Response time display + success/error notifications (M2)

Run:
1) uvicorn backend.main:app --reload
2) streamlit run frontend/app.py
"""

import uuid
import time
import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8000"
MAX_MESSAGE_CHARS = 4000  # mirror backend validation

st.set_page_config(page_title="Basic LLM Chatbot (Gemini)", page_icon="💬")
st.title("💬 Basic LLM Chatbot (Gemini)")

# Stable session id (lets the backend keep history + rate-limit per user)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Local UI history; we'll hydrate it from the server on first load
if "messages" not in st.session_state:
    st.session_state.messages = []

# Flag to ensure we only hydrate once per app session
if "hydrated" not in st.session_state:
    st.session_state.hydrated = False

def hydrate_from_server_history():
    """
    Fetch server-side history so conversation persists across browser refresh.
    Maps backend roles ('user' | 'model') to Streamlit roles
    ('user' | 'assistant') and rebuilds the UI state.
    """
    try:
        r = requests.get(
            f"{BACKEND_URL}/history",
            headers={"X-Session-Id": st.session_state.session_id},
            timeout=10,
        )
        if r.status_code == 200:
            items = r.json()
            # Only hydrate if our local list is empty to avoid dupes
            if not st.session_state.messages:
                for m in items:
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    st.session_state.messages.append({
                        "role": "assistant" if role == "model" else "user",
                        "content": content
                    })
    except Exception:
        # History fetch is best-effort; UI still works if this fails
        pass

# First render → hydrate once to satisfy “persist on page refresh”
if not st.session_state.hydrated:
    hydrate_from_server_history()
    st.session_state.hydrated = True

# Controls
left, right = st.columns([1, 9])
if left.button("Clear chat"):
    # Clear server-side history & local UI
    try:
        requests.post(
            f"{BACKEND_URL}/clear-session",
            headers={"X-Session-Id": st.session_state.session_id},
            timeout=10,
        )
        st.toast("Conversation cleared.", icon="🧹")
    except Exception as e:
        st.error(f"Failed to clear on server: {e}")
    st.session_state.messages = []
    st.experimental_rerun()

# Render past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input with basic client-side checks (UX polish; backend still enforces)
prompt = st.chat_input("Type your message...")
if prompt is not None:
    prompt = prompt.strip()
    if not prompt:
        st.warning("Please enter a message.")
    elif len(prompt) > MAX_MESSAGE_CHARS:
        st.warning(f"Message is too long (>{MAX_MESSAGE_CHARS} chars). Please shorten it.")
    else:
        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Ask backend
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                t0 = time.time()
                try:
                    r = requests.post(
                        f"{BACKEND_URL}/chat",
                        json={"message": prompt},
                        headers={"X-Session-Id": st.session_state.session_id},
                        timeout=20,
                    )
                    data = r.json()
                    reply = data.get("reply", "Error: No reply")
                    model = data.get("model") or "unknown-model"
                    latency_ms = data.get("latency_ms")
                    meta = f"\n\n<span style='opacity:0.6;font-size:0.85em'>⏱️ {latency_ms} ms · {model}</span>"
                    st.toast("Message sent.", icon="✅")
                except Exception as e:
                    reply = f"Error calling backend: {e}"
                    meta = ""
                    st.error(reply)

                st.markdown(reply + meta, unsafe_allow_html=True)

        # Save assistant message to UI history
        st.session_state.messages.append({"role": "assistant", "content": reply})
