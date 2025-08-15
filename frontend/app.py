import streamlit as st
import requests
import time



API_URL = "http://localhost:8000/api/chat"
BACKEND_URL = "http://localhost:8000"
LLM_PROVIDERS = ["openai", "gemini"]
DEFAULT_MODEL = "gpt-4o-mini"


st.set_page_config(page_title="LLM Chatbot", layout="wide")
st.title("🤖 LLM Chatbot")



# Persistent chat messages in session state (initialize once)
MAX_MESSAGES = 30  # Configurable: keep only last N messages
if "messages" not in st.session_state:
    st.session_state["messages"] = []



# Sidebar for actions and provider selection
with st.sidebar:
    provider = st.selectbox("Select LLM Provider", LLM_PROVIDERS, key="provider")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()
    st.markdown("---")
    st.markdown("**Milestone 2 UI**: Persistent chat, clear conversation.")


# Main chat column
chat_col, _ = st.columns([1, 2])
with chat_col:
    # Replay all messages from session state

    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f"<div style='background:#e6f7ff;padding:8px 12px;border-radius:8px;max-width:80%;margin-left:auto;text-align:right;'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
        elif msg["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(f"<div style='background:#f6f6f6;padding:8px 12px;border-radius:8px;max-width:80%;margin-right:auto;text-align:left;'><b>Assistant:</b> {msg['content']}</div>", unsafe_allow_html=True)
                meta = []
                if "latency_ms" in msg:
                    meta.append(f"⏱️ {msg['latency_ms']} ms")
                if "error" in msg and msg["error"]:
                    meta.append(f"❗ {msg['error']}")
                if meta:
                    st.caption(" | ".join(meta))

    # Chat input

    user_input = st.chat_input("Type your message...")
    if user_input:
        # Add user message
        st.session_state["messages"].append({"role": "user", "content": user_input})
        # Trim to last N messages
        if len(st.session_state["messages"]) > MAX_MESSAGES:
            st.session_state["messages"] = st.session_state["messages"][-MAX_MESSAGES:]
        with st.chat_message("user"):
            st.markdown(f"<div style='background:#e6f7ff;padding:8px 12px;border-radius:8px;max-width:80%;margin-left:auto;text-align:right;'><b>You:</b> {user_input}</div>", unsafe_allow_html=True)
        error_msg = None
        with st.spinner("Thinking..."):
            try:
                t0 = time.perf_counter()
                payload = {"message": user_input, "provider": provider}
                resp = requests.post(API_URL, json=payload, timeout=30)
                t1 = time.perf_counter()
                if resp.status_code == 200:
                    data = resp.json()
                    reply = data["reply"]
                    latency = data["latency_ms"]
                    error_msg = None
                else:
                    reply = resp.json().get("detail", "Error from backend.")
                    latency = int((t1 - t0) * 1000)
                    error_msg = reply
            except Exception as e:
                reply = f"Error: {e}"
                latency = 0
                error_msg = str(e)
        # Toast/alert for errors
        if error_msg:
            st.toast(f"Error: {error_msg}", icon="❗")
        st.session_state["messages"].append({"role": "assistant", "content": reply, "latency_ms": latency, "error": error_msg})
        # Trim again after assistant reply
        if len(st.session_state["messages"]) > MAX_MESSAGES:
            st.session_state["messages"] = st.session_state["messages"][-MAX_MESSAGES:]
        st.rerun()
# Status bar (bottom)
st.markdown(f"<div style='position:fixed;bottom:0;left:0;width:100vw;background:#f0f0f0;padding:4px 12px;font-size:12px;color:#888;border-top:1px solid #eee;z-index:999;'>"
            f"Backend: <b>{BACKEND_URL}</b> &nbsp;|&nbsp; Model: <b>{DEFAULT_MODEL if provider == 'openai' else 'gemini-1.5-pro-latest'}</b> &nbsp;|&nbsp; Provider: <b>{provider}</b>"
            "</div>", unsafe_allow_html=True)
