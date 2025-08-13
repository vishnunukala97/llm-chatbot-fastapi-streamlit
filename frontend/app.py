import os, time, requests, streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Basic LLM Chatbot", page_icon="💬", layout="centered")
st.title("💬 Basic LLM Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_msg = st.chat_input("Type your message…")
if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                t0 = time.perf_counter()
                r = requests.post(f"{API_URL}/chat", json={"message": user_msg}, timeout=30)
                elapsed = int((time.perf_counter() - t0) * 1000)
                if r.status_code == 200:
                    data = r.json()
                    reply = data.get("reply", "")
                    st.markdown(reply or "_(empty response)_")
                    st.caption(f"⏱️ {data.get('latency_ms', elapsed)} ms")
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                elif r.status_code in (401, 403):
                    st.error("Auth error: check your API key.")
                    st.session_state.messages.append({"role": "assistant", "content": "Auth error."})
                elif r.status_code == 429:
                    st.error("Rate limit. Please wait a moment and retry.")
                    st.session_state.messages.append({"role": "assistant", "content": "Rate limit; retry soon."})
                else:
                    st.error(f"Server error: {r.text[:200]}")
                    st.session_state.messages.append({"role": "assistant", "content": "Server error."})
            except requests.Timeout:
                st.error("Timeout—please try again.")
                st.session_state.messages.append({"role": "assistant", "content": "Timeout—please try again."})
            except Exception as e:
                st.error("Unexpected error.")
                st.session_state.messages.append({"role": "assistant", "content": "Unexpected error."})

st.divider()
if st.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()
