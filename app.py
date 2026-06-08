import streamlit as st
from dotenv import load_dotenv
import os

import core

load_dotenv()
st.set_page_config(page_title="Chat with YouTube (Free)", page_icon="🎬",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
    .stApp { background: #0d1117; color: #e6edf3; }
    .main-header { text-align: center; padding: 1.5rem 0 0.5rem; }
    .main-header h1 {
        font-size: 2.6rem; font-weight: 700;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .main-header p { color: #7d8590; margin-top: -0.4rem; }
    .bubble-user { background:#161b22; border-left:3px solid #00d2ff; padding:.7rem 1rem; border-radius:8px; margin:.4rem 0; }
    .bubble-ai   { background:#13171d; border-left:3px solid #3a7bd5; padding:.7rem 1rem; border-radius:8px; margin:.4rem 0; }
    .badge { display:inline-block; padding:.25rem .75rem; border-radius:20px; font-size:.78rem; font-weight:600; }
    .badge-ready { background:#0d2a1a; color:#3fb950; }
    .badge-idle  { background:#2a230d; color:#d29922; }
    .src-tag { font-size:.72rem; color:#7d8590; }
    div[data-testid="stSidebar"] { background:#0a0d12 !important; }
    .stButton > button {
        background: linear-gradient(90deg,#00d2ff,#3a7bd5) !important; color:#001018 !important;
        font-weight:700 !important; border:none !important; border-radius:8px !important; width:100%;
    }
    .stButton > button:hover { opacity:.9 !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading embedding model (first run only)…")
def get_embedder():
    return core.Embedder()


@st.cache_resource(show_spinner=False)
def get_index(url: str, _embedder, groq_key: str):
    """Cached per-URL so we never re-transcribe / re-embed the same video."""
    transcript, source = core.get_transcript(url, groq_key)
    store, n = core.build_index(transcript, _embedder)
    return store, n, source

defaults = {"messages": [], "store": None, "ready": False, "url": "",
            "chunks": 0, "source": "", "groq_key": os.getenv("GROQ_API_KEY", ""), "model": "llama-3.3-70b-versatile"}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

with st.sidebar:
    st.markdown("## 🎬 Setup")
    st.markdown("---")

    key_in = st.text_input("Groq API Key", type="password", placeholder="gsk_...",
                           value=st.session_state.groq_key,
                           help="Free at console.groq.com — no credit card needed.")
    if key_in:
        st.session_state.groq_key = key_in

    st.session_state.model = st.selectbox(
        "Chat model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"],
        index=0,
        help="70b = best answers · 8b = fastest / highest rate limit",
    )

    st.markdown("---")
    url_in = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=…",
                           value=st.session_state.url)

    if st.button("🚀 Load Video"):
        if not st.session_state.groq_key:
            st.error("Enter your Groq API key first.")
        elif not core.extract_video_id(url_in or ""):
            st.error("Enter a valid YouTube URL.")
        else:
            try:
                embedder = get_embedder()
                with st.spinner("⏳ Transcribing + indexing…"):
                    store, n, source = get_index(url_in, embedder, st.session_state.groq_key)
                st.session_state.update(store=store, chunks=n, source=source,
                                        url=url_in, ready=True, messages=[])
                st.success(f"✅ Ready! {n} chunks · via {source}")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    if st.session_state.ready:
        st.markdown('<span class="badge badge-ready">● Video Ready</span>', unsafe_allow_html=True)
        st.markdown(f"**Chunks:** {st.session_state.chunks}")
        st.markdown(f'<span class="src-tag">transcript source: {st.session_state.source}</span>',
                    unsafe_allow_html=True)
        vid = core.extract_video_id(st.session_state.url)
        if vid:
            st.image(f"https://img.youtube.com/vi/{vid}/mqdefault.jpg", use_column_width=True)
    else:
        st.markdown('<span class="badge badge-idle">● Waiting for video</span>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        "**Stack:** fastembed · FAISS · Groq (Llama&nbsp;3.3 + Whisper) · Streamlit  \n"
        "*100% free — no OpenAI key.*"
    )

st.markdown("""
<div class="main-header">
  <h1>🎬 Chat with YouTube</h1>
  <p>Ask anything about a video — free RAG with Groq + Whisper</p>
</div>
""", unsafe_allow_html=True)

for m in st.session_state.messages:
    cls = "bubble-user" if m["role"] == "user" else "bubble-ai"
    icon = "🧑" if m["role"] == "user" else "🤖"
    st.markdown(f'<div class="{cls}">{icon} {m["content"]}</div>', unsafe_allow_html=True)

if st.session_state.ready:
    q = st.chat_input("Ask something about the video…")
    if q:
        st.session_state.messages.append({"role": "user", "content": q})
        history = [
            (st.session_state.messages[i]["content"], st.session_state.messages[i + 1]["content"])
            for i in range(0, len(st.session_state.messages) - 1, 2)
            if i + 1 < len(st.session_state.messages)
        ]
        with st.spinner("Thinking…"):
            try:
                a = core.answer(q, st.session_state.store, get_embedder(),
                                history, st.session_state.groq_key, st.session_state.model)
            except Exception as e:
                a = f"⚠️ Error: {e}"
        st.session_state.messages.append({"role": "assistant", "content": a})
        st.rerun()
else:
    st.info("👈 Add your Groq key and a YouTube URL in the sidebar to begin.")
