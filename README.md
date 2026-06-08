# 🎬 Chat with YouTube — Free RAG Edition

Ask questions about **any YouTube video** using Retrieval-Augmented Generation.  
Runs on a **100% free stack** — one free Groq key, no OpenAI, no paid APIs.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-red?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-Llama_3.3-orange?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-vector_search-green?style=flat-square)

---

## ✨ Why this version is different

| | OpenAI version | **This (free) version** |
|---|---|---|
| LLM | GPT-3.5 (paid) | **Groq Llama 3.3 70B** (free) |
| Transcription | captions only | **captions + Whisper** (Groq, free) |
| Embeddings | OpenAI (paid) | **fastembed** (local, no torch, free) |
| API keys needed | OpenAI (billing) | **1 free Groq key** |
| Cost | $ per query | **$0** |

---

## 🧠 How It Works

```
YouTube URL
   │
   ├─ has captions? ──► youtube-transcript-api      (instant, no download)
   │
   └─ no captions?  ──► yt-dlp → Groq Whisper        (audio → text)
                          │
                          ▼
                 custom chunker (800 words, 120 overlap)
                          │
                          ▼
              fastembed  (BGE-small, local ONNX → 384-dim vectors)
                          │
                          ▼
                 FAISS  (cosine similarity search)
                          │
       question ──► top-4 chunks ──► Groq Llama 3.3 ──► answer
```

---

## 📂 Project Structure

```
youtube-rag-free/
├── app.py            # Streamlit UI
├── core.py           # transcript + RAG logic (the brains)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

`core.py` is fully decoupled from the UI — you can import it into a notebook,
a FastAPI service, or a CLI without touching Streamlit.

---

## ⚙️ Setup

```bash
# 1. Clone
git clone https://github.com/yourusername/youtube-rag-free.git
cd youtube-rag-free

# 2. Virtual env
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install
pip install -r requirements.txt

# 4. Add your free Groq key
cp .env.example .env              # then edit .env

# 5. Run
streamlit run app.py
```

> **ffmpeg** is only needed for the Whisper fallback (videos without captions).
> Install via `winget install ffmpeg` / `brew install ffmpeg` / `apt install ffmpeg`.
> The caption path needs no ffmpeg.

---

## 🔑 Getting a free Groq key

1. Go to **https://console.groq.com**
2. Sign in (no credit card)
3. **API Keys → Create API Key**
4. Paste it into the app sidebar or your `.env`

---

## 🧩 Key Concepts (interview-ready)

| Concept | One-liner |
|---|---|
| **RAG** | Retrieve relevant transcript chunks at query time, feed them to the LLM |
| **Chunking + overlap** | Long transcripts are split into 800-word windows with 120-word overlap so context survives boundaries |
| **Embeddings** | `fastembed` turns text into 384-dim vectors locally (ONNX, no torch) |
| **FAISS** | Inner-product search on L2-normalised vectors = cosine similarity |
| **Whisper fallback** | When a video has no captions, audio is transcribed via Groq Whisper |
| **Conversational memory** | Prior Q&A turns are replayed so follow-ups work |
| **Caching** | `@st.cache_resource` avoids re-embedding the same video and reloading the model |

---

## 🔧 Swap the embedding model

`core.py` uses `fastembed` by default. To use the classic `sentence-transformers`
model instead, replace the `Embedder` class:

```python
from sentence_transformers import SentenceTransformer
class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    def embed(self, texts):
        return self.model.encode(texts, normalize_embeddings=False).astype("float32")
```

(Heavier — pulls in torch — but very widely used.)

---

## 💡 Ideas to extend

- ⏱️ **Timestamped answers** — keep `start` times from captions, link to `youtube.com/watch?v=ID&t=SECONDS`
- 📥 **Export chat** to PDF/Markdown
- 📝 **Auto-summary** on load (5 bullet points)
- 🎚️ **Long-video Whisper** — split audio into <25 MB segments before transcribing
- 🌐 **Multi-video** — index several videos and search across them

---

## 📄 License

MIT — use and modify freely.

> Built by [Your Name] · AIML @ NHCE
