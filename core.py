
import os
import re
import tempfile
import numpy as np

def extract_video_id(url: str) -> str | None:
    """Pull the 11-char video ID out of any YouTube URL form."""
    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})", url)
    return match.group(1) if match else None


def _fetch_captions(video_id: str, languages=("en", "hi")) -> str | None:
    """Return caption text if the video has captions, else None."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        fetched = YouTubeTranscriptApi().fetch(video_id, languages=list(languages))
        raw = fetched.to_raw_data()            # [{'text':..., 'start':..., 'duration':...}, ...]
        text = " ".join(snippet["text"] for snippet in raw).strip()
        return text or None
    except Exception:
        return None


def _download_audio(url: str) -> str:
    """Download the audio track as a small mp3 (needs ffmpeg). Returns file path."""
    import yt_dlp

    out_dir = tempfile.mkdtemp()
    out_tmpl = os.path.join(out_dir, "audio.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_tmpl,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "64",          
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return os.path.join(out_dir, "audio.mp3")


def _whisper_transcribe(audio_path: str, groq_key: str) -> str:
    """Transcribe an audio file using Groq's hosted Whisper (free tier)."""
    from groq import Groq

    client = Groq(api_key=groq_key)
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=(os.path.basename(audio_path), f.read()),
        )
    return result.text.strip()


def get_transcript(url: str, groq_key: str) -> tuple[str, str]:
    """
    Returns (transcript_text, source) where source is 'captions' or 'whisper'.
    Raises ValueError if nothing could be transcribed.
    """
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL.")

    # (a) captions — fast, no download, works for long videos
    captions = _fetch_captions(video_id)
    if captions:
        return captions, "captions"

    # (b) Whisper fallback — for videos without captions
    audio_path = _download_audio(url)
    text = _whisper_transcribe(audio_path, groq_key)
    if not text:
        raise ValueError("Could not transcribe this video.")
    return text, "whisper"

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """
    Split text into word-based chunks with overlap so context isn't lost at the
    boundaries. chunk_size/overlap are measured in words.
    """
    words = text.split()
    if not words:
        return []

    chunks, start = [], 0
    step = max(1, chunk_size - overlap)
    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)
        start += step
    return chunks

class Embedder:
    """Wraps fastembed. Local ONNX model — no torch, no API key, ~100 MB download once."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = list(self.model.embed(texts))
        return np.array(vectors, dtype="float32")


class VectorStore:
    """Tiny FAISS wrapper. Cosine similarity via inner product on L2-normalised vectors."""

    def __init__(self, dim: int):
        import faiss
        self.faiss = faiss
        self.index = faiss.IndexFlatIP(dim)
        self.texts: list[str] = []

    def add(self, texts: list[str], embeddings: np.ndarray) -> None:
        emb = embeddings.copy()
        self.faiss.normalize_L2(emb)
        self.index.add(emb)
        self.texts.extend(texts)

    def search(self, query_emb: np.ndarray, k: int = 4) -> list[str]:
        q = query_emb.copy()
        self.faiss.normalize_L2(q)
        _, idx = self.index.search(q, k)
        return [self.texts[i] for i in idx[0] if i != -1]


def build_index(transcript: str, embedder: Embedder) -> tuple[VectorStore, int]:
    """Chunk -> embed -> store. Returns (store, n_chunks)."""
    chunks = chunk_text(transcript)
    if not chunks:
        raise ValueError("Transcript was empty after chunking.")

    embeddings = embedder.embed(chunks)
    store = VectorStore(dim=embeddings.shape[1])
    store.add(chunks, embeddings)
    return store, len(chunks)


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about a YouTube video. "
    "Use ONLY the transcript excerpts provided as context. "
    "If the answer is not in the context, say you couldn't find it in the video. "
    "Be concise and quote the video where useful."
)


def answer(
    question: str,
    store: VectorStore,
    embedder: Embedder,
    history: list[tuple[str, str]],
    groq_key: str,
    model: str = "llama-3.3-70b-versatile",
    k: int = 4,
) -> str:
    """Retrieve top-k chunks and ask Groq, keeping prior turns for follow-ups."""
    from groq import Groq

    # retrieve
    q_emb = embedder.embed([question])
    chunks = store.search(q_emb, k=k)
    context = "\n\n---\n\n".join(chunks)

    # build the message list (system + prior turns + current question)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for prev_q, prev_a in history:
        messages.append({"role": "user", "content": prev_q})
        messages.append({"role": "assistant", "content": prev_a})
    messages.append({
        "role": "user",
        "content": f"Context from the video:\n{context}\n\nQuestion: {question}",
    })

    # generate
    client = Groq(api_key=groq_key)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()
