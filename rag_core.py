"""
RAG engine - the same pipeline as the handwritten notes:

    Load (p.4) -> Chunk (p.5) -> Embed (p.6) -> Store (p.7)
    -> Hybrid retrieve + RRF (p.8) -> Prompt (p.9) -> Generate (p.10)
    + conversational query rewriting (p.13)

Runs fully locally with Ollama (no API key needed).
Author: Sumant Chakravarty
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import requests
from pypdf import PdfReader
from rank_bm25 import BM25Okapi


# --------------------------------------------------------------------------- config
def _ollama_url() -> str:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip()
    if not host.startswith("http"):
        host = "http://" + host
    if host.count(":") < 2:          # no port given
        host += ":11434"
    return host.replace("0.0.0.0", "localhost").rstrip("/")


OLLAMA = _ollama_url()
CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "gemma4:e4b")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "nomic-embed-text")
CHUNK_SIZE = 700        # characters (~150-180 tokens)
CHUNK_OVERLAP = 120     # characters carried into the next chunk
RRF_K = 60              # constant in Reciprocal Rank Fusion
CACHE_DIR = Path(__file__).parent / ".rag_cache"

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about the user's documents.
Rules:
- Answer ONLY using the numbered context passages provided.
- If the answer is not in the context, say: "I don't know - that isn't covered in the documents."
- Cite the passages you used like [1] or [2][3], right after the fact they support.
- Be concise and specific: quote exact numbers, dates and names from the context.
- Do not invent policies, numbers or contacts."""


class RAGError(RuntimeError):
    """Raised with a message that can be shown directly to the user."""


# --------------------------------------------------------------------------- data
@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_id: int


@dataclass
class Hit:
    chunk: Chunk
    score: float          # fused RRF score
    dense: float          # cosine similarity
    dense_rank: int
    bm25_rank: int


# --------------------------------------------------------------------------- Ollama
def _post(path: str, payload: dict, stream: bool = False, timeout: int = 300):
    try:
        r = requests.post(f"{OLLAMA}{path}", json=payload, stream=stream, timeout=timeout)
    except requests.ConnectionError:
        raise RAGError(f"Can't reach Ollama at {OLLAMA}. Start the Ollama app (or run `ollama serve`) and try again.")
    if r.status_code != 200:
        raise RAGError(f"Ollama returned {r.status_code}: {r.text[:300]}")
    return r


def check_ollama() -> None:
    """Make sure Ollama is running and both models are installed."""
    try:
        tags = requests.get(f"{OLLAMA}/api/tags", timeout=5).json()
    except requests.RequestException:
        raise RAGError(f"Can't reach Ollama at {OLLAMA}. Start the Ollama app (or run `ollama serve`).")
    names = {m["name"] for m in tags.get("models", [])}
    names |= {n.split(":")[0] for n in names if n.endswith(":latest")}
    missing = [m for m in (CHAT_MODEL, EMBED_MODEL) if m not in names]
    if missing:
        cmds = "  ".join(f"`ollama pull {m}`" for m in missing)
        raise RAGError(f"Missing Ollama model(s): {', '.join(missing)}. Install with {cmds}")


def embed(texts: list[str], kind: str = "document") -> np.ndarray:
    """Text -> L2-normalised vectors. kind = 'document' or 'query'."""
    if "nomic" in EMBED_MODEL:                       # nomic models expect task prefixes
        prefix = "search_document: " if kind == "document" else "search_query: "
        texts = [prefix + t for t in texts]
    vecs = []
    for i in range(0, len(texts), 32):
        r = _post("/api/embed", {"model": EMBED_MODEL, "input": texts[i:i + 32]})
        vecs.extend(r.json()["embeddings"])
    arr = np.asarray(vecs, dtype="float32")
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12   # normalise -> dot = cosine
    return arr


def chat(messages: list[dict], stream: bool = True, temperature: float = 0.2) -> Iterator[str]:
    """Yield answer text pieces from the chat model."""
    r = _post("/api/chat", {"model": CHAT_MODEL, "messages": messages, "stream": stream,
                            "options": {"temperature": temperature}}, stream=stream)
    if not stream:
        yield r.json()["message"].get("content", "")
        return
    for line in r.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        if "error" in data:
            raise RAGError(data["error"])
        piece = data.get("message", {}).get("content", "")
        if piece:
            yield piece
        if data.get("done"):
            break


# --------------------------------------------------------------------------- 1. load
def _clean(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)      # join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def load_pdf(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] for every page that has text."""
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = _clean(page.extract_text() or "")
        if text:
            pages.append((i + 1, text))
    if not pages:
        raise RAGError(f"No text found in {path.name}. Is it a scanned PDF? (needs OCR first)")
    return pages


# --------------------------------------------------------------------------- 2. chunk
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Sentence-aware chunking with overlap (never cuts a sentence in half)."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    chunks, current = [], []
    for sent in sentences:
        if current and len(" ".join(current + [sent])) > size:
            chunks.append(" ".join(current))
            tail, length = [], 0                      # carry the last sentences as overlap
            for s in reversed(current):
                if length + len(s) > overlap:
                    break
                tail.insert(0, s)
                length += len(s)
            current = tail
        current.append(sent)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _tokens(text: str) -> list[str]:
    """Lowercase word tokens; keeps codes like 'e-1043' as one token."""
    return re.findall(r"[a-z0-9]+(?:[-.][a-z0-9]+)*", text.lower())


# --------------------------------------------------------------------------- 3+4. index
class RAGIndex:
    """Embeds chunks once, caches them on disk, then does hybrid search."""

    def __init__(self, pdf_paths: list[Path]):
        if not pdf_paths:
            raise RAGError("No PDF documents to index. Add a PDF to the data folder or upload one.")
        self.chunks: list[Chunk] = []
        vec_parts = []
        for path in pdf_paths:
            chunks, vecs = self._load_or_build(Path(path))
            offset = len(self.chunks)
            for c in chunks:
                c.chunk_id += offset
            self.chunks += chunks
            vec_parts.append(vecs)
        self.vectors = np.vstack(vec_parts)
        self.bm25 = BM25Okapi([_tokens(c.text) for c in self.chunks])
        self.sources = sorted({c.source for c in self.chunks})

    @staticmethod
    def _load_or_build(path: Path) -> tuple[list[Chunk], np.ndarray]:
        key = hashlib.sha256(path.read_bytes() + f"{EMBED_MODEL}|{CHUNK_SIZE}|{CHUNK_OVERLAP}".encode()).hexdigest()[:16]
        CACHE_DIR.mkdir(exist_ok=True)
        meta_f, vec_f = CACHE_DIR / f"{key}.json", CACHE_DIR / f"{key}.npy"
        if meta_f.exists() and vec_f.exists():
            chunks = [Chunk(**d) for d in json.loads(meta_f.read_text(encoding="utf-8"))]
            return chunks, np.load(vec_f)

        chunks = []
        for page, text in load_pdf(path):
            for piece in chunk_text(text):
                chunks.append(Chunk(text=piece, source=path.name, page=page, chunk_id=len(chunks)))
        vecs = embed([c.text for c in chunks], kind="document")
        meta_f.write_text(json.dumps([asdict(c) for c in chunks], ensure_ascii=False), encoding="utf-8")
        np.save(vec_f, vecs)
        return chunks, vecs

    # ----------------------------------------------------------------------- 5. retrieve
    def search(self, query: str, k: int = 4, fetch: int = 20) -> list[Hit]:
        """Hybrid search: dense (meaning) + BM25 (keywords), fused with RRF."""
        fetch = min(fetch, len(self.chunks))
        q = embed([query], kind="query")[0]
        dense = self.vectors @ q
        sparse = self.bm25.get_scores(_tokens(query))

        dense_order = np.argsort(-dense)[:fetch]
        sparse_order = [i for i in np.argsort(-sparse)[:fetch] if sparse[i] > 0]
        dense_rank = {int(i): r + 1 for r, i in enumerate(dense_order)}
        sparse_rank = {int(i): r + 1 for r, i in enumerate(sparse_order)}

        fused: dict[int, float] = {}
        for ranks in (dense_rank, sparse_rank):
            for i, r in ranks.items():
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r)

        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [Hit(self.chunks[i], fused[i], float(dense[i]), dense_rank.get(i, 0), sparse_rank.get(i, 0))
                for i in best]


# --------------------------------------------------------------------------- 6. augment
def build_messages(question: str, hits: list[Hit], history: list[dict] | None = None) -> list[dict]:
    context = "\n\n".join(f"[{n}] ({h.chunk.source}, p.{h.chunk.page}) {h.chunk.text}"
                          for n, h in enumerate(hits, 1))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-4:]:                 # a little memory for follow-ups
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})
    return messages


def rewrite_question(question: str, history: list[dict]) -> str:
    """Turn a follow-up like 'and for interns?' into a standalone search query."""
    if not history:
        return question
    convo = "\n".join(f"{t['role']}: {t['content'][:400]}" for t in history[-4:])
    prompt = ("Rewrite the user's last question as one standalone search question, using the conversation "
              "for context. Reply with the question only.\n\n"
              f"Conversation:\n{convo}\n\nLast question: {question}\nStandalone question:")
    out = "".join(chat([{"role": "user", "content": prompt}], stream=False, temperature=0)).strip()
    out = out.strip('"').splitlines()[0].strip() if out else ""
    return out or question


# --------------------------------------------------------------------------- 7. one-call API
def answer(index: RAGIndex, question: str, history: list[dict] | None = None, k: int = 4):
    """Returns (search_query, hits, stream_of_answer_text)."""
    query = rewrite_question(question, history or [])
    hits = index.search(query, k=k)
    return query, hits, chat(build_messages(question, hits, history))
