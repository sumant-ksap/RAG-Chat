"""
RAG Chatbot - web interface.   Run:  streamlit run app.py
Author: Sumant Chakravarty
"""
from pathlib import Path

import streamlit as st

import rag_core as rag

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"

st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="centered")

SUGGESTIONS = [
    "How many paid leaves do I get per year?",
    "How many unused leave days can I carry over?",
    "My VPN shows error E-1043. How do I fix it?",
    "What is the internet allowance for work from home?",
]


@st.cache_resource(show_spinner=False)
def get_index(files: tuple[tuple[str, float], ...]) -> rag.RAGIndex:
    """Build (or load from cache) the index. Rebuilt only when the file list changes."""
    return rag.RAGIndex([Path(p) for p, _ in files])


def list_pdfs() -> list[Path]:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(DATA_DIR.glob("*.pdf")) + sorted(UPLOAD_DIR.glob("*.pdf"))


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("📄 Documents")
    uploaded = st.file_uploader("Add your own PDFs", type="pdf", accept_multiple_files=True)
    for f in uploaded or []:
        target = UPLOAD_DIR / Path(f.name).name
        if not target.exists():
            target.write_bytes(f.getbuffer())

    all_pdfs = list_pdfs()
    chosen = st.multiselect("Search in", [p.name for p in all_pdfs], default=[p.name for p in all_pdfs])
    pdfs = [p for p in all_pdfs if p.name in chosen]

    st.header("⚙️ Settings")
    top_k = st.slider("Chunks to retrieve (top-k)", 2, 8, 4)
    show_debug = st.toggle("Show how retrieval worked", value=False)
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(f"**Answer model:** {rag.CHAT_MODEL}  \n**Embedding model:** {rag.EMBED_MODEL}  \n"
               "**Search:** hybrid (vectors + BM25, RRF)  \nRuns locally with Ollama.")
    st.caption("Built by **Sumant Chakravarty**")

# ----------------------------------------------------------------------------- setup
st.title("📚 RAG Chatbot")
st.caption("Ask questions about your documents. Every answer cites the page it came from.")

try:
    rag.check_ollama()
    with st.spinner("Indexing documents (first time only)..."):
        index = get_index(tuple((str(p), p.stat().st_mtime) for p in pdfs))
except rag.RAGError as e:
    st.error(str(e))
    st.stop()

st.success(f"Ready - {len(index.chunks)} chunks from {len(index.sources)} document(s): {', '.join(index.sources)}",
           icon="✅")

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(hits: list[dict], query: str | None, debug: bool) -> None:
    with st.expander(f"📎 Sources ({len(hits)})"):
        if debug and query:
            st.markdown(f"**Search query used:** {query}")
        for n, h in enumerate(hits, 1):
            st.markdown(f"**[{n}] {h['source']} - page {h['page']}**")
            if debug:
                st.caption(f"RRF score {h['score']:.4f} · cosine {h['dense']:.2f} · "
                           f"dense rank {h['dense_rank'] or '-'} · BM25 rank {h['bm25_rank'] or '-'}")
            st.markdown(f"> {h['text']}")


# ----------------------------------------------------------------------------- history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("hits"):
            show_sources(m["hits"], m.get("query"), show_debug)

# ----------------------------------------------------------------------------- input
question = st.chat_input("Ask about your documents...")
if not st.session_state.messages and not question:
    st.markdown("**Try one of these:**")
    cols = st.columns(2)
    for i, s in enumerate(SUGGESTIONS):
        if cols[i % 2].button(s, key=f"sugg{i}", use_container_width=True):
            question = s

if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the documents..."):
                query, hits, stream = rag.answer(index, question, history, k=top_k)
            text = st.write_stream(stream)
        except rag.RAGError as e:
            st.error(str(e))
            st.stop()
        hit_dicts = [{"source": h.chunk.source, "page": h.chunk.page, "text": h.chunk.text, "score": h.score,
                      "dense": h.dense, "dense_rank": h.dense_rank, "bm25_rank": h.bm25_rank} for h in hits]
        show_sources(hit_dicts, query, show_debug)

    st.session_state.messages.append({"role": "assistant", "content": text, "hits": hit_dicts, "query": query})
