# RAG Chatbot

A working chat assistant that answers questions from your PDF documents and cites the page each answer came from. It is the runnable version of the handwritten notes *RAG - Retrieval-Augmented Generation* (`../RAG_Handwritten_Notes.pdf`).

It runs entirely on your own computer with [Ollama](https://ollama.com): no API key, no cost, and your documents never leave your machine.

*Author: Sumant Chakravarty*

## Quick start (Windows)

1. Install **Python 3.10+** and **Ollama**.
2. Download the two models (one time):
   ```
   ollama pull gemma4:e4b
   ollama pull nomic-embed-text
   ```
3. Double-click **`run_chatbot.bat`**. On first run it creates a Python environment and installs the packages, then opens the chatbot in your browser at http://localhost:8501.

Manual start instead of the .bat file:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py          # web chat
python chat_cli.py            # or: terminal chat
```

## Try it

A sample handbook for a fictional company is included in `data/hr_policy.pdf`. Ask:

- How many paid leaves do I get per year?  *(page 4)*
- How many unused leave days can I carry over?  *(page 5)*
- My VPN shows error E-1043. How do I fix it?  *(page 9, found by keyword search)*
- What is the dress code?  *(not in the document, so it answers "I don't know")*
- Follow-up: "and for interns?" after the leave question *(rewritten into a full question)*

## Use your own documents

- **Web app:** upload PDFs in the sidebar, or
- copy PDFs into the `data` folder and restart.

Documents are indexed once and cached in `.rag_cache/`, so later starts are fast. Scanned PDFs (images only) need OCR first.

## How it works (maps to the notes)

| Step | Notes page | Code |
|---|---|---|
| Load PDF, keep page numbers | 4 | `load_pdf()` |
| Sentence-aware chunks, 700 chars, 120 overlap | 5 | `chunk_text()` |
| Embeddings with `nomic-embed-text` | 6 | `embed()` |
| Vectors stored in NumPy + disk cache | 7 | `RAGIndex` |
| Hybrid search: vectors + BM25, fused with RRF | 8 | `RAGIndex.search()` |
| Grounded prompt with numbered sources | 9 | `build_messages()` |
| Answer streamed from `gemma4:e4b` | 10 | `chat()` |
| Follow-up question rewriting | 13 | `rewrite_question()` |

Turn on **"Show how retrieval worked"** in the sidebar to see each chunk's cosine score, dense rank, BM25 rank and fused RRF score.

## Settings

Change the models with environment variables:

```
set RAG_CHAT_MODEL=llama3.2
set RAG_EMBED_MODEL=nomic-embed-text
set OLLAMA_HOST=http://localhost:11434
```

Answers take about 20-60 seconds on a laptop CPU with `gemma4:e4b`. A GPU or a smaller chat model makes them faster.

## Files

```
rag_chatbot/
├── app.py             web chat (Streamlit)
├── chat_cli.py        terminal chat
├── rag_core.py        the RAG pipeline
├── run_chatbot.bat    double-click launcher
├── requirements.txt
└── data/
    ├── hr_policy.pdf  sample document
    └── source/        HTML the sample PDF was made from
```
