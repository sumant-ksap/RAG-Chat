"""
RAG Chatbot - terminal version.
Usage:  python chat_cli.py                 (uses every PDF in ./data)
        python chat_cli.py my_file.pdf     (use specific PDFs)
Author: Sumant Chakravarty
"""
import sys
from pathlib import Path

import rag_core as rag


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]] or sorted((Path(__file__).parent / "data").glob("*.pdf"))
    try:
        rag.check_ollama()
        print("Indexing documents...")
        index = rag.RAGIndex(paths)
    except rag.RAGError as e:
        sys.exit(f"Error: {e}")

    print(f"Ready: {len(index.chunks)} chunks from {', '.join(index.sources)}")
    print("Ask a question (type 'exit' to quit).\n")
    history: list[dict] = []
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue
        try:
            _, hits, stream = rag.answer(index, question, history)
            print("Bot: ", end="", flush=True)
            text = ""
            for piece in stream:
                print(piece, end="", flush=True)
                text += piece
        except rag.RAGError as e:
            print(f"\nError: {e}\n")
            continue
        print("\n   Sources: " + ", ".join(f"[{n}] {h.chunk.source} p.{h.chunk.page}"
                                         for n, h in enumerate(hits, 1)) + "\n")
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": text}]


if __name__ == "__main__":
    main()
