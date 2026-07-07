from __future__ import annotations

from pathlib import Path
import pickle

from llama_index.core import Document

from app.chunker import DocumentChunker


INPUT_PATH = Path("data/cleaned/llama_docs.pkl")
OUTPUT_PATH = Path("data/cleaned/chunks.pkl")


def load_llama_docs(input_path: Path) -> list[Document]:
    with input_path.open("rb") as f:
        return pickle.load(f)


def save_chunks(chunks: list[Document], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as f:
        pickle.dump(chunks, f)


def main() -> None:
    llama_docs = load_llama_docs(INPUT_PATH)

    chunker = DocumentChunker(
        chunk_size=800,
        chunk_overlap=120,
    )
    chunks = chunker.chunk_documents(llama_docs)

    save_chunks(chunks, OUTPUT_PATH)

    print(f"llama_docs loaded: {len(llama_docs)}")
    print(f"chunks created: {len(chunks)}")
    print(f"chunks saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()