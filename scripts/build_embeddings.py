from __future__ import annotations

from pathlib import Path
import pickle

from llama_index.core import Document

from app.embeddings import EmbeddedChunk, OpenAIEmbeddingClient


INPUT_PATH = Path("data/cleaned/chunks.pkl")
OUTPUT_PATH = Path("data/cleaned/embedded_chunks.pkl")


def load_chunks(input_path: Path) -> list[Document]:
    with input_path.open("rb") as f:
        return pickle.load(f)


def save_embedded_chunks(chunks: list[EmbeddedChunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as f:
        pickle.dump(chunks, f)


def main() -> None:
    chunks = load_chunks(INPUT_PATH)

    texts = [(chunk.text or "").strip() for chunk in chunks]
    texts = [text for text in texts if text]

    valid_chunks = [chunk for chunk in chunks if (chunk.text or "").strip()]

    embedding_client = OpenAIEmbeddingClient(
        model="text-embedding-3-small",
        batch_size=100,
    )
    vectors = embedding_client.embed_texts(texts)

    embedded_chunks: list[EmbeddedChunk] = []

    for chunk, vector in zip(valid_chunks, vectors):
        embedded_chunks.append(
            EmbeddedChunk(
                text=chunk.text,
                metadata=dict(chunk.metadata or {}),
                embedding=vector,
            )
        )

    save_embedded_chunks(embedded_chunks, OUTPUT_PATH)

    print(f"chunks loaded: {len(chunks)}")
    print(f"embedded chunks: {len(embedded_chunks)}")
    print(f"saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()