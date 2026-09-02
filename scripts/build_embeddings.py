from __future__ import annotations

import os
from pathlib import Path
import pickle

from dotenv import load_dotenv
from llama_index.core import Document

from app.embeddings import EmbeddedChunk, OpenAIEmbeddingClient

load_dotenv()


INPUT_PATH = Path("data/cleaned/chunks.pkl")
OUTPUT_PATH = Path("data/cleaned/embedded_chunks.pkl")


def load_chunks(input_path: Path) -> list[Document]:
    with input_path.open("rb") as f:
        return pickle.load(f)


def load_existing_embeddings(output_path: Path) -> list[EmbeddedChunk]:
    if not output_path.exists():
        return []

    with output_path.open("rb") as f:
        return pickle.load(f)


def save_embedded_chunks(chunks: list[EmbeddedChunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as f:
        pickle.dump(chunks, f)


def main() -> None:
    chunks = load_chunks(INPUT_PATH)

    valid_chunks = [chunk for chunk in chunks if (chunk.text or "").strip()]
    existing_chunks = load_existing_embeddings(OUTPUT_PATH)
    existing_vectors = {
        chunk.text: chunk.embedding
        for chunk in existing_chunks
        if chunk.text and chunk.embedding
    }

    missing_texts = list(dict.fromkeys(
        chunk.text
        for chunk in valid_chunks
        if chunk.text not in existing_vectors
    ))

    use_yandex = bool(os.getenv("YANDEX_API_KEY"))

    embedding_client = OpenAIEmbeddingClient(
        model=os.getenv("YANDEX_DOC_EMBEDDING_MODEL", "text-embedding-3-small"),
        batch_size=1 if use_yandex else 100,
        api_key=os.getenv("YANDEX_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("YANDEX_BASE_URL"),
        project=os.getenv("YANDEX_FOLDER_ID"),
        encoding_format="float" if use_yandex else None,
    )
    new_vectors = embedding_client.embed_texts(missing_texts) if missing_texts else []
    if len(new_vectors) != len(missing_texts):
        raise ValueError(
            f"Embedding count mismatch: expected {len(missing_texts)}, got {len(new_vectors)}"
        )

    vectors_by_text = dict(existing_vectors)
    vectors_by_text.update(zip(missing_texts, new_vectors))

    vector_sizes = {len(vector) for vector in vectors_by_text.values()}
    if len(vector_sizes) > 1:
        raise ValueError(
            f"Embedding vector size mismatch: found sizes {sorted(vector_sizes)}"
        )

    embedded_chunks: list[EmbeddedChunk] = []

    for chunk in valid_chunks:
        vector = vectors_by_text.get(chunk.text)
        if vector is None:
            raise ValueError("Embedding is missing for a valid chunk")

        metadata = dict(chunk.metadata or {})
        metadata["embedding_model"] = embedding_client.model

        embedded_chunks.append(
            EmbeddedChunk(
                text=chunk.text,
                metadata=metadata,
                embedding=vector,
            )
        )

    save_embedded_chunks(embedded_chunks, OUTPUT_PATH)

    print(f"chunks loaded: {len(chunks)}")
    print(f"existing embeddings reused: {len(valid_chunks) - len(missing_texts)}")
    print(f"new embeddings created: {len(missing_texts)}")
    print(f"embedded chunks: {len(embedded_chunks)}")
    print(f"embedding model: {embedding_client.model}")
    print(f"vector size: {len(embedded_chunks[0].embedding) if embedded_chunks else 0}")
    print(f"saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
