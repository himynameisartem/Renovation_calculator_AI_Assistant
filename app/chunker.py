from __future__ import annotations

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter


class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        chunks: list[Document] = []

        for doc in documents:
            text = (doc.text or "").strip()
            if not text:
                continue

            parts = self.splitter.split_text(text)
            total_parts = len(parts)

            for index, part in enumerate(parts):
                chunk_text = part.strip()
                if not chunk_text:
                    continue

                metadata = dict(doc.metadata or {})
                metadata["chunk_index"] = index
                metadata["chunk_total"] = total_parts

                if metadata.get("document_type") == "estimate":
                    chunk_text = self._prepend_estimate_context(
                        text=chunk_text,
                        metadata=metadata,
                    )

                chunks.append(
                    Document(
                        text=chunk_text,
                        metadata=metadata,
                    )
                )

        return chunks

    def _prepend_estimate_context(self, text: str, metadata: dict) -> str:
        context_parts = ["Исторический пример реальной сметы"]

        room_name = str(metadata.get("room_name") or "").strip()
        if room_name:
            context_parts.append(f"помещение или раздел: {room_name}")

        if metadata.get("estimate_document_type") == "room_summary":
            context_parts.append("тип документа: итог помещения")
        else:
            stage_name = str(metadata.get("stage_name") or "").strip()
            work_area_name = str(metadata.get("work_area_name") or "").strip()

            if stage_name:
                context_parts.append(f"этап: {stage_name}")
            if work_area_name:
                context_parts.append(f"раздел работ: {work_area_name}")

        prefix = "Контекст: " + "; ".join(context_parts) + "."
        return f"{prefix}\n\n{text}"
