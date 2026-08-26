import os
from pathlib import Path
import pickle

from dotenv import load_dotenv
from llama_index.core import Document

from app.estimate_loader import EstimateLoader
from app.parser import SiteParser
from app.pipeline import IngestionPipeline
from app.pricing_loader import PricingLoader


OUTPUT_PATH = Path("data/cleaned/llama_docs.pkl")

load_dotenv()


def to_llama_document(doc) -> Document:
    metadata = dict(doc.metadata)
    metadata.setdefault("url", doc.url)
    metadata.setdefault("title", doc.title)
    metadata.setdefault("h1", doc.h1)
    metadata.setdefault("headings", doc.headings)

    return Document(
        text=doc.text,
        metadata=metadata,
    )


def save_llama_docs(documents: list[Document], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as f:
        pickle.dump(documents, f)


def main() -> None:
    parser = SiteParser(
        junk_selectors=[
            ".w-grid-json",
            ".l-popup",
            ".g-preloader",
            ".w-btn-wrapper",
        ],
        junk_phrases=[],
        min_text_length=200,
    )

    pipeline = IngestionPipeline(
        sitemap_index_url="https://sk-family.ru/sitemap_index.xml",
        parser=parser,
    )

    website_docs = pipeline.collect_documents()

    pricing_loader = PricingLoader("https://sk-family.ru/db.json")
    pricing_docs = pricing_loader.load_documents()

    estimates_dir = os.getenv("ESTIMATES_DIR", "").strip()
    estimate_docs = (
        EstimateLoader(estimates_dir).load_documents()
        if estimates_dir
        else []
    )

    all_docs = website_docs + pricing_docs + estimate_docs
    llama_docs = [to_llama_document(doc) for doc in all_docs]

    save_llama_docs(llama_docs, OUTPUT_PATH)

    print(f"website_docs: {len(website_docs)}")
    print(f"pricing_docs: {len(pricing_docs)}")
    print(f"estimate_docs: {len(estimate_docs)}")
    print(f"all_docs: {len(all_docs)}")
    print(f"llama_docs saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
