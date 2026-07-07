from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable

from bs4 import BeautifulSoup, Tag


@dataclass(slots=True)
class ParsedDocument:
    url: str
    title: str
    h1: str
    text: str
    headings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        payload = f"{self.url}\n{self.title}\n{self.h1}\n{self.text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SiteParser:
    def __init__(
        self,
        junk_selectors: Iterable[str] | None = None,
        junk_phrases: Iterable[str] | None = None,
        min_text_length: int = 200,
    ) -> None:
        self.junk_selectors = list(junk_selectors or [])
        self.junk_phrases = list(junk_phrases or [])
        self.min_text_length = min_text_length

    def parse_page(self, url: str, html: str) -> ParsedDocument | None:
        if not html or not html.strip():
            return None

        soup = BeautifulSoup(html, "html.parser")

        self._remove_noise(soup)
        main_node = self._extract_main_node(soup)

        title = self._extract_title(soup)
        h1 = self._extract_h1(main_node, soup)
        headings = self._extract_headings(main_node)
        text = self._extract_structured_text(main_node)
        text = self._postprocess_text(text)

        if self._is_bad_document(text):
            return None

        metadata = self._build_metadata(
            url=url,
            title=title,
            h1=h1,
            headings=headings,
            text=text,
        )

        return ParsedDocument(
            url=url,
            title=title,
            h1=h1,
            text=text,
            headings=headings,
            metadata=metadata,
        )

    def _remove_noise(self, soup: BeautifulSoup) -> None:
        default_selectors = [
            "script",
            "style",
            "noscript",
            "iframe",
            "svg",
            "canvas",
            "form",
            "header",
            "footer",
            "nav",
            "aside",
            ".menu",
            ".header",
            ".footer",
            ".sidebar",
            ".popup",
            ".modal",
            ".breadcrumbs",
            ".breadcrumb",
            ".widget",
            ".socials",
            ".share",
            ".cookie",
            ".cookies",
            ".wpcf7",
        ]

        for selector in [*default_selectors, *self.junk_selectors]:
            for tag in soup.select(selector):
                tag.decompose()

    def _extract_main_node(self, soup: BeautifulSoup) -> Tag:
        candidates = [
            "main",
            "article",
            ".entry-content",
            ".post-content",
            ".page-content",
            ".content",
            "#content",
            ".site-content",
        ]

        for selector in candidates:
            node = soup.select_one(selector)
            if isinstance(node, Tag):
                return node

        if isinstance(soup.body, Tag):
            return soup.body

        return soup

    def _extract_title(self, soup: BeautifulSoup) -> str:
        if soup.title and soup.title.string:
            return self._normalize_inline_text(soup.title.string)
        return ""

    def _extract_h1(self, main_node: Tag, soup: BeautifulSoup) -> str:
        h1_tag = main_node.find("h1")
        if not h1_tag:
            h1_tag = soup.find("h1")
        if h1_tag:
            return self._normalize_inline_text(h1_tag.get_text(" ", strip=True))
        return ""

    def _extract_headings(self, main_node: Tag) -> list[str]:
        headings: list[str] = []

        for tag in main_node.find_all(["h1", "h2", "h3"]):
            text = self._normalize_inline_text(tag.get_text(" ", strip=True))
            if text:
                headings.append(text)

        for selector in [".title_h1", ".title_h2"]:
            for tag in main_node.select(selector):
                text = self._normalize_inline_text(tag.get_text(" ", strip=True))
                if text:
                    headings.append(text)

        return headings

    def _extract_structured_text(self, main_node: Tag) -> str:
        blocks: list[str] = []

        for tag in main_node.find_all(
            ["h1", "h2", "h3", "p", "li", "blockquote", "div"],
            recursive=True,
        ):
            classes = tag.get("class", [])
            text = self._normalize_inline_text(tag.get_text(" ", strip=True))
            if not text:
                continue

            if tag.name == "div" and "title_h1" in classes:
                blocks.append(f"## {text}")
            elif tag.name == "div" and "title_h2" in classes:
                blocks.append(f"### {text}")
            elif tag.name == "h1":
                blocks.append(f"# {text}")
            elif tag.name == "h2":
                blocks.append(f"## {text}")
            elif tag.name == "h3":
                blocks.append(f"### {text}")
            elif tag.name == "li":
                blocks.append(f"- {text}")
            elif tag.name == "div":
                continue
            else:
                blocks.append(text)

        return "\n\n".join(blocks)

    def _postprocess_text(self, text: str) -> str:
        cleaned = text

        for phrase in self.junk_phrases:
            cleaned = cleaned.replace(phrase, "\n")

        cleaned = re.sub(r"\xa0", " ", cleaned)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned.strip()

    def _is_bad_document(self, text: str) -> bool:
        if len(text) < self.min_text_length:
            return True

        if self._looks_like_serialized_garbage(text):
            return True

        return False

    def _looks_like_serialized_garbage(self, text: str) -> bool:
        markers = [
            r'\\u[0-9a-fA-F]{4}',
            r'https:\\/\\/',
            r'"img"',
            r'"alt"',
            r'"li_gallery"',
            r'"li_name"',
            r'"lid":"',
            r"\[\{",
            r"tildacdn",
        ]

        hits = sum(bool(re.search(pattern, text)) for pattern in markers)
        return hits >= 2

    def _build_metadata(
        self,
        url: str,
        title: str,
        h1: str,
        headings: list[str],
        text: str,
    ) -> dict:
        return {
            "url": url,
            "title": title,
            "h1": h1,
            "headings": headings,
            "source": self._extract_domain(url),
            "text_length": len(text),
        }

    def _extract_domain(self, url: str) -> str:
        match = re.match(r"^https?://([^/]+)", url)
        return match.group(1).lower() if match else ""

    def _normalize_inline_text(self, text: str) -> str:
        text = text.replace("\xa0", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()