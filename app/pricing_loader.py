import requests

from app.parser import ParsedDocument


class PricingLoader:
    def __init__(self, json_url: str, timeout: int = 20) -> None:
        self.json_url = json_url
        self.timeout = timeout

    def load_json(self) -> dict:
        response = requests.get(self.json_url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def load_documents(self) -> list[ParsedDocument]:
        data = self.load_json()
        documents: list[ParsedDocument] = []

        categories = data.get("categories", [])
        for category in categories:
            category_id = category.get("id", "")
            category_title = category.get("title", "Категория без названия")

            sections = category.get("sections", [])
            for section in sections:
                section_id = section.get("id", "")
                section_title = section.get("title", "Раздел без названия")

                items = section.get("items", [])
                for item in items:
                    doc = self._build_item_document(
                        category_id=category_id,
                        category_title=category_title,
                        section_id=section_id,
                        section_title=section_title,
                        item=item,
                    )
                    if doc is not None:
                        documents.append(doc)

        return documents

    def _build_item_document(
        self,
        category_id: str,
        category_title: str,
        section_id: str,
        section_title: str,
        item: dict,
    ) -> ParsedDocument | None:
        if not isinstance(item, dict):
            return None

        item_id = item.get("id", "")
        item_title = item.get("title", "Услуга без названия")
        unit = item.get("unit")
        price = item.get("price")
        description = item.get("description")
        photos = item.get("photos", [])

        lines = [
            f"Категория: {category_title}",
            f"Раздел: {section_title}",
            f"Услуга: {item_title}",
        ]

        if unit:
            lines.append(f"Единица измерения: {unit}")

        if price is not None:
            lines.append(f"Цена: {price}")

        if description:
            lines.append(f"Описание: {description}")

        text = "\n".join(lines).strip()

        if not text:
            return None

        return ParsedDocument(
            url=f"{self.json_url}#{item_id or item_title}",
            title=item_title,
            h1=item_title,
            text=text,
            headings=[category_title, section_title, item_title],
            metadata={
                "source": self.json_url,
                "document_type": "pricing",
                "category_id": category_id,
                "category_title": category_title,
                "section_id": section_id,
                "section_title": section_title,
                "item_id": item_id,
                "item_title": item_title,
                "unit": unit,
                "price": price,
                "description": description,
                "photos": photos,
            },
        )