import requests

from app.crawler import SitemapCrawler
from app.parser import SiteParser, ParsedDocument


class IngestionPipeline:
    def __init__(
        self,
        sitemap_index_url: str,
        parser: SiteParser,
        timeout: int = 20,
    ) -> None:
        self.crawler = SitemapCrawler(sitemap_index_url=sitemap_index_url, timeout=timeout)
        self.parser = parser
        self.timeout = timeout

    def fetch_html(self, url: str) -> str | None:
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException:
            return None

    def collect_documents(self) -> list[ParsedDocument]:
        urls = self.crawler.discover_urls()
        documents: list[ParsedDocument] = []

        for url in urls:
            html = self.fetch_html(url)
            if not html:
                continue

            doc = self.parser.parse_page(url, html)
            if doc is None:
                continue

            documents.append(doc)

        return documents