from urllib.parse import unquote, urlparse
import requests
import xml.etree.ElementTree as ET


class SitemapCrawler:
    def __init__(self, sitemap_index_url: str, timeout: int = 20) -> None:
        self.sitemap_index_url = sitemap_index_url
        self.timeout = timeout

    def fetch_xml(self, url: str) -> str:
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.text

    def parse_sitemap_index(self, xml_text: str) -> list[str]:
        root = ET.fromstring(xml_text)
        namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

        sitemap_urls = []

        for loc in root.findall(".//sm:loc", namespace):
            if loc.text:
                sitemap_urls.append(loc.text.strip())

        return sitemap_urls

    def filter_sitemaps(self, sitemap_urls: list[str]) -> list[str]:
        allowed_suffixes = (
            "page-sitemap.xml",
            "post-sitemap.xml",
            "us_portfolio-sitemap.xml",
        )

        return [url for url in sitemap_urls if url.endswith(allowed_suffixes)]

    def parse_url_sitemap(self, xml_text: str) -> list[str]:
        root = ET.fromstring(xml_text)
        namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

        page_urls = []

        for loc in root.findall(".//sm:loc", namespace):
            if loc.text:
                page_urls.append(loc.text.strip())

        return page_urls

    def normalize_url(self, url: str) -> str:
        return unquote(url).strip()

    def should_skip_url(self, url: str) -> bool:
        parsed = urlparse(url)
        path = parsed.path.lower()

        skip_patterns = [
            "/thankyou",
            "/quiz",
            "/калькулятор",
            "/калькурятор",
            "/author/",
            "/category/",
        ]

        for pattern in skip_patterns:
            if pattern in path:
                return True

        return False

    def filter_page_urls(self, urls: list[str]) -> list[str]:
        filtered = []
        seen = set()

        for url in urls:
            normalized = self.normalize_url(url)

            if self.should_skip_url(normalized):
                continue

            if normalized in seen:
                continue

            seen.add(normalized)
            filtered.append(normalized)

        return filtered

    def discover_urls(self) -> list[str]:
        sitemap_index_xml = self.fetch_xml(self.sitemap_index_url)
        sitemap_urls = self.parse_sitemap_index(sitemap_index_xml)
        filtered_sitemaps = self.filter_sitemaps(sitemap_urls)

        all_urls = []

        for sitemap_url in filtered_sitemaps:
            sitemap_xml = self.fetch_xml(sitemap_url)
            urls = self.parse_url_sitemap(sitemap_xml)
            all_urls.extend(urls)

        return self.filter_page_urls(all_urls)