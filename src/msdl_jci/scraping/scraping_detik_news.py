"""
Detik Finance IHSG Scraper - Parallel Processing Version
Scrapes search pages with concurrent requests and resilient parsing.
"""

import csv
import json
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


class DetikFinanceParallelScraper:
    def __init__(self, cache_dir: str = "./detik_cache", workers: int = 5):
        self.base_url = "https://www.detik.com/search/searchall"
        self.content_base_url = "https://finance.detik.com"

        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        self.params = {
            "result_type": "latest",
            "fromdatex": "01/08/2018",
            "todatex": "31/12/2025",
            "siteid": "29",
            "query": "ihsg",
        }

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.workers = workers
        self.max_retries = 2
        self.retry_delay = 1
        self.request_timeout = 15

    def _retry_request(self, url: str, params: dict = None) -> requests.Response | None:
        """Retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    url,
                    params=params,
                    headers=self.headers,
                    timeout=self.request_timeout,
                )
                response.raise_for_status()
                return response
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    page = params.get("page") if params else "?"
                    print(f"   Warning: request failed for page {page}: {e}")
                    return None

    def _normalize_url(self, url: str) -> str:
        if not url:
            return ""
        if url.startswith("/"):
            url = urljoin(self.content_base_url, url)
        return url.split("#", 1)[0]

    def _is_finance_article_url(self, url: str) -> bool:
        if not url:
            return False
        return "finance.detik.com/" in url or "/finance/" in url

    def _extract_text(self, node) -> str:
        return node.get_text(" ", strip=True) if node else ""

    def _text_matches(self, class_name, keywords) -> bool:
        if not class_name:
            return False
        if isinstance(class_name, str):
            haystack = class_name.lower()
        else:
            haystack = " ".join(class_name).lower()
        return any(keyword in haystack for keyword in keywords)

    def _parse_article_from_node(self, node) -> dict | None:
        link = node if getattr(node, "name", None) == "a" else node.find("a", href=True)
        if not link:
            return None

        url = self._normalize_url(link.get("href", ""))
        if not self._is_finance_article_url(url) or "/d-" not in url:
            return None

        title = self._extract_text(link)
        if not title:
            title_elem = node.find(["h1", "h2", "h3", "h4", "span"])
            title = self._extract_text(title_elem)
        if not title:
            title = re.sub(r"[-_]+", " ", url.rstrip("/").split("/")[-1]).strip()
        if not title:
            title = "N/A"

        snippet = ""
        published = ""

        search_node = node
        for _ in range(4):
            if not search_node:
                break

            if not snippet:
                snippet_elem = search_node.find(
                    ["p", "div", "span"],
                    class_=lambda c: self._text_matches(c, ["snippet", "summary", "desc", "description", "excerpt", "content"]),
                ) if hasattr(search_node, "find") else None
                snippet = self._extract_text(snippet_elem)

            if not published:
                date_elem = search_node.find(
                    ["time", "span", "div"],
                    class_=lambda c: self._text_matches(c, ["date", "time", "published", "meta"]),
                ) if hasattr(search_node, "find") else None
                if date_elem:
                    published = date_elem.get("datetime", "").strip() or self._extract_text(date_elem)

            if snippet and published:
                break
            search_node = search_node.parent

        return {
            "title": title,
            "url": url,
            "snippet": snippet,
            "published": published,
            "scraped_at": datetime.now().isoformat(),
        }

    def scrape_page(self, page_num: int) -> tuple[int, list[dict]]:
        """Scrape single page - returns (page_num, articles)."""
        cache_file = self.cache_dir / f"page_{page_num:04d}.json"
        if cache_file.exists():
            with open(cache_file, encoding="utf-8") as f:
                return page_num, json.load(f)

        params = self.params.copy()
        params["page"] = page_num

        response = self._retry_request(self.base_url, params)
        if not response:
            return page_num, []

        soup = BeautifulSoup(response.content, "html.parser")
        articles: list[dict] = []
        seen_urls = set()

        # Search page markup changes occasionally, so scan all anchors and dedupe by URL.
        for link in soup.find_all("a", href=True):
            try:
                article = self._parse_article_from_node(link)
                if not article:
                    continue
                if article["url"] in seen_urls:
                    continue
                seen_urls.add(article["url"])
                articles.append(article)
            except Exception:
                continue

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)

        return page_num, articles

    def scrape_all_parallel(self, total_pages: int = 588, max_pages: int | None = None):
        """Scrape all pages in parallel."""
        if max_pages:
            total_pages = min(total_pages, max_pages)

        print(f"\n{'=' * 70}")
        print("Detik Finance Parallel Scraper (2018-2025)")
        print(f"{'=' * 70}")
        print("Configuration:")
        print(f"   - Pages: {total_pages}")
        print(f"   - Workers: {self.workers}")
        print(f"   - Estimated time: {total_pages / self.workers / 2:.1f} seconds")
        print(f"{'=' * 70}\n")

        all_articles: list[dict] = []
        failed_pages: list[int] = []
        success_count = 0

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(self.scrape_page, page_num): page_num
                for page_num in range(1, total_pages + 1)
            }

            for i, future in enumerate(as_completed(futures), 1):
                try:
                    page_num, articles = future.result()

                    if articles:
                        all_articles.extend(articles)
                        success_count += 1
                        status = "OK"
                    else:
                        failed_pages.append(page_num)
                        status = "FAIL"

                    elapsed = time.time() - start_time
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (total_pages - i) / rate if rate > 0 else 0

                    print(
                        f"{status} Page {page_num:>4} | Progress {i}/{total_pages} | "
                        f"Rate: {rate:.1f} p/s | ETA: {eta:.0f}s",
                        end="\r",
                    )
                except Exception as e:
                    print(f"FAIL Error processing page: {str(e)}")

        elapsed = time.time() - start_time
        print(f"\n\n{'=' * 70}")
        print("Scraping complete!")
        print(f"  - Total articles: {len(all_articles)}")
        print(f"  - Successful pages: {success_count}/{total_pages}")
        print(f"  - Failed pages: {len(failed_pages)}")
        print(f"  - Time elapsed: {elapsed:.1f} seconds")
        print(f"  - Speed: {total_pages / elapsed:.1f} pages/second")
        if not all_articles:
            print("  - No articles were parsed. The search HTML may have changed or the site may have blocked the requests.")
        print(f"{'=' * 70}\n")

        return all_articles, failed_pages

    def export_json(self, articles: list[dict], filename: str = "detik_ihsg_articles.json"):
        """Export to JSON."""
        output_file = self.cache_dir / filename
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        print(f"OK JSON exported: {output_file} ({len(articles)} articles)")
        return output_file

    def export_csv(self, articles: list[dict], filename: str = "detik_ihsg_articles.csv"):
        """Export to CSV."""
        if not articles:
            print("Warning: no articles to export")
            return None

        output_file = self.cache_dir / filename
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["title", "url", "snippet", "published", "scraped_at"])
            writer.writeheader()
            writer.writerows(articles)

        print(f"OK CSV exported: {output_file} ({len(articles)} articles)")
        return output_file

    def export_stats(self, articles: list[dict], failed_pages: list[int]):
        """Export detailed statistics."""
        output_file = self.cache_dir / "statistics.json"

        years_count = defaultdict(int)
        for _article in articles:
            years_count["total"] += 1

        stats = {
            "total_articles": len(articles),
            "total_pages": 588,
            "successful_pages": 588 - len(failed_pages),
            "failed_pages": failed_pages,
            "scrape_timestamp": datetime.now().isoformat(),
            "date_range": {
                "start": "2018-08-01",
                "end": "2025-12-31",
            },
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print(f"OK Statistics saved: {output_file}")
        return output_file

    def export_sample(self, articles: list[dict], sample_size: int = 10):
        """Export sample articles for review."""
        output_file = self.cache_dir / "sample_articles.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(articles[:sample_size], f, ensure_ascii=False, indent=2)

        print(f"OK Sample exported: {output_file} ({min(sample_size, len(articles))} articles)")
        return output_file


def main():
    """Main execution."""
    scraper = DetikFinanceParallelScraper(workers=15)

    articles, failed_pages = scraper.scrape_all_parallel(
        total_pages=588,
        max_pages=None,
    )

    print("SAMPLE ARTICLES (first 5):\n")
    for i, article in enumerate(articles[:5], 1):
        print(f"{i}. {article['title']}")
        print(f"   URL: {article['url']}")
        print(f"   Date: {article['published']}\n")

    print("\nExporting data...\n")
    scraper.export_json(articles)
    scraper.export_csv(articles)
    scraper.export_sample(articles, sample_size=20)
    scraper.export_stats(articles, failed_pages)

    print(f"\nAll files saved to: {scraper.cache_dir}\n")


if __name__ == "__main__":
    main()
