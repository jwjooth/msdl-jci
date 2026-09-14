import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class KontanInvestasiParallelScraper:
    def __init__(self, cache_dir: str = "./kontan_cache", workers: int = 5):
        self.base_url = "https://www.kontan.co.id/search/"
        self.content_base_url = "https://investasi.kontan.co.id"

        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,id;q=0.8",
            "Referer": "https://www.kontan.co.id/",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }

        self.params = {
            "search": "ihsg",
            "per_page": "20",
        }

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.workers = workers
        self.max_retries = 2
        self.retry_delay = 1
        self.request_timeout = 15
        self.use_playwright = True
        self.session = requests.Session()

        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=workers, pool_maxsize=workers)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _normalize_url(self, url: str) -> str:
        if not url:
            return ""
        if url.startswith("/"):
            url = urljoin(self.content_base_url, url)
        return url.split("#", 1)[0]

    def _is_finance_article_url(self, url: str) -> bool:
        if not url:
            return False
        if "investasi.kontan.co.id/" not in url:
            return False

        excluded_keywords = ["/search", "/tag/", "/video/", "/author/"]
        if any(ex in url for ex in excluded_keywords):
            return False
        return "/news/" in url

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

    def _is_search_result_item(self, node) -> bool:
        if not node or getattr(node, "name", None) != "li":
            return False
        link = node.find("a", href=True)
        if not link:
            return False
        url = self._normalize_url(link.get("href", ""))
        return self._is_finance_article_url(url)

    def _extract_search_item(self, node) -> dict | None:
        link = node.find("a", href=True)
        if not link:
            return None

        url = self._normalize_url(link.get("href", ""))
        if not self._is_finance_article_url(url):
            return None

        title = ""
        title_link = node.select_one(".sp-hl a, h2 a, h3 a")
        if title_link:
            title = self._extract_text(title_link)
        if not title:
            title = self._extract_text(link)
        if not title:
            title = re.sub(r"[-_]+", " ", url.rstrip("/").split("/")[-1]).strip() or "N/A"

        category = ""
        cat_link = node.select_one(".fs14 a, .linkto-orange a, .link-orange2black a")
        if cat_link:
            category = self._extract_text(cat_link)

        published = ""
        font_gray = node.select_one(".fs14 .font-gray, .font-gray")
        if font_gray:
            published = self._extract_text(font_gray)
            published = re.sub(r"^\|\s*", "", published).strip()

        return {
            "title": title,
            "url": url,
            "snippet": "",
            "published": published,
            "category": category,
            "scraped_at": datetime.now().isoformat(),
        }

    def _fetch_detail_metadata(self, article_url: str) -> dict:
        try:
            response = self.session.get(
                article_url,
                headers=self.headers,
                timeout=self.request_timeout,
            )
            if response.status_code != 200:
                return {}
            soup = BeautifulSoup(response.text, "html.parser")

            title = ""
            title_elem = soup.select_one("h1.judul-artikel, h1.detail-desk, h1")
            if title_elem:
                title = self._extract_text(title_elem)
            if not title:
                meta_title = soup.find("meta", attrs={"property": "og:title"})
                if meta_title and meta_title.get("content"):
                    title = meta_title["content"].strip()

            published = ""
            published_elem = soup.select_one(".font-gray, time, .fs14.ff-opensans.font-gray")
            if published_elem:
                published = self._extract_text(published_elem)
                published = re.sub(r"^\|?\s*", "", published).strip()
            if not published:
                meta_time = soup.find("meta", attrs={"property": "article:published_time"})
                if meta_time and meta_time.get("content"):
                    published = meta_time["content"].strip()

            snippet = ""
            lead = soup.select_one(".tmpt-desk-kon p, .box-det-desk-2 .fs12, meta[property='og:description']")
            if lead:
                if getattr(lead, "name", "") == "meta":
                    snippet = lead.get("content", "").strip()
                else:
                    snippet = self._extract_text(lead)
            if not snippet:
                meta_desc = soup.find("meta", attrs={"property": "og:description"})
                if meta_desc and meta_desc.get("content"):
                    snippet = meta_desc["content"].strip()

            return {
                "detail_title": title,
                "detail_published": published,
                "detail_snippet": snippet,
            }
        except Exception:
            return {}

    def _build_search_url(self, page_num: int) -> str:
        params = self.params.copy()
        params["per_page"] = str(page_num * 20)
        query = "&".join(f"{k}={requests.utils.quote(str(v))}" for k, v in params.items())
        return f"{self.base_url}?{query}"

    def _fetch_page_html(self, page_num: int) -> tuple[str | None, str | None]:
        if not self.use_playwright:
            try:
                response = self.session.get(
                    self.base_url,
                    params={**self.params, "per_page": str(page_num * 20)},
                    headers=self.headers,
                    timeout=self.request_timeout,
                )
                if response.status_code != 200:
                    return None, f"HTTP {response.status_code}"
                return response.text, None
            except Exception as e:
                return None, str(e)

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent=self.headers["User-Agent"],
                    locale="id-ID",
                    extra_http_headers={
                        "Accept": self.headers["Accept"],
                        "Accept-Language": self.headers["Accept-Language"],
                        "Referer": self.headers["Referer"],
                        "Cache-Control": self.headers["Cache-Control"],
                        "Pragma": self.headers["Pragma"],
                    },
                )
                page = context.new_page()
                page.goto(self._build_search_url(page_num), wait_until="networkidle", timeout=45000)
                page.wait_for_timeout(1500)
                html = page.content()
                final_url = page.url
                context.close()
                browser.close()
                return html, final_url
        except Exception as e:
            playwright_error = str(e)
            # Fallback to plain requests so the scraper still works in environments
            # where Playwright browser networking is blocked.
            try:
                response = self.session.get(
                    self.base_url,
                    params={**self.params, "per_page": str(page_num * 20)},
                    headers=self.headers,
                    timeout=self.request_timeout,
                )
                if response.status_code != 200:
                    return None, f"Playwright failed: {playwright_error} | HTTP {response.status_code}"
                return response.text, f"fallback-requests: {playwright_error}"
            except Exception as request_error:
                return None, f"Playwright failed: {playwright_error} | Requests failed: {request_error}"

    def _parse_article_from_node(self, node) -> dict | None:
        link = node if getattr(node, "name", None) == "a" else node.find("a", href=True)
        if not link:
            return None

        url = self._normalize_url(link.get("href", ""))
        if not self._is_finance_article_url(url):
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
                snippet_elem = (
                    search_node.find(
                        ["p", "div", "span"],
                        class_=lambda c: self._text_matches(
                            c,
                            [
                                "snippet",
                                "summary",
                                "desc",
                                "description",
                                "excerpt",
                                "content",
                            ],
                        ),
                    )
                    if hasattr(search_node, "find")
                    else None
                )
                snippet = self._extract_text(snippet_elem)

            if not published:
                date_elem = (
                    search_node.find(
                        ["time", "span", "div"],
                        class_=lambda c: self._text_matches(
                            c, ["date", "time", "published", "meta"]
                        ),
                    )
                    if hasattr(search_node, "find")
                    else None
                )
                if date_elem:
                    published = date_elem.get(
                        "datetime", ""
                    ).strip() or self._extract_text(date_elem)

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
        cache_file = self.cache_dir / f"page_{page_num:04d}.json"
        if cache_file.exists():
            with open(cache_file, encoding="utf-8") as f:
                return page_num, json.load(f)

        html, error = self._fetch_page_html(page_num)
        if not html:
            print(f"   Warning: page {page_num} fetch failed: {error}")
            return page_num, []

        soup = BeautifulSoup(html, "html.parser")
        articles: list[dict] = []
        seen_urls = set()

        search_items = soup.select("li")
        for node in search_items:
            try:
                if not self._is_search_result_item(node):
                    continue
                article = self._extract_search_item(node)
                if not article or article["url"] in seen_urls:
                    continue
                detail = self._fetch_detail_metadata(article["url"])
                if detail:
                    article["snippet"] = detail.get("detail_snippet") or article["snippet"]
                    article["published"] = detail.get("detail_published") or article["published"]
                    article["title"] = detail.get("detail_title") or article["title"]
                seen_urls.add(article["url"])
                articles.append(article)
            except Exception:
                continue

        if not articles:
            card_like = soup.find_all(["article", "div"])
            for node in card_like:
                try:
                    article = self._parse_article_from_node(node)
                    if not article or article["url"] in seen_urls:
                        continue
                    detail = self._fetch_detail_metadata(article["url"])
                    if detail:
                        article["snippet"] = detail.get("detail_snippet") or article["snippet"]
                        article["published"] = detail.get("detail_published") or article["published"]
                        article["title"] = detail.get("detail_title") or article["title"]
                    seen_urls.add(article["url"])
                    articles.append(article)
                except Exception:
                    continue

        if not articles:
            preview = html[:500].replace(chr(10), " ").replace(chr(13), " ")
            print(f"   Debug page {page_num}: first html chunk={preview}")

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)

        return page_num, articles

    def scrape_all_parallel(self, total_pages: int = 24, max_pages: int | None = None):
        if max_pages:
            total_pages = min(total_pages, max_pages)

        print(f"\n{'=' * 70}")
        print("Kontan Investasi Parallel Scraper (IHSG)")
        print(f"{'=' * 70}\n")

        all_articles: list[dict] = []
        failed_pages: list[int] = []
        success_count = 0

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

                    print(
                        f"{status} Page {page_num:>4} | Progress {i}/{total_pages} | Articles found so far: {len(all_articles)}",
                        end="\r",
                    )
                except Exception as e:
                    print(f"FAIL Error processing page: {str(e)}")

        print(f"\n\nScraping complete! Total articles collected: {len(all_articles)}")
        if failed_pages:
            print(f"Failed pages: {failed_pages[:20]}")
        return all_articles, failed_pages

    def export_json(self, articles: list[dict], filename: str = "kontan_investasi_ihsg_articles.json"):
        output_file = self.cache_dir / filename
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        print(f"OK JSON exported: {output_file}")
        return output_file

    def export_csv(self, articles: list[dict], filename: str = "kontan_investasi_ihsg_articles.csv"):
        if not articles:
            print("Warning: no articles to export")
            return None
        output_file = self.cache_dir / filename
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["title", "url", "snippet", "published", "category", "scraped_at"],
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(articles)
        print(f"OK CSV exported: {output_file}")
        return output_file


def main():
    scraper = KontanInvestasiParallelScraper(workers=5)
    articles, failed_pages = scraper.scrape_all_parallel(total_pages=24)

    if articles:
        scraper.export_json(articles)
        scraper.export_csv(articles)
    else:
        print("Masih belum ada artikel yang tertangkap. Periksa kembali akses jaringan, struktur HTML pencarian, atau aktifkan fallback requests.")


if __name__ == "__main__":
    main()
