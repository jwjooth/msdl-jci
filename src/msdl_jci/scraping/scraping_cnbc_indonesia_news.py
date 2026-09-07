import csv
import json
import logging
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from bs4 import BeautifulSoup
from curl_cffi import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CNBC_Deep_Scraper")


class CNBCRSSDeepScraper:
    def __init__(self, output_dir: str = "./cnbc_rss_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.rss_endpoints = {
            "market": "https://www.cnbcindonesia.com/market/rss",
            "news": "https://www.cnbcindonesia.com/news/rss",
            "terbaru": "https://www.cnbcindonesia.com/rss",
        }

    def _get_rss_metadata(self) -> List[Dict[str, str]]:
        """Tahap 1: Mengambil metadata (URL) dari RSS."""
        all_articles = []
        seen_urls = set()
        session = requests.Session(impersonate="chrome120")

        for cat, url in self.rss_endpoints.items():
            logger.info(f"Fetching RSS: {cat.upper()}")
            try:
                response = session.get(url, timeout=10)
                if response.status_code != 200:
                    continue

                root = ET.fromstring(response.content)
                channel = root.find("channel")
                if channel is None:
                    continue

                for item in channel.findall("item"):
                    link = item.findtext("link", default="").strip()
                    if link and link not in seen_urls:
                        all_articles.append({
                            "title": item.findtext("title", default="").strip(),
                            "url": link,
                            "pub_date": item.findtext("pubDate", default="").strip(),
                            "category": cat
                        })
                        seen_urls.add(link)
            except Exception as e:
                logger.error(f"Error parsing RSS for {cat}: {e}")

        session.close()
        return all_articles

    def _fetch_content(self, article: Dict[str, str]) -> Dict[str, str]:
        """Tahap 2: Mengunjungi URL dan mengekstrak teks berita HTML."""
        url = article["url"]
        article["content"] = ""
        article["scraped_at"] = datetime.now().isoformat()

        try:
            # Jeda ringan agar tidak diblokir saat deep scrape
            time.sleep(0.5)
            session = requests.Session(impersonate="chrome120")
            response = session.get(url, timeout=15)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")

                # Class target utama konten berita CNBC
                body = soup.select_one(".detail_text, .text_detail, article")

                if body:
                    # Hapus elemen yang tidak relevan (script, iframe, iklan video)
                    for tag in body(["script", "style", "iframe", "div.video", "div.paragraf_bottom"]):
                        tag.decompose()

                    # Ekstrak teks bersih
                    article["content"] = " ".join(body.stripped_strings)

            session.close()

        except Exception as e:
            logger.error(f"Failed to fetch content for {url}: {e}")

        return article

    def execute_deep_scrape(self, limit: int = 10) -> List[Dict[str, str]]:
        # 1. Ambil semua link dari RSS
        metadata_list = self._get_rss_metadata()
        logger.info(f"Found {len(metadata_list)} unique URLs from RSS.")

        # Batasi jumlah agar tidak terlalu lama saat testing
        targets = metadata_list[:limit]
        logger.info(f"Starting deep scrape for {len(targets)} articles...")

        final_dataset = []

        # 2. Eksekusi Deep Scrape secara paralel dengan max 3 workers
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(self._fetch_content, art): art for art in targets}

            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                final_dataset.append(result)
                print(f"Deep Scrape Progress: {i}/{len(targets)}", end="\r")

        print()  # New line after progress
        return final_dataset

    def export_data(self, articles: List[Dict[str, str]], filename_prefix: str = "cnbc_deep_scrape"):
        if not articles:
            return

        json_file = self.output_dir / f"{filename_prefix}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)

        csv_file = self.output_dir / f"{filename_prefix}.csv"
        fieldnames = ["title", "url", "pub_date", "category", "content", "scraped_at"]
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(articles)

        logger.info(f"Full content saved to {json_file}")


if __name__ == "__main__":
    scraper = CNBCRSSDeepScraper()

    # Set limit=20 untuk mengambil isi konten penuh dari 20 berita terbaru saja sebagai uji coba
    # Ubah limit=1000 jika ingin memproses semua isi beritanya.
    dataset = scraper.execute_deep_scrape(limit=1000)

    scraper.export_data(dataset)