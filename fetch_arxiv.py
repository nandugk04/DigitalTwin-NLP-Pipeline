# fetch_arxiv.py
import time
import xml.etree.ElementTree as ET
import urllib.parse
import urllib.request
import pandas as pd
from config import KEYWORDS


def fetch_arxiv_data(pages_per_sector=50, max_per_page=100):
    all_papers = []

    for sector, queries in KEYWORDS.items():
        print(f"[arXiv] Fetching Academic Abstracts for: {sector.upper()}...")
        encoded_query = urllib.parse.quote(queries["arxiv"])
        sector_count = 0

        for page in range(pages_per_sector):
            start = page * max_per_page
            url = f"http://export.arxiv.org/api/query?search_query={encoded_query}&start={start}&max_results={max_per_page}"

            try:
                req = urllib.request.Request(url, headers={"User-Agent": "DigitalTwinResearch/1.0"})
                response = urllib.request.urlopen(req, timeout=15)
                xml_data = response.read()
                root = ET.fromstring(xml_data)
            except Exception as e:
                print(f"   [Error] Fetching page {page} for {sector}: {e}")
                continue

            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall("atom:entry", ns)
            if not entries:
                break

            for entry in entries:
                title_elem = entry.find("atom:title", ns)
                summary_elem = entry.find("atom:summary", ns)
                published_elem = entry.find("atom:published", ns)
                id_elem = entry.find("atom:id", ns)

                title = title_elem.text.strip().replace("\n", " ") if title_elem is not None and title_elem.text else ""
                summary = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None and summary_elem.text else ""
                pub_date = published_elem.text.strip()[:10] if published_elem is not None and published_elem.text else time.strftime("%Y-%m-%d")
                paper_url = id_elem.text.strip() if id_elem is not None and id_elem.text else ""

                if title and summary:
                    all_papers.append(
                        {
                            "sector": sector,
                            "source_type": "Academic Paper",
                            "source_name": "arXiv Repository",
                            "title": title,
                            "text": summary,
                            "url": paper_url,
                            "published_date": pub_date,
                            "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    )
                    sector_count += 1

            time.sleep(1)  # Respect arXiv API rate guidelines

        print(f"   Collected {sector_count} arXiv papers for {sector}")

    # Save to individual CSV
    if all_papers:
        df = pd.DataFrame(all_papers)
        df = df.drop_duplicates(subset=["text"])
        df.to_csv("arxiv_raw_data.csv", index=False, encoding="utf-8")
        print(
            f"Success! Saved {len(df)} unique academic rows to 'arxiv_raw_data.csv'\n"
        )
    else:
        print("No academic data collected.")


if __name__ == "__main__":
    fetch_arxiv_data()