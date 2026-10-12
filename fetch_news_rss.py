# fetch_news_rss.py
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import pandas as pd

# Define sectors and search terms for Google News RSS
SECTOR_RSS_QUERIES = {
    "healthcare": [
        "digital twin healthcare",
        "digital twin hospital medical",
        "virtual twin healthcare sustainability",
    ],
    "manufacturing": [
        "digital twin manufacturing",
        "industry 4.0 digital twin factory",
        "smart manufacturing digital twin",
    ],
    "business": [
        "digital twin supply chain",
        "enterprise twin logistics",
        "digital twin business operations sustainability",
    ],
}

# General Tech/Industry RSS feeds to parse
GENERAL_RSS_FEEDS = [
    ("IEEE Spectrum", "https://spectrum.ieee.org/rss/index.xml"),
    ("TechCrunch", "https://techcrunch.com/feed/"),
    ("MIT News", "https://news.mit.edu/rss/feed"),
]


import datetime
from email.utils import parsedate_to_datetime


def parse_rss_url(url, source_name, sector="business"):
    items = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=12) as resp:
            xml_data = resp.read()
        root = ET.fromstring(xml_data)

        # Standard RSS parser (channel -> item)
        for item in root.findall(".//item"):
            title_elem = item.find("title")
            desc_elem = item.find("description")
            source_elem = item.find("source")
            pubdate_elem = item.find("pubDate")
            link_elem = item.find("link")

            title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
            desc = desc_elem.text.strip() if desc_elem is not None and desc_elem.text else title
            link = link_elem.text.strip() if link_elem is not None and link_elem.text else ""
            
            # Clean basic HTML tags from RSS description if any
            if "<" in desc and ">" in desc:
                import re
                desc = re.sub(r"<[^>]+>", "", desc).strip()

            src = source_elem.text.strip() if source_elem is not None and source_elem.text else source_name

            pub_date = ""
            if pubdate_elem is not None and pubdate_elem.text:
                try:
                    dt = parsedate_to_datetime(pubdate_elem.text.strip())
                    pub_date = dt.strftime("%Y-%m-%d")
                except Exception:
                    pub_date = str(datetime.date.today())
            if not pub_date:
                pub_date = str(datetime.date.today())

            if title and len(desc) > 10:
                items.append(
                    {
                        "sector": sector,
                        "source_type": "News Media (RSS)",
                        "source_name": src,
                        "title": title,
                        "text": desc,
                        "url": link,
                        "published_date": pub_date,
                        "fetched_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
    except Exception as e:
        pass

    return items


def fetch_news_rss_data():
    all_rss_news = []
    print("[News RSS] Fetching Articles from Google News RSS & Industry Feeds...")

    # 1. Fetch Google News RSS for each sector
    for sector, queries in SECTOR_RSS_QUERIES.items():
        sector_items = 0
        for q in queries:
            encoded = urllib.parse.quote(q)
            rss_url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
            articles = parse_rss_url(rss_url, "Google News RSS", sector=sector)
            all_rss_news.extend(articles)
            sector_items += len(articles)

        print(f"   Found {sector_items} Google News RSS items for {sector.upper()}")

    # 2. Fetch General Tech RSS feeds
    for feed_name, feed_url in GENERAL_RSS_FEEDS:
        articles = parse_rss_url(feed_url, feed_name, sector="business")
        for art in articles:
            txt = (art["title"] + " " + art["text"]).lower()
            if any(k in txt for k in ["health", "medical", "hospital", "patient"]):
                art["sector"] = "healthcare"
            elif any(k in txt for k in ["manufactur", "factory", "industrial", "robot"]):
                art["sector"] = "manufacturing"
            else:
                art["sector"] = "business"
            all_rss_news.append(art)

    if all_rss_news:
        df = pd.DataFrame(all_rss_news)
        df = df.drop_duplicates(subset=["text"])
        df.to_csv("news_rss_raw_data.csv", index=False, encoding="utf-8")
        print(f"Success! Saved {len(df)} unique RSS news rows to 'news_rss_raw_data.csv'\n")
    else:
        print("No RSS news data collected.")


if __name__ == "__main__":
    fetch_news_rss_data()
