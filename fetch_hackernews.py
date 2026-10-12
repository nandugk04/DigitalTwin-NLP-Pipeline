# fetch_hackernews.py
import time
import requests
import pandas as pd

HN_QUERIES = {
    "healthcare": ["digital twin healthcare", "medical digital twin"],
    "manufacturing": ["digital twin manufacturing", "industry 4.0 factory"],
    "business": ["digital twin supply chain", "digital twin logistics enterprise"],
}


def fetch_hackernews_data(pages_per_query=30):
    all_hn = []
    print("[HackerNews] Fetching Tech Community Discussions...")

    headers = {
        "User-Agent": "DigitalTwinDataCollector/1.0"
    }

    for sector, queries in HN_QUERIES.items():
        sector_count = 0
        for q in queries:
            for page in range(pages_per_query):
                url = f"https://hn.algolia.com/api/v1/search?query={q}&hitsPerPage=100&page={page}"
                
                try:
                    response = requests.get(url, headers=headers, timeout=15)
                    if response.status_code != 200:
                        break

                    data = response.json()
                    hits = data.get("hits", [])
                    if not hits:
                        break

                    for hit in hits:
                        title = hit.get("title") or hit.get("story_title") or ""
                        story_text = hit.get("story_text") or hit.get("comment_text") or title

                        if "<" in story_text and ">" in story_text:
                            import re
                            story_text = re.sub(r"<[^>]+>", "", story_text).strip()

                        pub_date = str(hit.get("created_at", ""))[:10]
                        if not pub_date:
                            pub_date = time.strftime("%Y-%m-%d")

                        story_url = hit.get("url") or (f"https://news.ycombinator.com/item?id={hit.get('objectID')}" if hit.get("objectID") else "")

                        if title or len(story_text) > 15:
                            all_hn.append(
                                {
                                    "sector": sector,
                                    "source_type": "Tech Community Discussions",
                                    "source_name": "Hacker News (Algolia)",
                                    "title": title if title else story_text[:60] + "...",
                                    "text": story_text,
                                    "url": story_url,
                                    "published_date": pub_date,
                                    "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                }
                            )
                            sector_count += 1

                    time.sleep(0.3)
                except Exception as e:
                    print(f"   Error fetching Hacker News page {page} for '{q}': {e}")
                    break

        print(f"   Collected {sector_count} Hacker News records for {sector.upper()}")

    if all_hn:
        df = pd.DataFrame(all_hn)
        df = df.drop_duplicates(subset=["text"])
        df.to_csv("hackernews_raw_data.csv", index=False, encoding="utf-8")
        print(f"Success! Saved {len(df)} unique Hacker News rows to 'hackernews_raw_data.csv'\n")
    else:
        print("No Hacker News data collected.")


if __name__ == "__main__":
    fetch_hackernews_data()
