# fetch_gnews.py
import datetime
import pandas as pd
import requests
from config import GNEWS_API_KEY, GNEWS_KEYWORDS


def fetch_gnews_data():
    base_url = "https://gnews.io/api/v4/search"
    all_gnews = []

    for sector, query in GNEWS_KEYWORDS.items():
        print(f"[GNews] Fetching GNews for: {sector.upper()}...")

        params = {
            "q": query,
            "lang": "en",
            "max": 10,
            "apikey": GNEWS_API_KEY,
        }

        response = requests.get(base_url, params=params)
        
        if response.status_code != 200:
            error_details = response.json().get("errors", "Unknown Error")
            print(f"   Error fetching {sector} from GNews: {response.status_code}")
            print(f"   Reason: {error_details}")
            continue

        data = response.json()
        articles = data.get("articles", [])
        print(f"   Found {len(articles)} articles for {sector}")

        for art in articles:
            pub_date = str(art.get("publishedAt", ""))[:10]  # Extract YYYY-MM-DD
            all_gnews.append(
                {
                    "sector": sector,
                    "source_type": "News Media (GNews)",
                    "source_name": art.get("source", {}).get("name", "GNews"),
                    "title": art.get("title", ""),
                    "text": art.get("description") or art.get("title", ""),
                    "url": art.get("url", ""),
                    "published_date": pub_date,
                    "fetched_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

    if all_gnews:
        df = pd.DataFrame(all_gnews)
        df.to_csv("gnews_raw_data.csv", index=False, encoding="utf-8")
        print(f"Success! Saved {len(df)} rows to 'gnews_raw_data.csv'\n")
    else:
        print("No GNews data collected.")


if __name__ == "__main__":
    fetch_gnews_data()