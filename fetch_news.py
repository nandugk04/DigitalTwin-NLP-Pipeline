# fetch_news.py
import datetime
import pandas as pd
import requests
from config import KEYWORDS, NEWS_API_KEY


def fetch_news_data():
    base_url = "https://newsapi.org/v2/everything"
    all_news = []

    # Free tier allows looking back 28-30 days
    from_date = (datetime.date.today() - datetime.timedelta(days=28)).isoformat()

    for sector, queries in KEYWORDS.items():
        print(f"[NewsAPI] Fetching News Articles for: {sector.upper()}...")
        sector_count = 0

        for page in range(1, 6):
            params = {
                "q": queries["news"],
                "from": from_date,
                "sortBy": "relevancy",
                "language": "en",
                "pageSize": 100,
                "page": page,
                "apiKey": NEWS_API_KEY,
            }

            response = requests.get(base_url, params=params)
            if response.status_code != 200:
                print(f"   Status {response.status_code} for {sector} page {page}")
                break

            articles = response.json().get("articles", [])
            if not articles:
                break

            for art in articles:
                pub_date = str(art.get("publishedAt", ""))[:10]  # Extract YYYY-MM-DD
                all_news.append(
                    {
                        "sector": sector,
                        "source_type": "News Media",
                        "source_name": art.get("source", {}).get("name", "NewsAPI"),
                        "title": art.get("title", ""),
                        "text": art.get("description") or art.get("title", ""),
                        "url": art.get("url", ""),
                        "published_date": pub_date,
                        "fetched_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
                sector_count += 1

        print(f"   Collected {sector_count} news articles for {sector.upper()}")

    # Save to individual CSV
    if all_news:
        df = pd.DataFrame(all_news)
        df.to_csv("news_raw_data.csv", index=False, encoding="utf-8")
        print(
            f"Success! Saved {len(df)} news rows to 'news_raw_data.csv'\n"
        )
    else:
        print("No news data collected. Check API key.")


if __name__ == "__main__":
    fetch_news_data()