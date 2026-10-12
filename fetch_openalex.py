# fetch_openalex.py
import time
import requests
import pandas as pd

OPENALEX_QUERIES = {
    "healthcare": "digital twin healthcare medical",
    "manufacturing": "digital twin manufacturing factory industry 4.0",
    "business": "digital twin supply chain logistics enterprise",
}


def reconstruct_abstract(inverted_index):
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort(key=lambda x: x[0])
    return " ".join([word for _, word in word_positions])


def fetch_openalex_data(pages_per_sector=50, per_page=200):
    all_works = []
    print("[OpenAlex] Fetching Global Research & Industry Literature...")

    headers = {
        "User-Agent": "DigitalTwinResearchPipeline/1.0 (mailto:researcher@digitaltwin.org)"
    }

    for sector, query in OPENALEX_QUERIES.items():
        sector_count = 0
        for page in range(1, pages_per_sector + 1):
            url = f"https://api.openalex.org/works?filter=default.search:{query}&per-page={per_page}&page={page}"
            
            try:
                response = requests.get(url, headers=headers, timeout=15)
                if response.status_code != 200:
                    print(f"   OpenAlex status {response.status_code} for {sector} page {page}")
                    break

                data = response.json()
                results = data.get("results", [])
                if not results:
                    break

                for work in results:
                    title = work.get("title") or work.get("display_name") or ""
                    inverted_abstract = work.get("abstract_inverted_index")
                    abstract = reconstruct_abstract(inverted_abstract)
                    text_content = abstract if len(abstract) > 30 else title

                    pub_date = work.get("publication_date")
                    if not pub_date and work.get("publication_year"):
                        pub_date = f"{work.get('publication_year')}-01-01"
                    if not pub_date:
                        pub_date = str(time.strftime("%Y-%m-%d"))

                    work_url = work.get("doi") or work.get("id") or ""

                    if title and text_content:
                        all_works.append(
                            {
                                "sector": sector,
                                "source_type": "Research & Industry Index",
                                "source_name": "OpenAlex Repository",
                                "title": title,
                                "text": text_content,
                                "url": work_url,
                                "published_date": pub_date,
                                "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            }
                        )
                        sector_count += 1

                time.sleep(0.5)
            except Exception as e:
                print(f"   Error fetching OpenAlex page {page} for {sector}: {e}")
                break

        print(f"   Collected {sector_count} OpenAlex records for {sector.upper()}")

    if all_works:
        df = pd.DataFrame(all_works)
        df = df.drop_duplicates(subset=["text"])
        df.to_csv("openalex_raw_data.csv", index=False, encoding="utf-8")
        print(f"Success! Saved {len(df)} unique OpenAlex rows to 'openalex_raw_data.csv'\n")
    else:
        print("No OpenAlex data collected.")


if __name__ == "__main__":
    fetch_openalex_data()
