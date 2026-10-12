# generate_reference.py
import os
import pandas as pd


def create_source_reference_table():
    print("Generating Project Data Sources Reference Table...")

    # Load cumulative cleaned data if available to count actual preserved records
    clean_df = None
    if os.path.exists("master_cleaned_data.csv"):
        try:
            clean_df = pd.read_csv("master_cleaned_data.csv")
        except Exception:
            clean_df = None

    def get_source_count(source_keyword, fallback_file):
        if clean_df is not None and not clean_df.empty:
            if "source_type" in clean_df.columns or "source_name" in clean_df.columns:
                match = clean_df[
                    clean_df["source_type"].astype(str).str.contains(source_keyword, case=False, na=False) |
                    clean_df["source_name"].astype(str).str.contains(source_keyword, case=False, na=False)
                ]
                return len(match)
        if os.path.exists(fallback_file):
            try:
                return len(pd.read_csv(fallback_file))
            except Exception:
                return 0
        return 0

    news_count = get_source_count("NewsAPI|News Media", "news_raw_data.csv")
    gnews_count = get_source_count("GNews", "gnews_raw_data.csv")
    arxiv_count = get_source_count("arXiv", "arxiv_raw_data.csv")
    news_rss_count = get_source_count("RSS", "news_rss_raw_data.csv")
    openalex_count = get_source_count("OpenAlex", "openalex_raw_data.csv")
    hackernews_count = get_source_count("Hacker News|Algolia", "hackernews_raw_data.csv")

    reference_data = [
        {
            "Data Source Name": "NewsAPI",
            "Source Type": "Commercial & Market Media",
            "Keywords / Query Strings Used": '"Digital Twin" AND (healthcare OR manufacturing OR business)',
            "Target Domains": "Healthcare, Manufacturing, Business",
            "Sample Size (Rows)": news_count,
            "Access Type": "Developer API Key (Free Tier)",
        },
        {
            "Data Source Name": "GNews API",
            "Source Type": "Global Tech News Headlines",
            "Keywords / Query Strings Used": "digital twin healthcare / manufacturing / logistics",
            "Target Domains": "Healthcare, Manufacturing, Business",
            "Sample Size (Rows)": gnews_count,
            "Access Type": "Developer API Key (Free Tier)",
        },
        {
            "Data Source Name": "arXiv Repository API",
            "Source Type": "Academic & Scientific Abstracts",
            "Keywords / Query Strings Used": 'all:"Digital Twin" AND (all:healthcare OR all:manufacturing OR all:business)',
            "Target Domains": "Smart Health, Industry 4.0, Logistics",
            "Sample Size (Rows)": arxiv_count,
            "Access Type": "Open-Access XML Query (No Key)",
        },
        {
            "Data Source Name": "Google News RSS & Tech Feeds",
            "Source Type": "Global News & Industry RSS Feeds",
            "Keywords / Query Strings Used": "digital twin healthcare / manufacturing / supply chain",
            "Target Domains": "Healthcare, Manufacturing, Business",
            "Sample Size (Rows)": news_rss_count,
            "Access Type": "Open RSS Endpoint (No Key)",
        },
        {
            "Data Source Name": "OpenAlex Index API",
            "Source Type": "Global Science & Tech Literature",
            "Keywords / Query Strings Used": "digital twin healthcare / manufacturing / supply chain",
            "Target Domains": "Healthcare, Manufacturing, Business",
            "Sample Size (Rows)": openalex_count,
            "Access Type": "Open REST API (No Key)",
        },
        {
            "Data Source Name": "Hacker News (Algolia API)",
            "Source Type": "Tech Community Discussions",
            "Keywords / Query Strings Used": "digital twin / industry 4.0",
            "Target Domains": "Healthcare, Manufacturing, Business",
            "Sample Size (Rows)": hackernews_count,
            "Access Type": "Open Algolia Search API (No Key)",
        },
    ]

    ref_df = pd.DataFrame(reference_data)
    output_filename = "data_sources_reference.csv"
    ref_df.to_csv(output_filename, index=False, encoding="utf-8")

    print(f"\nSuccess! Documentation table saved to '{output_filename}'")
    print("\n--- PROJECT REFERENCE TABLE PREVIEW ---")
    print(ref_df[["Data Source Name", "Source Type", "Sample Size (Rows)"]])


if __name__ == "__main__":
    create_source_reference_table()