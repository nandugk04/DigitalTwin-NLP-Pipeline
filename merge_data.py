# merge_data.py
import os
import pandas as pd


def merge_all_sources():
    print("Merging all raw datasets into Master File...")

    datasets = []

    # 1. Check & load NewsAPI Data
    if os.path.exists("news_raw_data.csv"):
        datasets.append(pd.read_csv("news_raw_data.csv"))
    else:
        print("Warning: 'news_raw_data.csv' missing.")

    # 2. Check & load GNews Data
    if os.path.exists("gnews_raw_data.csv"):
        datasets.append(pd.read_csv("gnews_raw_data.csv"))
    else:
        print("Warning: 'gnews_raw_data.csv' missing.")

    # 3. Check & load arXiv Data
    if os.path.exists("arxiv_raw_data.csv"):
        datasets.append(pd.read_csv("arxiv_raw_data.csv"))
    else:
        print("Warning: 'arxiv_raw_data.csv' missing.")

    # 4. Check & load News RSS Data
    if os.path.exists("news_rss_raw_data.csv"):
        datasets.append(pd.read_csv("news_rss_raw_data.csv"))
    else:
        print("Warning: 'news_rss_raw_data.csv' missing.")

    # 5. Check & load OpenAlex Data
    if os.path.exists("openalex_raw_data.csv"):
        datasets.append(pd.read_csv("openalex_raw_data.csv"))
    else:
        print("Warning: 'openalex_raw_data.csv' missing.")

    # 6. Check & load Hacker News Data
    if os.path.exists("hackernews_raw_data.csv"):
        datasets.append(pd.read_csv("hackernews_raw_data.csv"))
    else:
        print("Warning: 'hackernews_raw_data.csv' missing.")

    if not datasets:
        print("Error: No raw CSV files found to merge.")
        return

    # Combine all loaded datasets
    master_df = pd.concat(datasets, ignore_index=True)

    # Ensure all expected columns exist
    for col in ["published_date", "url", "fetched_at"]:
        if col not in master_df.columns:
            master_df[col] = ""

    # Remove blank text entries
    master_df = master_df.dropna(subset=["text"])
    master_df = master_df[master_df["text"].astype(str).str.strip() != ""]

    # Drop exact duplicate rows based on text content
    master_df = master_df.drop_duplicates(subset=["text"])

    # Save to master raw dataset
    master_df.to_csv("master_raw_data.csv", index=False, encoding="utf-8")

    print("\nMASTER RAW DATASET GENERATION COMPLETE!")
    print("File saved as: 'master_raw_data.csv'")
    print(f"Total Unique Raw Data Points In Current Batch: {len(master_df)}")
    print("\n--- Distribution Across Domains ---")
    print(master_df["sector"].value_counts())
    return master_df


if __name__ == "__main__":
    merge_all_sources()