# preprocess_data.py
import os
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK assets silently
try:
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))


def clean_text(text):
    if not isinstance(text, str):
        return ""

    # 1. Convert text to lowercase
    text = text.lower()

    # 2. Remove web links (URLs)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # 3. Remove HTML tags if any exist
    text = re.sub(r"<.*?>", "", text)

    # 4. Remove special characters, numbers, and punctuation
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # 5. Tokenize the text (split into individual words)
    try:
        words = word_tokenize(text)
    except Exception:
        words = text.split()

    # 6. Remove Stopwords
    cleaned_words = [w for w in words if w not in STOP_WORDS]

    # Join the words back together into a clean sentence
    return " ".join(cleaned_words)


def cleanup_temporary_raw_files():
    """Deletes temporary raw staging files after successful cleaning."""
    raw_files = [
        "news_raw_data.csv",
        "gnews_raw_data.csv",
        "arxiv_raw_data.csv",
        "news_rss_raw_data.csv",
        "openalex_raw_data.csv",
        "hackernews_raw_data.csv",
        "master_raw_data.csv",
    ]
    print("\nCleaning up temporary raw staging files...")
    for f in raw_files:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception as e:
                print(f"   Could not remove {f}: {e}")
    print("Raw staging files cleaned up successfully.")


def run_preprocessing_pipeline(cleanup_raw=True):
    input_file = "master_raw_data.csv"
    output_file = "master_cleaned_data.csv"

    if not os.path.exists(input_file):
        print(f"Error: '{input_file}' not found. Please run merge_data.py first!")
        return None

    print(f"Loading raw batch from '{input_file}'...")
    new_df = pd.read_csv(input_file)

    print("Cleaning text data (lowercasing, tokenizing, removing stopwords)...")
    new_df["cleaned_text"] = new_df["text"].apply(clean_text)

    # Filter out rows where cleaning resulted in an empty string
    new_df = new_df[new_df["cleaned_text"].astype(str).str.strip() != ""]
    new_df = new_df.drop_duplicates(subset=["cleaned_text"])

    initial_existing_count = 0
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            initial_existing_count = len(existing_df)
            print(f"Existing clean dataset found with {initial_existing_count} records.")
            
            # Append new cleaned rows BELOW the existing clean rows
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            # Deduplicate so previously cleaned text is not repeated
            combined_df = combined_df.drop_duplicates(subset=["cleaned_text"], keep="first")
        except Exception as e:
            print(f"Warning reading existing {output_file}: {e}. Overwriting with current batch.")
            combined_df = new_df
    else:
        combined_df = new_df

    total_clean_count = len(combined_df)
    new_added_count = total_clean_count - initial_existing_count

    print(f"Saving cumulative cleaned dataset to '{output_file}'...")
    combined_df.to_csv(output_file, index=False, encoding="utf-8")

    print("\nPREPROCESSING & CUMULATIVE CLEANING COMPLETE!")
    print(f"New Clean Records Added in this Run: {new_added_count}")
    print(f"Total Cumulative Clean Records for Model Training: {total_clean_count}")

    if cleanup_raw:
        cleanup_temporary_raw_files()

    return {
        "new_records": new_added_count,
        "total_records": total_clean_count,
        "combined_df": combined_df,
        "new_batch_df": new_df,
    }


if __name__ == "__main__":
    run_preprocessing_pipeline(cleanup_raw=False)