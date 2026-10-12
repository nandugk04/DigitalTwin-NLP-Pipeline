# main.py
import time
import traceback

# 1. Data Collection & Preparation imports
from fetch_arxiv import fetch_arxiv_data
from fetch_gnews import fetch_gnews_data
from fetch_news import fetch_news_data
from fetch_news_rss import fetch_news_rss_data
from fetch_openalex import fetch_openalex_data
from fetch_hackernews import fetch_hackernews_data
from generate_reference import create_source_reference_table
from merge_data import merge_all_sources
from preprocess_data import run_preprocessing_pipeline
from pipeline_tracker import log_data_collection_run

# 2. NLP & Sentiment Analysis imports
from aspect_sentiment import run_aspect_sentiment_analysis
from ml_perception_model import run_ml_perception_pipeline
from advanced_sentiment import run_advanced_sentiment as run_roberta_sentiment
from perception_summary import summarize_perceptions
from bertopic_clustering import run_bertopic_clustering
from topic_extraction import run_topic_extraction
from executive_summary import generate_sector_summaries


def run_and_track_preprocessing():
    res = run_preprocessing_pipeline(cleanup_raw=True)
    if res is not None:
        log_data_collection_run(
            batch_df=res.get("new_batch_df"),
            cumulative_df=res.get("combined_df"),
            new_records_count=res.get("new_records", 0),
        )
    return res


def main():
    print("\nStarting full data collection and analysis pipeline...\n")
    start_time = time.time()

    step_order = [
        # --- Data Collection & Prep ---
        ("NewsAPI collection", fetch_news_data),
        ("GNews collection", fetch_gnews_data),
        ("Google RSS collection", fetch_news_rss_data),
        ("OpenAlex collection", fetch_openalex_data),
        ("Hacker News collection", fetch_hackernews_data),
        ("arXiv collection", fetch_arxiv_data),
        ("Master dataset merge", merge_all_sources),
        ("Data preprocessing & Cumulative Logging", run_and_track_preprocessing),
        
        # --- Sentiment, ML Embeddings & Perception ---
        ("VADER Aspect-Based Sentiment Analysis", run_aspect_sentiment_analysis),
        ("Word2Vec & ML Perception Modeling", run_ml_perception_pipeline),
        ("RoBERTa Deep Sentiment Analysis", run_roberta_sentiment),
        ("Market Perception Aggregation", summarize_perceptions),
        
        # --- Advanced Semantic Topics & Summaries ---
        ("BERTopic Semantic Clustering", run_bertopic_clustering),
        ("Baseline Topic Extraction", run_topic_extraction),
        ("AI Executive Summarizer", generate_sector_summaries),
        
        # --- Documentation Reference ---
        ("Reference table generation", create_source_reference_table),
    ]

    pipeline_status = {}

    for title, func in step_order:
        print(f"---\nStep: {title}")
        step_start = time.time()
        try:
            func()
            duration = round(time.time() - step_start, 2)
            print(f"Step succeeded: {title} ({duration}s)\n")
            pipeline_status[title] = "SUCCESS"
        except Exception as exc:
            print(f"Step failed: {title}")
            print(f"   {type(exc).__name__}: {exc}")
            traceback.print_exc()
            print("   Continuing with the next step...\n")
            pipeline_status[title] = f"FAILED ({type(exc).__name__})"

    total_time = round(time.time() - start_time, 2)
    
    print("=" * 50)
    print("PIPELINE EXECUTION SUMMARY")
    print(f"Total Execution Time: {total_time} seconds\n")
    for step, status in pipeline_status.items():
        icon = "[OK]" if status == "SUCCESS" else "[FAIL]"
        print(f" {icon} {step}: {status}")
    print("=" * 50)


if __name__ == "__main__":
    main()