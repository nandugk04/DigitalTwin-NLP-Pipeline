# pipeline_tracker.py
import datetime
import os
import pandas as pd


def log_data_collection_run(
    batch_df=None,
    cumulative_df=None,
    new_records_count=0,
    log_file="data_collection_log.csv",
):
    """
    Appends a new execution record below the existing records in the tracking table.
    
    Fields:
    - execution_time: Exact timestamp of run
    - news_start_date: Earliest date from which news/data is collected
    - news_end_date: Latest date up to which news/data is collected
    - batch_new_records: Count of new unique clean records added this run
    - dataset_size_at_execution: Full cumulative clean dataset size at execution
    - source_breakdown: Distribution across sources
    """
    execution_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    today = datetime.date.today()
    default_start = (today - datetime.timedelta(days=28)).strftime("%Y-%m-%d")
    default_end = today.strftime("%Y-%m-%d")

    start_date = default_start
    end_date = default_end

    # Check if there are news records in the current batch within the recent period
    if batch_df is not None and not batch_df.empty and "published_date" in batch_df.columns:
        news_mask = batch_df["source_type"].astype(str).str.contains("News|Media|RSS|GNews", case=False, na=False)
        news_df = batch_df[news_mask] if news_mask.any() else batch_df
        
        valid_dates = news_df["published_date"].dropna().astype(str)
        valid_dates = valid_dates[valid_dates.str.match(r"^\d{4}-\d{2}-\d{2}")]
        
        # Filter for dates within recent collection timeframe (last 35 days)
        recent_cutoff = (today - datetime.timedelta(days=35)).strftime("%Y-%m-%d")
        recent_dates = valid_dates[valid_dates >= recent_cutoff]
        
        if not recent_dates.empty:
            start_date = recent_dates.min()
            end_date = recent_dates.max()

    # Determine total size
    total_size = len(cumulative_df) if cumulative_df is not None else (len(batch_df) if batch_df is not None else 0)

    # Determine source breakdown
    source_summary = "N/A"
    df_for_sources = cumulative_df if cumulative_df is not None else batch_df
    if df_for_sources is not None and "source_name" in df_for_sources.columns:
        counts = df_for_sources["source_name"].value_counts().to_dict()
        source_summary = ", ".join([f"{k}: {v}" for k, v in list(counts.items())[:6]])

    new_log_entry = {
        "execution_time": execution_time,
        "news_start_date": start_date,
        "news_end_date": end_date,
        "batch_new_records": new_records_count,
        "dataset_size_at_execution": total_size,
        "source_breakdown": source_summary,
    }

    new_row_df = pd.DataFrame([new_log_entry])

    # Append below existing rows
    if os.path.exists(log_file):
        try:
            existing_log = pd.read_csv(log_file)
            updated_log = pd.concat([existing_log, new_row_df], ignore_index=True)
        except Exception:
            updated_log = new_row_df
    else:
        updated_log = new_row_df

    updated_log.to_csv(log_file, index=False, encoding="utf-8")

    print("\n" + "=" * 70)
    print("DATA COLLECTION & EXECUTION TRACKING LOG UPDATED")
    print(f"Log Table: '{log_file}'")
    print("-" * 70)
    print(f" Execution Time             : {execution_time}")
    print(f" News/Data Date Range       : From {start_date} to {end_date}")
    print(f" New Clean Records Added    : {new_records_count}")
    print(f" Total Dataset Size for ML  : {total_size}")
    print(f" Source Breakdown           : {source_summary}")
    print("=" * 70 + "\n")

    return updated_log


if __name__ == "__main__":
    if os.path.exists("master_cleaned_data.csv"):
        cdf = pd.read_csv("master_cleaned_data.csv")
        log_data_collection_run(cumulative_df=cdf, new_records_count=len(cdf))
    else:
        print("No master_cleaned_data.csv found to test tracking log.")
