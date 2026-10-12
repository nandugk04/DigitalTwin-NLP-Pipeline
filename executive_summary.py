import pandas as pd

def generate_sector_summaries():
    print("Generating Executive Summaries using AI...")
    try:
        from transformers import pipeline
    except ImportError:
        print("[Skipped] 'transformers' library not installed. Install with: pip install transformers torch")
        return

    try:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        df = pd.read_csv("master_cleaned_data.csv")
        sectors = df['sector'].unique()
        
        for sector in sectors:
            print(f"\nWriting summary for {str(sector).upper()}...")
            sector_text = " ".join(df[df['sector'] == sector]['cleaned_text'].astype(str).tolist())
            chunk_to_summarize = sector_text[:3000] 
            
            try:
                summary = summarizer(chunk_to_summarize, max_length=50, min_length=15, do_sample=False)
                print(f"AI Insight: {summary[0]['summary_text']}")
            except Exception as e:
                print(f"Could not summarize {sector}: {e}")
    except Exception as e:
        print(f"[Warning] AI Summary failed: {e}")

if __name__ == "__main__":
    generate_sector_summaries()