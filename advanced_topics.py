import pandas as pd

def run_bertopic_extraction():
    print("Running Semantic Topic Modeling (BERTopic)...")
    try:
        from bertopic import BERTopic
    except ImportError:
        print("[Skipped] 'bertopic' library not installed. Install with: pip install bertopic")
        return
    
    try:
        df = pd.read_csv("master_cleaned_data.csv")
        docs = df['cleaned_text'].dropna().tolist()
        
        topic_model = BERTopic(language="english", calculate_probabilities=False)
        topics, probs = topic_model.fit_transform(docs)
        topic_info = topic_model.get_topic_info()
        
        print("\nTop AI-Discovered Topics across all domains:")
        print(topic_info[topic_info['Topic'] != -1].head(5)[['Count', 'Name']])
        
        topic_info.to_csv("master_ai_topics.csv", index=False, encoding="utf-8")
        print("\nBERTopic extraction complete. Saved to 'master_ai_topics.csv'")
    except Exception as e:
        print(f"[Warning] BERTopic extraction failed: {e}")

if __name__ == "__main__":
    run_bertopic_extraction()