# bertopic_clustering.py
import pandas as pd


def run_bertopic_clustering():
  print('Running Advanced BERTopic Semantic Clustering...')
  try:
    from bertopic import BERTopic
  except ImportError:
    print("[Skipped] 'bertopic' library not installed. Install with: pip install bertopic")
    return
  try:
    df = pd.read_csv('master_cleaned_data.csv')
  except FileNotFoundError:
    print("[Error] 'master_cleaned_data.csv' not found.")
    return

  docs = df['cleaned_text'].astype(str).tolist()

  # Initialize and fit BERTopic model
  topic_model = BERTopic(language='english', calculate_probabilities=True, verbose=True)
  topics, probs = topic_model.fit_transform(docs)

  # Display top discovered topics
  print('\n--- Top Discovered Semantic Topics ---')
  print(topic_model.get_topic_info().head(10))

  # Save topic info
  topic_info = topic_model.get_topic_info()
  topic_info.to_csv('bertopic_summary.csv', index=False)
  print("BERTopic results saved to 'bertopic_summary.csv'.")


if __name__ == '__main__':
  run_bertopic_clustering()