# advanced_sentiment.py
import os
import pandas as pd


def run_advanced_sentiment():
  print('Running Deep Learning Sentiment Analysis (RoBERTa)...')
  try:
    from transformers import pipeline
  except ImportError:
    print("[Skipped] 'transformers' library not installed. Install with: pip install transformers torch")
    return
  
  # Load aspect or cleaned dataset
  if os.path.exists('master_nlp_aspects.csv'):
    df = pd.read_csv('master_nlp_aspects.csv')
  elif os.path.exists('master_cleaned_data.csv'):
    df = pd.read_csv('master_cleaned_data.csv')
  else:
    print('[Error] No cleaned dataset found to run sentiment analysis.')
    return

  # Initialize the Hugging Face RoBERTa sentiment classification pipeline
  print('Loading cardiffnlp/twitter-roberta-base-sentiment-latest model...')
  try:
    sentiment_pipeline = pipeline(
        'sentiment-analysis',
        model='cardiffnlp/twitter-roberta-base-sentiment-latest',
    )
  except Exception as e:
    print(f"[Warning] Could not initialize RoBERTa model: {e}")
    return

  texts = df['cleaned_text'].astype(str).tolist()
  results = []

  print('Scoring documents with RoBERTa transformer (processing records)...')
  for text in texts:
    try:
      # Truncate text to 512 tokens max limit for transformers
      res = sentiment_pipeline(text[:512])[0]
      results.append(res['label'])
    except Exception:
      results.append('Neutral')

  # Assign results to DataFrame column
  df['roberta_sentiment'] = results

  # Save updated file back to master aspects file for integration
  output_file = 'master_nlp_aspects.csv'
  df.to_csv(output_file, index=False)
  print(f"RoBERTa sentiment results successfully saved to '{output_file}'.")


if __name__ == '__main__':
  run_advanced_sentiment()