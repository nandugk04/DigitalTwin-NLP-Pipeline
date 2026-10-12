# topic_sentiment_mapping.py
import pandas as pd


def run_topic_sentiment_mapping():
  print('Loading dataset for topic-sentiment correlation analysis...')
  try:
    df = pd.read_csv('master_nlp_aspects.csv')
  except FileNotFoundError:
    print(
        "[Error] 'master_nlp_aspects.csv' not found. Ensure RoBERTa and BERTopic"
        ' have run.'
    )
    return

  if 'roberta_sentiment' not in df.columns or 'topic' not in df.columns:
    print(
        '[Error] Required columns ("roberta_sentiment", "topic") missing from'
        ' dataframe.'
    )
    return

  # Create cross-tabulation mapping sentiment polarity across topics
  mapping_table = pd.crosstab(
      df['topic'],
      df['roberta_sentiment'],
      normalize='index',
  ) * 100

  print('\n--- Topic-Sentiment Proportions (%) per Topic ---')
  print(mapping_table.round(2).to_string())

  # Save results for thesis charts
  mapping_table.to_csv('topic_sentiment_correlation.csv')
  print(
      '\nTopic-sentiment mapping successfully saved to'
      " 'topic_sentiment_correlation.csv'."
  )


if __name__ == '__main__':
  run_topic_sentiment_mapping()