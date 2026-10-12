# word2vec_sentiment_model.py
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def run_word2vec_sentiment():
  print('Loading dataset...')
  try:
    df = pd.read_csv('master_nlp_aspects.csv')
  except FileNotFoundError:
    print("[Error] 'master_nlp_aspects.csv' not found.")
    return

  # Ensure text and a sentiment label column exist
  if 'cleaned_text' not in df.columns:
    print('[Error] "cleaned_text" column missing.')
    return

  # Create a target sentiment label using VADER compound scores if explicit label is missing
  if 'sentiment_label' not in df.columns and 'Cost & ROI' in df.columns:
    # Use Cost & ROI pillar score as a proxy sentiment label for demonstration
    df['sentiment_label'] = df['Cost & ROI'].apply(
        lambda c: (
            'Positive'
            if float(c) >= 0.05
            else ('Negative' if float(c) <= -0.05 else 'Neutral')
        )
    )

  df = df.dropna(subset=['cleaned_text', 'sentiment_label'])

  # 1. Train Word2Vec on the corpus
  print('Training Word2Vec model on text corpus...')
  tokenized_sentences = [
      str(text).lower().split() for text in df['cleaned_text']
  ]
  w2v_model = Word2Vec(
      sentences=tokenized_sentences, vector_size=100, window=5, min_count=1
  )

  # 2. Convert each document into an average vector
  def get_document_vector(words):
    valid_words = [w for w in words if w in w2v_model.wv]
    if not valid_words:
      return np.zeros(100)
    return np.mean([w2v_model.wv[w] for w in valid_words], axis=0)

 # Convert documents into Word2Vec feature vectors and labels into numpy arrays
  print('Converting documents into Word2Vec feature vectors...')
  X = np.array([get_document_vector(text) for text in tokenized_sentences])
  y = np.array(df['sentiment_label'].tolist())

  # 3. Train/Test Split & Machine Learning Classification
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  print('Training Logistic Regression classifier on Word2Vec vectors...')
  clf = LogisticRegression(max_iter=1000)
  clf.fit(X_train, y_train)

  y_pred = clf.predict(X_test)
  acc = accuracy_score(y_test, y_pred)

  print('\n=== WORD2VEC + ML SENTIMENT RESULTS ===')
  print(f'Model Accuracy: {acc * 100:.2f}%')
  print('\nClassification Report:')
  print(classification_report(y_test, y_pred))


if __name__ == '__main__':
  run_word2vec_sentiment()