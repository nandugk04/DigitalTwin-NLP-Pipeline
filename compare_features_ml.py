# compare_features_ml.py
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def run_side_by_side_comparison():
  print('Loading dataset...')
  try:
    df = pd.read_csv('master_nlp_aspects.csv')
  except FileNotFoundError:
    print("[Error] 'master_nlp_aspects.csv' not found.")
    return

  if 'sector' not in df.columns or 'cleaned_text' not in df.columns:
    print(
        '[Error] Required columns ("sector" or "cleaned_text") missing from'
        ' CSV.'
    )
    return

  pillar_cols = [
      'Cost & ROI',
      'Security & Privacy',
      'Scalability & Tech',
      'Sustainability & Green',
  ]

  # Fill missing VADER values with 0.0 instead of dropping the whole row
  for col in pillar_cols:
    if col in df.columns:
      df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    else:
      df[col] = 0.0

  df_clean = df.dropna(subset=['sector', 'cleaned_text']).copy()
  y = np.array(df_clean['sector'].tolist())

  print(f'Total samples used for fair comparison: {len(df_clean)}\n')

  # ==========================================
  # EXPERIMENT 1: ML using VADER Pillar Features
  # ==========================================
  print('--- Running ML Model with VADER Features ---')
  X_vader = df_clean[pillar_cols].values

  X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(
      X_vader, y, test_size=0.2, random_state=42
  )
  clf_vader = LogisticRegression(max_iter=1000)
  clf_vader.fit(X_train_v, y_train_v)
  acc_vader = accuracy_score(y_test_v, clf_vader.predict(X_test_v))
  print(f'VADER Features Accuracy: {acc_vader * 100:.2f}%\n')

  # ==========================================
  # EXPERIMENT 2: ML using Word2Vec Features
  # ==========================================
  print('--- Running ML Model with Word2Vec Features ---')
  tokenized_texts = [
      str(text).lower().split() for text in df_clean['cleaned_text']
  ]
  w2v_model = Word2Vec(
      sentences=tokenized_texts, vector_size=100, window=5, min_count=1
  )

  def get_doc_vector(words):
    valid_words = [w for w in words if w in w2v_model.wv]
    if not valid_words:
      return np.zeros(100)
    return np.mean([w2v_model.wv[w] for w in valid_words], axis=0)

  X_w2v = np.array([get_doc_vector(text) for text in tokenized_texts])

  X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
      X_w2v, y, test_size=0.2, random_state=42
  )
  clf_w2v = LogisticRegression(max_iter=1000)
  clf_w2v.fit(X_train_w, y_train_w)
  acc_w2v = accuracy_score(y_test_w, clf_w2v.predict(X_test_w))
  print(f'Word2Vec Features Accuracy: {acc_w2v * 100:.2f}%\n')

  # ==========================================
  # FINAL SUMMARY
  # ==========================================
  print('==========================================')
  print('           FINAL COMPARISON SUMMARY       ')
  print('==========================================')
  print(f'1. VADER Features Model Accuracy:   {acc_vader * 100:.2f}%')
  print(f'2. Word2Vec Features Model Accuracy: {acc_w2v * 100:.2f}%')
  print('==========================================')


if __name__ == '__main__':
  run_side_by_side_comparison()