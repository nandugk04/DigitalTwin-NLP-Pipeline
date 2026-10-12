# ml_perception_model.py
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

try:
    from gensim.models import Word2Vec
except ImportError:
    Word2Vec = None


def get_document_vector(doc_tokens, model, vector_size):
  """Generate an average vector for a document using trained Word2Vec vectors."""
  valid_words = [
      word for word in doc_tokens if word in model.wv.key_to_index
  ]
  if not valid_words:
    return np.zeros(vector_size)
  return np.mean(model.wv[valid_words], axis=0)


def run_ml_perception_pipeline():
  print("Running Word2Vec + ML Perception Classification Pipeline...")
  if Word2Vec is None:
    print("[Skipped] 'gensim' library not installed. Install with: pip install gensim")
    return

  if os.path.exists("master_nlp_aspects.csv"):
    df = pd.read_csv("master_nlp_aspects.csv")
  elif os.path.exists("master_cleaned_data.csv"):
    df = pd.read_csv("master_cleaned_data.csv")
  else:
    print(
        "[Error] No cleaned or aspect dataset found. Run preprocessing first."
    )
    return

  # Prepare tokenized sentences for Word2Vec training
  df["cleaned_text"] = df["cleaned_text"].astype(str)
  tokenized_corpus = [text.split() for text in df["cleaned_text"]]

  # Train Word2Vec model on your dataset (vector_size=100, window=5)
  print("Training custom Word2Vec embeddings on corpus...")
  w2v_model = Word2Vec(
      sentences=tokenized_corpus,
      vector_size=100,
      window=5,
      min_count=2,
      workers=4,
      epochs=10,
  )

  # Convert documents into averaged feature vectors
  print("Generating aggregated sentence vectors...")
  vector_size = 100
  X = np.array(
      [
          get_document_vector(tokens, w2v_model, vector_size)
          for tokens in tokenized_corpus
      ]
  )
  y = df["sector"]

  # Train/Test Split
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  # Train Logistic Regression Classifier on Word2Vec features
  clf = LogisticRegression(max_iter=1000)
  clf.fit(X_train, y_train)

  predictions = clf.predict(X_test)
  print("\n--- Word2Vec + ML Perception Model Classification Report ---")
  print(classification_report(y_test, predictions))

  print("Word2Vec ML Perception Model trained and evaluated successfully.")


if __name__ == "__main__":
  run_ml_perception_pipeline()
  