# cross_domain_test.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def run_generalization_test():
  print('Loading dataset for cross-domain generalization test...')
  df = pd.read_csv('master_cleaned_data.csv')
  df = df.dropna(subset=['cleaned_text', 'sector', 'source_type'])

  # Split into Academic training corpus vs. Real-world news/social test corpus
  academic_sources = ['OpenAlex Repository', 'arXiv Repository']
  train_df = df[df['source_type'].isin(academic_sources)]
  test_df = df[~df['source_type'].isin(academic_sources)]

  if train_df.empty or test_df.empty:
    print(
        '[Warning] Source type column data missing or split is empty. Using'
        ' random split simulation instead.'
    )
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

  X_train, y_train = (
      train_df['cleaned_text'].astype(str),
      train_df['sector'],
  )
  X_test, y_test = test_df['cleaned_text'].astype(str), test_df['sector']

  print(
      f'Training corpus size (Academic/Structured): {len(X_train)} records'
  )
  print(f'Testing corpus size (Noisy/Real-world): {len(X_test)} records')

  vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
  X_train_vec = vectorizer.fit_transform(X_train)
  X_test_vec = vectorizer.transform(X_test)

  model = LogisticRegression(max_iter=1000)
  model.fit(X_train_vec, y_train)

  y_pred = model.predict(X_test_vec)
  acc = accuracy_score(y_test, y_pred)

  print('\n--- Cross-Domain Generalization Results ---')
  print(f'Generalization Accuracy: {acc:.4f}')
  print(classification_report(y_test, y_pred))


if __name__ == '__main__':
  run_generalization_test()