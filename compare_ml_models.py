# compare_ml_models.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


def run_model_comparison():
  print('Loading dataset for ML algorithm comparison...')
  try:
    df = pd.read_csv('master_cleaned_data.csv')
  except FileNotFoundError:
    print("[Error] 'master_cleaned_data.csv' not found. Run preprocessing first.")
    return

  # Drop rows missing text or sector labels
  df = df.dropna(subset=['cleaned_text', 'sector'])

  X = df['cleaned_text'].astype(str)
  y = df['sector']

  # Train-test split (80% train, 20% test matching standard evaluation guidelines)
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  # Vectorize text using TF-IDF features
  print('Extracting TF-IDF features from text corpus...')
  vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
  X_train_vec = vectorizer.fit_transform(X_train)
  X_test_vec = vectorizer.transform(X_test)

  # Define the models to compare
  models = {
      'Logistic Regression': LogisticRegression(max_iter=1000),
      'Linear Support Vector Machine': LinearSVC(),
      'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
      'Multinomial Naive Bayes': MultinomialNB(),
  }

  comparison_results = []

  print('\n' + '=' * 60)
  print('MACHINE LEARNING ALGORITHM COMPARISON FOR PERCEPTION ANALYSIS')
  print('=' * 60)

  for name, model in models.items():
    print(f'\nTraining and evaluating: {name}...')
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    comparison_results.append(
        {'Algorithm': name, 'Accuracy (%)': round(acc * 100, 2)}
    )

    print(f'--- {name} Performance Report ---')
    print(f'Accuracy: {acc:.4f}')
    print(classification_report(y_test, y_pred))

  # Compile summary table
  results_df = pd.DataFrame(comparison_results)
  print('\n' + '=' * 60)
  print('FINAL SUMMARY ACCURACY TABLE')
  print('=' * 60)
  print(results_df.to_string(index=False))
  print('=' * 60)

  # Save results for documentation
  results_df.to_csv('model_comparison_results.csv', index=False)
  print("Comparison summary successfully saved to 'model_comparison_results.csv'.")


if __name__ == '__main__':
  run_model_comparison()