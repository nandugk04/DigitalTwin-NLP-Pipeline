# optimize_hyperparameters.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline


def run_grid_search():
  print('Loading dataset for hyperparameter optimization...')
  df = pd.read_csv('master_cleaned_data.csv')
  df = df.dropna(subset=['cleaned_text', 'sector'])

  X = df['cleaned_text'].astype(str)
  y = df['sector']

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  # Create a pipeline combining vectorizer and classifier
  pipeline = Pipeline([
      ('tfidf', TfidfVectorizer()),
      ('clf', LogisticRegression(max_iter=1000)),
  ])

  # Define parameter grid for grid search
  param_grid = {
      'tfidf__max_features': [3000, 5000, 10000],
      'tfidf__ngram_range': [(1, 1), (1, 2)],
      'clf__C': [0.1, 1.0, 10.0],
  }

  print(
      'Running 5-fold GridSearchCV across parameter combinations (this may take'
      ' a moment)...'
  )
  grid_search = GridSearchCV(
      pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1
  )
  grid_search.fit(X_train, y_train)

  print('\n--- Optimal Hyperparameters Found ---')
  print(grid_search.best_params_)
  print(f'Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}')

  # Evaluate on test set
  test_acc = grid_search.score(X_test, y_test)
  print(f'Test Set Accuracy with Best Parameters: {test_acc:.4f}')


if __name__ == '__main__':
  run_grid_search()