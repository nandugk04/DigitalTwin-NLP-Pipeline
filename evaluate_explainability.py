# evaluate_explainability.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


def run_explainability_analysis():
  print('Loading dataset for advanced evaluation and explainability...')
  df = pd.read_csv('master_cleaned_data.csv')
  df = df.dropna(subset=['cleaned_text', 'sector'])

  X = df['cleaned_text'].astype(str)
  y = df['sector']

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  # Vectorize text using TF-IDF
  vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
  X_train_vec = vectorizer.fit_transform(X_train)
  X_test_vec = vectorizer.transform(X_test)

  # Train the winning model (Logistic Regression)
  print('Training optimal Logistic Regression classifier...')
  model = LogisticRegression(max_iter=1000)
  model.fit(X_train_vec, y_train)

  # --- 1. Confusion Matrix Generation ---
  print('\nGenerating confusion matrix evaluation...')
  y_pred = model.predict(X_test_vec)
  cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
  
  # Save raw confusion matrix values to CSV for thesis tables
  cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
  cm_df.to_csv('confusion_matrix_results.csv')
  print("Confusion matrix saved to 'confusion_matrix_results.csv'.")

  # Plot and save confusion matrix figure
  fig, ax = plt.subplots(figsize=(8, 6))
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
  disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
  plt.title('Confusion Matrix - Sector Classification (Logistic Regression)')
  plt.savefig('confusion_matrix_plot.png', dpi=300, bbox_inches='tight')
  print("Confusion matrix plot saved as 'confusion_matrix_plot.png'.")
  plt.close()

  # --- 2. Explainable AI: Feature Importance Coefficients ---
  print('\nExtracting top predictive features per sector (Explainable AI)...')
  feature_names = np.array(vectorizer.get_feature_names_out())
  classes = model.classes_

  top_features = {}
  for i, class_name in enumerate(classes):
    # Get top 10 coefficients for each sector class
    top_indices = model.coef_[i].argsort()[-10:][::-1]
    top_features[class_name] = feature_names[top_indices]

  features_df = pd.DataFrame(top_features)
  print('\nTop Predictive Keywords per Sector:')
  print(features_df.to_string())
  features_df.to_csv('top_sector_features_xai.csv', index=False)
  print("\nExplainable AI feature weights saved to 'top_sector_features_xai.csv'.")


if __name__ == '__main__':
  run_explainability_analysis()