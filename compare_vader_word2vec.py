# compare_vader_word2vec.py
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from xgboost import XGBClassifier

# Download VADER lexicon silently
nltk.download('vader_lexicon', quiet=True)


def tokenize_text(text):
    """Simple lowercasing and word tokenization."""
    text_clean = re.sub(r'[^a-zA-Z0-9\s]', '', str(text).lower())
    return text_clean.split()


def extract_vader_features(texts):
    """Extract global 4D VADER sentiment features (neg, neu, pos, compound)."""
    sia = SentimentIntensityAnalyzer()
    vader_features = []
    for text in texts:
        scores = sia.polarity_scores(str(text))
        vader_features.append([
            scores['neg'],
            scores['neu'],
            scores['pos'],
            scores['compound']
        ])
    return np.array(vader_features)


def train_word2vec_and_vectorize(tokenized_texts, vector_size=100, window=5, min_count=2):
    """Train Word2Vec model and generate document-level mean embeddings."""
    print(f"Training Word2Vec model (vector_size={vector_size}, window={window})...")
    w2v_model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        seed=42
    )

    document_vectors = []
    for tokens in tokenized_texts:
        vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        if vectors:
            doc_vec = np.mean(vectors, axis=0)
        else:
            doc_vec = np.zeros(vector_size)
        document_vectors.append(doc_vec)

    return np.array(document_vectors), w2v_model


def run_vader_word2vec_comparison():
    data_file = 'master_cleaned_data.csv'
    print(f"Loading dataset from '{data_file}'...")
    if not os.path.exists(data_file):
        print(f"[Error] '{data_file}' not found. Please ensure data preprocessing has run.")
        return

    df = pd.read_csv(data_file)
    df = df.dropna(subset=['cleaned_text', 'sector']).reset_index(drop=True)

    print(f"Loaded {len(df)} valid records across sectors: {df['sector'].nunique()} unique classes.")

    texts = df['cleaned_text'].astype(str)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['sector'])

    # 1. Feature Extraction
    print("\n--- Extracting VADER Global Sentiment Features (4D) ---")
    X_vader = extract_vader_features(texts)

    print("\n--- Tokenizing Text for Word2Vec ---")
    tokenized_texts = [tokenize_text(t) for t in texts]

    print("\n--- Generating Word2Vec Document Embeddings (100D) ---")
    X_w2v, _ = train_word2vec_and_vectorize(tokenized_texts, vector_size=100)

    print("\n--- Combining Word2Vec + VADER Features (104D) ---")
    X_combined = np.hstack([X_w2v, X_vader])

    feature_sets = {
        'VADER Only (4D)': X_vader,
        'Word2Vec Only (100D)': X_w2v,
        'Word2Vec + VADER (104D)': X_combined
    }

    # 2. ML Models Setup
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Linear SVC': LinearSVC(random_state=42, dual='auto'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(eval_metric='mlogloss', random_state=42)
    }

    results = []

    print("\n" + "=" * 80)
    print("RUNNING FEATURE REPRESENTATION VS ML MODEL COMPARISON")
    print("=" * 80)

    for feature_name, X_data in feature_sets.items():
        print(f"\n>>> Evaluating Feature Set: {feature_name} <<<")
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y, test_size=0.2, random_state=42, stratify=y
        )

        for model_name, model in classifiers.items():
            print(f"  Training {model_name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='macro', zero_division=0
            )

            results.append({
                'Feature Representation': feature_name,
                'Algorithm': model_name,
                'Accuracy (%)': round(acc * 100, 2),
                'Precision': round(precision, 4),
                'Recall': round(recall, 4),
                'F1-Score': round(f1, 4)
            })

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("FINAL COMPARISON RESULTS TABLE")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80)

    # Save to CSV
    output_csv = 'vader_word2vec_comparison.csv'
    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved detailed tabular results to '{output_csv}'.")

    # 3. Visualization Chart
    print("Generating model comparison chart...")
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(
        data=results_df,
        x='Algorithm',
        y='Accuracy (%)',
        hue='Feature Representation',
        palette='viridis'
    )
    plt.title('ML Model Performance Comparison: VADER vs Word2Vec vs Combined Features', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('ML Algorithm', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.legend(title='Feature Set', loc='lower right')
    plt.ylim(0, 105)

    # Add numeric labels on top of bars
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f'{height:.1f}%',
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom',
                fontsize=8, fontweight='bold',
                xytext=(0, 3), textcoords='offset points'
            )

    plt.tight_layout()
    chart_output = 'vader_word2vec_comparison_chart.png'
    plt.savefig(chart_output, dpi=300)
    plt.close()
    print(f"Comparison chart saved to '{chart_output}'.")


if __name__ == '__main__':
    run_vader_word2vec_comparison()
