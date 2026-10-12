import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.models import Word2Vec

# Ensure VADER lexicon is available
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

def run_global_comparison():
    print("Loading dataset ('master_cleaned_data.csv')...")
    df = pd.read_csv('master_cleaned_data.csv')
    df = df.dropna(subset=['cleaned_text', 'sector'])
    print(f"Loaded {len(df)} valid records.")

    # 1. Extract Global VADER Sentiment Score (Single Compound Score)
    print("\n--- Extracting Global VADER Sentiment (1D) ---")
    sia = SentimentIntensityAnalyzer()
    global_vader_scores = [[sia.polarity_scores(str(text))['compound']] for text in df['cleaned_text']]
    X_vader = np.array(global_vader_scores, dtype=np.float32)
    
    # Encode target labels to numeric (0, 1, 2) for XGBoost compatibility
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(df['sector'].to_numpy())

    # 2. Extract Word2Vec Document Embeddings (100D)
    print("\n--- Training Word2Vec for Global Document Embeddings (100D) ---")
    tokenized_corpus = [str(text).split() for text in df['cleaned_text']]
    w2v_model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=2, workers=4)
    
    def get_document_embedding(tokens, model, vector_size=100):
        valid_vectors = [model.wv[word] for word in tokens if word in model.wv]
        return np.mean(valid_vectors, axis=0) if len(valid_vectors) > 0 else np.zeros(vector_size)

    w2v_embeddings = [get_document_embedding(tokens, w2v_model, 100) for tokens in tokenized_corpus]
    X_w2v = np.array(w2v_embeddings, dtype=np.float32)

    # Define the 4 algorithms
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Linear SVC": LinearSVC(random_state=42, dual='auto'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=42)
    }

    feature_sets = {
        "Global VADER (1D)": X_vader,
        "Global Word2Vec (100D)": X_w2v
    }

    results = []

    print("\n========================================================")
    print(" EVALUATING GLOBAL MODELS ACROSS 4 ALGORITHMS")
    print("========================================================")

    for feat_name, X in feature_sets.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            acc = accuracy_score(y_test, pred)
            results.append({
                "Feature Set": feat_name,
                "Algorithm": model_name,
                "Accuracy (%)": round(acc * 100, 2)
            })

    results_df = pd.DataFrame(results)
    pivot_table = results_df.pivot(index="Algorithm", columns="Feature Set", values="Accuracy (%)")
    print("\n--- Global Features: Multi-Algorithm Comparison ---")
    print(pivot_table.to_string())
    print("========================================================")

if __name__ == "__main__":
    run_global_comparison()