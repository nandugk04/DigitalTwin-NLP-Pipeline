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

def run_aspect_comparison():
    print("Loading dataset ('master_cleaned_data.csv')...")
    df = pd.read_csv('master_cleaned_data.csv')
    df = df.dropna(subset=['cleaned_text', 'sector'])
    print(f"Loaded {len(df)} valid records.")

    # FIX: Encode target labels to numeric integers (0, 1, 2) right here at the start
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(df['sector'].to_numpy())
    
    tokenized_corpus = [str(text).split() for text in df['cleaned_text']]
    # ... rest of your aspect feature extraction code ...

    pillar_keywords = {
        'Cost_ROI': ['cost', 'roi', 'price', 'budget', 'saving', 'investment', 'financial', 'expense'],
        'Security_Privacy': ['security', 'privacy', 'secure', 'data', 'breach', 'compliance', 'threat', 'protection'],
        'Scalability_Tech': ['scalable', 'scalability', 'technology', 'tech', 'performance', 'system', 'architecture', 'infrastructure'],
        'Sustainability_Green': ['sustainability', 'sustainable', 'green', 'environmental', 'carbon', 'energy', 'eco', 'emission']
    }

    sia = SentimentIntensityAnalyzer()

    print("\n--- Extracting 4-Pillar VADER Aspect Scores (4D) ---")
    vader_aspect_features = []
    for text in df['cleaned_text']:
        words = str(text).split()
        pillar_scores = []
        for pillar, keywords in pillar_keywords.items():
            relevant_text = " ".join([w for w in words if w in keywords]) or str(text)
            pillar_scores.append(sia.polarity_scores(relevant_text)['compound'])
        vader_aspect_features.append(pillar_scores)

    X_vader_aspects = np.array(vader_aspect_features, dtype=np.float32)

    print("\n--- Training Word2Vec and Extracting 4-Pillar Aspect Embeddings (400D) ---")
    w2v_model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=2, workers=4)

    word2vec_aspect_features = []
    for tokens in tokenized_corpus:
        doc_aspect_vecs = []
        for pillar, keywords in pillar_keywords.items():
            pillar_tokens = [w for w in tokens if w in keywords and w in w2v_model.wv]
            if len(pillar_tokens) > 0:
                pillar_vec = np.mean([w2v_model.wv[w] for w in pillar_tokens], axis=0)
            else:
                all_valid = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv]
                pillar_vec = np.mean(all_valid, axis=0) if len(all_valid) > 0 else np.zeros(100)
            doc_aspect_vecs.append(pillar_vec)
        word2vec_aspect_features.append(np.concatenate(doc_aspect_vecs))

    X_w2v_aspects = np.array(word2vec_aspect_features, dtype=np.float32)

    # Define the 4 algorithms
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Linear SVC": LinearSVC(random_state=42, dual='auto'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=42)
    }

    feature_sets = {
        "Aspect VADER (4D)": X_vader_aspects,
        "Aspect Word2Vec (400D)": X_w2v_aspects
    }

    results = []

    print("\n========================================================")
    print(" EVALUATING ASPECT MODELS ACROSS 4 ALGORITHMS")
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
    print("\n--- Aspect Features: Multi-Algorithm Comparison ---")
    print(pivot_table.to_string())
    print("========================================================")

if __name__ == "__main__":
    run_aspect_comparison()