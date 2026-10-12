from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def extract_topics_with_both_noms(df):
    print("Running balanced unigram and bigram topic extraction via TF-IDF...")
    
    # 1. Configure vectorizer to capture BOTH unigrams and bigrams (ngram_range=(1,2))
    # min_df and max_df help filter out extremely rare or overly common filler words
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.85, stop_words='english')
    
    # Group text by sector to extract sector-specific trends
    sectors = df['sector'].unique()
    
    for sector in sectors:
        print(f"\n==========================================")
        print(f"Top Combined Unigram & Bigram Trends for: {sector.upper()}")
        print(f"==========================================")
        
        sector_texts = df[df['sector'] == sector]['cleaned_text']
        
        # Fit TF-IDF on this sector's corpus
        tfidf_matrix = vectorizer.fit_transform(sector_texts.astype(str))
        feature_names = vectorizer.get_feature_names_out()
        
        # Sum TF-IDF scores across all documents for each term
        mean_tfidf = tfidf_matrix.mean(axis=0).A1
        term_scores = list(zip(feature_names, mean_tfidf))
        
        # Sort by highest TF-IDF score (takes the absolute highest of both unigrams & bigrams)
        sorted_terms = sorted(term_scores, key=lambda x: x[1], reverse=True)
        
        # Print top 10 terms (mixing unigrams and bigrams fairly)
        for term, score in sorted_terms[:10]:
            # Simple check to label them nicely in output
            ngram_type = "2-gram" if " " in term else "1-gram"
            print(f"  * {term} ({ngram_type}) — Score: {score:.4f}")

if __name__ == "__main__":
    # Assuming df is loaded from your master cleaned data CSV
    df = pd.read_csv('master_cleaned_data.csv')
    df = df.dropna(subset=['cleaned_text', 'sector'])
    extract_topics_with_both_noms(df)