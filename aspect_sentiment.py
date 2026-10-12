import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# Define aspect dictionary
ASPECTS = {
    "Cost & ROI": ["cost", "investment", "price", "budget", "roi", "expensive", "capital"],
    "Security & Privacy": ["security", "privacy", "cyber", "data leak", "encryption", "gdpr"],
    "Scalability & Tech": ["scale", "integration", "cloud", "latency", "interoperability", "infrastructure"],
    "Sustainability & Green": ["energy", "carbon", "sustainability", "waste", "green", "emissions"]
}

def extract_aspect_sentiments(text):
    text_lower = str(text).lower()
    sentences = text_lower.split('.')

    aspect_scores = {aspect: [] for aspect in ASPECTS}

    for sentence in sentences:
        for aspect, keywords in ASPECTS.items():
            if any(kw in sentence for kw in keywords):
                score = sia.polarity_scores(sentence)["compound"]
                aspect_scores[aspect].append(score)

    result = {}
    for aspect, scores in aspect_scores.items():
        if scores:
            avg_score = sum(scores) / len(scores)
            result[aspect] = round(avg_score, 3)
        else:
            result[aspect] = None

    return result


def run_aspect_sentiment_analysis():
    input_file = "master_cleaned_data.csv"
    output_file = "master_nlp_aspects.csv"

    print("Running aspect-based sentiment analysis on cleaned dataset...")
    df = pd.read_csv(input_file)
    aspect_results = df["cleaned_text"].apply(extract_aspect_sentiments)
    aspect_df = pd.DataFrame(list(aspect_results))

    final_df = pd.concat([df, aspect_df], axis=1)
    final_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Aspect-based sentiment analysis complete. Output saved to '{output_file}'")


if __name__ == "__main__":
    run_aspect_sentiment_analysis()