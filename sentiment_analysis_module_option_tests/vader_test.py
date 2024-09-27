# sentiment_analysis.py
# install with 
# pip install nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer

class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, texts):
        """
        Analyze sentiment for a list of texts.

        Parameters:
        - texts: List of text strings.

        Returns:
        - List of dictionaries containing sentiment scores.
        """
        sentiment_scores = []
        for text in texts:
            scores = self.sia.polarity_scores(text)
            sentiment_scores.append(scores)
        return sentiment_scores

# Test the module
if __name__ == "__main__":
    texts = [
        "I love this product!",
        "This is the worst thing I've ever tasted.",
        "It's okay, not great but not bad either."
    ]
    analyzer = SentimentAnalyzer()
    results = analyzer.analyze_sentiment(texts)
    for text, scores in zip(texts, results):
        print(f"Text: {text}")
        print(f"Scores: {scores}\n")
