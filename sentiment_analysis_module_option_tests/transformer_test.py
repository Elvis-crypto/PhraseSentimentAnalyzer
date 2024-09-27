# sentiment_analysis_transformers.py
# install with 
# pip install transformers

from transformers import pipeline

class TransformerSentimentAnalyzer:
    def __init__(self):
        self.sentiment_pipeline = pipeline('sentiment-analysis')

    def analyze_sentiment(self, texts):
        """
        Analyze sentiment for a list of texts.

        Parameters:
        - texts: List of text strings.

        Returns:
        - List of dictionaries containing sentiment labels and scores.
        """
        results = self.sentiment_pipeline(texts)
        return results

# Test the module
if __name__ == "__main__":
    texts = [
        "I love this product!",
        "This is the worst thing I've ever tasted.",
        "It's okay, not great but not bad either."
    ]
    analyzer = TransformerSentimentAnalyzer()
    results = analyzer.analyze_sentiment(texts)
    for text, result in zip(texts, results):
        print(f"Text: {text}")
        print(f"Result: {result}\n")
