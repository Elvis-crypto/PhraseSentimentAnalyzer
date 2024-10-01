# sentiment_analysis.py
# Install with:
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
        # **Strong Positive Sentiments**
        "I love this product!",
        "Absolutely fantastic service, exceeded my expectations!",
        "What a wonderful experience, I will come back again.",
        "I'm extremely satisfied with the quality.",
        "Brilliant execution, well done!",
        "I love the design, it's so sleek.",
        "I'm thrilled with the results.",
        "It's perfect for my needs.",
        "The team did an outstanding job on the project. Their dedication is unparalleled!",
        "This is by far the best investment I've made this year. Highly recommended to everyone.",

        # **Strong Negative Sentiments**
        "This is the worst thing I've ever tasted.",
        "Terrible, I want a refund.",
        "This is awful, totally disappointed.",
        "Horrible mistake on your part.",
        "I hate how this works.",
        "Disgusting behavior, very unprofessional.",
        "I'm not happy with the customer support provided.",
        "The service was dreadful and the staff were unhelpful.",
        "This product failed to meet any of my expectations. Completely useless!",
        "I'm extremely disappointed with the quality and performance of this item.",

        # **Neutral or Mixed Sentiments**
        "It's okay, not great but not bad either.",
        "The new update is okay, but it could be better.",
        "It's neither good nor bad, just average.",
        "Pretty decent performance, could use some improvements.",
        "Not what I expected, but it's fine.",
        "It's fine, nothing special.",
        "Could be better, but not too bad.",
        "I'm indifferent about this product.",
        "The event was decent overall, though there were some minor issues.",
        "I have mixed feelings about the recent changes. Some aspects are good, while others need work.",

        # **Sentiments with Intensifiers and Negations**
        "I'm not happy with the customer support provided.",
        "I'm not happy with the customer support provided.",
        "Not what I expected, but it's fine.",
        "I do not like this at all.",
        "The movie was absolutely amazing, but the ending was disappointing.",
        "I can't say I'm satisfied with the results, despite the effort put in.",
        "She's extremely talented, yet sometimes lacks punctuality.",
        "It's very good, but I wish there were more features.",
        "I didn't enjoy the book as much as I thought I would.",
        "The meal was delicious; however, the service was slow."
    ]
    analyzer = SentimentAnalyzer()
    results = analyzer.analyze_sentiment(texts)
    for text, scores in zip(texts, results):
        print(f"Text: {text}")
        print(f"Scores: {scores}\n")
