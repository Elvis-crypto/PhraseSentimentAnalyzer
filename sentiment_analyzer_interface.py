# sentiment_analyzer_interface.py

from abc import ABC, abstractmethod

class SentimentAnalyzerInterface(ABC):
    @abstractmethod
    def analyze_sentiment(self, texts):
        """
        Analyze sentiment for a list of texts.

        Parameters:
        - texts: List of text strings.

        Returns:
        - List of sentiment analysis results.
        """
        pass
