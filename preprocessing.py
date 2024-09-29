# preprocessing.py

import spacy
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess
import gensim.corpora as corpora

class Preprocessor:
    def __init__(self):
        """
        Initialize the Preprocessor by loading the spaCy English language model.
        """
        # Load the small English model
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    def preprocess_for_topic_modeling(self, texts):
        """
        Preprocess texts for topic modeling.
        """
        # Tokenize and clean up texts
        processed_texts = []
        for text in texts:
            # Remove stopwords
            text_nostops = remove_stopwords(text.lower())
            # Tokenize
            tokens = simple_preprocess(text_nostops, deacc=True)
            processed_texts.append(tokens)
        return processed_texts

    def preprocess_for_sentiment_analysis(self, texts):
        """
        Preprocess texts for sentiment analysis.
        """
        # Minimal preprocessing for sentiment analysis
        # VADER performs better with less aggressive preprocessing
        return texts

    def create_dictionary_corpus(self, texts):
        """
        Create a dictionary and corpus for topic modeling.
        """
        # Create Dictionary
        id2word = corpora.Dictionary(texts)
        # Create Corpus
        corpus = [id2word.doc2bow(text) for text in texts]
        return id2word, corpus

    def lemmatize_text(self, text):
        """
        Lemmatize a single text string.
        """
        doc = self.nlp(text)
        return ' '.join([token.lemma_.lower() for token in doc if not token.is_punct])

# Test the module
if __name__ == "__main__":
    # Sample texts
    texts = [
        "I love the natural flavoring in this product!",
        "Artificial flavoring substances can be harmful.",
        "What do you think about polyphenols?",
        "This yeast extract tastes amazing."
    ]
    preprocessor = Preprocessor()
    processed_texts = preprocessor.preprocess_for_topic_modeling(texts)
    print("Processed texts for topic modeling:")
    print(processed_texts)
    id2word, corpus = preprocessor.create_dictionary_corpus(processed_texts)
    print("\nDictionary tokens:")
    print(id2word.token2id)
    # Test lemmatization
    test_text = "Natural flavors are added to enhance the taste."
    lemmatized = preprocessor.lemmatize_text(test_text)
    print("\nLemmatized text:")
    print(lemmatized)
