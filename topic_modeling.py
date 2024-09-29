# topic_modeling.py

from gensim.models import LdaModel
from gensim import corpora
import numpy as np
import logging

class TopicModeler:
    def __init__(self, num_topics=5):
        self.num_topics = num_topics
        self.model = None
        self.id2word = None
        self.corpus = None

    def build_lda_model(self, processed_texts, no_below=5, no_above=0.5):
        """
        Build LDA model from processed texts.
        Returns the LDA model if successful, else None.
        """
        if not processed_texts:
            logging.warning("No processed texts provided for LDA modeling.")
            return None

        self.id2word, self.corpus = self.create_dictionary_corpus(processed_texts, no_below, no_above)

        if not self.corpus:
            logging.warning("The corpus is empty after creating the dictionary.")
            return None

        # Check if dictionary is empty
        if len(self.id2word) == 0:
            logging.warning("The dictionary is empty. Cannot build LDA model.")
            return None

        try:
            self.model = LdaModel(
                corpus=self.corpus,
                id2word=self.id2word,
                num_topics=self.num_topics,
                random_state=42,
                passes=10,
                iterations=100,
                alpha='auto',
                eta='auto'
            )
            logging.info("LDA model built successfully.")
            return self.model
        except Exception as e:
            logging.error(f"Failed to build LDA model: {e}")
            return None

    def create_dictionary_corpus(self, texts, no_below=5, no_above=0.5):
        """
        Create a dictionary and corpus for topic modeling.
        """
        # Create Dictionary
        id2word = corpora.Dictionary(texts)
        # Filter extremes to remove very rare and very common words
        id2word.filter_extremes(no_below=no_below, no_above=no_above)
        # Create Corpus
        corpus = [id2word.doc2bow(text) for text in texts]
        return id2word, corpus

    def compute_topic_distribution(self, text_tokens):
        """
        Compute the topic distribution for a given text.
        """
        if not self.model or not self.id2word:
            logging.error("LDA model or dictionary not initialized.")
            return np.zeros(self.num_topics)

        bow = self.id2word.doc2bow(text_tokens)
        topic_dist = self.model.get_document_topics(bow)
        # Convert to dense vector
        topic_vector = np.zeros(self.num_topics)
        for idx, val in topic_dist:
            topic_vector[idx] = val
        return topic_vector

    def compute_relevance_score(self, doc_topic_vector, phrase_topic_vector):
        """
        Compute relevance score between document topic vector and phrase topic vector.
        """
        if not np.any(doc_topic_vector) or not np.any(phrase_topic_vector):
            return 0.0
        # Use cosine similarity
        numerator = np.dot(doc_topic_vector, phrase_topic_vector)
        denominator = np.linalg.norm(doc_topic_vector) * np.linalg.norm(phrase_topic_vector)
        if denominator == 0:
            return 0.0
        else:
            return numerator / denominator

# Test the module
if __name__ == "__main__":
    # Sample texts
    texts = [
        "I love the natural flavoring in this product!",
        "Artificial flavoring substances can be harmful.",
        "What do you think about polyphenols?",
        "This yeast extract tastes amazing."
    ]
    # Preprocess texts
    from preprocessing import Preprocessor
    preprocessor = Preprocessor()
    processed_texts = preprocessor.preprocess_for_topic_modeling(texts)

    # Build LDA model with adjusted filter parameters for testing
    topic_modeler = TopicModeler(num_topics=2)
    lda_model = topic_modeler.build_lda_model(processed_texts, no_below=1, no_above=0.5)

    # Compute topic distributions
    if lda_model:
        text_tokens = processed_texts[0]
        doc_topic_vector = topic_modeler.compute_topic_distribution(text_tokens)
        print("Document topic vector:")
        print(doc_topic_vector)
    else:
        print("LDA model was not built.")
