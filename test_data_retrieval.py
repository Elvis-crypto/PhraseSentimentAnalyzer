# test_data_retrieval.py

import unittest
from data_retrieval import DataRetriever
from phrase_config import Config

class TestDataRetriever(unittest.TestCase):
    def setUp(self):
        """
        Set up the DataRetriever instance with a test configuration.
        """
        # Create a test configuration with target phrases
        self.config = Config()
        self.config.phrases = ["natural flavoring"]
        self.retriever = DataRetriever(self.config)
    
    def test_phrase_in_text_true_exact(self):
        text = "This product contains natural flavoring."
        self.assertTrue(self.retriever.phrase_in_text("natural flavoring", text))
    
    def test_phrase_in_text_true_plural(self):
        text = "This product contains natural flavors."
        self.assertTrue(self.retriever.phrase_in_text("natural flavoring", text))
    
    def test_phrase_in_text_true_base_form(self):
        text = "Natural flavor is added to enhance taste."
        self.assertTrue(self.retriever.phrase_in_text("natural flavoring", text))
    
    def test_phrase_in_text_false(self):
        text = "This product contains artificial flavors."
        self.assertFalse(self.retriever.phrase_in_text("natural flavoring", text))
    
    def test_phrase_case_insensitivity(self):
        text = "NATURAL FLAVORING enhances taste."
        self.assertTrue(self.retriever.phrase_in_text("natural flavoring", text))
    
    def test_phrase_with_punctuation(self):
        text = "Contains natural flavoring!"
        self.assertTrue(self.retriever.phrase_in_text("natural flavoring", text))
    
    def test_phrase_partial_matching_false(self):
        text = "This flavoring is natural."
        self.assertFalse(self.retriever.phrase_in_text("natural flavoring", text))
    
    def test_phrase_substring_false(self):
        text = "Contains natural flavorings."
        self.assertTrue(self.retriever.phrase_in_text("natural flavoring", text))
    
    def test_phrase_multiple_occurrences(self):
        text = "Natural flavoring is used. Natural flavoring improves taste."
        self.assertTrue(self.retriever.phrase_in_text("natural flavoring", text))
    
    def test_phrase_not_present(self):
        text = "No mention of the key phrase here."
        self.assertFalse(self.retriever.phrase_in_text("natural flavoring", text))

if __name__ == '__main__':
    unittest.main()
