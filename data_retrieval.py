# data_retrieval.py

import praw
from phrase_config import Config
from preprocessing import Preprocessor  # Import Preprocessor
import logging
import spacy
from spacy.matcher import PhraseMatcher

class DataRetriever:
    def __init__(self, config):
        """
        Initialize the DataRetriever with Reddit API credentials and configuration settings.
        """
        self.config = config

        # Configure logging once at the top
        logging.basicConfig(
            filename='data_retrieval.log',
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.debug("Logging configured.")

        self.reddit = praw.Reddit(
            client_id=self.config.reddit_config['client_id'],
            client_secret=self.config.reddit_config['client_secret'],
            user_agent=self.config.reddit_config['user_agent']
        )
        self.preprocessor = Preprocessor()  # Initialize Preprocessor

        # Load spaCy model
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

        # Initialize PhraseMatcher with 'LOWER' attribute for case-insensitive matching
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")

        # Prepare patterns for PhraseMatcher
        self.prepare_phrase_patterns()

    def prepare_phrase_patterns(self):
        """
        Prepare and add phrase patterns to the PhraseMatcher.
        Handles plural forms and lemmatization.
        """
        for phrase in self.config.phrases:
            # Lemmatize the phrase to handle different grammatical forms
            lemmatized_phrase = self.preprocessor.lemmatize_text(phrase)
            doc = self.nlp(lemmatized_phrase)

            # Add the original lemmatized phrase
            patterns = [doc]

            # Handle plural forms by adding 's' to the last token if it's a noun
            if doc[-1].pos_ == "NOUN":
                plural_with_ing = doc[-1].text + 's'
                plural_with_ing_phrase = ' '.join([token.text for token in doc[:-1]] + [plural_with_ing])
                plural_with_ing_doc = self.nlp(plural_with_ing_phrase)
                patterns.append(plural_with_ing_doc)

            # Handle base forms (e.g., "natural flavor" for "natural flavoring")
            base_phrase = self.create_base_form(lemmatized_phrase)
            if base_phrase:
                base_doc = self.nlp(base_phrase)
                patterns.append(base_doc)

                # Handle plural of the base form (e.g., "natural flavors")
                if base_doc[-1].pos_ == "NOUN":
                    plural_of_base = base_doc[-1].text + 's'
                    plural_of_base_phrase = ' '.join([token.text for token in base_doc[:-1]] + [plural_of_base])
                    plural_of_base_doc = self.nlp(plural_of_base_phrase)
                    patterns.append(plural_of_base_doc)

            # Add patterns to the matcher
            self.matcher.add("PHRASE", patterns)

            # Logging the added patterns for verification
            logging.debug(f"Added patterns for phrase '{phrase}': {[str(p) for p in patterns]}")

    def create_base_form(self, lemmatized_phrase):
        """
        Create the base form of a lemmatized phrase by removing suffixes like 'ing' if applicable.
        """
        tokens = self.nlp(lemmatized_phrase)
        if not tokens:
            logging.debug("No tokens found for lemmatized_phrase.")
            return None

        last_token = tokens[-1]
        if last_token.text.endswith('ing') and last_token.pos_ == "NOUN":
            # Manually remove 'ing' to form the base
            base_last_token = last_token.text[:-3]  # Remove 'ing'
            base_phrase = ' '.join([token.text for token in tokens[:-1]] + [base_last_token])
            logging.debug(f"Created base form: '{base_phrase}' from '{lemmatized_phrase}'")
            return base_phrase
        logging.debug(f"No base form created for phrase: '{lemmatized_phrase}'")
        return None

    def phrase_in_text(self, phrase, text):
        """
        Check if the phrase exists in the text using spaCy's PhraseMatcher.
        """
        doc = self.nlp(text)
        matches = self.matcher(doc)

        # Log the text being analyzed
        logging.debug(f"Analyzing text: '{text}'")

        if matches:
            matched_phrases = [doc[start:end].text for match_id, start, end in matches]
            logging.debug(f"Matched phrases in text: {matched_phrases}")
        else:
            logging.debug("No matched phrases found in text.")

        return bool(matches)

    def fetch_submissions(self, phrase, limit=100):
        """
        Fetch submissions containing the specified phrase.
        """
        logging.info(f"Fetching submissions for phrase: {phrase}")
        subreddit_list = '+'.join(self.config.reddit_config.get('subreddits', ['all']))
        query = f'"{phrase}"'
        submissions = self.reddit.subreddit(subreddit_list).search(
            query=query,
            sort='new',
            limit=limit
        )
        return list(submissions)

    def fetch_comments_and_replies(self, submissions, phrase):
        """
        Fetch comments containing the phrase and their replies from the given submissions.
        Record the depth of each comment/reply.
        """
        logging.info(f"Fetching comments and replies for phrase: {phrase}")
        matching_comments = []
        processed_comment_ids = set()

        for submission in submissions:
            # Fetch all comments in the submission
            submission.comments.replace_more(limit=None)
            # Initialize the queue with the top-level comments
            comment_queue = [(comment, 1, submission.id) for comment in submission.comments]
            while comment_queue:
                current_comment, depth, parent_id = comment_queue.pop(0)
                if current_comment.id in processed_comment_ids:
                    continue
                processed_comment_ids.add(current_comment.id)
                # Determine if the comment contains the phrase using PhraseMatcher
                contains_phrase = self.phrase_in_text(phrase, current_comment.body)
                # Log the phrase detection result
                logging.debug(f"Comment ID: {current_comment.id}")
                logging.debug(f"Depth: {depth}")
                logging.debug(f"Contains Phrase: {contains_phrase}")
                # Collect comment data with timestamp and author
                comment_data = {
                    'id': current_comment.id,
                    'parent_id': parent_id,
                    'body': current_comment.body,
                    'score': current_comment.score,
                    'depth': depth,
                    'contains_phrase': contains_phrase,
                    'timestamp': current_comment.created_utc,  # Added timestamp
                    'author': current_comment.author.name if current_comment.author else '[deleted]'
                }
                matching_comments.append(comment_data)
                # Add replies to the queue
                for reply in current_comment.replies:
                    comment_queue.append((reply, depth + 1, current_comment.id))
        return matching_comments

    def fetch_posts_comments_and_replies(self, phrase, limit=100):
        """
        Fetch submissions, comments containing the phrase, and their replies.
        """
        submissions = self.fetch_submissions(phrase, limit)
        comments_and_replies = self.fetch_comments_and_replies(submissions, phrase)
        # Collect submission data with timestamp and author
        submission_data = []
        for submission in submissions:
            contains_phrase = self.phrase_in_text(phrase, submission.title + ' ' + (submission.selftext or ''))
            submission_info = {
                'id': submission.id,
                'title': submission.title,
                'selftext': submission.selftext,
                'score': submission.score,
                'depth': 0,  # Assuming depth 0 for submissions
                'contains_phrase': contains_phrase,
                'timestamp': submission.created_utc,  # Added timestamp
                'author': submission.author.name if submission.author else '[deleted]'
            }
            submission_data.append(submission_info)
        return submission_data, comments_and_replies

# Test the module
if __name__ == "__main__":
    config = Config()
    retriever = DataRetriever(config)
    test_phrase = config.phrases[0]
    submissions, comments_and_replies = retriever.fetch_posts_comments_and_replies(test_phrase, limit=5)
    print(f"Retrieved {len(submissions)} submissions and {len(comments_and_replies)} comments/replies for phrase '{test_phrase}'.")
    # Print sample data
    if submissions:
        print("\nSample submission data:")
        print(submissions[0])
    if comments_and_replies:
        print("\nSample comment data:")
        print(comments_and_replies[0])
