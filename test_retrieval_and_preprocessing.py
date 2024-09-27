# main.py

from phrase_config import Config
from data_retrieval import DataRetriever
from preprocessing import Preprocessor

def main():
    # Load configuration and initialize modules.
    config = Config()
    retriever = DataRetriever(config)
    preprocessor = Preprocessor()

    # Iterate over each phrase in the configuration.
    for phrase in config.phrases:
        print(f"\nProcessing phrase: {phrase}")
        # Fetch posts and comments (with replies) containing the phrase.
        posts, comments_and_replies = retriever.fetch_posts_comments_and_replies(phrase, limit=5)
        # Combine posts and comments into a single list.
        combined = posts + comments_and_replies
        # Preprocess the combined documents.
        preprocessed_texts = preprocessor.preprocess_documents(combined)
        print(f"Number of documents retrieved: {len(combined)}")
        print("Sample preprocessed text:")
        # Print a sample of the preprocessed texts.
        for text in preprocessed_texts[:3]:
            print(f"- {text}")

        # Retrieve comment scores (upvotes - downvotes) for future evaluation.
        comment_scores = [comment.score for comment in comments_and_replies if hasattr(comment, 'score')]
        if comment_scores:
            average_score = sum(comment_scores) / len(comment_scores)
            print(f"Average comment score: {average_score:.2f}")
        else:
            print("No comments retrieved for this phrase.")

if __name__ == "__main__":
    main()
