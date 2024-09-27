# data_retrieval.py

import praw
from phrase_config import Config

class DataRetriever:
    def __init__(self, config):
        """
        Initialize the DataRetriever with Reddit API credentials and configuration settings.
        """
        self.config = config
        self.reddit = praw.Reddit(
            client_id=self.config.reddit_config['client_id'],
            client_secret=self.config.reddit_config['client_secret'],
            user_agent=self.config.reddit_config['user_agent']
        )

    def fetch_submissions(self, phrase, limit=100):
        """
        Fetch submissions containing the specified phrase.
        """
        print(f"Fetching submissions for phrase: {phrase}")
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
        print(f"Fetching comments and replies for phrase: {phrase}")
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
                # Check if the comment or submission contains the phrase
                if depth == 1 and phrase.lower() in (submission.title + submission.selftext).lower():
                    contains_phrase = True
                else:
                    contains_phrase = phrase.lower() in current_comment.body.lower()
                # Collect comment data
                comment_data = {
                    'id': current_comment.id,
                    'parent_id': parent_id,
                    'body': current_comment.body,
                    'score': current_comment.score,
                    'depth': depth,
                    'contains_phrase': contains_phrase
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
        # Collect submission data
        submission_data = []
        for submission in submissions:
            submission_info = {
                'id': submission.id,
                'title': submission.title,
                'selftext': submission.selftext,
                'score': submission.score,
                'contains_phrase': phrase.lower() in (submission.title + submission.selftext).lower(),
                'depth': 0  # Depth of submission is 0
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
    print("\nSample submission data:")
    print(submissions[0])
    print("\nSample comment data:")
    print(comments_and_replies[0])
