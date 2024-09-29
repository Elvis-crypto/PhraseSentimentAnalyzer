# analysis.py

import pickle
import pandas as pd
import logging
from visualization import (
    visualize_sentiment_over_time,
    visualize_sentiment_per_thread,
    visualize_sentiment_distribution,
    visualize_word_cloud,
    visualize_stacked_bar_graph,
    visualize_exponential_moving_average,
    visualize_alternate_score,
    average_sentiment_per_topic,
    visualize_time_based_activity,
    user_engagement_metrics,
    visualize_top_contributors,
    visualize_word_frequency
)

def count_phrases_detected(df, phrase):
    """
    Count how many times the target phrase appears in submissions and comments.
    """
    count = df['contains_phrase'].sum()
    total = len(df)
    logging.info(f"Phrase '{phrase}' detected {count} times out of {total} documents.")
    print(f"Phrase '{phrase}' detected {count} times out of {total} documents.")


def main():
    # Load processed data
    with open('processed_data.pkl', 'rb') as f:
        all_results = pickle.load(f)

    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)

    with open('lda_model.pkl', 'rb') as f:
        lda_model = pickle.load(f)
    with open('lda_dictionary.pkl', 'rb') as f:
        dictionary = pickle.load(f)
    # Perform visualizations
    phrases = df_results['phrase'].unique()
    for phrase in phrases:
        df_phrase = df_results[df_results['phrase'] == phrase]

        # Existing visualizations
        visualize_sentiment_over_time(df_phrase, phrase)
        visualize_sentiment_per_thread(df_phrase, phrase)

        # New visualizations
        visualize_stacked_bar_graph(df_phrase, phrase)
        visualize_exponential_moving_average(df_phrase, phrase)
        visualize_alternate_score(df_phrase, phrase)
        visualize_sentiment_distribution(df_phrase, phrase)
        visualize_word_cloud(df_phrase, phrase)
        visualize_time_based_activity(df_phrase, phrase)
        # average_sentiment_per_topic(df_phrase, lda_model, dictionary, phrase)
        user_engagement_metrics(df_phrase, phrase)
        visualize_top_contributors(df_phrase, phrase)
        visualize_word_frequency(df_phrase, phrase)
if __name__ == "__main__":
    main()
