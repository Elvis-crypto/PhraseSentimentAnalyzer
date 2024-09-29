# visualization.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from wordcloud import WordCloud
import numpy as np
from preprocessing import Preprocessor  # Import Preprocessor

# Configure logging
logging.basicConfig(
    filename='app.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logging.debug("Logging configured.")

def visualize_sentiment_over_time(df, phrase):
    """
    Visualize how sentiment connected to a phrase changes over time using Exponential Moving Average.
    """
    # Convert Unix timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df_sorted = df.sort_values('datetime')

    # Calculate Exponential Moving Average with a span corresponding to 3 months (~90 days)
    df_sorted.set_index('datetime', inplace=True)
    ema_span = 90  # days
    sentiment_ema = df_sorted['sentiment_score'].ewm(span=ema_span, adjust=False).mean()

    plt.figure(figsize=(12, 6))
    sns.lineplot(x=sentiment_ema.index, y=sentiment_ema.values, label='Exponential Moving Average')
    plt.title(f"Exponential Moving Average of Sentiment Over Time for '{phrase}'")
    plt.xlabel("Date")
    plt.ylabel("Sentiment Score (EMA)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'sentiment_ema_over_time_{phrase.replace(" ", "_")}.png')
    plt.close()
    logging.info(f"Saved EMA sentiment over time visualization as 'sentiment_ema_over_time_{phrase.replace(' ', '_')}.png'.")

def visualize_sentiment_per_thread(df, phrase):
    """
    Visualize the individual contributions of posts on a given day using stacked bar graphs.
    """
    # Convert Unix timestamp to datetime
    df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date

    # Aggregate sentiment scores by date and thread
    sentiment_thread = df.groupby(['date', 'id'])['sentiment_score'].sum().reset_index()

    # Pivot the data to have threads as columns
    sentiment_pivot = sentiment_thread.pivot(index='date', columns='id', values='sentiment_score').fillna(0)

    # Plot stacked bar chart
    plt.figure(figsize=(14, 8))
    sentiment_pivot.plot(kind='bar', stacked=True, figsize=(14, 8))
    plt.title(f"Individual Contributions of Posts Over Time for '{phrase}'")
    plt.xlabel("Date")
    plt.ylabel("Sentiment Score")
    plt.xticks(rotation=45)
    plt.legend(title='Thread ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'stacked_sentiment_per_thread_{phrase.replace(" ", "_")}.png')
    plt.close()
    logging.info(f"Saved stacked sentiment per thread visualization as 'stacked_sentiment_per_thread_{phrase.replace(' ', '_')}.png'.")

def visualize_sentiment_distribution(df, phrase):
    """
    Visualize the distribution of sentiment scores.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df['sentiment_score'], bins=20, kde=True)
    plt.title(f"Sentiment Score Distribution for '{phrase}'")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f'sentiment_distribution_{phrase.replace(" ", "_")}.png')
    plt.close()
    logging.info(f"Saved sentiment distribution visualization as 'sentiment_distribution_{phrase.replace(' ', '_')}.png'.")

def visualize_word_cloud(df, phrase):
    """
    Generate a word cloud of the most frequent words in the texts.
    """
    text = ' '.join(df['text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(15, 7.5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for '{phrase}'")
    plt.tight_layout()
    plt.savefig(f'word_cloud_{phrase.replace(" ", "_")}.png')
    plt.close()
    logging.info(f"Saved word cloud visualization as 'word_cloud_{phrase.replace(' ', '_')}.png'.")

def visualize_alternate_score(df, phrase):
    """
    Show the exponential moving average of the alternate score over time.
    """
    df['date'] = pd.to_datetime(df['timestamp'], unit='s')
    df_sorted = df.sort_values('date')

    # Set the decay constant (span) for 3 months (~90 days)
    df_sorted.set_index('date', inplace=True)
    ema_span = 90  # Adjust based on data frequency

    # Calculate EMA of alternate scores
    df_sorted['ema_alternate'] = df_sorted['alternate_score'].ewm(span=ema_span).mean()

    # Plot the EMA
    plt.figure(figsize=(12, 7))
    plt.plot(df_sorted.index, df_sorted['ema_alternate'], label='EMA of Alternate Score')
    plt.title(f"Exponential Moving Average of Alternate Score for '{phrase}'")
    plt.xlabel("Date")
    plt.ylabel("EMA of Alternate Score")
    plt.legend()

    # Display the formula on the graph
    formula = "Alternate Score = Combined Sentiment Weight + Upvote Score"
    plt.figtext(0.5, 0.01, formula, wrap=True, horizontalalignment='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'ema_alternate_score_{phrase.replace(" ", "_")}.png')
    plt.close()
    logging.info(f"Saved EMA of alternate score visualization as 'ema_alternate_score_{phrase.replace(' ', '_')}.png'.")

def visualize_number_of_phrases_detected(df, phrase):
    """
    Visualize the number of phrases detected in submissions and comments.
    """
    phrase_counts = df['contains_phrase'].value_counts().reset_index()
    phrase_counts.columns = ['contains_phrase', 'count']

    plt.figure(figsize=(6, 6))
    sns.barplot(x='contains_phrase', y='count', data=phrase_counts, palette='viridis')
    plt.title(f"Number of Phrases Detected for '{phrase}'")
    plt.xlabel("Contains Phrase")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f'number_of_phrases_detected_{phrase.replace(" ", "_")}.png')
    plt.close()
    logging.info(f"Saved number of phrases detected visualization as 'number_of_phrases_detected_{phrase.replace(' ', '_')}.png'.")

def visualize_average_sentiment_per_topic(df, phrase, topic_modeler):
    """
    Analyze and visualize average sentiment scores within each identified topic.
    """
    # Assuming TopicModeler can assign topics to documents
    topics = topic_modeler.assign_topics(df['text'].tolist())
    df['topic'] = topics

    average_sentiment = df.groupby('topic')['sentiment_score'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(x='topic', y='sentiment_score', data=average_sentiment, palette='coolwarm')
    plt.title(f"Average Sentiment per Topic for '{phrase}'")
    plt.xlabel("Topic")
    plt.ylabel("Average Sentiment Score")
    plt.tight_layout()
    plt.savefig(f'average_sentiment_per_topic_{phrase.replace(" ", "_")}.png')
    plt.close()
    logging.info(f"Saved average sentiment per topic visualization as 'average_sentiment_per_topic_{phrase.replace(' ', '_')}.png'.")

def visualize_time_based_activity(df, phrase):
    """
    Track and visualize the number of posts and comments over different  periods to identify peaks in discussions.
    """
    # Convert Unix timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['date'] = df['datetime'].dt.date

    activity = df.groupby(['date', 'type']).size().reset_index(name='count')

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=activity, x='date', y='count', hue='type', marker='o')
    plt.title(f"Time-Based Activity for '{phrase}'")
    plt.xlabel("Date")
    plt.ylabel("Number of Posts/Comments")
    plt.legend(title='Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'time_based_activity_{phrase.replace(" ", "_")}.png')
    plt.close()
    logging.info(f"Saved time-based activity visualization as 'time_based_activity_{phrase.replace(' ', '_')}.png'.")

def visualize_user_engagement(df, phrase):
    """
    Monitor and visualize user engagement metrics like number of replies per comment and depth of comment threads.
    """
    # Number of replies per comment
    replies_per_comment = df[df['type'] == 'comment'].groupby('id')['parent_id'].count().reset_index(name='num_replies')

    plt.figure(figsize=(10, 6))
    sns.histplot(replies_per_comment['num_replies'], bins=20, kde=True, color='salmon')
    plt.title(f"Distribution of Replies per Comment for '{phrase}'")
    plt.xlabel("Number of Replies")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f'replies_per_comment_distribution_{phrase.replace(" ", "_")}.png')
    plt.close()
    logging.info(f"Saved replies per comment distribution visualization as 'replies_per_comment_distribution_{phrase.replace(' ', '_')}.png'.")

    # Depth of comment threads
    plt.figure(figsize=(10, 6))
    sns.histplot(df['depth'], bins=20, kde=True, color='lightgreen')
    plt.title(f"Distribution of Comment Thread Depth for '{phrase}'")
    plt.xlabel("Depth")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f'comment_thread_depth_distribution_{phrase.replace(" ", "_")}.png')
    plt.close()
    logging.info(f"Saved comment thread depth distribution visualization as 'comment_thread_depth_distribution_{phrase.replace(' ', '_')}.png'.")

def visualize_top_contributors(df, phrase):
    """
    Identify and visualize the top contributors who frequently discuss the target phrase.
    """
    top_contributors = df['author'].value_counts().head(10).reset_index()
    top_contributors.columns = ['author', 'count']

    plt.figure(figsize=(12, 6))
    sns.barplot(x='author', y='count', data=top_contributors, palette='magma')
    plt.title(f"Top Contributors for '{phrase}'")
    plt.xlabel("Author")
    plt.ylabel("Number of Contributions")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'top_contributors_{phrase.replace(" ", "_")}.png')
    plt.close()
    logging.info(f"Saved top contributors visualization as 'top_contributors_{phrase.replace(' ', '_')}.png'.")

def visualize_word_frequency(df, phrase):
    """
    Analyze and visualize the most common words or n-grams associated with each phrase using a wordcloud.
    """
    text = ' '.join(df['text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=200).generate(text)

    plt.figure(figsize=(15, 7.5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Frequency Analysis for '{phrase}'")
    plt.tight_layout()
    plt.savefig(f'word_frequency_analysis_{phrase.replace(" ", "_")}.png')
    plt.close()
    logging.info(f"Saved word frequency analysis visualization as 'word_frequency_analysis_{phrase.replace(' ', '_')}.png'.")

def visualize_average_sentiment_per_topic(df, phrase, topic_modeler):
    """
    Analyze sentiment scores within each identified topic to understand topic-specific sentiments.
    """
    # Assign topics to each document
    topics = topic_modeler.assign_topics(df['text'].tolist())
    df['topic'] = topics

    average_sentiment = df.groupby('topic')['sentiment_score'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(x='topic', y='sentiment_score', data=average_sentiment, palette='coolwarm')
    plt.title(f"Average Sentiment per Topic for '{phrase}'")
    plt.xlabel("Topic")
    plt.ylabel("Average Sentiment Score")
    plt.tight_layout()
    plt.savefig(f'average_sentiment_per_topic_{phrase.replace(" ", "_")}.png')
    plt.close()
    logging.info(f"Saved average sentiment per topic visualization as 'average_sentiment_per_topic_{phrase.replace(' ', '_')}.png'.")

def visualize_all_metrics(df, phrase, topic_modeler):
    """
    Generate all required visualizations for the given phrase.
    """
    visualize_sentiment_over_time(df, phrase)
    visualize_sentiment_per_thread(df, phrase)
    visualize_sentiment_distribution(df, phrase)
    visualize_word_cloud(df, phrase)
    visualize_alternate_score(df, phrase)
    visualize_number_of_phrases_detected(df, phrase)
    visualize_average_sentiment_per_topic(df, phrase, topic_modeler)
    visualize_time_based_activity(df, phrase)
    visualize_user_engagement(df, phrase)
    visualize_top_contributors(df, phrase)
    visualize_word_frequency(df, phrase)

def visualize_stacked_bar_graph(df, phrase):
    """
    Show individual contributions of posts on a given day in stacked bar graphs.
    """
    df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date
    df_sorted = df.sort_values('date')

    # Pivot the data to get sentiment scores per post per day
    pivot_df = df_sorted.pivot_table(
        index='date',
        columns='id',
        values='sentiment_score',
        aggfunc='sum',
        fill_value=0
    )

    # Plot the stacked bar graph
    pivot_df.plot(kind='bar', stacked=True, figsize=(12, 7))

    plt.title(f"Individual Post Contributions per Day for '{phrase}'")
    plt.xlabel("Date")
    plt.ylabel("Sentiment Score")
    plt.legend(title='Post ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'stacked_bar_{phrase.replace(" ", "_")}.png')
    plt.close()
    logging.info(f"Saved stacked bar graph as 'stacked_bar_{phrase.replace(' ', '_')}.png'.")

def visualize_exponential_moving_average(df, phrase):
    """
    Show the exponential moving average of sentiment scores over time.
    """
    df['date'] = pd.to_datetime(df['timestamp'], unit='s')
    df_sorted = df.sort_values('date')

    # Set the decay constant (span) for 3 months (~90 days)
    df_sorted.set_index('date', inplace=True)
    ema_span = 90  # Adjust based on data frequency

    # Calculate EMA of sentiment scores
    df_sorted['ema_sentiment'] = df_sorted['sentiment_score'].ewm(span=ema_span).mean()

    # Plot the EMA
    plt.figure(figsize=(12, 7))
    plt.plot(df_sorted.index, df_sorted['ema_sentiment'], label='EMA of Sentiment Score')
    plt.title(f"Exponential Moving Average of Sentiment for '{phrase}'")
    plt.xlabel("Date")
    plt.ylabel("EMA of Sentiment Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'ema_sentiment_{phrase.replace(" ", "_")}.png')
    plt.close()
    logging.info(f"Saved EMA of sentiment visualization as 'ema_sentiment_{phrase.replace(' ', '_')}.png'.")

def average_sentiment_per_topic(df, lda_model, dictionary, phrase):
    """
    Analyze sentiment scores within each identified topic.
    """
    preprocessor = Preprocessor()
    # Compute topic distribution for each document
    df['tokens'] = df['text'].apply(lambda x: preprocessor.preprocess_for_topic_modeling([x])[0])
    df['topic_distribution'] = df['tokens'].apply(lambda tokens: lda_model.get_document_topics(dictionary.doc2bow(tokens), minimum_probability=0))

    # Assign the dominant topic to each document
    df['dominant_topic'] = df['topic_distribution'].apply(lambda x: max(x, key=lambda item: item[1])[0])

    # Compute average sentiment per topic
    topic_sentiment = df.groupby('dominant_topic')['sentiment_score'].mean().reset_index()

    # Plot the average sentiment per topic
    plt.figure(figsize=(10, 6))
    sns.barplot(x='dominant_topic', y='sentiment_score', data=topic_sentiment)
    plt.title(f"Average Sentiment per Topic for '{phrase}'")
    plt.xlabel("Topic")
    plt.ylabel("Average Sentiment Score")
    plt.tight_layout()
    plt.savefig(f'average_sentiment_per_topic_{phrase.replace(" ", "_")}.png')
    plt.close()
    logging.info(f"Saved average sentiment per topic visualization as 'average_sentiment_per_topic_{phrase.replace(' ', '_')}.png'.")

def user_engagement_metrics(df, phrase):
    """
    Monitor metrics like number of replies per comment and depth.
    """
    # Average depth
    avg_depth = df['depth'].mean()
    logging.info(f"Average depth for '{phrase}' is {avg_depth:.2f}")
    print(f"Average depth for '{phrase}' is {avg_depth:.2f}")

    # Distribution of depth
    plt.figure(figsize=(10, 6))
    sns.histplot(df['depth'], bins=15, kde=False)
    plt.title(f"Distribution of Comment Depth for '{phrase}'")
    plt.xlabel("Depth")
    plt.ylabel("Number of Comments")
    plt.tight_layout()
    plt.savefig(f'depth_distribution_{phrase.replace(" ", "_")}.png')
    plt.close()
    logging.info(f"Saved depth distribution visualization as 'depth_distribution_{phrase.replace(' ', '_')}.png'.")