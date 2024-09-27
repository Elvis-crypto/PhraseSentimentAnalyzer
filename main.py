# main.py

import time
import logging
from phrase_config import Config
from data_retrieval import DataRetriever
from preprocessing import Preprocessor
from topic_modeling import TopicModeler
import pandas as pd

def main():
    # Configure logging
    logging.basicConfig(
        filename='app.log',
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    # Record overall start time
    overall_start_time = time.time()
    logging.info("Starting the sentiment analysis pipeline.")

    # Load configuration and initialize modules
    config_start_time = time.time()
    config = Config()
    retriever = DataRetriever(config)
    preprocessor = Preprocessor()
    topic_modeler = TopicModeler(num_topics=5)
    config_end_time = time.time()
    config_time = config_end_time - config_start_time
    logging.info(f"Configuration and initialization took {config_time:.2f} seconds.")

    # Initialize lists to store results and performance data
    all_results = []
    perf_data = []

    # Iterate over each phrase in the configuration
    for phrase in config.phrases:
        logging.info(f"Processing phrase: {phrase}")
        phrase_start_time = time.time()

        # Data Retrieval
        retrieval_start_time = time.time()
        submissions, comments_and_replies = retriever.fetch_posts_comments_and_replies(phrase, limit=100)
        retrieval_end_time = time.time()
        retrieval_time = retrieval_end_time - retrieval_start_time
        logging.info(f"Data retrieval for phrase '{phrase}' took {retrieval_time:.2f} seconds.")

        # Combine all texts for topic modeling
        all_texts = []
        for sub in submissions:
            text = sub['title'] + ' ' + (sub['selftext'] or '')
            all_texts.append(text)
        for comment in comments_and_replies:
            all_texts.append(comment['body'])

        # Preprocessing for topic modeling
        preprocessing_start_time = time.time()
        processed_texts = preprocessor.preprocess_for_topic_modeling(all_texts)
        preprocessing_end_time = time.time()
        preprocessing_time = preprocessing_end_time - preprocessing_start_time
        logging.info(f"Preprocessing for topic modeling took {preprocessing_time:.2f} seconds.")

        # Check if processed_texts is empty
        if not processed_texts:
            logging.warning(f"No processed texts available for phrase '{phrase}'. Skipping LDA modeling.")
            perf_data.append({
                'phrase': phrase,
                'retrieval_time': retrieval_time,
                'preprocessing_time': preprocessing_time,
                'topic_modeling_time': 0,
                'phrase_total_time': time.time() - phrase_start_time,
                'num_documents': 0
            })
            continue

        # Build LDA model
        topic_modeling_start_time = time.time()
        lda_model = topic_modeler.build_lda_model(processed_texts)
        topic_modeling_end_time = time.time()
        topic_modeling_time = topic_modeling_end_time - topic_modeling_start_time
        logging.info(f"Topic modeling took {topic_modeling_time:.2f} seconds.")

        if lda_model is None:
            logging.warning(f"LDA model was not built for phrase '{phrase}'. Skipping sentiment analysis.")
            perf_data.append({
                'phrase': phrase,
                'retrieval_time': retrieval_time,
                'preprocessing_time': preprocessing_time,
                'topic_modeling_time': topic_modeling_time,
                'phrase_total_time': time.time() - phrase_start_time,
                'num_documents': 0
            })
            continue

        # Compute topic distribution for the original phrase
        phrase_tokens = preprocessor.preprocess_for_topic_modeling([phrase])[0]
        phrase_topic_vector = topic_modeler.compute_topic_distribution(phrase_tokens)

        # Prepare data for each document
        doc_data = []
        idx = 0  # Index to access processed_texts
        # Process submissions
        for sub in submissions:
            text = sub['title'] + ' ' + (sub['selftext'] or '')
            tokens = processed_texts[idx]
            idx += 1
            doc_topic_vector = topic_modeler.compute_topic_distribution(tokens)
            relevance_score = topic_modeler.compute_relevance_score(doc_topic_vector, phrase_topic_vector)
            # Thread-based weight (depth 0)
            depth_weight = 1.0
            # Final weight
            final_weight = relevance_score * depth_weight
            doc_info = {
                'phrase': phrase,
                'id': sub['id'],
                'type': 'submission',
                'text': text,
                'depth': sub['depth'],
                'score': sub['score'],
                'contains_phrase': sub['contains_phrase'],
                'relevance_score': relevance_score,
                'depth_weight': depth_weight,
                'final_weight': final_weight
            }
            doc_data.append(doc_info)
        # Process comments and replies
        for comment in comments_and_replies:
            text = comment['body']
            tokens = processed_texts[idx]
            idx += 1
            doc_topic_vector = topic_modeler.compute_topic_distribution(tokens)
            relevance_score = topic_modeler.compute_relevance_score(doc_topic_vector, phrase_topic_vector)
            # Thread-based weight
            depth_weight = 1 / (comment['depth'] + 1)
            # Final weight
            final_weight = relevance_score * depth_weight
            doc_info = {
                'phrase': phrase,
                'id': comment['id'],
                'parent_id': comment['parent_id'],
                'type': 'comment',
                'text': text,
                'depth': comment['depth'],
                'score': comment['score'],
                'contains_phrase': comment['contains_phrase'],
                'relevance_score': relevance_score,
                'depth_weight': depth_weight,
                'final_weight': final_weight
            }
            doc_data.append(doc_info)

        # Collect performance data for this phrase
        phrase_end_time = time.time()
        phrase_total_time = phrase_end_time - phrase_start_time
        perf_data.append({
            'phrase': phrase,
            'retrieval_time': retrieval_time,
            'preprocessing_time': preprocessing_time,
            'topic_modeling_time': topic_modeling_time,
            'phrase_total_time': phrase_total_time,
            'num_documents': len(doc_data)
        })
        logging.info(f"Total time for processing phrase '{phrase}' was {phrase_total_time:.2f} seconds.")

        # Append to all_results
        all_results.extend(doc_data)

    # Convert results to a DataFrame for further analysis
    df_results = pd.DataFrame(all_results)
    logging.info("Converted results to DataFrame.")

    # Save results to CSV
    df_results.to_csv('data_with_weights.csv', index=False)
    logging.info("Saved data with weights to 'data_with_weights.csv'.")

    # Convert performance data to a DataFrame
    df_perf = pd.DataFrame(perf_data)
    logging.info("Converted performance data to DataFrame.")

    # Save performance data to Excel
    df_perf.to_excel('perf_results.xlsx', index=False)
    logging.info("Saved performance results to 'perf_results.xlsx'.")

    # Record overall end time
    overall_end_time = time.time()
    overall_time = overall_end_time - overall_start_time
    logging.info(f"Total execution time: {overall_time:.2f} seconds.")

    # Print summary
    print("\nPerformance Summary:")
    print(df_perf)

if __name__ == "__main__":
    main()
