# Phrase Sentiment Analyzer

## **Overview**

**Phrase Sentiment Analyzer** is a Python-based tool designed to analyze sentiment associated with specific phrases by leveraging Reddit data. The tool fetches relevant Reddit submissions and comments, preprocesses the text data, performs topic modeling to determine the relevance of each comment, and calculates sentiment scores. This project aims to provide insights into public sentiment on various topics by analyzing discussions on Reddit.

## **Table of Contents**

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
  - [Main Modules](#main-modules)
  - [Planned Modules](#planned-modules)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Creating a GitHub Repository](#creating-a-github-repository)

## **Features**

- **Data Retrieval:** Fetches Reddit submissions and comments related to specified phrases.
- **Text Preprocessing:** Cleans and preprocesses text data using spaCy.
- **Topic Modeling:** Utilizes Latent Dirichlet Allocation (LDA) to identify topics and relevance.
- **Sentiment Analysis:** (Planned) Analyzes sentiment of the text data.
- **Performance Logging:** Tracks and logs the performance of each processing step.
- **Reproducible Environment:** Uses `requirements.txt` for easy setup.

## **Installation**

### **Prerequisites**

- **Python 3.9** installed on your system.
- **Anaconda** or **Miniconda** for environment management.
- **Git** installed (optional, for version control).

### **Setup Instructions**

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/PhraseSentimentAnalyzer.git
   cd PhraseSentimentAnalyzer
   ```

2. **Create a Conda Environment:**

   ```conda create -n phrase_env python=3.9
   conda activate phrase_env
   ```

3. **Install Required Packages:**
   ```pip install -r requirements.txt

   ```
4. **Download NLTK Data (VADER Lexicon):**
   ```python -m nltk.downloader vader_lexicon

   ```
5. **Verify Installations:**
   ```python verify_installations.py

   ```
   **Expected Output:**
   ```Python version: 3.9.19 (main, May  6 2024, 20:12:36) [MSC v.1916 64 bit (AMD64)]
   praw version: 7.7.1
   spaCy version: 3.7.6
   gensim version: 4.1.2
   scipy version: 1.7.3
   pandas version: 1.3.5
   numpy version: 1.21.4
   nltk version: 3.6.7
   matplotlib version: 3.5.1
   openpyxl version: 3.0.9
   pydantic version: 1.10.9
   ```
6. ** Test Loading the spaCy Model:**
   ```python test_spacy.py

   ```
   **Expected Output:**
   ```spaCy model loaded successfully.

   ```

## **Usage**

After setting up the environment and verifying installations, you can run the main script to perform sentiment analysis.

### **Running the Main Script**

Execute the `main.py` script using Python:

```
python main.py
```

### **Output Files**

Upon successful execution, the following files will be generated:

- **`data_with_weights.csv`**

  Contains the processed data with relevance and depth-based weights for each document (submission or comment).

- **`perf_results.xlsx`**

  Logs performance metrics for each processing step, including data retrieval time, preprocessing time, topic modeling time, and total time per phrase.

- **`app.log`**

  Detailed logs of the application's execution, useful for debugging and monitoring the pipeline's progress.

### **Understanding the Output**

1. **`data_with_weights.csv`**

   This CSV file includes the following columns:

   - `phrase`: The phrase being analyzed.
   - `id`: Unique identifier of the Reddit submission or comment.
   - `type`: Indicates whether the entry is a `submission` or a `comment`.
   - `text`: The text content of the submission or comment.
   - `depth`: The depth level in the comment thread (useful for weighting).
   - `score`: The score (upvotes - downvotes) of the submission or comment.
   - `contains_phrase`: Boolean indicating if the text contains the target phrase.
   - `relevance_score`: The relevance score derived from topic modeling.
   - `depth_weight`: Weight based on the depth of the comment in the thread.
   - `final_weight`: Combined weight used for sentiment analysis.

2. **`perf_results.xlsx`**

   This Excel file provides a summary of the performance for each phrase analyzed, including:

   - `phrase`: The phrase being analyzed.
   - `retrieval_time`: Time taken to fetch data from Reddit.
   - `preprocessing_time`: Time taken to preprocess the fetched data.
   - `topic_modeling_time`: Time taken to build and apply the LDA model.
   - `phrase_total_time`: Total time taken to process the phrase.
   - `num_documents`: Number of documents (submissions and comments) processed.

3. **`app.log`**

   This log file records detailed information about the application's execution, including:

   - Start and end times of each processing step.
   - Warnings for any issues encountered, such as empty collections.
   - Errors if any exceptions occur during processing.

### **Example Workflow**

1. **Start the Analysis:**

   Run the main script:

   ```
   python main.py
   ```

2. **Monitor Progress:**

   - Check the console for a performance summary after the script completes.
   - Review `app.log` for detailed logs and any potential warnings or errors.

3. **Review Results:**

   - Open `data_with_weights.csv` to examine the weighted data.
   - Analyze `perf_results.xlsx` to understand the performance metrics of each phrase analysis.

### **Troubleshooting**

- **Empty Collections Error:**

  If you encounter an error stating `ValueError: cannot compute LDA over an empty collection (no terms)`, it means that no valid data was available for topic modeling. Ensure that:

  - The phrases you're analyzing have sufficient data on Reddit.
  - The `limit` parameter in the data retrieval module is set appropriately to fetch enough submissions and comments.

- **spaCy Model Loading Issues:**

  If the spaCy model fails to load, verify that you've correctly downloaded the `en_core_web_sm` model:

  ```
  python -m spacy download en_core_web_sm
  ```

- **Package Compatibility:**

  Ensure that all packages are installed with the versions specified in `requirements.txt`. You can verify installed package versions using:

  ```
  pip list
  ```

### **Next Steps**

- **Implement Sentiment Analysis:**

  The sentiment analysis module is planned and will utilize tools like NLTK's VADER to assign sentiment scores to each document, adjusted by relevance and depth-based weights.

- **Enhance Validation:**

  Develop a validation module to test and ensure the accuracy and reliability of sentiment analysis results.

- **Develop User Interface:**

  Create a user-friendly interface, such as a CLI or GUI, to make the tool more accessible.
