---

# Naive Machine Translation and LSH

This repository contains a Jupyter Notebook and a utility script that demonstrate Naive Machine Translation using embeddings and Locality-Sensitive Hashing (LSH). The project guides you through the process of translating text using basic methods and efficient similarity search techniques.

## Contents

- **Introduction**: Overview of the notebook and its objectives.
- **Setup**: Installing necessary libraries and importing datasets.
- **Data Preparation**: Loading and preprocessing the dataset for translation tasks.
- **Embedding and Transform Matrices**: Creating matrices from word embeddings.
- **Naive Machine Translation**: Implementing a basic translation algorithm using word embeddings.
- **Locality-Sensitive Hashing (LSH)**: Applying LSH for efficient nearest neighbor similarity search.
- **Evaluation**: Assessing the performance of the translation model.
- **Conclusion**: Summary of findings and potential improvements.

## Prerequisites

To run this project, you need the following libraries:
- numpy
- pandas
- nltk
- sklearn
- gensim
- matplotlib

You can install these libraries using pip:
```bash
pip install numpy pandas nltk sklearn gensim matplotlib
```

## Running the Notebook

1. Clone the repository:
   ```bash
   git clone https://github.com/Shreyash-Gaur/Naive_Machine_Translation_LSH.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Naive_Machine_Translation_LSH
   ```
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook "Naive Machine Translation and LSH.ipynb"
   ```

## Utility Script

The repository includes a utility script `utils.py` that contains the following functions:

### `process_tweet`
Cleans and tokenizes tweet text.
```python
def process_tweet(tweet):
    '''
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet
    '''
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    
    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and word not in string.punctuation):
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)
    
    return tweets_clean
```

### `get_dict`
Loads an English-to-French dictionary from a file.

### `cosine_similarity`
Computes the cosine similarity between two vectors.

## Key Sections

### Introduction
Provides an overview of the goals and methodologies used in the notebook. This section introduces the concept of using word embeddings for translation and the benefits of using LSH for efficient similarity search.

### Setup
Details the installation of required libraries and the loading of datasets. This section ensures that you have all the necessary tools and data to replicate the experiments and results presented in the notebook.

### Data Preparation
Covers the preprocessing steps necessary for preparing the dataset for translation tasks. This includes cleaning the data, tokenizing text, and other preparatory steps.

### Embedding and Transform Matrices
This section explains how to create matrices `X` and `Y` from word embeddings for English and French words, respectively. The function `get_matrices` takes an English-to-French dictionary and dictionaries of word embeddings for both languages and returns the matrices `X` and `Y`, where each row in `X` corresponds to the word embedding of an English word and the same row in `Y` corresponds to the word embedding of the French translation.

### Naive Machine Translation
Shows the implementation of a simple translation algorithm based on word embeddings. This section walks through the process of using cosine similarity to find the closest translation in the target language.

### Locality-Sensitive Hashing (LSH)
Describes the application of LSH for fast similarity search, crucial for translating large datasets. This section provides an in-depth look at how Hashing & LSH works and how it can be used to speed up the translation process.

### Evaluation
Includes methods for evaluating the accuracy and performance of the translation model. This section covers various metrics and approaches to assess how well the translation model performs.

### Conclusion
Summarizes the results and suggests potential improvements for future work. This section reflects on the findings of the project and discusses possible enhancements to the methodology.

## Results

The notebook demonstrates a basic approach to machine translation and highlights the benefits of using LSH for efficient similarity search. The results section provides an evaluation of the translation model's performance, including accuracy metrics and potential areas for improvement.

## Acknowledgments

Special thanks to the authors and contributors of the libraries and datasets used in this project.

---