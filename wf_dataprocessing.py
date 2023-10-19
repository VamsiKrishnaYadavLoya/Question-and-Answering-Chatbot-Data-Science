import numpy as np 
import pandas as pd 
import nltk
from nltk.util import ngrams
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
# Download the averaged_perceptron_tagger data if you haven't already
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')


#count unique words
def count_unique_words(text):
    return len(set(text.split()))

#count the characters
def count_chars(text):
    return len(text)

#count number of words
def count_words(text):
    return len(text.split())


#Tranformations
# N-grams: Extract n-grams (sequences of n words) 
def generate_ngrams(text, n=2):
    words = word_tokenize(text)
    n_grams = list(ngrams(words, n))
    return n_grams

# Part-of-Speech Tags: Use natural language processing tools to extract part-of-speech tags for words in questions and answers. This can help the model understand the grammatical structure.
#each word is tagged as NN(Noun),ADJ(adjective) etc,.
def tag_text(text):
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    return pos_tags

# Named Entity Recognition (NER): Apply NER to identify and label entities (e.g., names, dates, locations) in the text. This can be useful for handling specific information.
def perform_ner_nltk(text):
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    ner_tree = ne_chunk(pos_tags)
    return ner_tree

# Sentiment Analysis: Determine the sentiment of the questions and answers using sentiment analysis tools. This can help the model respond with an appropriate tone or emotion.
#1. if neg(negative) > 0.5 the senence is in a negative tone
#2. if pos(positive) > 0.5 the senence is in a positive tone
#3. if neu(neutral) > 0.5 the senence is in a neutral tone
def perform_sentiment_analysis_nltk(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment

# It takes some raw data, processes it, and prepares it for visualization.
def process_data(df):
    df=df.drop(['Unnamed: 0', 'id'],axis=1)
    df['question tokens']=df['question'].apply(lambda x:x.split())
    df['answer tokens']=df['answers'].apply(lambda x:x.split())
    df['char_count'] = df["question"].apply(lambda x:count_chars(x))
    df['word_count'] = df["question"].apply(lambda x:count_words(x))
    df['unique_word_count'] = df["question"].apply(lambda x:count_unique_words(x))
    # Apply POS tagging to each row in the 'text' column
    df['pos_tags'] = df['question'].apply(tag_text)
    df['bigrams'] = df['question'].apply(lambda x: generate_ngrams(x, n=2))
    df['ner_nltk'] = df['question'].apply(perform_ner_nltk)
    df['sentiment_nltk'] = df['question'].apply(perform_sentiment_analysis_nltk)
    return df

