# Preprocessing 

# this file is for preprocessing of text
import pandas as pd
import numpy as np
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import logging
import pickle


# nltk.download('wordnet')
# nltk.download('stopwords')


# convert to lower case
def lower_case(text: str) -> str:
    try:
        return text.lower()
    except AttributeError:
        return ""


# remove punctuation and special characters
def remove_punctuation(text: str) -> str:
    try:
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    except TypeError:
        return ""


# remove numbers
def remove_numbers(text: str) -> str:
    try:
        return re.sub(r'\d+', '', text)
    except TypeError:
        return ""


# remove URLs
def removing_urls(text: str) -> str:
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except TypeError:
        return ""


# tokenization
def tokenization(text: str) -> list:
    try:
        tokens = text.split()

        # fallback (IMPORTANT)
        if len(tokens) == 0:
            return [text] if text else []

        return tokens

    except Exception as e:
        return [text] if text else []


# remove stopwords
def remove_stopwords(text: list) -> list:
    try:
        stop_words = set(stopwords.words('english'))
        text = [word for word in text if word not in stop_words]
        text = [word for word in text if len(word) > 2]  # remove short words
        return text
    except Exception as e:
        print(f"Error during stopword removal: {e}")
        return []


# lemmatization
def lemmatizer(text: list) -> list:
    try:
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(word) for word in text]
    except Exception as e:
        print(f"Error during lemmatization: {e}")
        return []


# join tokens back to string
def join_words(text: list) -> str:
    try:
        return " ".join(text).strip()
    except Exception as e:
        print(f"Error while joining words: {e}")
        return ""



# Preprocess text
def PreprocessText(text:str)->str:
        
        text = lower_case(text)
        text = remove_punctuation(text)
        text = remove_numbers(text)
        text = removing_urls(text)
        text = tokenization(text)
        text = remove_stopwords(text)
        text = lemmatizer(text)
        text = join_words(text)
        return text


    

