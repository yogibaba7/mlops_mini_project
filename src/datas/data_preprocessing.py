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

nltk.download('wordnet')
nltk.download('stopwords')

# configure logging
logger = logging.getLogger('data_preprocessing_log')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('logging.log')
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# read train and test data
def read_train_test(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        logger.debug('data loading')
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        return train, test
        logger.debug('data_ingestion successull')

    except Exception as e:
        logger.error('data_ingestion unsuccessull')
        print(f"Error while reading data: {e}")
        return pd.DataFrame(), pd.DataFrame()


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
        return word_tokenize(text)
    except TypeError:
        return []


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


# save data
def save_data(path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        os.makedirs(path, exist_ok=True)
        train_data.to_csv(os.path.join(path, 'train_preprocessed.csv'), index=False)
        test_data.to_csv(os.path.join(path, 'test_preprocessed.csv'), index=False)
        print(f"Preprocessed data saved successfully in '{path}'")

    except Exception as e:
        print(f"Error while saving data: {e}")


# main
def main():
    try:
        train_path = 'data/raw/train_data.csv'
        test_path = 'data/raw/test_data.csv'
        train_data, test_data = read_train_test(train_path, test_path)

        logger.debug('applying preprocessing')
        # Apply preprocessing
        for data in [train_data, test_data]:
            
            data['content'] = data['content'].apply(lower_case)
            logger.debug('data converted in lower casel')
            data['content'] = data['content'].apply(remove_punctuation)
            logger.debug('punctuation removed')
            data['content'] = data['content'].apply(remove_numbers)
            logger.debug('numbers removed')
            data['content'] = data['content'].apply(removing_urls)
            logger.debug('url removed')
            data['content'] = data['content'].apply(tokenization)
            logger.debug('tokenization completed')
            data['content'] = data['content'].apply(remove_stopwords)
            logger.debug('stop words removed')
            data['content'] = data['content'].apply(lemmatizer)
            logger.debug('lemmatizer applied')
            data['content'] = data['content'].apply(join_words)
            logger.debug('words joined')
 
        # Save preprocessed data
        data_path = os.path.join('data', 'interim')
        save_data(data_path, train_data, test_data)
        logger.debug(f'data successfully saved in {data_path}')
        logger.debug('data preprocessing successfully completed')
    except Exception as e:
        logger.error('data preprocessing unsucessfull')
        print(f"Unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
