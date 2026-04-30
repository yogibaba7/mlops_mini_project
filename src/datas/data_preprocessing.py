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

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# ---------------- LOGGING SETUP ---------------- #

logger = logging.getLogger('data_preprocessing_log')
logger.setLevel(logging.DEBUG)

if not logger.handlers:

    # 📁 File handler
    file_handler = logging.FileHandler('data_preprocessing.log')
    file_handler.setLevel(logging.DEBUG)

    # 💻 Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# ---------------- FUNCTIONS ---------------- #

def read_train_test(train_path: str, test_path: str):
    try:
        logger.debug('Loading train and test data')
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        logger.info('Train & test data loaded successfully')
        return train, test
    except Exception as e:
        logger.error(f'Reading data failed: {e}')
        return pd.DataFrame(), pd.DataFrame()


def lower_case(text: str) -> str:
    try:
        return text.lower()
    except:
        return ""


def remove_punctuation(text: str) -> str:
    try:
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    except:
        return ""


def remove_numbers(text: str) -> str:
    try:
        return re.sub(r'\d+', '', text)
    except:
        return ""


def removing_urls(text: str) -> str:
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except:
        return ""


def tokenization(text: str) -> list:
    try:
        return word_tokenize(text)
    except:
        return []


def remove_stopwords(text: list) -> list:
    try:
        stop_words = set(stopwords.words('english'))
        text = [word for word in text if word not in stop_words]
        text = [word for word in text if len(word) > 2]
        return text
    except Exception as e:
        logger.error(f'Stopword removal failed: {e}')
        return []


def lemmatizer(text: list) -> list:
    try:
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(word) for word in text]
    except Exception as e:
        logger.error(f'Lemmatization failed: {e}')
        return []


def join_words(text: list) -> str:
    try:
        return " ".join(text).strip()
    except Exception as e:
        logger.error(f'Join words failed: {e}')
        return ""


def save_data(path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        os.makedirs(path, exist_ok=True)
        train_data.to_csv(os.path.join(path, 'train_preprocessed.csv'), index=False)
        test_data.to_csv(os.path.join(path, 'test_preprocessed.csv'), index=False)
        logger.info(f"Preprocessed data saved at {path}")
    except Exception as e:
        logger.error(f'Saving data failed: {e}')


# ---------------- MAIN ---------------- #

def main():
    try:
        logger.info('Data preprocessing started')

        train_path = 'data/raw/train_data.csv'
        test_path = 'data/raw/test_data.csv'

        train_data, test_data = read_train_test(train_path, test_path)

        # 🔥 DEBUG: BEFORE preprocessing
        logger.info("🔍 BEFORE PREPROCESSING")
        if not train_data.empty:
            logger.info(f"Sample BEFORE: {train_data['content'].iloc[0]}")
        else:
            logger.warning("⚠️ Train data is empty BEFORE preprocessing")

        logger.debug('Applying preprocessing steps')

        for data in [train_data, test_data]:

            data['content'] = data['content'].apply(lower_case)
            data['content'] = data['content'].apply(remove_punctuation)
            data['content'] = data['content'].apply(remove_numbers)
            data['content'] = data['content'].apply(removing_urls)
            data['content'] = data['content'].apply(tokenization)
            data['content'] = data['content'].apply(remove_stopwords)
            data['content'] = data['content'].apply(lemmatizer)
            data['content'] = data['content'].apply(join_words)

        # 🔥 DEBUG: AFTER preprocessing
        logger.info("🔍 AFTER PREPROCESSING")

        if not train_data.empty:
            logger.info(f"Sample AFTER: {train_data['content'].iloc[0]}")
        else:
            logger.warning("⚠️ Train data is empty AFTER preprocessing")

        # 🔥 DEBUG: empty rows count
        empty_count = train_data['content'].astype(str).str.strip().eq('').sum()
        logger.info(f"Empty rows after preprocessing: {empty_count}")

        # 🔥 DEBUG: show few rows
        logger.info(f"Top 3 processed rows: {train_data['content'].head(3).tolist()}")

        data_path = os.path.join('data', 'interim')
        save_data(data_path, train_data, test_data)

        logger.info('Data preprocessing completed successfully')

    except Exception as e:
        logger.error(f'Data preprocessing failed: {e}')
        raise
if __name__ == "__main__":
    main()