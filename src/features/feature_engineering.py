
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import yaml
import logging

# configure logging

logger = logging.getLogger('feature_engineering_log')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('logging.log')
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# load params
def load_params():
    logger.debug('loading params')
    try:
        logger.debug('opening params.yaml')
        with open('params.yaml', 'r') as file:
            params = yaml.safe_load(file)
            max_features = params['feature_engineering']['max_features']
            return max_features
        logger.debug('params loaded successfully')
    except Exception as e:
        logger.error('params fetching unsuccessfull')
        print(f"Unexpected error while loading params: {e}")
        return 1000


# read train test data
def read_train_test_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        logger.debug('reading training and testing data')
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        return train_data, test_data
        logger.debug('training and testing data fetched successfully')

    except Exception as e:
        logger.error('training and testing data fetching unsuccessfull')
        print(f"Error while reading data: {e}")
        return pd.DataFrame(), pd.DataFrame()


# drop null values
def dropnull(data: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.debug('dropping null values')
        data.dropna(inplace=True)
        return data
        logger.debug('null values dropped')
    except Exception as e:
        logger.error('null values not dropped')
        print(f"Error while dropping null values: {e}")
        return data


# apply tfidf vectorizer
def tfidf(max_features: int, x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        logger.debug('applying  tfidf vectorizer')
        vectorizer = TfidfVectorizer(max_features=max_features)
        x_train_tfidf = vectorizer.fit_transform(x_train)
        x_test_tfidf = vectorizer.transform(x_test)

        # make train dataframe
        train_df = pd.DataFrame(x_train_tfidf.toarray())
        train_df['label'] = y_train

        # make test dataframe
        test_df = pd.DataFrame(x_test_tfidf.toarray())
        test_df['label'] = y_test

        return train_df, test_df
        logger.debug('tfidf successfully applied')
    except Exception as e:
        logger.error('tfidf unsuccessfull')
        print(f"Unexpected error during feature extraction: {e}")
        return pd.DataFrame(), pd.DataFrame()


# save data
def save_data(path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        logger.debug('saving the data')
        os.makedirs(path, exist_ok=True)
        train_data.to_csv(os.path.join(path, 'train_tfidf.csv'), index=False)
        test_data.to_csv(os.path.join(path, 'test_tfidf.csv'), index=False)
        logger.debug(f"Data saved successfully in '{path}'")
    
    except Exception as e:
        logger.error(f"Error while saving data: {e}")


# main
def main():
    try:
        logger.debug('feature engineering process started')
        max_features = load_params()
        train_path = 'data/interim/train_preprocessed.csv'
        test_path = 'data/interim/test_preprocessed.csv'

        train_data, test_data = read_train_test_data(train_path, test_path)


        train_data = dropnull(train_data)
        test_data = dropnull(test_data)

        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values

        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values

        train_data, test_data = tfidf(max_features, X_train, y_train, X_test, y_test)

        file_path = os.path.join('data', 'processed')
        save_data(file_path, train_data, test_data)

        # Check if feature extraction was successful
        logger.debug('feature engineering process successfully completed')

    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")


if __name__ == "__main__":
    main()