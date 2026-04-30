import logging
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml

# ---------------- LOGGING SETUP ---------------- #

logger = logging.getLogger('data_ingestion_log')
logger.setLevel(logging.DEBUG)

# Prevent duplicate logs
if not logger.handlers:

    # 📁 File handler (separate log file)
    file_handler = logging.FileHandler('data_ingestion.log')
    file_handler.setLevel(logging.DEBUG)

    # 💻 Console handler (terminal logs)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# ---------------- FUNCTIONS ---------------- #

def load_params() -> float:
    try:
        logger.debug('Opening params.yaml')
        with open('params.yaml', 'r') as file:
            params = yaml.safe_load(file)
            test_size = params['data_ingestion']['test_size']
            logger.info('Params loaded successfully')
            return test_size
    except Exception as e:
        logger.error(f'Params loading failed: {e}')
        return 0.2


def read_data(url: str) -> pd.DataFrame:
    try:
        logger.debug('Loading data from URL')
        data = pd.read_csv(url)
        logger.info('Data loaded successfully')
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()


def process_data(data: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.debug('Processing data')
        data.drop(columns=['tweet_id'], inplace=True)

        data = data[data['sentiment'].isin(['sadness', 'happiness'])]
        data.replace({'sadness': 0, 'happiness': 1}, inplace=True)

        logger.info('Data processed successfully')
        return data

    except Exception as e:
        logger.error(f'Processing failed: {e}')
        return pd.DataFrame()


def save_data(path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        logger.debug('Creating directory')
        os.makedirs(path, exist_ok=True)

        logger.debug('Saving train and test data')
        train_data.to_csv(os.path.join(path, 'train_data.csv'), index=False)
        test_data.to_csv(os.path.join(path, 'test_data.csv'), index=False)

        logger.info('Data saved successfully')

    except Exception as e:
        logger.error(f'Data saving failed: {e}')


# ---------------- MAIN ---------------- #

def main():
    try:
        logger.info('Data ingestion started')

        test_size = load_params()

        df = read_data(
            'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
        )

        df = process_data(df)

        data_path = os.path.join('data', 'raw')

        train_data, test_data = train_test_split(
            df, test_size=test_size, random_state=42
        )

        save_data(data_path, train_data, test_data)

        logger.info('Data ingestion completed successfully')

    except Exception as e:
        logger.error(f'Data ingestion failed: {e}')


if __name__ == "__main__":
    main()