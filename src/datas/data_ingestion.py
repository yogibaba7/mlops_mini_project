import logging
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml

# configure logging
logger = logging.getLogger('data_ingestion_log')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('logging.log')
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# load params
def load_params() -> float:
    try:
        logger.debug('params file opening')
        with open('params.yaml', 'r') as file:
            logger.debug('params fetching')
            params = yaml.safe_load(file)
            test_size = params['data_ingestion']['test_size']
            return test_size
            logger.debug("parameters sucessfully fetched")
    except Exception as e:
        logger.error('params not fetched')
        print(f"{e}")
        return 0.2  # Default value if file is missing



# read_data
def read_data(url: str) -> pd.DataFrame:
    try:
        logger.debug('data loading')
        data = pd.read_csv(url)
        return data
        logger.debug('data loaded sucessfully')
    except Exception as e:
        logger.debug('loading data unsucessfull')
        print(f"Error: Unable to load data from URL: {e}")
        return pd.DataFrame()


# process_data
def process_data(data: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.debug('data processing started')
        data.drop(columns=['tweet_id'], inplace=True)        
        # make it a binary problem
        data = data[data['sentiment'].isin(['sadness', 'happiness'])]
        data.replace({'sadness': 0, 'happiness': 1}, inplace=True)
        logger.debug('data processing sucessfull')
        return data

    except Exception as e:
        logger.error('data processing unsucessful')
        print(f"Error during data processing: {e}")
        return pd.DataFrame()


# save data
def save_data(path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        logger.debug('creating a directory')
        # make a directory/folder
        os.makedirs(path, exist_ok=True)
        logger.debug('directory created sucessfully')
        # save the train and test data
        logger.debug('saving data in created directory')
        train_data.to_csv(os.path.join(path, 'train_data.csv'), index=False)
        test_data.to_csv(os.path.join(path, 'test_data.csv'), index=False)
        logger.debug('data saved successfully')
        
    except Exception as e:
        logger.error('data saving unsuccessfull')
        print(f"Error while saving data: {e}")


# main
def main():
    try:
        test_size = load_params()
        df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        df = process_data(df)
        # make a path
        data_path = os.path.join('data', 'raw')
        # train-test split the data
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        # save the data
        save_data(data_path, train_data, test_data)
        logger.debug('data_ingestion successull')
    except Exception as e:
        logger.debug('data_ingestion unsuccessull')
        print(f"Unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
