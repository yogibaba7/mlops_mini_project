


import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
import yaml
import logging
import os



# configure logging
logger = logging.getLogger('model_building_log')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('logging.log')
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# load params
def load_params():
    logger.debug('fetching params for model')
    try:
        with open('params.yaml', 'r') as file:
            params = yaml.safe_load(file)
            n_estimators = params['model_building']['n_estimators']
            return n_estimators

    except Exception as e:
        logger.error(f"Unexpected error while loading params: {e}")
        return 100


# read train data
def read_train(train_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        logger.debug('loading training data')
        train_data = pd.read_csv(train_path)

        # input and output
        x_train = train_data.drop(columns=['label'])
        y_train = train_data['label']
        return x_train, y_train

    except Exception as e:
        logger.error(f"Error while reading training data: {e}")
        return pd.DataFrame(), pd.DataFrame()


# create a model
def fit_model(x_train: pd.DataFrame, y_train: pd.DataFrame, n_estimators: int) -> LogisticRegression:
    try:
        logger.debug('training the model')
        lor = LogisticRegression(C=10,solver='liblinear',penalty='l2')
        lor.fit(x_train, y_train)
        return lor

    except Exception as e:
        logger.error(f"Error while fitting the model: {e}")
        return None


# dump the model in a pickle file
def dump_model(model: LogisticRegression, file_path: str) -> None:
    try:
        logger.debug(f"dump model path {file_path}")
        # Save the model to a pickle file
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
    except Exception as e:
        logger.error(f"Error while saving model: {e}")


# main
def main():

    try:
        n_estimators = load_params()
        train_path = 'data/processed/train_tfidf.csv'

        x_train, y_train = read_train(train_path)

        model = fit_model(x_train, y_train, n_estimators)

        # create models folder if not exist
        os.makedirs('models',exist_ok=True)
        
        # model file path
        model_path = 'models/model.pkl'
        dump_model(model, model_path)

        logger.debug(f'model build and dump sucessfully on path {model_path}')

    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")


if __name__ == '__main__':
    main()