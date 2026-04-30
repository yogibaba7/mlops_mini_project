import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
import logging
import mlflow
import dagshub
from dotenv import load_dotenv
import os

load_dotenv()

token = os.getenv("DAGSHUB_PAT")

if not token:
    raise Exception("DAGSHUB_PAT not set")

# set MLflow credentials
os.environ["MLFLOW_TRACKING_USERNAME"] = "yogibaba7"
os.environ["MLFLOW_TRACKING_PASSWORD"] = token


dagshub_url = "https://dagshub.com"
repo_owner = "yogibaba7"
repo_name = "mlops_mini_project"
# setup mlflow tracking 
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# mlflow.set_tracking_uri("https://dagshub.com/yogibaba7/mlops_mini_project.mlflow")
# dagshub.init(repo_owner='yogibaba7', repo_name='mlops_mini_project', mlflow=True)



# configure logging
logger = logging.getLogger('model_evaluation_log')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('logging.log')
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# read test data
def read_test(test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.debug(f"loading test data from {test_path}")
    try:
        test_df = pd.read_csv(test_path)
        x_test = test_df.drop(columns=['label'])
        y_test = test_df['label']
        return x_test, y_test
    except Exception as e:
        logger.error(f"Unexpected error while reading test data: {e}")
        return pd.DataFrame(), pd.DataFrame()


# Load the model from the pickle file
def load_model(model_path: str) -> GradientBoostingClassifier:
    logger.debug(f"loading the model from {model_path}")
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model

    except Exception as e:
        logger.error(f"Unexpected error while loading model: {e}")
        return None


# make prediction
def predict(model: GradientBoostingClassifier, x_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    logger.debug('Making prediction')
    try:
        y_pred = model.predict(x_test)
        y_pred_proba = model.predict_proba(x_test)
        return y_pred, y_pred_proba
    except Exception as e:
        logger.error(f"Error while making predictions: {e}")
        return np.array([]), np.array([])


# scores
def store_result(file_path: str, y_test: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> None:

    try:
        logger.debug('calculating results')
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_pred_proba[:, 1])

        score_dict = {
            'accuracy_score': accuracy,
            'precision_score': precision,
            'recall_score': recall,
            'roc_score': roc
        }
        logger.debug(f"storing the result on file {file_path}")

        with open(file_path, 'w') as file:
            json.dump(score_dict, file, indent=4)
        return score_dict
        logger.debug(f"result successfully stored on file {file_path}")

    except Exception as e:
        logger.error(f"Error while storing results: {e}")

# SaveModelInfo
def SaveModelInfo(model_uri: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'model_uri': model_uri, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise






# main
def main():
    mlflow.set_experiment("DVC-Pipeline")
    with mlflow.start_run() as run:
        try:
            test_path = 'data/processed/test_tfidf.csv'
            model_path = 'models/model.pkl'

            x_test, y_test = read_test(test_path)
            model = load_model(model_path)
            y_pred, y_pred_proba = predict(model, x_test)
            file_path = 'reports/metrics.json'
            metrics = store_result(file_path, y_test, y_pred, y_pred_proba)
            logger.debug('model evaluation successfully completed')

            mlflow.log_param("model", "Logistic Regression")
            mlflow.log_param("feature_engineering", "TF-IDF")
            mlflow.log_param("C", 10)
            mlflow.log_param("solver", 'liblinear')
            mlflow.log_param("penalty", 'l2')

            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            mlflow.log_artifact("reports/metrics.json")

            print("Before model logging")
            # log the model
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model"
            )

            print(model_info.model_uri)
            
            
            # Save model info
            SaveModelInfo(model_info.model_uri,"model","reports/model_info.json")

            # Log the model info file to MLflow
            mlflow.log_artifact('reports/model_info.json')

            # Log vectorizer 
            mlflow.log_artifact("models/vectorizer.pkl", artifact_path="preprocessor")

            
        except Exception as e:
            logger.error(f"Unexpected error occurred: {e}")


if __name__ == '__main__':
    main()
    # print("Artifact URI:", mlflow.get_artifact_uri())