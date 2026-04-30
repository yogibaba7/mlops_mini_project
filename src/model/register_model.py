
import json
import mlflow
from mlflow.tracking import MlflowClient
import logging
import dagshub
import os
from dotenv import load_dotenv

load_dotenv()

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "yogibaba7"
repo_name = "mlops_mini_project"
# setup mlflow tracking 
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_name}/{repo_name}.mlflow")



# configure logging
logger = logging.getLogger('model_registry_log')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('logging.log')
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

import time

def wait_for_model_ready(client, name, version):
    for _ in range(10):
        mv = client.get_model_version(name, version)
        if mv.status == "READY":
            return
        time.sleep(2)
    raise Exception("Model version not ready")


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        # model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        client = MlflowClient()
        # Register the model
        result = mlflow.register_model(model_info['model_uri'], model_name)

        version = result.version

        #  WAIT until ready
        wait_for_model_ready(client, model_name, version)
        
        # Transition the model to "Staging" stage
        
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging"
        )
        
        logger.debug(f'Model {model_name} version {version} registered and transitioned to Staging.')
    
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        

def main():
    try:
        model_info_path = 'reports/model_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "my_model"
        register_model(model_name, model_info)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()