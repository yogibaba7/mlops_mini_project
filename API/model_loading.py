
# this file loads model from model registry that is in production

import mlflow 
import mlflow.pyfunc
import json
import pickle
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

def SetupDagshub():
     # setup dagshub credentials for mlflow tracking
    dagshub_token = os.getenv("DAGSHUB_PAT")

    os.environ["MLFLOW_TRACKING_USERNAME"] = "yogibaba7"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    # uri
    dagshub_url = "https://dagshub.com"
    repo_owner = "yogibaba7"
    repo_name = "mlops_mini_project"
    

    uri = f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
    mlflow.set_tracking_uri(uri)


def LoadModel(ModelName:str,ModelStage:str)->mlflow.pyfunc:

    SetupDagshub()

    # prepare model uri
    model_uri = model_uri = f"models:/{ModelName}/{ModelStage}"

    # load model 
    model = mlflow.pyfunc.load_model(model_uri)

    return model


# load Vector
def LoadVector():
    SetupDagshub()

    client = mlflow.MlflowClient()

    model_name = "my_model"
    # latest production version
    latest_versions = client.get_latest_versions(model_name, stages=["Production"])

    run_id = latest_versions[0].run_id
    

    local_path = mlflow.artifacts.download_artifacts(run_id=run_id)

    # vectorizer load
    vec_path = os.path.join(local_path, "preprocessor/vectorizer.pkl")

    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)

    return vectorizer














