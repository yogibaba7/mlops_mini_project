
# this file loads model from model registry that is in production

import mlflow
import mlflow.pyfunc
import json
import dagshub
import pickle
import os
import numpy as np

def LoadModel(ModelName:str,ModelStage:str)->mlflow.pyfunc:

    # setup mlflow connection
    mlflow.set_tracking_uri("https://dagshub.com/yogibaba7/mlops_mini_project.mlflow")
    dagshub.init(repo_owner='yogibaba7', repo_name='mlops_mini_project', mlflow=True)

    # prepare model uri
    model_uri = model_uri = f"models:/{ModelName}/{ModelStage}"

    # load model 
    model = mlflow.pyfunc.load_model(model_uri)

    return model


# load Vector
def LoadVector():
    from mlflow.tracking import MlflowClient
 
    # setup mlflow connection
    mlflow.set_tracking_uri("https://dagshub.com/yogibaba7/mlops_mini_project.mlflow")
    dagshub.init(repo_owner='yogibaba7', repo_name='mlops_mini_project', mlflow=True)

    client = MlflowClient()

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


# apply vectorization
def Vectorization(text:str)->np.array:
    vector = LoadVector()
    result_array = vector.transform([text])
    return result_array 











