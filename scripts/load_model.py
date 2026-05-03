
# this file loads model from model registry that is in production

import mlflow 
import mlflow.pyfunc
import json
import pickle
import numpy as np
import os
import shutil
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

    
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)

    shutil.copytree(local_path, "API/Production_Model_artifacts/model", dirs_exist_ok=True)


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

def main():
    # create model saving dir
    dir = "API/Production_Model_artifacts"
    os.makedirs(dir,exist_ok=True)
    # load model
    model = LoadModel("my_model","Production")
    vector = LoadVector()

    # save model and vector
    # model_path = os.path.join(dir,"model.pkl")
    vector_path = os.path.join(dir,"vector.pkl")


    with open(vector_path,"wb") as f:
        pickle.dump(vector,f)

    print(f"model and vector saved to {dir}")



if __name__=="__main__":
    main()













