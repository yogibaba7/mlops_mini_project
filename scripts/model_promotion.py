# promote model

import mlflow
import os
from dotenv import load_dotenv

load_dotenv()



def promote_model():
    # setup dagshub credentials for mlflow tracking
    dagshub_token = os.getenv("DAGSHUB_PAT")

    os.environ["MLFLOW_TRACKING_USERNAME"] = "yogibaba7"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    # uri
    dagshub_url = "https://dagshub.com"
    repo_owner = "yogibaba7"
    repo_name = "mlops_mini_project"
    model_name = "my_model"

    uri = f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
    mlflow.set_tracking_uri(uri)

    client = mlflow.MlflowClient()

    versions = client.get_latest_versions(model_name,stages=["Staging"])
    latest_version = versions[0]

    # archive the current production model
    prod_version = client.get_latest_versions(model_name,stages=['Production'])
    # assuming there is only one model in production
    client.transition_model_version_stage(
        name=model_name,
        version = prod_version[0],
        stage='Archived'
    )

    # promote the new model to production
    client.transition_model_version_stage(
        name=model_name,
        version = latest_version,
        stage="Production"

    )

    print(f"Model version {latest_version} promoted to Production stage")

if __name__=="__main__":
    promote_model()



