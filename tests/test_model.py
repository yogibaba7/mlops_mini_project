
import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

from dotenv import load_dotenv
import os

load_dotenv()


class BaseTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # ---------------------------
        # 1. Setup MLflow connection
        # ---------------------------
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        # set MLflow credentials
        os.environ["MLFLOW_TRACKING_USERNAME"] = "yogibaba7"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "yogibaba7"
        repo_name = "mlops_mini_project"
        # setup mlflow tracking 
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        # ---------------------------
        # 2. Load Model
        # ---------------------------
        cls.model_name = "my_model"
        cls.model_version = cls.get_latest_model_version(cls.model_name)

        if cls.model_version is None:
            raise Exception("No model found in registry")

        cls.model_uri = f"models:/{cls.model_name}/{cls.model_version}"
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)

        # ---------------------------
        # 3. Load Vectorizer FROM MLFLOW (FIXED)
        # ---------------------------
        client = mlflow.MlflowClient()
        mv = client.get_model_version(cls.model_name, cls.model_version)

        run_id = mv.run_id

        local_path = mlflow.artifacts.download_artifacts(run_id=run_id)

        vec_path = os.path.join(local_path, "preprocessor/vectorizer.pkl")

        with open(vec_path, "rb") as f:
            cls.vectorizer = pickle.load(f)

        # ---------------------------
        # 4. Load RAW Test Data (FIXED)
        # ---------------------------
        cls.test_data = pd.read_csv("data/interim/test_preprocessed.csv")

        cls.text_col = cls.test_data.columns[1]
        cls.target_col = cls.test_data.columns[0]

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[stage])
        return versions[0].version if versions else None


# =====================================================
# ✅ 1. FUNCTIONAL TEST
# =====================================================
class TestFunctional(BaseTest):

    def test_model_loaded(self):
        self.assertIsNotNone(self.model)

    def test_prediction_output(self):
        sample_text = ["this product is amazing"]

        vec = self.vectorizer.transform(sample_text)
        df = pd.DataFrame(vec.toarray())

        preds = self.model.predict(df)

        self.assertEqual(len(preds), 1)
        self.assertTrue(preds[0] in [0, 1])


# =====================================================
# ✅ 2. DATA VALIDATION TEST
# =====================================================
class TestDataValidation(BaseTest):

    def test_no_nulls(self):
        self.assertEqual(self.test_data.isnull().sum().sum(), 0)

    def test_text_type(self):
        text_data = self.test_data[self.text_col]
        self.assertTrue(isinstance(text_data.iloc[0], str))

    def test_labels_exist(self):
        labels = self.test_data[self.target_col]
        self.assertTrue(len(labels) > 0)


# =====================================================
# ✅ 3. PERFORMANCE TEST
# =====================================================
class TestPerformance(BaseTest):

    def test_model_performance(self):

        raw_text = self.test_data[self.text_col]
        y = self.test_data[self.target_col]

        # 🔥 APPLY VECTORIZER HERE (MAIN FIX)
        X_vec = self.vectorizer.transform(raw_text)
        X_vec = pd.DataFrame(X_vec.toarray())

        preds = self.model.predict(X_vec)

        acc = accuracy_score(y, preds)
        precision = precision_score(y, preds)
        recall = recall_score(y, preds)
        f1 = f1_score(y, preds)

        print("\nModel Metrics:")
        print(f"Accuracy: {acc}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

        self.assertGreaterEqual(acc, 0.40)
        self.assertGreaterEqual(precision, 0.40)
        self.assertGreaterEqual(recall, 0.40)
        self.assertGreaterEqual(f1, 0.40)


if __name__ == "__main__":
    unittest.main()