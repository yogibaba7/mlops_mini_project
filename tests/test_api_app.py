from fastapi.testclient import TestClient
from API.app import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "Message" in response.json()


def test_predict_valid_input():
    payload = {"sentiment": "I am very happy today"}

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "Prediction" in data
    assert data["Prediction"] in [0, 1]


def test_predict_invalid_input():
    payload = {"sentiment": 123}  # wrong type

    response = client.post("/predict", json=payload)

    assert response.status_code == 422