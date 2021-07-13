from fastapi.testclient import TestClient
from main import app

# test to check the correct functioning of the /ping route
def test_ping():
    with TestClient(app) as client:
        response = client.get("/ping")
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"ping": "pong"}


# test to check if Iris Virginica is classified correctly
def test_pred_virginica():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 6.1,
        "sepal_width": 3.0,
        "petal_length": 4.9,
        "petal_width": 1.8,
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        #assert response.json() == {"flower_class": "Iris Virginica"}
        assert response.json()["flower_class"] == "Iris Virginica"
        
        

def test_pred_setosa():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 4.9,
        "sepal_width": 3.0,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        #assert response.json() == {"flower_class": "Iris Setosa"}
        assert response.json()["flower_class"] == "Iris Setosa"
        

def test_pred_versicolour():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 7.0,
        "sepal_width": 3.2,
        "petal_length": 4.7,
        "petal_width": 1.4,
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        #assert response.json() == {"flower_class": "Iris Versicolour"}
        assert response.json()["flower_class"] == "Iris Versicolour"
        

