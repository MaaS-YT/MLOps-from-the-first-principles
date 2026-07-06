# app.py - FastAPI inference service
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.datasets import load_iris

# Define the input data model (4 features for Iris)


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Load the trained model
model = joblib.load("iris_model.pkl")
iris = load_iris()

app = FastAPI(title="Iris Species Prediction API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_species(features: IrisFeatures):
    # Convert input features to the format for model (numpy array of shape (1,4))
    data = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]]
    data_np = np.array(data)
    pred_class = model.predict(data_np)[0]        # numeric class (e.g., 0,1,2)
    # probabilities for each class
    pred_proba = model.predict_proba(data_np)[0]
    species_name = iris.target_names[pred_class]  # e.g., "setosa"
    # probability of the predicted class
    confidence = max(pred_proba)
    
    return {
        "predicted_species": species_name,
        "confidence": round(float(confidence), 3)
    }
