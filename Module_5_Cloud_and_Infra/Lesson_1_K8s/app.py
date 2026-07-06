# app.py - FastAPI inference service for model.pkl (y=2x, trained in train.py)
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Define the input data model using Pydantic


class InputData(BaseModel):
    x: float


# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model = joblib.load('model.pkl')


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok"}


@app.post("/predict")
def predict(data: InputData):
    """Make a prediction using the trained model"""
    x_value = np.array([[data.x]])
    prediction = model.predict(x_value)
    return {"prediction": prediction[0]}
