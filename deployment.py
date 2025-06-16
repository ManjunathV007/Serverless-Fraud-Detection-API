from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os
from typing import List, Dict

# Initialize FastAPI app
app = FastAPI(title="Credit Card Fraud Detection API")

# Load latest model
def load_model():
    model_dir = "models"
    try:
        models = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
        latest_model = max(models)
        model_path = os.path.join(model_dir, latest_model)
        return joblib.load(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

# Input validation schema
class TransactionData(BaseModel):
    features: List[float]

# Load model at startup
model = load_model()

@app.get("/")
def home():
    return {"message": "Credit Card Fraud Detection API"}

@app.post("/predict")
def predict(data: TransactionData):
    try:
        # Convert input to numpy array
        features = np.array(data.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "fraud_status": "Fraudulent" if prediction == 1 else "Legitimate"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")