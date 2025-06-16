from flask import Flask, request, jsonify
import numpy as np
import joblib
import os
from datetime import datetime
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the saved model
MODEL_PATH = "models/fraud_detection_model_20250125_222537.joblib"

def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

# Load model at startup
model = load_model()

@app.route('/')
def home():
    return jsonify({
        "status": "active",
        "model_path": MODEL_PATH,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({"error": "No features provided"}), 400
            
        # Convert features to numpy array
        features = np.array(data['features']).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        # Return prediction
        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability),
            "status": "fraudulent" if prediction == 1 else "legitimate",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)