from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__, static_folder='static')
CORS(app)

# Load models and scalers
heart_model = None
heart_scaler = None
diabetes_model = None
diabetes_scaler = None

def load_models():
    global heart_model, heart_scaler, diabetes_model, diabetes_scaler
    try:
        heart_model = joblib.load('heart_model.pkl')
        heart_scaler = joblib.load('heart_scaler.pkl')
        diabetes_model = joblib.load('diabetes_model.pkl')
        diabetes_scaler = joblib.load('diabetes_scaler.pkl')
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")

# Load models on startup
load_models()

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/predict/heart', methods=['POST'])
def predict_heart():
    try:
        if heart_model is None or heart_scaler is None:
            return jsonify({'error': 'Heart model not loaded. Please train the model first.'}), 500
        
        data = request.json
        
        # Extract features in correct order
        features = [
            float(data.get('age', 0)),
            float(data.get('sex', 0)),
            float(data.get('cp', 0)),
            float(data.get('trestbps', 0)),
            float(data.get('chol', 0)),
            float(data.get('fbs', 0)),
            float(data.get('restecg', 0)),
            float(data.get('thalach', 0)),
            float(data.get('exang', 0)),
            float(data.get('oldpeak', 0)),
            float(data.get('slope', 0)),
            float(data.get('ca', 0)),
            float(data.get('thal', 0))
        ]
        
        # Scale features
        features_scaled = heart_scaler.transform([features])
        
        # Make prediction
        prediction = heart_model.predict(features_scaled)[0]
        probability = heart_model.predict_proba(features_scaled)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability[1]),
            'message': 'Heart disease detected' if prediction == 1 else 'No heart disease detected'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    try:
        if diabetes_model is None or diabetes_scaler is None:
            return jsonify({'error': 'Diabetes model not loaded. Please train the model first.'}), 500
        
        data = request.json
        
        # Extract features in correct order
        features = [
            float(data.get('Pregnancies', 0)),
            float(data.get('Glucose', 0)),
            float(data.get('BloodPressure', 0)),
            float(data.get('SkinThickness', 0)),
            float(data.get('Insulin', 0)),
            float(data.get('BMI', 0)),
            float(data.get('DiabetesPedigreeFunction', 0)),
            float(data.get('Age', 0))
        ]
        
        # Scale features
        features_scaled = diabetes_scaler.transform([features])
        
        # Make prediction
        prediction = diabetes_model.predict(features_scaled)[0]
        probability = diabetes_model.predict_proba(features_scaled)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability[1]),
            'message': 'Diabetes detected' if prediction == 1 else 'No diabetes detected'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)

