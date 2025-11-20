# Health Prediction System

A machine learning-powered web application for predicting heart disease and diabetes using Random Forest Classifier. This project provides an intuitive web interface for healthcare professionals and individuals to assess health risks based on clinical parameters.

## 🎯 Features

- **Heart Disease Prediction**: Predicts the likelihood of cardiovascular disease based on 13 clinical features
- **Diabetes Prediction**: Predicts the likelihood of diabetes based on 8 health metrics
- **Modern Web Interface**: Beautiful, responsive UI with dark glass-morphism design
- **Real-time Predictions**: Instant results with probability scores
- **Health Recommendations**: Personalized health suggestions based on prediction results
- **Field Information**: Detailed explanations for each input parameter
- **RESTful API**: Backend API for integration with other applications

## 🛠️ Technologies Used

- **Backend**: Flask (Python web framework)
- **Machine Learning**: scikit-learn (Random Forest Classifier)
- **Data Processing**: pandas, numpy
- **Model Persistence**: joblib
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **CORS**: flask-cors for cross-origin requests

## 📋 Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

## 🚀 Installation

1. **Clone the repository** (or navigate to the project directory):
   ```bash
   cd ml
   ```

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models** (if model files don't exist):
   ```bash
   python train_models.py
   ```
   
   This will:
   - Load and merge multiple datasets for heart disease and diabetes
   - Preprocess the data (handle missing values, normalize features)
   - Train Random Forest models
   - Save models and scalers as `.pkl` files

4. **Start the Flask application**:
   ```bash
   python app.py
   ```

5. **Access the application**:
   Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

## 📁 Project Structure

```
ml/
├── app.py                          # Flask application and API endpoints
├── train_models.py                 # Model training script
├── model.ipynb                     # Jupyter notebook for model development
├── requirements.txt                # Python dependencies
├── static/
│   └── index.html                  # Web interface
├── heart_model.pkl                 # Trained heart disease model
├── heart_scaler.pkl                # Feature scaler for heart disease
├── diabetes_model.pkl              # Trained diabetes model
├── diabetes_scaler.pkl              # Feature scaler for diabetes
├── heart.csv                       # Heart disease dataset
├── Cardiovascular_Disease_Dataset.csv
├── diabetes.csv                    # Diabetes dataset
├── diabetes_all_2016.csv
└── Dataset of Diabetes .csv
```

## 🔌 API Endpoints

### Heart Disease Prediction
- **URL**: `/predict/heart`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
  }
  ```
- **Response**:
  ```json
  {
    "prediction": 1,
    "probability": 0.85,
    "message": "Heart disease detected"
  }
  ```

### Diabetes Prediction
- **URL**: `/predict/diabetes`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
  }
  ```
- **Response**:
  ```json
  {
    "prediction": 1,
    "probability": 0.78,
    "message": "Diabetes detected"
  }
  ```

## 📊 Model Information

### Heart Disease Model
- **Algorithm**: Random Forest Classifier
- **Features**: 13 features
  - Age, Sex, Chest Pain Type, Resting Blood Pressure, Serum Cholesterol
  - Fasting Blood Sugar, Resting ECG, Maximum Heart Rate
  - Exercise Induced Angina, ST Depression, Slope, Number of Vessels, Thalassemia
- **Preprocessing**: StandardScaler for feature normalization
- **Datasets**: Combined from `heart.csv` and `Cardiovascular_Disease_Dataset.csv`

### Diabetes Model
- **Algorithm**: Random Forest Classifier
- **Features**: 8 features
  - Pregnancies, Glucose, Blood Pressure, Skin Thickness
  - Insulin, BMI, Diabetes Pedigree Function, Age
- **Preprocessing**: StandardScaler for feature normalization
- **Datasets**: Combined from multiple diabetes datasets

## 💻 Usage

### Web Interface

1. **Start the application** (if not already running):
   ```bash
   python app.py
   ```

2. **Open the web interface**:
   - Navigate to `http://localhost:5000` in your browser

3. **Select a prediction type**:
   - Click on "❤️ Heart Disease" or "🩺 Diabetes" tab

4. **Fill in the form**:
   - Enter all required clinical parameters
   - Click the info icon (ℹ️) next to any field for detailed information

5. **Get predictions**:
   - Click the "Predict" button
   - View the prediction result, probability score, and health recommendations

### Programmatic API Usage

You can also use the API programmatically:

```python
import requests

# Heart disease prediction
response = requests.post('http://localhost:5000/predict/heart', json={
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
})

result = response.json()
print(f"Prediction: {result['message']}")
print(f"Probability: {result['probability']:.2%}")
```

## 🔧 Configuration

- **Port**: Default port is 5000. To change it, modify `app.py`:
  ```python
  app.run(debug=True, port=YOUR_PORT)
  ```

- **Model Parameters**: To adjust model hyperparameters, edit `train_models.py`:
  ```python
  RandomForestClassifier(n_estimators=100, random_state=42)
  ```

## ⚠️ Important Notes

- **Medical Disclaimer**: This application is for educational and research purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

- **Model Accuracy**: Model performance depends on data quality and may vary. Check the training output for accuracy scores.

- **Data Privacy**: Ensure patient data is handled according to applicable privacy regulations (HIPAA, GDPR, etc.) when deploying in production.

## 🧪 Retraining Models

To retrain the models with updated data:

1. Place new datasets in the project directory
2. Update `train_models.py` to include new dataset files
3. Run the training script:
   ```bash
   python train_models.py
   ```
4. The new models will overwrite existing `.pkl` files

## 📝 Dependencies

See `requirements.txt` for the complete list. Key dependencies:
- Flask 3.0.0
- scikit-learn 1.4.0
- pandas 2.1.4
- numpy 1.26.3
- joblib 1.3.2
- flask-cors 4.0.0

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is for educational purposes as part of a Statistical Machine Learning course.

## 👨‍💻 Author

Developed as part of SEM 3 - Statistical Machine Learning course project at Bennett University.

---

**Note**: This is an academic project. For production use, ensure proper validation, security measures, and compliance with healthcare regulations.

