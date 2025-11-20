import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ==========================================
# ========== HEART DISEASE TRAINING =========
# ==========================================

print("Loading heart disease dataset...")

heart_files = [
    "heart.csv",
    "Cardiovascular_Disease_Dataset.csv"
]

heart_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                  'restecg', 'thalach', 'exang', 'oldpeak',
                  'slope', 'ca', 'thal', 'num']

heart_frames = []

for file in heart_files:
    try:
        df = pd.read_csv(file)
        print(f"Loaded: {file}, shape={df.shape}")

        df.columns = df.columns.str.strip()

        # Add missing columns (option 2)
        for col in heart_features:
            if col not in df.columns:
                df[col] = np.nan

        df = df[heart_features]
        heart_frames.append(df)

    except:
        print(f"Failed to load: {file}")

# Merge datasets
heart_df = pd.concat(heart_frames, ignore_index=True)

# Clean dataset
heart_df = heart_df.replace("?", np.nan)

for col in heart_features[:-1]:
    heart_df[col] = pd.to_numeric(heart_df[col], errors='coerce')
    heart_df[col] = heart_df[col].fillna(heart_df[col].median())

# ===== SAFE BINARY CLEANING FOR num ===== #
heart_df['num'] = pd.to_numeric(heart_df['num'], errors='coerce')
heart_df['num'] = heart_df['num'].fillna(0)
heart_df['num'] = (heart_df['num'] > 0).astype(int)

# Final X, y
X_heart = heart_df[heart_features[:-1]]
y_heart = heart_df['num']

# Train-test split
X_heart_train, X_heart_test, y_heart_train, y_heart_test = train_test_split(
    X_heart, y_heart, test_size=0.2, random_state=42
)

# Scale
heart_scaler = StandardScaler()
X_heart_train_scaled = heart_scaler.fit_transform(X_heart_train)
X_heart_test_scaled = heart_scaler.transform(X_heart_test)

# Train model
print("Training heart disease model...")
heart_model = RandomForestClassifier(n_estimators=100, random_state=42)
heart_model.fit(X_heart_train_scaled, y_heart_train)

# Accuracy
heart_score = heart_model.score(X_heart_test_scaled, y_heart_test)
print(f"Heart disease model accuracy: {heart_score:.4f}")

# Save model
joblib.dump(heart_model, 'heart_model.pkl')
joblib.dump(heart_scaler, 'heart_scaler.pkl')
print("Heart disease model saved!")



# ==========================================
# ============= DIABETES TRAINING ===========
# ==========================================

print("\nLoading diabetes dataset...")

diabetes_files = [
    "diabetes.csv",
    "diabetes_all_2016.csv",
    "Dataset of Diabetes .csv"
]

diabetes_features = ['Pregnancies', 'Glucose', 'BloodPressure',
                     'SkinThickness', 'Insulin', 'BMI',
                     'DiabetesPedigreeFunction', 'Age', 'Outcome']

diabetes_frames = []

for file in diabetes_files:
    try:
        df = pd.read_csv(file)
        print(f"Loaded: {file}, shape={df.shape}")

        df.columns = df.columns.str.strip()

        # Rename different labels to your fixed names
        rename_map = {}
        for col in df.columns:
            c = col.lower()
            if "preg" in c:
                rename_map[col] = "Pregnancies"
            elif "glucose" in c:
                rename_map[col] = "Glucose"
            elif "bloodpressure" in c or c == "bp":
                rename_map[col] = "BloodPressure"
            elif "thick" in c:
                rename_map[col] = "SkinThickness"
            elif "insulin" in c:
                rename_map[col] = "Insulin"
            elif c == "bmi":
                rename_map[col] = "BMI"
            elif "pedigree" in c:
                rename_map[col] = "DiabetesPedigreeFunction"
            elif c == "age":
                rename_map[col] = "Age"
            elif "outcome" in c or "diabetic" in c or "class" in c:
                rename_map[col] = "Outcome"

        df = df.rename(columns=rename_map)

        # Add missing columns
        for col in diabetes_features:
            if col not in df.columns:
                df[col] = np.nan

        df = df[diabetes_features]
        diabetes_frames.append(df)

    except:
        print(f"Failed: {file}")

# Merge diabetes datasets
diabetes_df = pd.concat(diabetes_frames, ignore_index=True)

# Replace zeros
for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    diabetes_df[col] = diabetes_df[col].replace(0, np.nan)

# Fill missing
for col in diabetes_features[:-1]:
    diabetes_df[col] = diabetes_df[col].fillna(diabetes_df[col].median())

# ===== UNIVERSAL SAFE OUTCOME CLEANER ===== #
diabetes_df['Outcome'] = diabetes_df['Outcome'].astype(str).str.strip().str.lower()

positive_values = ['yes', 'y', 'positive', 'pos', 'true', 't', '1']
negative_values = ['no', 'n', 'negative', 'neg', 'false', 'f', '0', '', 'nan', 'none']

diabetes_df['Outcome'] = diabetes_df['Outcome'].replace(positive_values, 1)
diabetes_df['Outcome'] = diabetes_df['Outcome'].replace(negative_values, 0)

# Convert leftover unknown values
diabetes_df['Outcome'] = pd.to_numeric(diabetes_df['Outcome'], errors='coerce')
diabetes_df['Outcome'] = diabetes_df['Outcome'].fillna(0).astype(int)
# ====================================================== #

# Final X, y
X_diabetes = diabetes_df[diabetes_features[:-1]]
y_diabetes = diabetes_df['Outcome']

# Train-test split
X_diabetes_train, X_diabetes_test, y_diabetes_train, y_diabetes_test = train_test_split(
    X_diabetes, y_diabetes, test_size=0.2, random_state=42
)

# Scale
diabetes_scaler = StandardScaler()
X_diabetes_train_scaled = diabetes_scaler.fit_transform(X_diabetes_train)
X_diabetes_test_scaled = diabetes_scaler.transform(X_diabetes_test)

# Train model
print("Training diabetes model...")
diabetes_model = RandomForestClassifier(n_estimators=100, random_state=42)
diabetes_model.fit(X_diabetes_train_scaled, y_diabetes_train)

# Accuracy
diabetes_score = diabetes_model.score(X_diabetes_test_scaled, y_diabetes_test)
print(f"Diabetes model accuracy: {diabetes_score:.4f}")

# Save model
joblib.dump(diabetes_model, 'diabetes_model.pkl')
joblib.dump(diabetes_scaler, 'diabetes_scaler.pkl')

print("Diabetes model saved!")
print("\nAll models trained and saved successfully!")
