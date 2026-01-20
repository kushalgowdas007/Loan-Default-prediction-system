import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Paths (ABSOLUTE & SAFE)
# -----------------------------
PROJECT_ROOT = os.getcwd()
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "processed_data.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

print("📁 Project root:", PROJECT_ROOT)
print("📄 Data path:", DATA_PATH)
print("📦 Model dir:", MODEL_DIR)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH)

print("✅ Dataset loaded")
print("Shape:", df.shape)
print("Class distribution:")
print(df["loan_status"].value_counts())

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# -----------------------------
# Train-test split (small dataset safe)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)
print("⚠️ Small dataset detected → stratify disabled")

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Scaling completed")

# -----------------------------
# Train model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train_scaled, y_train)
print("✅ Model training completed")

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test_scaled)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# -----------------------------
# Save artifacts
# -----------------------------
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, "model.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print("\n✅ Model saved at:", model_path)
print("✅ Scaler saved at:", scaler_path)
print("🎉 TRAINING PIPELINE COMPLETED")
