from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# ✅ CORRECT MODEL PATH
model_path = os.path.join("models", "model.pkl")
model = joblib.load(model_path)

@app.route("/")
def home():
    return "Loan Default Prediction API is running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features = [
        data["loan_amount"],
        data["interest_rate"],
        data["income"],
        data["credit_score"],
        data["loan_term"],
        data["loan_to_income_ratio"]
    ]

    prediction = model.predict([features])[0]
    result = "Defaulter" if prediction == 1 else "Non-Defaulter"

    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
