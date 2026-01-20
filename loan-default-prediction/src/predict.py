import joblib
import numpy as np

model = joblib.load("model.pkl")

def predict_default(features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]
