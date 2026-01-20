import streamlit as st
import joblib
import numpy as np
import os

# Load model & scaler
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.set_page_config(page_title="Loan Default Prediction", layout="centered")

st.title("🏦 Loan Default Prediction System")
st.write("Enter applicant details to predict loan default risk.")

st.markdown("---")

loan_amount = st.number_input("Loan Amount", min_value=0.0, value=500000.0)
interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=15.0)
income = st.number_input("Annual Income", min_value=0.0, value=300000.0)
credit_score = st.number_input("Credit Score", min_value=300.0, max_value=900.0, value=600.0)
loan_term = st.number_input("Loan Term (months)", min_value=1.0, value=48.0)

loan_to_income_ratio = loan_amount / income if income != 0 else 0

if st.button("Predict Loan Risk"):
    features = np.array([[
        loan_amount,
        interest_rate,
        income,
        credit_score,
        loan_term,
        loan_to_income_ratio
    ]])

    features_scaled = scaler.transform(features)
    default_probability = model.predict_proba(features_scaled)[0][1]

    st.markdown("### 📊 Prediction Result")
    st.write(f"**Default Risk Probability:** `{default_probability:.2f}`")

    if default_probability >= 0.30:
        st.error("❌ High Risk: Applicant is likely to DEFAULT")
    else:
        st.success("✅ Low Risk: Applicant is NOT likely to default")

st.markdown("---")
st.caption("End-to-End ML Project | Streamlit + Scikit-learn")
