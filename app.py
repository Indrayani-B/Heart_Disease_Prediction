import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("svm_house_price_model.joblib")
scaler = joblib.load("scaler.joblib")

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to predict heart disease risk")

# -------------------------------
# USER INPUT
# -------------------------------
age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex", [0,1], format_func=lambda x: "Female" if x==0 else "Male")
cp = st.selectbox("Chest Pain Type (cp)", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120", [0,1])
restecg = st.selectbox("Resting ECG", [0,1,2])
thalach = st.number_input("Max Heart Rate", 70, 210, 150)
exang = st.selectbox("Exercise Induced Angina", [0,1])
oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope", [0,1,2])
ca = st.selectbox("Number of Vessels (ca)", [0,1,2,3])
thal = st.selectbox("Thal", [1,2,3])

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict"):

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    # Scale numerical features (same as training)
    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)
    probability = model.predict_proba(input_data_scaled)[0][1]

    # -------------------------------
    # OUTPUT
    # -------------------------------
    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of Heart Disease\n\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Low Risk of Heart Disease\n\nProbability: {probability:.2f}")

    st.subheader("🧠 Interpretation")
    st.write("This prediction is based on clinical features like chest pain, heart rate, and ECG results.")