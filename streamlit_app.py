import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load("model.pkl")

st.title("Logistic Regression Predictor")

# User input fields
f1 = st.number_input("Enter Feature 1")
f2 = st.number_input("Enter Feature 2")

if st.button("Predict"):
    prediction = model.predict(np.array([[f1, f2]]))
    st.success(f"Prediction: {prediction[0]}")
