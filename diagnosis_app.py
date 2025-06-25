import streamlit as st
import numpy as np
import pickle
from model_training import Node

# Load models and scalers
with open("trained_nodes.pkl", "rb") as f:
    nodes, scalers = pickle.load(f)

hospital_map = {
    "Heart Hospital": {
        "node": nodes[0],
        "scaler": scalers[0],
        "features": [
            "age", "sex", "resting_bp", "cholesterol",
            "max_heart_rate", "fasting_blood_sugar", "exercise_angina"
        ]
    },
    "Kidney Hospital": {
        "node": nodes[1],
        "scaler": scalers[1],
        "features": [
            "age", "blood_pressure", "specific_gravity", "albumin",
            "serum_creatinine", "hemoglobin", "diabetes_mellitus"
        ]
    },
    "Lung Hospital": {
        "node": nodes[2],
        "scaler": scalers[2],
        "features": [
            "age", "smoking_years", "pack_years", "cough_severity",
            "fev1_percent", "chest_pain_level", "exposure_to_pollution"
        ]
    }
}

st.title("🩺 Disease Predictor")

# Dropdown to select hospital
hospital = st.selectbox("Choose Hospital Type", ["Select a Hospital"] + list(hospital_map.keys()))

# 🧠 Only show further inputs if a valid hospital is selected
if hospital != "Select a Hospital":
    st.subheader(f"Enter Patient Details for {hospital}:")
    inputs = []
    for feature in hospital_map[hospital]["features"]:
        val = st.number_input(f"{feature.replace('_', ' ').title()}", step=0.1)
        inputs.append(val)

    if st.button("Predict"):
        X = np.array(inputs).reshape(1, -1)
        scaler = hospital_map[hospital]["scaler"]
        X_scaled = scaler.transform(X)
        clf = hospital_map[hospital]["node"].get_predictor()
        prediction = clf.predict(X_scaled)[0]
        result = "✅ No Disease" if prediction == 0 else "⚠️ Disease Detected"
        st.success(f"Prediction: {result}")
