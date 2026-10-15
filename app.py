import streamlit as st
import numpy as np
import joblib

# Load the saved model and scaler
model = joblib.load('trained_model.sav')
scaler = joblib.load('scaler.joblib')

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")

st.title("🩺 Diabetes Prediction System")
st.write("This app predicts whether a person is diabetic or not using SVM.")

st.subheader("Enter Patient Details:")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=300, value=110)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

if st.button("Predict Diabetes"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    
    # Standardize the input
    std_data = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(std_data)
    
    st.markdown("---")
    if prediction[0] == 0:
        st.success("✅ The person is **Not Diabetic**")
    else:
        st.error("⚠️ The person is **Diabetic**")
