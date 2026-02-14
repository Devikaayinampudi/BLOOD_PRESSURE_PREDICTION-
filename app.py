import streamlit as st
import numpy as np
import pickle
import os

# ----------------------------
# Load or create model and scaler
# ----------------------------
model_file = "bp_model.pkl"
scaler_file = "scaler.pkl"

if not os.path.exists(model_file) or not os.path.exists(scaler_file):
    st.warning("Model files not found! Please train your model first.")
    st.stop()

model = pickle.load(open(model_file, "rb"))
scaler = pickle.load(open(scaler_file, "rb"))

# ----------------------------
# Streamlit Page Setup
# ----------------------------
st.set_page_config(
    page_title="💓 Blood Pressure Predictor",
    layout="centered"
)

st.title("💓 Blood Pressure Predictor")
st.markdown("Enter your details below to predict your blood pressure!")

# ----------------------------
# User Input Section
# ----------------------------
st.subheader("📝 Enter Your Details:")

age = st.slider("Age (years)", 20, 80, 30)
weight = st.slider("Weight (kg)", 40, 120, 70)
height = st.slider("Height (cm)", 140, 210, 165)
cholesterol = st.slider("Cholesterol (mg/dL)", 100, 300, 180)
glucose = st.slider("Glucose (mg/dL)", 60, 200, 90)
smoking = st.selectbox("Do you smoke? 🚬", ["No", "Yes"])
alcohol = st.selectbox("Do you drink alcohol? 🍷", ["No", "Yes"])
physical_activity = st.slider("Physical Activity (hrs/week)", 0, 10, 3)
gender = st.selectbox("Gender 👤", ["Female", "Male"])

# Fill remaining features with 0 if unused
feature10 = 0
feature11 = 0
feature12 = 0
feature13 = 0
feature14 = 0

# ----------------------------
# Prepare input for prediction
# ----------------------------
input_data = np.array([
    age,
    weight,
    height,
    cholesterol,
    glucose,
    1 if smoking == "Yes" else 0,
    1 if alcohol == "Yes" else 0,
    physical_activity,
    0 if gender == "Female" else 1,
    feature10,
    feature11,
    feature12,
    feature13,
    feature14
]).reshape(1, -1)

# Scale the input
input_scaled = scaler.transform(input_data)

# Predict BP
prediction = model.predict(input_scaled)[0]

# ----------------------------
# Display Prediction with Colors & Emojis
# ----------------------------
st.subheader("💡 Predicted Blood Pressure:")

if prediction < 120:
    st.success(f"Normal: {prediction:.1f} mmHg ✅")
elif prediction < 130:
    st.info(f"Elevated: {prediction:.1f} mmHg ⚠️")
elif prediction < 140:
    st.warning(f"High BP Stage 1: {prediction:.1f} mmHg ⚠️")
else:
    st.error(f"High BP Stage 2: {prediction:.1f} mmHg 🚨")

# ----------------------------
# Health Tips
# ----------------------------
st.markdown("---")
st.subheader("💡 Tips for Healthy Blood Pressure")
st.write("""
- Eat a balanced diet 🥗  
- Exercise regularly 🏃‍♂️  
- Avoid smoking 🚬 and limit alcohol 🍷  
- Monitor your BP regularly 🩺  
""")