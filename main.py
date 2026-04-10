import streamlit as st
import pickle
import numpy as np
import os

# Page config
st.set_page_config(page_title="Churn Prediction", page_icon="📊")

st.title("📊 Telecom Churn Prediction App")

# Load model safely
model_path = os.path.join("model", "churn_model.pkl")

if not os.path.exists(model_path):
    st.error("❌ Model file not found! Please train and save the model first.")
else:
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    st.success("✅ Model loaded successfully!")

    # User Inputs
    st.header("Enter Customer Details")

    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
    total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

    # Prediction button
    if st.button("Predict Churn"):

        input_data = np.array([[tenure, monthly_charges, total_charges]])

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("⚠️ Customer is likely to CHURN")
        else:
            st.success("✅ Customer is NOT likely to churn")

            