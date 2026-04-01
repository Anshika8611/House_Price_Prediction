import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 House Price Prediction")
st.write("Enter house details to predict price")
from model import train_model, predict
import os

MODEL_PATH = "model.pkl"

# Agar model nahi hai → auto train karo
if not os.path.exists(MODEL_PATH):
    st.warning("⚙️ Training model for first time...")
    train_model()
    st.success("✅ Model trained successfully!")

# ================================
# Load Model (if saved)
# ================================

MODEL_PATH = "model.pkl"

if os.path.exists(MODEL_PATH):
    model = pickle.load(open(MODEL_PATH, "rb"))
else:
    st.error("❌ Model not found! Please train model first.")
    st.stop()

# ================================
# User Inputs (basic features)
# ================================


overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)

gr_liv_area = st.number_input(
    "Living Area (sq ft)", 500, 5000, 1500)

garage_cars = st.slider(
    "Garage Capacity (cars)", 0, 5, 1)

total_bsmt_sf = st.number_input(
    "Basement Area (sq ft)", 0, 3000, 800)

year_built = st.number_input(
    "Year Built", 1900, 2025, 2000)

full_bath = st.slider(
    "Full Bathrooms", 0, 5, 2)

tot_rms = st.slider(
    "Total Rooms Above Ground", 2, 15, 6)

lot_area = st.number_input(
    "Lot Area (sq ft)", 1000, 100000, 8000)

# ================================
# Prediction Button
# ================================

if st.button("🔮 Predict Price"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "OverallQual": [overall_qual],
        "GrLivArea": [gr_liv_area],
        "GarageCars": [garage_cars],
        "TotalBsmtSF": [total_bsmt_sf],
        "YearBuilt": [year_built]
    })

    try:
        prediction = model.predict(input_data)[0]

        st.success(f"💰 Estimated House Price: ₹ {int(prediction):,}")

    except Exception as e:
        st.error("❌ Feature mismatch! Model needs same features as training data.")
