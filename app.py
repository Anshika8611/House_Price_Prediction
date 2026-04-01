import streamlit as st
import pandas as pd
import os

from model import train_model, predict

st.set_page_config(page_title="House Price Predictor")

st.title("🏠 House Price Prediction")
st.write("Enter house details")

# ================================
# Auto Train Model
# ================================
if not os.path.exists("model.pkl"):
    st.warning("Training model for first time...")
    train_model()
    st.success("Model trained!")

# ================================
# Inputs
# ================================

overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
gr_liv_area = st.number_input("Living Area (sq ft)", 500, 5000, 1500)
garage_cars = st.slider("Garage Capacity", 0, 5, 1)
total_bsmt_sf = st.number_input("Basement Area", 0, 3000, 800)
year_built = st.number_input("Year Built", 1900, 2025, 2000)
full_bath = st.slider("Full Bathrooms", 0, 5, 2)
rooms = st.slider("Total Rooms Above Ground", 1, 15, 6)
lot_area = st.number_input("Lot Area", 1000, 20000, 8000)

# ================================
# Prediction
# ================================

if st.button("🔮 Predict Price"):

    input_data = pd.DataFrame({
        "OverallQual": [overall_qual],
        "GrLivArea": [gr_liv_area],
        "GarageCars": [garage_cars],
        "TotalBsmtSF": [total_bsmt_sf],
        "YearBuilt": [year_built],
        "FullBath": [full_bath],
        "TotRmsAbvGrd": [rooms],
        "LotArea": [lot_area]
    })

    try:
        price = predict(input_data)
        st.success(f"💰 Estimated Price: ₹ {int(price):,}")
    except Exception as e:
        st.error("❌ Something went wrong. Check feature inputs.")
