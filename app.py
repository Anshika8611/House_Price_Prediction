import streamlit as st
import pandas as pd

from model import load_data, preprocess, train_model, predict

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 House Price Prediction App")

st.write("Train model and generate predictions using Kaggle dataset")

if st.button("🚀 Run Model"):
    train, test = load_data()
    X, y, test_processed, test_ids = preprocess(train, test)

    model = train_model(X, y)
    submission = predict(model, test_processed, test_ids)

    st.success("✅ Prediction Done!")

    st.write("### 📊 Sample Output:")
    st.dataframe(submission.head())

    st.download_button(
        label="📥 Download Submission",
        data=submission.to_csv(index=False),
        file_name="submission.csv",
        mime="text/csv"
    )
