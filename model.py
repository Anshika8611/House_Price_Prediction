# ================================
# House Price Model (UI Prediction)
# ================================

import pandas as pd
import numpy as np
import os
import pickle

from sklearn.ensemble import RandomForestRegressor

# ================================
# 1. Load Data
# ================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data():
    train_path = os.path.join(BASE_DIR, "data/train.csv")
    train = pd.read_csv(train_path)
    return train


# ================================
# 2. Preprocess
# ================================

def preprocess(train):
    train = train.drop("Id", axis=1)

    y = train["SalePrice"]
    X = train.drop("SalePrice", axis=1)

    # Fill missing values
    X = X.fillna(X.median(numeric_only=True))

    # Encode categorical
    X = pd.get_dummies(X)

    return X, y


# ================================
# 3. Train Model + Save
# ================================

def train_model():
    train = load_data()
    X, y = preprocess(train)

    model = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    # Save model + columns (VERY IMPORTANT)
    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(X.columns, open("columns.pkl", "wb"))

    return model


# ================================
# 4. Predict (Single Input)
# ================================

def predict(input_df):
    # Load model
    model = pickle.load(open("model.pkl", "rb"))
    columns = pickle.load(open("columns.pkl", "rb"))

    # Preprocess input
    input_df = input_df.fillna(0)
    input_df = pd.get_dummies(input_df)

    # Align with training columns
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Prediction
    prediction = model.predict(input_df)

    return prediction[0]
