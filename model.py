# ================================
# House Price Model (Fixed Version)
# ================================

import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor

# ================================
# 1. Load Data (FIXED PATH)
# ================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data():
    train_path = os.path.join(BASE_DIR, "data/train.csv")
    test_path = os.path.join(BASE_DIR, "data/test.csv")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    return train, test


# ================================
# 2. Preprocessing
# ================================

def preprocess(train, test):
    test_ids = test["Id"]

    # Drop Id
    train = train.drop("Id", axis=1)
    test = test.drop("Id", axis=1)

    # Target
    y = train["SalePrice"]
    X = train.drop("SalePrice", axis=1)

    # Fill missing values
    X = X.fillna(X.median(numeric_only=True))
    test = test.fillna(test.median(numeric_only=True))

    # Encode categorical
    X = pd.get_dummies(X)
    test = pd.get_dummies(test)

    # Align columns
    X, test = X.align(test, join="left", axis=1, fill_value=0)

    return X, y, test, test_ids


# ================================
# 3. Train Model
# ================================

def train_model(X, y):
    model = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model


# ================================
# 4. Predict + Save
# ================================

def predict(model, test, test_ids):
    preds = model.predict(test)

    submission = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": preds
    })

    # Save file
    submission_path = os.path.join(BASE_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)

    return submission
