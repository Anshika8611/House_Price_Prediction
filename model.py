import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Features used (same everywhere)
FEATURES = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "YearBuilt",
    "FullBath",
    "TotRmsAbvGrd",
    "LotArea"
]

# ================================
# Train Model
# ================================
def train_model():
    train = pd.read_csv(os.path.join(BASE_DIR, "data/train.csv"))

    train = train[FEATURES + ["SalePrice"]]

    X = train[FEATURES]
    y = train["SalePrice"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    pickle.dump(model, open("model.pkl", "wb"))


# ================================
# Predict
# ================================
def predict(input_df):
    model = pickle.load(open("model.pkl", "rb"))

    # Ensure same column order
    input_df = input_df[FEATURES]

    prediction = model.predict(input_df)

    return prediction[0]
