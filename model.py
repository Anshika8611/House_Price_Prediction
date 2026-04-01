import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def train_model():
    train = pd.read_csv(os.path.join(BASE_DIR, "data/train.csv"))

    features = [
        "OverallQual",
        "GrLivArea",
        "GarageCars",
        "TotalBsmtSF",
        "YearBuilt",
        "FullBath",
        "TotRmsAbvGrd",
        "LotArea"
    ]

    train = train[features + ["SalePrice"]]

    X = train[features]
    y = train["SalePrice"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    pickle.dump(model, open("model.pkl", "wb"))

def predict(input_df):
    model = pickle.load(open("model.pkl", "rb"))
    prediction = model.predict(input_df)
    return prediction[0]
