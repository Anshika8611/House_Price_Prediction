import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def load_data():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    return train, test

def preprocess(train, test):
    test_ids = test["Id"]

    train.drop("Id", axis=1, inplace=True)
    test.drop("Id", axis=1, inplace=True)

    y = train["SalePrice"]
    X = train.drop("SalePrice", axis=1)

    X = X.fillna(X.median(numeric_only=True))
    test = test.fillna(test.median(numeric_only=True))

    X = pd.get_dummies(X)
    test = pd.get_dummies(test)

    X, test = X.align(test, join="left", axis=1, fill_value=0)

    return X, y, test, test_ids

def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict(model, test, test_ids):
    preds = model.predict(test)

    submission = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": preds
    })

    submission.to_csv("submission.csv", index=False)
    return submission
