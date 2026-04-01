def preprocess(train):
    # Only selected features
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

    y = train["SalePrice"]
    X = train[features]

    return X, y
