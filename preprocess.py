import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df, target_column="Rainfall"):
    df = df.dropna()
    X = df.drop(columns=[target_column])
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler
