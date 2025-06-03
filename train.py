import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from preprocess import preprocess_data

# Load dataset
df = pd.read_csv("data/rainfall.csv")  # Columns: temp, humidity, pressure, ..., Rainfall

# Preprocess
X, y, scaler = preprocess_data(df)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False)}")
print(f"R² Score: {r2_score(y_test, y_pred)}")

# Save model and scaler
joblib.dump(model, "models/rainfall_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
