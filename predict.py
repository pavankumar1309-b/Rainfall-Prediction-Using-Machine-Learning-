import joblib
import pandas as pd

def predict_rainfall(input_dict):
    model = joblib.load("models/rainfall_model.pkl")
    scaler = joblib.load("models/scaler.pkl")

    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    return prediction[0]

if __name__ == "__main__":
    sample_input = {
        "temperature": 25,
        "humidity": 80,
        "pressure": 1012,
        "wind_speed": 10
    }
    print(f"Predicted Rainfall (mm): {predict_rainfall(sample_input)}")
