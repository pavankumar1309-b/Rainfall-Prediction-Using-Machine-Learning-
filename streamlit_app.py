import streamlit as st
from src.predict import predict_rainfall

st.title("🌧️ Rainfall Prediction App")

temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 80.0)
pressure = st.number_input("Pressure (hPa)", 900.0, 1100.0, 1012.0)
wind_speed = st.number_input("Wind Speed (km/h)", 0.0, 100.0, 10.0)

if st.button("Predict"):
    input_data = {
        "temperature": temperature,
        "humidity": humidity,
        "pressure": pressure,
        "wind_speed": wind_speed
    }
    prediction = predict_rainfall(input_data)
    st.success(f"Predicted Rainfall: **{prediction:.2f} mm**")
