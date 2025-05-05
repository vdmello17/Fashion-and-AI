import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import gdown
import os
from PIL import Image

# --- Utility: Download from Google Drive ---
def download_if_not_exists(url, filename):
    if not os.path.exists(filename):
        gdown.download(url, filename, quiet=False)

# --- Replace these with your actual Drive file IDs ---
cnn_model_url = "https://drive.google.com/uc?id=YOUR_CNN_MODEL_ID"
sentiment_model_url = "https://drive.google.com/uc?id=YOUR_SENTIMENT_MODEL_ID"
arima_forecast_url = "https://drive.google.com/uc?id=YOUR_FORECAST_CSV_ID"

# --- Download necessary files ---
download_if_not_exists(cnn_model_url, "cnn_model.keras")
download_if_not_exists(sentiment_model_url, "sentiment_model.pkl")
download_if_not_exists(arima_forecast_url, "arima_forecast.csv")

# --- Load models and data ---
cnn_model = tf.keras.models.load_model("cnn_model.keras", compile=False)

import joblib
sentiment_model = joblib.load("sentiment_model.pkl")
forecast_df = pd.read_csv("arima_forecast.csv")

# --- Streamlit App ---
st.set_page_config(page_title="AI in Fashion", layout="wide")
st.title("👗 AI in Fashion: Trends, Classification & Sentiment")

# --- Section 1: Image Classification ---
st.header("1. Fashion Image Classification")
uploaded_file = st.file_uploader("Upload a fashion image (28x28 grayscale)", type=["jpg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("L").resize((28, 28))
    st.image(image, caption="Uploaded Image", use_column_width=False)
    img_array = np.array(image).reshape(1, 28, 28, 1) / 255.0
    prediction = cnn_model.predict(img_array)
    predicted_class = int(np.argmax(prediction))
    st.success(f"Predicted Class Index: **{predicted_class}**")

# --- Section 2: Review Sentiment Analysis ---
st.header("2. Sentiment Analysis of a Fashion Review")
user_review = st.text_area("Type a customer review:")
if user_review:
    sentiment = sentiment_model.predict([user_review])[0]
    label = "Positive 😊" if sentiment == 1 else "Negative 😞"
    st.write(f"Predicted Sentiment: **{label}**")

# --- Section 3: Trend Forecasting ---
st.header("3. Fashion Trend Forecast (ARIMA)")
if st.button("Show Trend Forecast"):
    st.line_chart(forecast_df.values.flatten())
    st.caption("Synthetic fashion demand forecast using ARIMA model.")
