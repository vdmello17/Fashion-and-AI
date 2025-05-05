import streamlit as st
import joblib

st.set_page_config(layout="centered")
st.title("💬 Fashion Review Sentiment Analyzer")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("sentiment_model.pkl")

model = load_model()

# Text input
st.subheader("📝 Enter a product review:")
user_input = st.text_area("Write a review about fashion, clothing, or accessories", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review first.")
    else:
        prediction = model.predict([user_input])[0]
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
        st.success(f"**Sentiment:** {sentiment}")
