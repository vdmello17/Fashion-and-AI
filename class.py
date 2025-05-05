import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(layout="wide")
st.title("🧠 Fashion MNIST Image Classifier")

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cnn_model.keras")

model = load_model()

# Class labels for Fashion MNIST
class_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Preprocessing function
def preprocess(image: np.ndarray) -> np.ndarray:
    if image.shape != (28, 28):
        image = tf.image.resize(image[..., np.newaxis], (28, 28)).numpy().squeeze()
    image = image / 255.0
    image = image.reshape(1, 28, 28, 1)
    return image

# Image source
st.sidebar.header("📤 Upload Image or Use Random Sample")

uploaded_file = st.sidebar.file_uploader("Upload a 28x28 grayscale image", type=["png", "jpg", "jpeg"])

# Display & predict
if uploaded_file:
    img = Image.open(uploaded_file).convert("L")  # convert to grayscale
    img_array = np.array(img)
    input_image = preprocess(img_array)
    prediction = model.predict(input_image)[0]
    predicted_label = class_labels[np.argmax(prediction)]

    st.subheader("🖼 Uploaded Image")
    st.image(img_array, width=150, caption="Uploaded")

    st.subheader("🔮 Prediction")
    st.write(f"**Predicted Label:** {predicted_label}")
    st.bar_chart(prediction)

else:
    # Load test set if no image uploaded
    st.info("No image uploaded. Showing random test image.")
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (_, _), (x_test, y_test) = fashion_mnist.load_data()
    index = st.sidebar.slider("Pick a test image index", 0, len(x_test) - 1, 0)
    test_image = x_test[index]
    input_image = preprocess(test_image)
    prediction = model.predict(input_image)[0]
    predicted_label = class_labels[np.argmax(prediction)]

    st.subheader("🖼 Sample Test Image")
    st.image(test_image, width=150, caption=f"True Label: {class_labels[y_test[index]]}")

    st.subheader("🔮 Prediction")
    st.write(f"**Predicted Label:** {predicted_label}")
    st.bar_chart(prediction)
