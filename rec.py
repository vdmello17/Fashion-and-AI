import streamlit as st
import numpy as np
import h5py
import os
import gdown
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide")
st.title("👗 Fashion MNIST Recommender (Google Drive-Based)")

# File ID from Google Drive
FILE_ID = "YOUR_FILE_ID_HERE"
FILE_NAME = "fashion_data.h5"
https://drive.google.com/file/d/1pWU1Vjj0m_ZPIc0Ce_JNafBMQzkd1JcN/view?usp=sharing
DOWNLOAD_URL = f"https://drive.google.com/uc?id={1pWU1Vjj0m_ZPIc0Ce_JNafBMQzkd1JcN}"

# Download HDF5 file if not already present
if not os.path.exists(FILE_NAME):
    with st.spinner("📥 Downloading dataset from Google Drive..."):
        gdown.download(DOWNLOAD_URL, FILE_NAME, quiet=False)

@st.cache_resource
def load_data():
    f = h5py.File(FILE_NAME, "r")
    x_test = f["x_test"]
    y_test = f["y_test"]
    embeddings = f["embeddings"]
    similarity = cosine_similarity(embeddings[:].astype("float32"))
    return f, x_test, y_test, embeddings, similarity

h5_file, x_test, y_test, embeddings, similarity_matrix = load_data()

# User controls
query_index = st.sidebar.slider("Select image index", 0, len(x_test) - 1, 0)
top_n = st.sidebar.slider("Number of recommendations", 1, 10, 5)

# Show query image
st.subheader("🧩 Query Image")
st.image(x_test[query_index].reshape(28, 28), width=150, caption=f"Label: {y_test[query_index]}")

# Show recommendations
st.subheader(f"🎯 Top {top_n} Recommendations")
cols = st.columns(top_n)
similar_indices = similarity_matrix[query_index].argsort()[::-1][1:top_n + 1]

for i, idx in enumerate(similar_indices):
    with cols[i]:
        st.image(x_test[idx].reshape(28, 28), width=100, caption=f"Label: {y_test[idx]}")
