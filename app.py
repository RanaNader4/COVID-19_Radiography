import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Load the trained model
import requests

model_url = "https://drive.google.com/file/d/1a8ZYKXN3aEIjffazMMj8Xlea5JNgAtek/view?usp=sharing"
model_path = "best_model.keras"

if not os.path.exists(model_path):
    with open(model_path, "wb") as f:
        f.write(requests.get(model_url).content)

# Preprocessing function
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((128, 128))
    image = np.array(image)
    if image.ndim == 2:  # grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# Streamlit UI
st.set_page_config(page_title="COVID-19 Radiography Classifier", layout="centered")
st.title("COVID-19 Radiography Image Classifier")
st.markdown("Upload a chest X-ray image, and the model will predict the condition.")

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.markdown("### Prediction")
    st.success(f"**Class:** {label_map[predicted_class]}")
    st.info(f"**Confidence:** {confidence * 100:.2f}%")
