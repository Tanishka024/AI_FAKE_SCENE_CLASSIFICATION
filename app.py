import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from utils import preprocess_image

st.set_page_config(page_title="AI Fake Scene Detection", layout="centered")

model = tf.keras.models.load_model("model/fake_scene_model.h5")

st.title("AI-Based Fake Scene Image Classification")
st.write("Upload an image to check whether it is **REAL** or **FAKE**")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)[0][0]

    confidence = prediction if prediction > 0.5 else 1 - prediction
    confidence = round(confidence * 100, 2)

    if prediction > 0.5:
        st.error(f"⚠️ FAKE IMAGE DETECTED\nConfidence: {confidence}%")
    else:
        st.success(f"✅ REAL IMAGE DETECTED\nConfidence: {confidence}%")
