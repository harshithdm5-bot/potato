import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("model.keras")

# Prediction function
def prediction(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))  # Adjust based on your model input
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    result = model.predict(image_array)
    st.image(image, caption="Input Image", use_column_width=True)
    st.write(f"Prediction: {np.argmax(result)}")  # Customize label mapping if needed

# App layout
st.set_page_config(page_title="Potato Disease Classifier", layout="centered")
st.title("🥔 Potato Disease Classifier")
st.markdown("Upload an image of a potato leaf to detect possible diseases.")

# Columns for upload and demo
col1, col2 = st.columns([3, 1])

# File uploader
with col1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        with st.spinner("Classifying uploaded image..."):
            prediction(uploaded_file.read())
            st.success("Upload prediction complete!")

# Demo button
with col2:
    if st.button("Try Demo Image"):
        with st.spinner("Classifying demo image..."):
            try:
                with open("demo.jpg", "rb") as f:  # Update path if needed
                    demo_image = f.read()
                prediction(demo_image)
                st.success("Demo prediction complete!")
            except Exception as e:
                st.error(f"Demo failed: {e}")
