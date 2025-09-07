import numpy as np
import PIL.Image as Image
import tensorflow as tf
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from warnings import filterwarnings
filterwarnings('ignore')


def streamlit_config():
    # Page configuration
    st.set_page_config(page_title='Classification', layout='centered')

    # Transparent header
    page_background_color = """
    <style>
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }
    </style>
    """
    st.markdown(page_background_color, unsafe_allow_html=True)

    # Title
    st.markdown('<h1 style="text-align: center;">Potato Disease Classification</h1>', unsafe_allow_html=True)
    add_vertical_space(4)


# Streamlit Configuration Setup
streamlit_config()


def prediction(uploaded_file, class_names=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']):
    # Validate file type
    if not uploaded_file.name.lower().endswith(('.jpg', '.jpeg', '.png')):
        st.error("Unsupported file type. Please upload a JPG or PNG image.")
        return

    try:
        # Try opening the image
        img = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("Invalid image file. Please upload a valid JPG or PNG.")
        return

    try:
        # Preprocess image
        img_resized = img.resize((256, 256))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Load model
        model = tf.keras.models.load_model("model.keras")

        # Predict
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = round(np.max(prediction) * 100, 2)

        # Display results
        add_vertical_space(1)
        st.markdown(
            f'<h4 style="color: orange;">Predicted Class : {predicted_class}<br>Confidence : {confidence}%</h4>',
            unsafe_allow_html=True
        )
        add_vertical_space(1)
        st.image(img.resize((400, 300)))

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# File uploader UI
col1, col2, col3 = st.columns([0.1, 0.9, 0.1])
with col2:
    input_image = st.file_uploader(label='Upload the Image', type=['jpg', 'jpeg', 'png'])

# Run prediction if image is uploaded
if input_image is not None:
    with st.spinner("Classifying..."):
        prediction(input_image)
    st.success("Prediction complete!")
