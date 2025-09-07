import numpy as np
import PIL.Image as Image
import tensorflow as tf
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from warnings import filterwarnings
filterwarnings('ignore')


def streamlit_config():
    st.set_page_config(page_title='Classification', layout='centered')

    page_background_color = """
    <style>
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }
    </style>
    """
    st.markdown(page_background_color, unsafe_allow_html=True)

    st.markdown('<h1 style="text-align: center;">Potato Disease Classification</h1>', unsafe_allow_html=True)
    add_vertical_space(4)


# Streamlit Configuration Setup
streamlit_config()


def prediction(uploaded_file, class_names=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']):
    # Debug info
    st.write("📄 File name:", uploaded_file.name)
    st.write("📦 File type:", uploaded_file.type)

    # Validate MIME type
    if uploaded_file.type not in ["image/jpeg", "image/png"]:
        st.error("Unsupported file type. Please upload a valid JPG or PNG image.")
        return

    try:
        img = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("Invalid image file. It may be corrupted or incomplete.")
        return

    try:
        img_resized = img.resize((256, 256))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        model = tf.keras.models.load_model("model.keras")

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = round(np.max(prediction) * 100, 2)

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

if input_image is not None:
    with st.spinner("Classifying..."):
        prediction(input_image)
    st.success("Prediction complete!")
