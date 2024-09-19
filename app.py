import streamlit as st
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import io
import os
import random

def load_model():
    model_dir = 'saved_model'
    return tf.saved_model.load(model_dir)

model = load_model()

def load_image(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    return img

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

def perform_style_transfer(content_image, style_image):
    content_image = np.array(content_image)
    style_image = np.array(style_image)

    content_image = tf.image.convert_image_dtype(content_image, tf.float32)
    content_image = tf.expand_dims(content_image, axis=0)

    style_image = tf.image.convert_image_dtype(style_image, tf.float32)
    style_image = tf.expand_dims(style_image, axis=0)

    stylized_image = model(content_image, style_image)[0]

    return tensor_to_image(stylized_image)

def get_random_image(image_type):
    img_folder = 'imgs'
    image_files = [f for f in os.listdir(img_folder) if f.startswith(image_type) and f.endswith(('jpg', 'jpeg', 'png'))]
    if image_files:
        return os.path.join(img_folder, random.choice(image_files))
    return None

def image_to_bytes(image):
    """Convert PIL Image to bytes."""
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    return img_bytes.getvalue()

st.title("Style Transfer using TensorFlow Hub")
st.write("Upload content and style images, or use default images from the 'imgs' folder.")

use_default_images = st.checkbox("Use default images from folder")

if 'content_image_path' not in st.session_state:
    st.session_state.content_image_path = get_random_image('content_')
if 'style_image_path' not in st.session_state:
    st.session_state.style_image_path = get_random_image('style_')

if use_default_images:
    if st.button("Change Images"):
        st.session_state.content_image_path = get_random_image('content_')
        st.session_state.style_image_path = get_random_image('style_')

    if st.session_state.content_image_path and st.session_state.style_image_path:
        content_image = load_image(st.session_state.content_image_path)
        style_image = load_image(st.session_state.style_image_path)

        st.image([content_image, style_image], caption=["Content Image", "Style Image"], width=300)
        
        if st.button("Stylize"):
            with st.spinner("Processing..."):
                stylized_image = perform_style_transfer(content_image, style_image)
                st.image(stylized_image, caption="Stylized Image", use_column_width=True)
                
                stylized_image_bytes = image_to_bytes(stylized_image)
                st.download_button(
                    label="Download Stylized Image",
                    data=stylized_image_bytes,
                    file_name="stylized_image.png",
                    mime="image/png"
                )
    else:
        st.warning("Default images not found in the 'imgs' folder.")
else:
    content_image_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
    style_image_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

    if content_image_file and style_image_file:
        content_image = load_image(content_image_file)
        style_image = load_image(style_image_file)

        st.image([content_image, style_image], caption=["Content Image", "Style Image"], width=300)

        if st.button("Stylize"):
            with st.spinner("Processing..."):
                stylized_image = perform_style_transfer(content_image, style_image)
                st.image(stylized_image, caption="Stylized Image", use_column_width=True)
                
                stylized_image_bytes = image_to_bytes(stylized_image)
                st.download_button(
                    label="Download Stylized Image",
                    data=stylized_image_bytes,
                    file_name="stylized_image.png",
                    mime="image/png"
                )
