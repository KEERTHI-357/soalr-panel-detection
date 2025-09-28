import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Load the trained model
@st.cache_resource
def load_model():
    model_path = 'solar_panel_classifier.h5'
    if not os.path.exists(model_path):
        st.error("Model file not found. Please place 'solar_panel_classifier.h5' in the same directory.")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Load the model
model = load_model()

# Define the class names
class_names = ['Bird-Drop', 'Clean', 'Dusty', 'Electrical-Damage', 'Physical-Damage', 'Snow-Covered']

st.title("Solar Panel Condition Classifier")
st.write("Upload an image of a solar panel to classify its condition.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image for prediction
    def preprocess_image(img):
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Rescale the image
        return img_array

    preprocessed_img = preprocess_image(img)

    if st.button("Classify"):
        if model is not None:
            # Make prediction
            prediction = model.predict(preprocessed_img)
            predicted_class_index = np.argmax(prediction)
            predicted_class = class_names[predicted_class_index]
            confidence = np.max(prediction) * 100

            # Display results
            st.success(f"The solar panel condition is: **{predicted_class}**")
            st.info(f"Confidence: {confidence:.2f}%")
        else:
            st.warning("Model is not loaded. Please ensure 'solar_panel_classifier.h5' is in the correct path.")