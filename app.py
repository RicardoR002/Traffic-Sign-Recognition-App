import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from model import load_model, get_class_label

# Set page config
st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide"
)

# Load the pre-trained model
@st.cache_resource
def get_model():
    try:
        with st.spinner('Loading model... This may take a few minutes on first run.'):
            model = load_model()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please try refreshing the page. If the problem persists, contact support.")
        return None

# Preprocess image for model input
def preprocess_image(image):
    image = cv2.resize(image, (32, 32))
    image = image.astype('float32') / 255.0
    return np.expand_dims(image, axis=0)

# Main app
def main():
    st.title("🚦 Traffic Sign Recognition")
    st.write("Upload an image or use your camera to classify traffic signs")

    # Load model
    model = get_model()
    
    if model is None:
        st.stop()

    # Create two columns for input methods
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Convert PIL image to numpy array
                image_np = np.array(image)
                processed_image = preprocess_image(image_np)
                
                # Make prediction
                with st.spinner('Analyzing image...'):
                    prediction = model.predict(processed_image)
                    predicted_class = np.argmax(prediction)
                    confidence = prediction[0][predicted_class] * 100
                
                st.success(f"Predicted Traffic Sign: {get_class_label(predicted_class)}")
                st.info(f"Confidence: {confidence:.2f}%")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

    with col2:
        st.subheader("Camera Input")
        camera_input = st.camera_input("Take a picture")
        
        if camera_input is not None:
            try:
                image = Image.open(camera_input)
                st.image(image, caption="Camera Image", use_column_width=True)
                
                # Convert PIL image to numpy array
                image_np = np.array(image)
                processed_image = preprocess_image(image_np)
                
                # Make prediction
                with st.spinner('Analyzing image...'):
                    prediction = model.predict(processed_image)
                    predicted_class = np.argmax(prediction)
                    confidence = prediction[0][predicted_class] * 100
                
                st.success(f"Predicted Traffic Sign: {get_class_label(predicted_class)}")
                st.info(f"Confidence: {confidence:.2f}%")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main() 