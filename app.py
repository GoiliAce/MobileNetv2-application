import streamlit as st
import tensorflow as tf
import pathlib
from rembg import remove
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np

fruit_classes = ['chuoi', 'oi', 'quyt', 'tao', 'xoai']
# load the model
model = tf.keras.models.load_model('model.h5')

def preprocess_image(image):
    image = image.resize((224, 224))  # Resize the image to match the model's input size
    image = remove(image)
    
    
    image = np.array(image)  # Convert PIL Image to NumPy array
    image = image[:, :, :3]
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit app
def main():
    st.title("Image Classification App")

    uploaded_image = st.camera_input("Take a picture")

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image")

        # Preprocess the image and make a prediction
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_class = fruit_classes[np.argmax(predictions)]

        # Display the prediction
        st.subheader("Prediction:")
        st.write(f"The uploaded image is classified as: {predicted_class}")
        st.write(f"Confidence: {np.max(predictions)}")

if __name__ == "__main__":
    main()