import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("autoencoder_model.h5", custom_objects={"mse": tf.keras.losses.MeanSquaredError()})

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((128, 128))
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to calculate Mean Squared Error (MSE)
def calculate_mse(original, reconstructed):
    return np.mean(np.square(original - reconstructed))

# Streamlit UI
st.title("Brain Tumor Detection using Autoencoder")

uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess image
    image = Image.open(uploaded_file).convert("RGB")
    processed_image = preprocess_image(image)

    # Get reconstructed image
    reconstructed = model.predict(processed_image)

    # Calculate MSE
    mse = calculate_mse(processed_image, reconstructed)
    threshold = 0.01  # Define a threshold (adjust based on experiments)
    
    # Display original and reconstructed images
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    with col2:
        st.image(reconstructed[0], caption="Reconstructed Image", use_column_width=True)

    # Display MSE score
    st.write(f"**Reconstruction Error (MSE):** {mse:.5f}")

    # Tumor Detection
    if mse > threshold:
        st.error("Tumor Detected! 🚨")
    else:
        st.success("No Tumor Detected ✅")
