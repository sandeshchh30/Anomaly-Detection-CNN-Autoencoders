# Anomaly Detection

## Overview
This project focuses on detecting anomalies in brain MRI images using **Convolutional Autoencoders (CNN-based Autoencoders)**. The model learns to reconstruct normal brain images and detects anomalies based on reconstruction loss, making it useful for identifying brain tumors.

## Features
- Uses **Autoencoder-based Anomaly Detection**
- Trained on **Brain MRI Dataset**
- Built with **TensorFlow & Keras**
- Deployed using **Streamlit** for easy user interaction
- App link - [Anomaly Detection](https://anomaly-detection-cnn-autoencoders-xm9tx6yjzshvgrgfedn6bx.streamlit.app)

## Dataset
Dataset: [Brain MRI Images for Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
- **Normal** and **Anomalous (Tumor)** images
- Preprocessed to **128x128 resolution**

## Installation
### **1. Clone the Repository**
```bash
git clone https://github.com/sandeshchh30/Anomaly-Detection-CNN-Autoencoders.git
cd Anomaly-Detection-CNN-Autoencoders
```
### **2. Create a Virtual Environment**
```bash
conda create --name venv python=3.9 -y
conda activate venv
```
### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

## Running the Streamlit App
```bash
streamlit run app.py
```

## Results
- The model successfully identifies brain tumors based on reconstruction error.
- Higher error = Higher anomaly score, indicating a possible tumor.

## Future Improvements
- Improve model performance with more layers and hyperparameter tuning.
- Implement Grad-CAM visualization for explainability.
- Optimize for real-time inference.
