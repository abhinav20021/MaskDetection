# 😷 Face Mask Detection using CNN

A deep learning-based image classification project to detect whether a person is **wearing a face mask or not**, using **TensorFlow** and **Keras CNNs**. The model is trained on thousands of labeled images and achieves high accuracy with real-time prediction capabilities.

---

## 📌 Overview

This project involves:
- Building a **Convolutional Neural Network (CNN)** from scratch
- **Classifying images** as either "With Mask" or "Without Mask"
- Training on **resized 128x128 RGB images**
- Using **Dropout layers** to prevent overfitting
- Evaluating model accuracy and visualizing training history
- Predicting on custom images with OpenCV

---


## 🗂️ Dataset

The dataset contains two categories:
- `with_mask`: 3725 images
- `without_mask`: 3828 images

Each image is resized to **128x128** and converted to **RGB format** before training.

---

## 🔧 How it Works

1. **Data Preprocessing**
   - Load and resize images to 128x128
   - Convert to NumPy arrays
   - Normalize image pixel values

2. **Model Architecture**
   - 2 × Conv2D + MaxPooling layers
   - Flatten → Dense(128) → Dropout → Dense(64) → Dropout
   - Final Dense layer with sigmoid activation for binary classification

3. **Training**
   - Loss Function: `sparse_categorical_crossentropy`
   - Optimizer: `Adam`
   - Metric: Accuracy
   - Epochs: 5

4. **Evaluation**
   - Evaluate on test split
   - Visualize loss and accuracy curves

5. **Prediction**
   - Accepts image path from user
   - Shows prediction result: Mask or No Mask

---

## 📊 Results

- 📈 Achieved high accuracy on both training and validation sets
- 📷 Can detect mask status in any custom image (128x128)

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-High-success?style=for-the-badge&logo=google" />
</p>

---

