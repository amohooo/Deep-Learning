# 🧠 Handwritten Digit Classification Using SVHN and Deep Learning

## 📘 Project Overview

This project focuses on building and evaluating **machine learning and deep learning models** to classify digits from the **Street View House Numbers (SVHN)** dataset — a real-world alternative to the MNIST dataset. It explores traditional machine learning using **SVM**, and advanced neural network-based approaches using **VGG-style convolutional neural networks (CNNs)**.

---

## 🎯 Objectives

- Load and preprocess SVHN image data from `.mat` files
- Explore data through visual inspection and shape transformations
- Train and evaluate:
  - 🧮 A **Support Vector Machine (SVM)** using vectorized image features
  - 🧠 A **VGG-style CNN** with and without data augmentation
- Track and compare:
  - Accuracy and classification performance
  - Training time and inference time
- Visualize model performance with confusion matrices and loss/accuracy plots

---

## 🧠 What I Learned

Through this project, I gained practical experience in:

- 🗂 Data loading and preprocessing (reshaping, grayscale conversion, normalization)
- 📉 Dimensionality reduction via image resizing
- 🔍 Implementing **Support Vector Machine (SVM)** classification using raw pixel vectors
- 🧪 Building, training, and validating a **deep convolutional neural network**
- ⚙️ Applying **data augmentation** to improve generalization
- ⏱ Measuring and comparing **training and inference times**
- 📊 Evaluating models using confusion matrices, F1 scores, and classification reports
- 🧱 Designing CNN architectures using `Keras` and `TensorFlow`

---

## 🛠️ Tools & Technologies

- **Programming Language**: Python
- **Libraries Used**:
  - `TensorFlow`, `Keras` – Deep learning
  - `scikit-learn` – SVM, evaluation metrics
  - `OpenCV` – Grayscale conversion
  - `Seaborn`, `Matplotlib` – Visualization
  - `NumPy`, `SciPy` – Data manipulation
  - `cv2`, `loadmat` – Image and `.mat` file handling

---

## 🔍 Model Summary

### 🔹 Support Vector Machine (SVM)
- Vectorized RGB images into flat feature vectors
- Trained a linear SVM using `sklearn`
- Evaluated on both training and test sets
- Visualized performance via confusion matrix and classification report

### 🔹 VGG-Style CNN (Without Augmentation)
- Custom architecture using Conv2D, MaxPooling, Dropout, and Dense layers
- Used categorical cross-entropy and accuracy metrics
- Class-weight balancing to handle imbalanced classes
- Trained with early stopping

### 🔹 VGG-Style CNN (With Data Augmentation)
- Applied random flipping, rotation, zoom, contrast, and brightness transformations
- Used `tf.data.Dataset` API for efficient loading and augmentation
- Demonstrated improved generalization and performance on unseen data

---

## 🧪 Evaluation & Results

All models were evaluated using:

- ✅ Accuracy on both training and test sets
- 📊 Confusion matrices (normalized)
- 📈 F1-score (macro and weighted)
- 📝 Classification reports
- ⏱ Training and inference times (in seconds)

Visual outputs include:
- Confusion matrix heatmaps
- Training/validation accuracy and loss curves
- Grids of classified images with labels

---

## 🚀 How to Run

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install numpy scipy pandas matplotlib seaborn opencv-python tensorflow scikit-learn
3. Run the SVM and VGG model scripts to reproduce training and evaluation

