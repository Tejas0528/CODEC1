# 🔢 Handwritten Digit Recognizer using MNIST (Pure Python)

## 📌 Project Overview

This project implements a **Handwritten Digit Recognition System** using the **MNIST dataset**, built completely from scratch using **Python and NumPy**. The model uses a **K-Nearest Neighbors (KNN)** classifier to identify digits (0-9) based on pixel data from 28x28 grayscale images.

No deep learning libraries like `TensorFlow`, `Keras`, or `scikit-learn` were used — the algorithm logic was coded manually for a better understanding of the ML fundamentals.


## ✅ Features

- Classifies handwritten digits (0 to 9)
- Custom-built KNN classifier using only NumPy
- Reads and processes MNIST dataset in CSV format
- Calculates accuracy based on correct predictions


## 📂 Dataset

The project uses the **CSV version of the MNIST dataset**:

- `mnist_train.csv`: Training data (label + 784 pixel values)
- `mnist_test.csv`: Testing data (label + 784 pixel values)

Each row contains:
```csv
label, pixel1, pixel2, ..., pixel784
