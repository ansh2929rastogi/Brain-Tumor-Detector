# 🧠 NeuroScan: Advanced Brain Tumor Classifier

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange.svg)](https://www.tensorflow.org/)

NeuroScan is a state-of-the-art medical imaging application that utilizes Deep Learning to detect and classify brain tumors from MRI scans. It provides a fast, reliable, and highly visual analysis tool using a custom-trained CNN.

---

## 🚀 Key Features

- **Multi-Class Detection**: Precisely identifies 4 distinct categories:
  - **Glioma**: Tumors arising from glial cells.
  - **Meningioma**: Tumors in the protective membranes.
  - **Pituitary**: Glandular tumors at the brain base.
  - **No Tumor**: Healthy brain scan verification.
- **Premium User Experience**: 
  - Interactive **Glassmorphism** UI.
  - Real-time probability visualization via dynamic charts.
  - Intelligent confidence-scoring system.
- **Custom Keras Compatibility**: Auto-patches legacy model weights to work with modern Keras 3 environments.

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Model** | Convolutional Neural Network (CNN) |
| **Backend** | Flask (Python) |
| **Deep Learning** | TensorFlow & Keras |
| **Image Processing** | PIL (Pillow) & NumPy |
| **Frontend** | HTML5, CSS3, JavaScript |

## 📥 Local Installation & Setup

1. **Clone the Project**
   ```bash
   git clone https://github.com/ansh2929rastogi/Brain-Tumor-Detector.git
   cd Brain-Tumor-Detector
   ```

2. **Environment Setup**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   python main.py
   ```
   *Open your browser and navigate to `http://127.0.0.1:5000`*

## 🧬 Model Details

The model processes images at **128x128** resolution. It includes custom layer wrappers (`FixedDense`, `FixedInputLayer`) to ensure seamless loading of models trained in legacy Keras versions on newer TensorFlow systems.

---

## 📜 Disclaimer
*This tool is for educational purposes only. Always consult a medical professional for clinical diagnosis.*

---
Created by [Sanskar Rastogi](https://github.com/ansh2929rastogi)