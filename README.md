# 🧠 NeuroScan: Advanced Brain Tumor Classifier

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

NeuroScan is a state-of-the-art medical imaging application that utilizes Deep Learning to detect and classify brain tumors from MRI scans. It provides healthcare professionals and researchers with a fast, reliable, and highly visual preliminary analysis tool.

---

## 🌟 Live Demo
🔗 **[Live Application Link will be here]**

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
- **Developer-Centric Architecture**:
  - Legacy model compatibility (Keras 2 to Keras 3 auto-patching).
  - Production-ready with Gunicorn support.

## 🛠️ Architecture & Tech Stack

| Component | Technology |
| :--- | :--- |
| **Model** | Convolutional Neural Network (CNN) / VGG16 based |
| **Backend** | Flask (Python) |
| **Deep Learning** | TensorFlow & Keras |
| **Image Processing** | PIL (Pillow) & NumPy |
| **Frontend** | HTML5, CSS3 (Vanilla), JavaScript |
| **Deployment** | Render / Gunicorn |

## 📥 Installation & Setup

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

3. **Run Locally**
   ```bash
   python main.py
   ```
   *Access at `http://127.0.0.1:5000`*

## 🧬 Model Insights

The underlying model uses a deep architecture optimized for MRI textures. It processes images at **128x128** resolution and outputs soft-max probabilities across all classes. To handle modern environment mismatches, the app includes a custom layer deserializer to ensure the legacy weights load perfectly.

---

## 🤝 Contributing
Contributions are welcome! If you have suggestions for model improvement or UI enhancements, feel free to fork this repo and submit a PR.

## 📜 Disclaimer
*This tool is for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a physician for any questions regarding a medical condition.*

---
Created with ❤️ by [Sanskar Rastogi](https://github.com/ansh2929rastogi)