import os
import numpy as np
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import base64

app = Flask(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Models", "model.h5")
IMG_SIZE   = (128, 128)   # Corrected from 224 based on model config

CLASS_LABELS = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
CONFIDENCE_THRESHOLD = 60.0

CLASS_INFO = {
    "Glioma": {
        "description": "Gliomas are tumors that arise from glial cells in the brain or spine.",
        "severity": "high", "icon": "🔴",
    },
    "Meningioma": {
        "description": "Meningiomas originate in the meninges — the protective membranes surrounding the brain.",
        "severity": "medium", "icon": "🟡",
    },
    "No Tumor": {
        "description": "No signs of a brain tumor were detected. Always consult a professional.",
        "severity": "none", "icon": "🟢",
    },
    "Pituitary": {
        "description": "Pituitary tumors develop in the pituitary gland at the base of the brain.",
        "severity": "medium", "icon": "🟡",
    },
}

# ─── Custom Layer Fix for Keras 3 Compatibility ───────────────────────────────
class FixedDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

class FixedInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, *args, **kwargs):
        kwargs.pop('optional', None)
        if 'batch_shape' in kwargs and 'input_shape' not in kwargs:
            kwargs['input_shape'] = kwargs.pop('batch_shape')[1:]
        super().__init__(*args, **kwargs)

# ─── Load model once at startup ───────────────────────────────────────────────
print(f"[INFO] Loading model from: {MODEL_PATH}")
try:
    custom_objects = {'Dense': FixedDense, 'InputLayer': FixedInputLayer}
    model = load_model(MODEL_PATH, compile=False, custom_objects=custom_objects)
    print("[INFO] Model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    model = None

# ─── Helpers ──────────────────────────────────────────────────────────────────
def preprocess_image(img_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def image_to_base64(img_bytes: bytes) -> str:
    encoded = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template("index.html", error="Model not loaded. Check server logs.")
    if "image" not in request.files:
        return render_template("index.html", error="No image file provided.")
    file = request.files["image"]
    if file.filename == "":
        return render_template("index.html", error="No file selected.")
    img_bytes = file.read()
    try:
        img_array = preprocess_image(img_bytes)
        predictions = model.predict(img_array)[0]
        predicted_idx = int(np.argmax(predictions))
        predicted_label = CLASS_LABELS[predicted_idx]
        confidence = float(predictions[predicted_idx]) * 100
        result = {
            "prediction": predicted_label,
            "confidence": round(confidence, 2),
            "low_confidence": confidence < CONFIDENCE_THRESHOLD,
            "class_probs": [
                {"label": CLASS_LABELS[i], "prob": round(float(predictions[i]) * 100, 2)}
                for i in range(len(CLASS_LABELS))
            ],
            "info": CLASS_INFO[predicted_label],
            "image_data": image_to_base64(img_bytes),
        }
        return render_template("index.html", result=result)
    except Exception as e:
        return render_template("index.html", error=f"Processing error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
