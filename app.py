import os
import io
import base64

import numpy as np
import tensorflow as tf

from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = 299

CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

CLASS_LABELS = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary"
}

CLASS_INFO = {
    "glioma":
        "Arises from glial cells. Most common primary brain tumor. "
        "Can be low-grade or high-grade.",

    "meningioma":
        "Grows from the meninges and is commonly slow-growing.",

    "notumor":
        "No tumor class was detected by the trained model.",

    "pituitary":
        "Tumor involving the pituitary gland."
}


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "InceptionResNetV2_BrainTumor.keras"
)

print("=" * 60)
print("Brain Tumor MRI Detection")
print("=" * 60)

print(f"Loading model from:")
print(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")
print(f"Input shape: {model.input_shape}")
print("=" * 60)


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "model": "InceptionResNetV2_BrainTumor",
        "classes": CLASSES
    })


# ============================================================
# IMAGE TO BASE64
# ============================================================

def image_to_base64(image):

    buffer = io.BytesIO()

    image.save(buffer, format="PNG")

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({
            "error": "No image file uploaded."
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "error": "No file selected."
        }), 400

    try:

        # ----------------------------------------------------
        # Read uploaded image
        # ----------------------------------------------------

        image = Image.open(
            io.BytesIO(file.read())
        ).convert("RGB")

        original_width = image.width
        original_height = image.height

        # ----------------------------------------------------
        # Resize exactly like the original prediction code
        # ----------------------------------------------------

        resized_image = image.resize(
            (IMG_SIZE, IMG_SIZE)
        )

        # ----------------------------------------------------
        # Convert to NumPy
        # ----------------------------------------------------

        image_array = np.array(
            resized_image,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Normalize exactly like original code
        # ----------------------------------------------------

        image_array = image_array / 255.0

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        probabilities = model.predict(
            image_array,
            verbose=0
        )[0]

        predicted_index = int(
            np.argmax(probabilities)
        )

        predicted_class = CLASSES[
            predicted_index
        ]

        confidence = float(
            probabilities[predicted_index] * 100
        )

        # ----------------------------------------------------
        # All probabilities
        # ----------------------------------------------------

        class_probabilities = {}

        for class_name, probability in zip(
            CLASSES,
            probabilities
        ):

            class_probabilities[class_name] = round(
                float(probability * 100),
                2
            )

        # ----------------------------------------------------
        # Thumbnail
        # ----------------------------------------------------

        thumbnail = image.copy()

        thumbnail.thumbnail(
            (400, 400)
        )

        image_b64 = image_to_base64(
            thumbnail
        )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "prediction": predicted_class,

            "label": CLASS_LABELS[
                predicted_class
            ],

            "confidence": round(
                confidence,
                2
            ),

            "probabilities":
                class_probabilities,

            "info":
                CLASS_INFO[
                    predicted_class
                ],

            "image_b64":
                image_b64,

            "filename":
                file.filename,

            "original_size":
                f"{original_width}x{original_height}"
        })

    except Exception as error:

        print("Prediction error:", error)

        return jsonify({
            "error": str(error)
        }), 500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )