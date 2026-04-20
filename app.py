import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from gradcam import get_gradcam_heatmap, overlay_heatmap

app = Flask(__name__)

# Folder for uploaded images
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Image size used during training
IMG_SIZE = 128

# Load trained model
model = load_model("skin_disease_model.h5")

# Class labels
class_names = [
    "Actinic Keratosis",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic Nevi",
    "Vascular Lesions"
]

# Home page
@app.route("/")
def index():
    return render_template("index.html")


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No selected file"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Image preprocessing
    img = image.load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Model prediction
    predictions = model.predict(img_array)

    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]

    # Confidence score (used as Accuracy Score)
    accuracy_score = round(np.max(predictions[0]) * 100, 2)

    # GradCAM visualization
    heatmap = get_gradcam_heatmap(model, img_array, "block5_conv4")
    gradcam_path = overlay_heatmap(heatmap, filepath)

    return render_template(
        "result.html",
        prediction=predicted_class,
        accuracy=accuracy_score,   # <-- send accuracy score
        uploaded_image=filepath,
        gradcam_image=gradcam_path
    )


if __name__ == "__main__":
    app.run(debug=True)