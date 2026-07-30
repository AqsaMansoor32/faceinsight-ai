import os
os.environ["TF_USE_LEGACY_KERAS"] = "0"

import time
import cv2
import gradio as gr
import numpy as np
from keras.applications import EfficientNetB0
from keras.applications.efficientnet import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model

# =====================================================
# Configuration
# =====================================================

WEIGHTS_PATH = "age_gender_weights.weights.h5"
IMG_SIZE = (200, 200)
MAX_AGE = 100

# =====================================================
# Rebuild Model Architecture + Load Weights
# =====================================================

def build_model():
    base = EfficientNetB0(include_top=False, input_shape=(200, 200, 3), weights=None)

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)

    age_out = Dense(128, activation="relu")(x)
    age_out = Dropout(0.2)(age_out)
    age_out = Dense(1, activation="linear", name="age")(age_out)

    gender_out = Dense(128, activation="relu")(x)
    gender_out = Dropout(0.2)(gender_out)
    gender_out = Dense(1, activation="sigmoid", name="gender")(gender_out)

    return Model(inputs=base.input, outputs=[age_out, gender_out])

model = build_model()
model.load_weights(WEIGHTS_PATH)

# =====================================================
# Haar Cascade Face Detector
# =====================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =====================================================
# Detect Face
# =====================================================

def detect_face(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )
    if len(faces) == 0:
        return None, None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return img_rgb[y:y+h, x:x+w], (x, y, w, h)

# =====================================================
# Prediction
# =====================================================

def predict(image):
    if image is None:
        return None, None, "", "", ""

    img_rgb = np.array(image.convert("RGB"))
    face, box = detect_face(img_rgb)

    if face is None:
        return (
            image,
            image,
            "No Face Detected",
            "-",
            "Please upload a clear frontal face."
        )

    x, y, w, h = box
    display = img_rgb.copy()
    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 3)

    face = cv2.resize(face, IMG_SIZE).astype(np.float32)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)

    start = time.time()
    pred_age, pred_gender = model.predict(face, verbose=0)
    elapsed = time.time() - start

    age = round(float(pred_age[0][0] * MAX_AGE))
    gender_prob = float(pred_gender[0][0])

    if gender_prob >= 0.5:
        gender = "Female"
        confidence = gender_prob
    else:
        gender = "Male"
        confidence = 1 - gender_prob

    return (
        image,
        display,
        f"{age} Years",
        gender,
        f"{confidence:.1%}\n⏱ {elapsed:.2f} sec"
    )

# =====================================================
# Interface
# =====================================================

with gr.Blocks(title="FaceInsight AI") as demo:

    gr.Markdown("""
# 🤖 FaceInsight AI

### AI-Powered Age & Gender Prediction

Upload a clear face image.

**Features**

- 🎂 Age Prediction
- 👤 Gender Prediction
- 📸 Automatic Face Detection
- ⚡ Fast Inference using EfficientNetB0
""")

    with gr.Row():
        input_image = gr.Image(type="pil", label="Upload Image")
        detected_face = gr.Image(label="Detected Face")

    predict_btn = gr.Button("🔍 Predict", variant="primary")

    with gr.Row():
        age_box = gr.Textbox(label="Estimated Age")
        gender_box = gr.Textbox(label="Gender")
        confidence_box = gr.Textbox(label="Confidence")

    predict_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[input_image, detected_face, age_box, gender_box, confidence_box]
    )

    gr.Markdown("""
---

### Built With

- TensorFlow / Keras
- EfficientNetB0
- OpenCV
- Gradio

Developed by **Aqsa Mansoor**
""")

if __name__ == "__main__":
    demo.queue().launch()
