---
title: Age Gender Prediction
emoji: 🧑
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# FaceInsight AI — Age & Gender Prediction

A **Gradio-based computer vision demo** that detects a face in an uploaded image and returns an estimated age and a binary gender-classification label.

The goal of this project is to demonstrate an end-to-end deep-learning inference pipeline, from input validation and face detection through preprocessing, prediction, and interactive visualization.

## Features

- **Image upload:** submit a photograph through the Gradio interface
- **Automatic face detection:** OpenCV Haar cascade finds the largest detected face
- **Age prediction:** displays an estimated age in years
- **Binary classification:** maps the model's sigmoid output to a Male/Female label
- **Visual feedback:** draws a bounding box around the detected face
- **Inference details:** displays the predicted-class score and timing

## How it works

```text
Uploaded image
     ↓
OpenCV face detection (Haar cascade)
     ↓
Crop largest detected face and resize to 200 × 200
     ↓
EfficientNetB0-based multitask model
     ├── Age regression output
     └── Binary classification output
     ↓
Gradio results and annotated image
```

The inference implementation is in [`app.py`](app.py). The EfficientNetB0 model is rebuilt in code and trained weights are loaded from [`age_gender_weights.weights.h5`](age_gender_weights.weights.h5).

## Run locally

Use a Python environment compatible with [`requirements.txt`](requirements.txt):

```bash
python -m pip install -r requirements.txt
python app.py
```

Gradio will print a local address (typically `http://127.0.0.1:7860`). Upload a clear image with a frontal face and select **Predict**.

## Repository structure

```text
faceinsight-ai/
├── app.py                         # Gradio UI, face detection and inference
├── age_gender_weights.weights.h5  # Saved model weights
├── requirements.txt               # Dependencies
└── README.md                      # This documentation
```

**Note:** The model-training notebook and experimental validation logs are not included in this repository. The public app files support reviewing the inference design, but by themselves do not establish a particular test-set accuracy or mean absolute error.

## Tools

Python · TensorFlow / Keras · EfficientNetB0 · OpenCV · NumPy · Gradio

## Important limitations

- Age is an **estimate**, not a reliable measure of a person's age.
- The classifier uses **binary training labels**; its output does **not** determine someone's gender identity.
- The displayed class score is derived from model output, not independently demonstrated to be a calibrated probability.
- Face detection and predictions may be less reliable with multiple faces, poor lighting, non-frontal views, occlusion, or unusual image conditions.
- The demo selects the **largest detected face** if multiple faces are present.
- Model performance should be communicated with a held-out evaluation before citing specific metrics.

## Author

**[Aqsa Mansoor](https://github.com/AqsaMansoor32)** — Data Analytics · Machine Learning · Computer Vision
