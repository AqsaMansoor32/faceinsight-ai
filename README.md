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

# Age & Gender Prediction

Predicts a person's age and gender from a photo using deep learning.

## Model

- **Architecture:** EfficientNetB0 (transfer learning, last 30 layers fine-tuned)
- **Dataset:** UTKFace (~23,700 images)
- **Tasks:** Age regression (MAE ~7-9 years) + Gender classification (~91-93% accuracy)
- **Training:** Multi-output model with early stopping, learning rate scheduling, and age-group oversampling to improve accuracy on children and teens

## How it works

1. Upload any photo containing a face
2. OpenCV detects and crops the face automatically
3. The model predicts age and gender from the cropped face

## Project structure

```
├── app.py                          # Gradio app
├── age_gender_prediction.ipynb     # Training notebook (Google Colab)
├── age_gender_efficientnetb0.keras # Trained model weights
├── requirements.txt                # Dependencies
└── README.md
```

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:7860` in your browser.

## Limitations

- Accuracy is lower on young children (under 12) due to limited training examples in UTKFace
- Works best on front-facing, well-lit photos
- Face detection may fail on extreme angles or very small faces

## Tech stack

Python · TensorFlow · Keras · EfficientNetB0 · OpenCV · Gradio
