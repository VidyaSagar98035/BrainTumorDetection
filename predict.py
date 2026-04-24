import tensorflow as tf
import numpy as np
import cv2

MODEL_PATH = "brain_tumor_model.h5"
LABELS_PATH = "labels.txt"
IMG_SIZE = 224

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load labels
with open(LABELS_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

print("✅ Loaded Classes:", class_names)

# Input image path
img_path = input("Enter MRI Image Path: ")

# Read and preprocess image
img = cv2.imread(img_path)

if img is None:
    print("❌ Image not found. Check path!")
    exit()

img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img)
class_index = np.argmax(prediction)
confidence = np.max(prediction) * 100

print("🧠 Prediction:", class_names[class_index])
print("🎯 Confidence:", round(confidence, 2), "%")