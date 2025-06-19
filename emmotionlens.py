# -*- coding: utf-8 -*-

# !pip install kagglehub
# !pip install ultralytics
# !apt-get update && apt-get install libgl1 --yes

import kagglehub
path = kagglehub.dataset_download("jonathanoheix/face-expression-recognition-dataset")

print("Path to dataset files:", path)

# from ultralytics import YOLO

# Create YOLOv8 classification model
# model = YOLO("yolov8x-cls")  # Or use yolov8n-cls for lighter model

# Train model
# model.train(
#     data="/kaggle/input/face-expression-recognition-dataset/images",
#     epochs=10,
#     batch=32,
#     pretrained=True
# )

# model = YOLO("/content/runs/classify/train/weights/best.pt")

# Upload an image to test
from google.colab import files
uploaded = files.upload()
test_img_path = list(uploaded.keys())[0]

# Predict
model.predict(source=test_img_path, show=True)

from ultralytics import YOLO
from google.colab import files
import matplotlib.pyplot as plt
import cv2

# Load trained model
model = YOLO("/content/runs/classify/train/weights/best.pt")

# Upload a test image
uploaded = files.upload()
img_path = list(uploaded.keys())[0]

# Predict
results = model.predict(source=img_path)

# Get label
pred_label = results[0].names[results[0].probs.top1]  # Class name
conf = results[0].probs.top1conf.item()               # Confidence

# Load image using OpenCV
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Put label on image
label_text = f"{pred_label} ({conf:.2f})"
cv2.putText(img, label_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

# Display image with label
plt.figure(figsize=(6,6))
plt.imshow(img)
plt.axis("off")
plt.title("Predicted Expression")
plt.show()