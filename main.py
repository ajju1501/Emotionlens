import cv2
from utils.inference import classify_frame
from ultralytics import YOLO

# Load trained model
model = YOLO("models/best.pt")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run classification
    label_text = classify_frame(model, frame)

    # Overlay prediction
    cv2.putText(frame, label_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Display the frame
    cv2.imshow("YOLOv8 Facial Expression Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
