import cv2

def classify_frame(model, frame):
    # Save frame temporarily for prediction
    temp_path = "temp.jpg"
    cv2.imwrite(temp_path, frame)

    # Predict using YOLOv8 classification
    results = model.predict(source=temp_path, verbose=False)

    # Extract label and confidence
    pred_label = results[0].names[results[0].probs.top1]
    conf = results[0].probs.top1conf.item()
    label_text = f"{pred_label} ({conf:.2f})"

    return label_text
