#!/usr/bin/env python3

import torch
import numpy as np
import cv2
import tensorflow as tf

# YOLOv10n inference class
class YoloV10nDetector:
    def __init__(self, model_path, class_names):
        model = tf.saved_model.load(model_path)
        self.infer = model.signatures["serving_default"]
        
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model = torch.load(model_path, map_location=self.device)
        # self.model.eval()
        self.class_names = class_names

    def preprocess(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (640, 640))
        img_transposed = img_resized.transpose(2, 0, 1)
        img_normalized = img_transposed / 255.0
        img_tensor = torch.from_numpy(img_normalized).float().to(self.device)
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor

    def postprocess(self, preds, confidence_threshold=0.5):
        detections = []
        for pred in preds:
            scores = pred[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id] * pred[4]
            if confidence > confidence_threshold:
                x1, y1, x2, y2 = map(int, pred[:4])
                detections.append({
                    'box': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': self.class_names[class_id]
                })
        return detections

    def detect(self, image):
        img_tensor = self.preprocess(image)
        preds = self.infer(tf.constand(img_tensor))
        # with torch.no_grad():
        #     preds = self.model(img_tensor)[0]
        detections = self.postprocess(preds)
        return detections

# Main function to test YOLOv10n on a single image
if __name__ == "__main__":
    # Set the path to the model and the class names
    model_path = "src/rock_yolo_model.pt"
    class_names = ['DANGER_ROCK', 'SAFE_ROCK']

    # Create a detector instance
    detector = YoloV10nDetector(model_path, class_names)

    # Load an image
    image_path = "src/frame_3714_jpg.rf.7235131bcbe324c35fd5c50373af1632.jpg"
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        exit(1)

    # Perform detection
    detections = detector.detect(image)

    # Check for dangerous rocks
    dangerous_rock = any(d['class_name'] == 'DANGER_ROCK' and d['confidence'] > 0.4 for d in detections)

    # Output results
    if dangerous_rock:
        print("Dangerous rock detected!")
    else:
        print("No dangerous rocks detected.")

    # Optionally, draw detections on the image and display/save
    for d in detections:
        x1, y1, x2, y2 = d['box']
        label = f"{d['class_name']} ({d['confidence']:.2f})"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image with detections
    cv2.imshow("Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
