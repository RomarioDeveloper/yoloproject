import cv2
import numpy as np
from ultralytics import YOLO
import torch

class DefectDetector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.6, target_classes=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading YOLO model on {self.device}...")
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to a lightweight model if custom path fails or download fails
            self.model = YOLO("yolov8n.pt") 
            
        self.conf_threshold = conf_threshold
        self.class_names = self.model.names
        self.target_classes = target_classes # List of class names to detect (None = all)

    def process_frame(self, frame):
        if frame is None:
            return None, []

        # Run inference
        results = self.model.predict(frame, conf=self.conf_threshold, device=self.device, verbose=False)
        result = results[0]
        
        detections = []
        annotated_frame = frame.copy()

        # Iterate through detections
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = self.class_names.get(cls_id, "Unknown")

            # Filter by target classes if set
            if self.target_classes is not None and class_name not in self.target_classes:
                continue

            detections.append({
                "type": class_name,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2)
            })

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw label
            label = f"{class_name} {conf:.2f}"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(annotated_frame, (x1, y1), c2, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(annotated_frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated_frame, detections
