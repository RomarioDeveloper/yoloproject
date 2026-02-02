import csv
import json
import os
from datetime import datetime
from threading import Lock

class DefectLogger:
    def __init__(self, log_dir="logs", log_format="csv"):
        self.log_dir = log_dir
        self.log_format = log_format
        self.lock = Lock()
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        self.current_log_file = os.path.join(
            self.log_dir, 
            f"defects_{datetime.now().strftime('%Y%m%d')}.{self.log_format}"
        )
        
        if self.log_format == "csv" and not os.path.exists(self.current_log_file):
            with open(self.current_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Defect_Type", "Confidence", "Coordinates"])

    def log_defect(self, defect_type, confidence, bbox):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = {
            "timestamp": timestamp,
            "type": defect_type,
            "confidence": f"{confidence:.2f}",
            "bbox": str(bbox)
        }
        
        with self.lock:
            if self.log_format == "csv":
                self._log_csv(data)
            elif self.log_format == "json":
                self._log_json(data)

    def _log_csv(self, data):
        with open(self.current_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                data["timestamp"], 
                data["type"], 
                data["confidence"], 
                data["bbox"]
            ])

    def _log_json(self, data):
        # Simply appending JSON objects for streaming read support
        with open(self.current_log_file, 'a') as f:
            f.write(json.dumps(data) + "\n")
