import cv2
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QTextEdit, QGroupBox)
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QObject
from PyQt6.QtGui import QPixmap, QImage

from core.detector import DefectDetector
from core.video_stream import VideoThread
from core.conveyor import ConveyorController
from utils.logger import DefectLogger

class InferenceWorker(QObject):
    finished = pyqtSignal()
    result_ready = pyqtSignal(object, list) # frame, detections
    defect_detected_signal = pyqtSignal(str, object) # Сигнал: (тип_дефекта, кадр_с_рамкой)

    def __init__(self, detector):
        super().__init__()
        self.detector = detector

    @pyqtSlot(object)
    def process_frame(self, frame):
        annotated_frame, detections = self.detector.process_frame(frame)
        self.result_ready.emit(annotated_frame, detections)
        
        # Проверяем наличие целевых классов для остановки
        for d in detections:
            # Если нашли дефект - отправляем сигнал С КАДРОМ
            self.defect_detected_signal.emit(d['type'], annotated_frame)
            break


class MainWindow(QMainWindow):
    request_inference = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RollScan - Defect Detection System")
        self.resize(1200, 800)

        # Initialize core components
        # Определяем список классов-дефектов (для теста: телефон, бутылка, машина)
        # Все остальные классы (например, человек) будут игнорироваться визуально и логически.
        self.defect_classes = ['cell phone', 'bottle', 'cup', 'car']
        
        self.detector = DefectDetector(
            model_path="yolov8n.pt", 
            target_classes=self.defect_classes
        ) 
        self.logger = DefectLogger()
        self.conveyor = ConveyorController()
        
        # State
        self.processing_in_progress = False
        self.conveyor_stopped_due_to_defect = False
        self.last_log_time = {} # Словарь для хранения времени последнего лога класса: {"person": time.time()}
        
        # UI Components Setup
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Left Side - Video Feed
        self.video_group = QGroupBox("Live Feed")
        self.video_layout = QVBoxLayout()
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setText("Camera Offline")
        self.image_label.setStyleSheet("background-color: #222; color: #eee; font-size: 16px;")
        self.image_label.setMinimumSize(640, 480)
        self.video_layout.addWidget(self.image_label)
        self.video_group.setLayout(self.video_layout)

        # Right Side - Controls & Stats
        self.stats_group = QGroupBox("Controls & Statistics")
        self.stats_layout = QVBoxLayout()
        
        self.start_btn = QPushButton("Start System")
        self.start_btn.setCheckable(True)
        self.start_btn.clicked.connect(self.toggle_system)
        self.start_btn.setStyleSheet("padding: 10px; font-size: 14px; font-weight: bold;")
        
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setPlaceholderText("Defect logs will appear here...")
        
        self.stats_layout.addWidget(self.start_btn)
        self.stats_layout.addWidget(QLabel("Recent Detections:"))
        self.stats_layout.addWidget(self.log_display)
        self.stats_group.setLayout(self.stats_layout)

        # Add to main layout
        self.main_layout.addWidget(self.video_group, stretch=3)
        self.main_layout.addWidget(self.stats_group, stretch=1)

        # Threading Setup
        self.video_thread = None
        self.inference_thread = QThread()
        self.inference_worker = InferenceWorker(self.detector)
        self.inference_worker.moveToThread(self.inference_thread)
        
        # Connect signals
        self.request_inference.connect(self.inference_worker.process_frame)
        self.inference_worker.result_ready.connect(self.update_ui)
        self.inference_worker.defect_detected_signal.connect(self.handle_defect_stop)
        self.inference_thread.start()

    def toggle_system(self):
        if self.start_btn.isChecked():
            self.start_btn.setText("Stop System")
            self.conveyor_stopped_due_to_defect = False
            self.conveyor.start()
            self.start_video()
            self.image_label.setStyleSheet("background-color: #222; color: #eee;") # Reset style
        else:
            self.start_btn.setText("Start System")
            self.stop_video()
            self.conveyor.stop()

    def handle_defect_stop(self, defect_type, evidence_frame):
        """Вызывается, когда worker нашел критический дефект"""
        if not self.conveyor_stopped_due_to_defect:
            self.conveyor_stopped_due_to_defect = True
            self.conveyor.stop()
            
            # 1. Принудительно показываем кадр-улику (evidence_frame)
            # Это гарантирует, что мы увидим именно то, что увидела нейросеть
            if evidence_frame is not None:
                rgb_image = cv2.cvtColor(evidence_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                p = qt_img.scaled(self.image_label.width(), self.image_label.height(), Qt.AspectRatioMode.KeepAspectRatio)
                self.image_label.setPixmap(QPixmap.fromImage(p))
            
            # 2. Визуальная сигнализация
            self.log_display.append(f"🔴 КРИТИЧЕСКИЙ ДЕФЕКТ: {defect_type}! КОНВЕЙЕР ОСТАНОВЛЕН!")
            self.image_label.setStyleSheet("border: 5px solid red; background-color: #300;")
            
            # 3. Останавливаем видеопоток
            if self.video_thread:
                self.video_thread.stop()
                self.video_thread = None
            self.processing_in_progress = False

            # 4. Меняем кнопку
            self.start_btn.setChecked(False)
            self.start_btn.setText("RESET & START")

    def start_video(self):
        self.video_thread = VideoThread(source=0) # Webcam 0
        self.video_thread.change_pixmap_signal.connect(self.handle_new_frame)
        self.video_thread.connection_lost_signal.connect(self.handle_connection_loss)
        self.video_thread.start()
        self.log_display.append(f"System started at {self.get_timestamp()}")

    def stop_video(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
        self.image_label.setText("System Stopped")
        self.log_display.append(f"System stopped at {self.get_timestamp()}")
        self.processing_in_progress = False

    def handle_connection_loss(self):
        self.image_label.setText("Connection Lost - Retrying...")
        self.log_display.append("Warning: Camera connection lost!")

    @pyqtSlot(object)
    def handle_new_frame(self, frame):
        # Flow control: Drop frame if previous one is still processing
        if not self.processing_in_progress:
            self.processing_in_progress = True
            self.request_inference.emit(frame)

    @pyqtSlot(object, list)
    def update_ui(self, frame, detections):
        self.processing_in_progress = False
        
        # Convert CV2 Image to QImage
        # Optimization: Scale before converting to QImage if image is huge, but here we scale after.
        # Ensure frame is valid
        if frame is None:
            return

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.image_label.width(), self.image_label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label.setPixmap(QPixmap.fromImage(p))

        # Log detections with throttle
        import time
        current_time = time.time()
        
        if detections:
            for d in detections:
                class_name = d['type']
                # Логируем только если прошло больше 2 секунд с последнего лога этого класса
                # ИЛИ если это новый класс
                last_time = self.last_log_time.get(class_name, 0)
                
                if current_time - last_time > 2.0: # 2 секунды задержка
                    self.last_log_time[class_name] = current_time
                    
                    log_msg = f"[{self.get_timestamp()}] {class_name} ({d['confidence']:.2f})"
                    self.log_display.append(log_msg)
                    # Also log to file
                    self.logger.log_defect(class_name, d['confidence'], d['bbox'])

    def get_timestamp(self):
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")

    def closeEvent(self, event):
        self.stop_video()
        if self.inference_thread.isRunning():
            self.inference_thread.quit()
            self.inference_thread.wait()
        event.accept()
