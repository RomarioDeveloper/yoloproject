import cv2
import time
from PyQt6.QtCore import QThread, pyqtSignal

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(object)
    connection_lost_signal = pyqtSignal()

    def __init__(self, source=0):
        super().__init__()
        self.source = source
        self._run_flag = True
        self.cap = None

    def run(self):
        # Initialize camera
        self.cap = cv2.VideoCapture(self.source)
        
        # Retry connection loop if initial fails
        while not self.cap.isOpened() and self._run_flag:
            self.connection_lost_signal.emit()
            time.sleep(2)
            self.cap = cv2.VideoCapture(self.source)

        while self._run_flag:
            ret, frame = self.cap.read()
            if ret:
                self.change_pixmap_signal.emit(frame)
            else:
                self.connection_lost_signal.emit()
                # Attempt to reconnect
                self.cap.release()
                time.sleep(1)
                self.cap = cv2.VideoCapture(self.source)
            
            # Control frame rate slightly to avoid 100% CPU on capture loop if needed, 
            # though usually read() blocks for appropriate time.
            # time.sleep(0.001)

        if self.cap:
            self.cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()
