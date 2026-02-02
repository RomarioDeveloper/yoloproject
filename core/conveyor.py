import time
from threading import Lock

class ConveyorController:
    """
    Класс для управления конвейером (симуляция).
    В реальном проекте здесь будет код для управления USB-реле, PLC или Arduino.
    """
    def __init__(self):
        self.is_running = True
        self.lock = Lock()

    def start(self):
        """Запускает конвейер"""
        with self.lock:
            if not self.is_running:
                print("[HARDWARE] Сигнал на реле: ЗАПУСК КОНВЕЙЕРА")
                # Здесь код для замыкания реле: relay.on()
                self.is_running = True

    def stop(self):
        """Останавливает конвейер"""
        with self.lock:
            if self.is_running:
                print("\n" + "!"*50)
                print("[HARDWARE] Сигнал на реле: ЭКСТРЕННАЯ ОСТАНОВКА")
                print("!"*50 + "\n")
                # Здесь код для размыкания реле: relay.off()
                self.is_running = False

    def get_status(self):
        """Возвращает текущий статус конвейера"""
        with self.lock:
            return self.is_running
