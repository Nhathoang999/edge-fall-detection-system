import time
import logging
import threading
import queue
from datetime import datetime

import os

# Ẩn lời chào của thư viện pygame
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

logger = logging.getLogger("voice_alert")
logger.setLevel(logging.INFO)
# Console handler for voice alert
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)

class VoiceAlert:
    """
    Module phát âm thanh cảnh báo bằng giọng nói (Voice Alert).
    Chạy trên background thread sử dụng Queue để đảm bảo Thread-safe và non-blocking.
    Tương thích: Windows, Linux, Raspberry Pi.
    """
    def __init__(self, cooldown: int = 30):
        self.cooldown = cooldown
        self.last_alert_time = 0.0
        self.lock = threading.Lock()
        
        # Hàng đợi giúp quản lý các yêu cầu đọc văn bản theo thứ tự, không bị đè lên nhau
        self.msg_queue = queue.Queue()
        
        # Pre-generate file MP3 để phát tức thì (0 giây độ trễ)
        self.alert_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alert_voice.mp3")
        self._ensure_audio_file()
        
        # Khởi chạy Worker Thread ngay khi khởi tạo class
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        
    def _ensure_audio_file(self):
        """Tạo sẵn file âm thanh 1 lần duy nhất để khi ngã có thể phát luôn không cần đợi tải từ mạng."""
        if not os.path.exists(self.alert_file):
            logger.info("[VOICE ALERT] Đang tạo sẵn file âm thanh cảnh báo (chỉ chạy 1 lần duy nhất)...")
            text = "Cảnh báo. Phát hiện té ngã. Cần hỗ trợ ngay lập tức."
            # Thêm thông số --rate=+35% (đọc nhanh hơn) và --volume=+50% (to hơn)
            command = f'edge-tts --voice vi-VN-HoaiMyNeural --rate=+35% --volume=+50% --text "{text}" --write-media "{self.alert_file}"'
            os.system(command)
            logger.info("[VOICE ALERT] Đã tạo xong file âm thanh!")
        
    def _worker(self):
        """Tiến trình ngầm (Worker) xử lý việc phát audio bằng giọng Microsoft Edge TTS."""
        try:
            pygame.mixer.init()
        except Exception as e:
            logger.error(f"[VOICE ALERT] Could not initialize pygame mixer: {e}")
            
        while True:
            text = self.msg_queue.get()
            if text is None:
                break
                
            try:
                # Phát MP3 NGAY LẬP TỨC (File đã được tạo sẵn từ trước)
                pygame.mixer.music.load(self.alert_file)
                pygame.mixer.music.play()
                
                # Đợi cho đến khi audio phát xong
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                    
                # Giải phóng tài nguyên
                pygame.mixer.music.unload()
                
            except Exception as e:
                logger.error(f"[VOICE ALERT] Playback error: {e}")
                
            self.msg_queue.task_done()

    def speak(self, text: str) -> None:
        """Đẩy văn bản vào hàng đợi để thread ngầm phát âm thanh."""
        self.msg_queue.put(text)

    def fall_alert(self, confidence: float) -> None:
        """Kích hoạt Voice Alert nếu thỏa mãn điều kiện độ tin cậy và cooldown."""
            
        current_time = time.time()
        
        # Sử dụng Lock để đảm bảo thread-safe khi truy xuất và cập nhật last_alert_time
        with self.lock:
            if current_time - self.last_alert_time < self.cooldown:
                return  # Đang trong thời gian cooldown
            
            self.last_alert_time = current_time
            
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Ghi log theo đúng yêu cầu
        log_message = (
            f"\n[VOICE ALERT]\n"
            f"Fall detected at {timestamp_str}\n"
            f"Confidence: {confidence*100:.0f}%\n"
        )
        logger.info(log_message)
        
        text = "Cảnh báo. Phát hiện té ngã. Cần hỗ trợ ngay lập tức."
        self.speak(text)

# Tạo một Singleton instance để import và sử dụng chung trên toàn backend
voice_notifier = VoiceAlert(cooldown=30)
