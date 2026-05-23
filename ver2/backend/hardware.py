import time
import logging

# Thiết lập logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hardware")

try:
    import RPi.GPIO as GPIO
    IS_RPI = True
except ImportError:
    IS_RPI = False

BUZZER_PIN = 18

def setup_hardware():
    """Khởi tạo các chân GPIO cho đèn/còi."""
    if IS_RPI:
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(BUZZER_PIN, GPIO.OUT)
        GPIO.output(BUZZER_PIN, GPIO.LOW)
        logger.info("RPi.GPIO initialized successfully. Hardware is ready.")
    else:
        logger.info("RPi.GPIO not found. Running in simulation mode (Hardware mock).")

def trigger_alarm(duration_seconds=1.0):
    """Kích hoạt còi báo động trong khoảng thời gian nhất định."""
    if IS_RPI:
        GPIO.output(BUZZER_PIN, GPIO.HIGH)
        time.sleep(duration_seconds)
        GPIO.output(BUZZER_PIN, GPIO.LOW)
    else:
        # Giả lập còi báo động trên Windows/Mac
        logger.warning(f"🚨 [MOCK HARDWARE] BUZZER ACTIVATED for {duration_seconds}s!")

def cleanup_hardware():
    """Giải phóng tài nguyên GPIO khi tắt server."""
    if IS_RPI:
        GPIO.cleanup()
        logger.info("RPi.GPIO cleaned up.")
