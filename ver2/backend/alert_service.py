import os
import time
import logging
import threading
from datetime import datetime
import cv2
import requests
import smtplib
from email.message import EmailMessage


# ================= Configuration =================
TELEGRAM_BOT_TOKEN = "8556127824:AAGvVm4xllt0fepUa9o0dNPG_Fzog6VxNgk"
TELEGRAM_CHAT_ID = "6054055103"  # Đã hardcode cố định Chat ID để không bị lỗi sau 24h

# Cấu hình Email (Bạn cần thay đổi email và App Password của bạn)
EMAIL_SENDER = "hoathanhan333@gmail.com"
EMAIL_PASSWORD = "kdoj ypxr npiw vdgt"  # Sử dụng App Password của Google, không phải mật khẩu gốc
EMAIL_RECEIVER = "thanhanpt1909@gmail.com"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALERTS_DIR = os.path.join(BASE_DIR, "alerts")

# Tự động tạo thư mục chứa ảnh cảnh báo nếu chưa có
os.makedirs(ALERTS_DIR, exist_ok=True)

# ================= Setup Logging =================
logger = logging.getLogger("alert_service")
logger.setLevel(logging.INFO)

# Ghi log ra file alerts.log
log_path = os.path.join(BASE_DIR, "alerts.log")
file_handler = logging.FileHandler(log_path, encoding="utf-8")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Cũng in ra console
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# ================= Helper Functions =================
def get_telegram_chat_id():
    global TELEGRAM_CHAT_ID
    if TELEGRAM_CHAT_ID:
        return TELEGRAM_CHAT_ID
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
        resp = requests.get(url, timeout=5)
        data = resp.json()
        if data.get("ok") and len(data["result"]) > 0:
            for update in reversed(data["result"]):
                if "message" in update and "chat" in update["message"]:
                    TELEGRAM_CHAT_ID = update["message"]["chat"]["id"]
                    logger.info(f"Đã tự động fetch Telegram Chat ID: {TELEGRAM_CHAT_ID}")
                    return TELEGRAM_CHAT_ID
    except Exception as e:
        logger.error(f"Lỗi khi lấy Telegram chat_id: {e}")
    return None

def send_telegram_alert(message: str, image_path: str = None) -> None:
    chat_id = get_telegram_chat_id()
    if not chat_id:
        logger.warning("Không tìm thấy Chat ID Telegram. Nhắn /start cho bot trước.")
        return

    try:
        if image_path and os.path.exists(image_path):
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            with open(image_path, "rb") as f:
                files = {"photo": f}
                data = {"chat_id": chat_id, "caption": message}
                requests.post(url, data=data, files=files, timeout=10)
        else:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {"chat_id": chat_id, "text": message}
            requests.post(url, data=data, timeout=10)
        logger.info("Đã gửi cảnh báo qua Telegram.")
    except Exception as e:
        logger.error(f"Lỗi gửi Telegram: {e}")

def send_email_alert(subject: str, body: str, image_path: str = None) -> None:
    if not EMAIL_SENDER or not EMAIL_PASSWORD or "your_email" in EMAIL_SENDER:
        logger.warning("Bỏ qua gửi Email vì chưa cấu hình đúng thông tin đăng nhập trong alert_service.py.")
        return

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg.set_content(body)

    if image_path and os.path.exists(image_path):
        with open(image_path, 'rb') as f:
            img_data = f.read()
            msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename=os.path.basename(image_path))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        logger.info("Đã gửi cảnh báo qua Email.")
    except Exception as e:
        logger.error(f"Lỗi gửi Email: {e}")

def save_fall_frame(frame) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"fall_{timestamp}.jpg"
    filepath = os.path.join(ALERTS_DIR, filename)
    cv2.imwrite(filepath, frame)
    logger.info(f"Đã lưu ảnh frame tại: {filepath}")
    return filepath

def log_alert(confidence: float, filepath: str) -> None:
    logger.info(f"FALL DETECTED | Confidence: {confidence:.4f} | Saved Image: {filepath}")

# ================= Main Orchestrator =================
def _handle_fall_process(frame, confidence: float):
    # 0. Phát cảnh báo giọng nói (Voice Alert) NGAY LẬP TỨC
    from backend.voice_alert import voice_notifier
    voice_notifier.fall_alert(confidence)

    # 1. Lưu ảnh frame hiện tại
    filepath = save_fall_frame(frame)
    
    # 2. Ghi log
    log_alert(confidence, filepath)
    
    # Chuẩn bị nội dung cảnh báo
    message = f"🚨 CẢNH BÁO: Phát hiện người té ngã!"
    subject = "⚠️ Cảnh báo khẩn cấp: Phát hiện té ngã trên Camera"
    
    # 3. Gửi cảnh báo Telegram
    send_telegram_alert(message, filepath)
    
    # 4. Gửi email
    send_email_alert(subject, message, filepath)

def handle_fall_event(frame, confidence: float) -> None:
    """
    Hàm entry point để điều phối quy trình cảnh báo ngầm, tránh block luồng xử lý video.
    """
    threading.Thread(target=_handle_fall_process, args=(frame, confidence), daemon=True).start()
