import sys
import os
import shutil

# Đảm bảo import được các module từ thư mục gốc ver2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import base64
from collections import deque
import time
import threading

import config
from src.keypoints import get_kpt_indices
from backend.hardware import setup_hardware, trigger_alarm, cleanup_hardware
from backend.skeleton import normalize_skeleton_frame

app = FastAPI(title="Edge IoT Fall Detection")

# Cho phép React/Vite kết nối
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
INPUT_TIMESTEPS = 30
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    import json
    with open(os.path.join(BASE_DIR, "threshold.json"), "r") as f:
        FALL_CONFIDENCE_THRESHOLD = json.load(f).get("threshold", 0.10)
except Exception:
    FALL_CONFIDENCE_THRESHOLD = 0.10
MODEL_PATH = os.path.join(BASE_DIR, "model.tflite")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load TFLite Model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

global_video_source = 0 # 0 là Webcam, nếu là string thì là file path

@app.post("/upload-video")
def upload_video(file: UploadFile = File(...)):
    global global_video_source
    print("Received upload request for", file.filename)
    file_location = os.path.join(BASE_DIR, "temp_video.mp4")
    try:
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        global_video_source = file_location
        print("Upload successful, switching video source to", file_location)
        return {"info": f"Video uploaded", "source": file_location}
    except Exception as e:
        print("Upload failed:", e)
        return {"error": str(e)}

@app.post("/reset-camera")
def reset_camera():
    global global_video_source
    print("Resetting camera to webcam")
    global_video_source = 0
    return {"info": "Switched to Webcam", "source": 0}


def extract_features(results):
    features = np.zeros(17 * 3, dtype=np.float32)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        mapping = {
            mp_pose.PoseLandmark.NOSE: 'Nose', mp_pose.PoseLandmark.LEFT_EYE: 'Left Eye',
            mp_pose.PoseLandmark.RIGHT_EYE: 'Right Eye', mp_pose.PoseLandmark.LEFT_EAR: 'Left Ear',
            mp_pose.PoseLandmark.RIGHT_EAR: 'Right Ear', mp_pose.PoseLandmark.LEFT_SHOULDER: 'Left Shoulder',
            mp_pose.PoseLandmark.RIGHT_SHOULDER: 'Right Shoulder', mp_pose.PoseLandmark.LEFT_ELBOW: 'Left Elbow',
            mp_pose.PoseLandmark.RIGHT_ELBOW: 'Right Elbow', mp_pose.PoseLandmark.LEFT_WRIST: 'Left Wrist',
            mp_pose.PoseLandmark.RIGHT_WRIST: 'Right Wrist', mp_pose.PoseLandmark.LEFT_HIP: 'Left Hip',
            mp_pose.PoseLandmark.RIGHT_HIP: 'Right Hip', mp_pose.PoseLandmark.LEFT_KNEE: 'Left Knee',
            mp_pose.PoseLandmark.RIGHT_KNEE: 'Right Knee', mp_pose.PoseLandmark.LEFT_ANKLE: 'Left Ankle',
            mp_pose.PoseLandmark.RIGHT_ANKLE: 'Right Ankle'
        }
        for mp_id, name in mapping.items():
            if name in config.SORTED_KEYPOINT_NAMES:
                try:
                    idx, idy, idc = get_kpt_indices(name)
                    lm = landmarks[mp_id.value]
                    features[idx], features[idy], features[idc] = lm.x, lm.y, lm.visibility
                except Exception:
                    pass
    return normalize_skeleton_frame(features)

@app.on_event("startup")
def startup_event():
    setup_hardware()
    print("Edge AI Backend Started!")

@app.on_event("shutdown")
def shutdown_event():
    cleanup_hardware()
    print("Edge AI Backend Shutdown.")

@app.websocket("/ws/video")
async def websocket_video_endpoint(websocket: WebSocket):
    global global_video_source
    await websocket.accept()
    
    current_source = global_video_source
    if current_source == 0 and os.name == 'nt':
        cap = cv2.VideoCapture(current_source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(current_source)
        
    if current_source == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    feature_sequence = deque(maxlen=INPUT_TIMESTEPS)
    last_fall_time = 0
    FALL_COOLDOWN = 5.0
    
    try:
        while True:
            # Check nếu nguồn video thay đổi (ví dụ user vừa upload file)
            if current_source != global_video_source:
                cap.release()
                current_source = global_video_source
                if current_source == 0 and os.name == 'nt':
                    cap = cv2.VideoCapture(current_source, cv2.CAP_DSHOW)
                else:
                    cap = cv2.VideoCapture(current_source)
                    
                if current_source == 0:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                feature_sequence.clear()
            
            ret, frame = cap.read()
            if not ret:
                if current_source != 0:
                    # Video đã hết, lặp lại từ đầu
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    await asyncio.sleep(0.1)
                    continue
            
            # Frame resize nhẹ để AI chạy nhanh hơn nếu video gốc 4K/1080p
            frame_resized = cv2.resize(frame, (640, 480))
            image_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            feats = extract_features(results)
            feature_sequence.append(feats)
            
            label = "non_fall"
            conf = 0.0
            
            if len(feature_sequence) == INPUT_TIMESTEPS:
                input_data = np.array(feature_sequence, dtype=np.float32)
                input_data = np.expand_dims(input_data, axis=0)
                
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                out = interpreter.get_tensor(output_details[0]['index'])[0][0]
                
                if out > FALL_CONFIDENCE_THRESHOLD:
                    label = "fall"
                    conf = float(out)
                    current_time = time.time()
                    if current_time - last_fall_time > FALL_COOLDOWN:
                        print("TRIGGER ALARM!")
                        threading.Thread(target=trigger_alarm, args=(2.0,), daemon=True).start()
                        last_fall_time = current_time
                else:
                    label = "non_fall"
                    conf = float(1.0 - out)
            
            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame_resized, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 60])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            payload = {
                "image": jpg_as_text,
                "label": label,
                "confidence": conf
            }
            await websocket.send_json(payload)
            
            # Nhường CPU cho luồng event loop (0.001s để không làm chậm video)
            await asyncio.sleep(0.001)
            
    except WebSocketDisconnect:
        print("Frontend Client Disconnected.")
    finally:
        cap.release()

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
