import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
print("tf imported", flush=True)
tflite = tf.lite
import time
import json
from collections import deque
import gradio as gr
import os
import shutil # Dùng để sao chép file thư mục
import uuid
import threading
import queue
import atexit

# Server-side session store to keep heavy objects (feature sequences) out of Gradio's JSON state
SESSION_STORE = {}

# Worker settings
WORKER_QUEUE_MAXSIZE = 1
SESSION_TIMEOUT_SECONDS = 300  # auto-clean sessions after inactivity (seconds)


def _ensure_session_worker(session_id):
    """Ensure per-session worker and queue exist and start worker thread if needed."""
    session = SESSION_STORE.get(session_id)
    if session is None:
        return
    if 'in_queue' not in session:
        session['in_queue'] = queue.Queue(maxsize=WORKER_QUEUE_MAXSIZE)
    if 'running' not in session or not session['running']:
        session['running'] = True
        t = threading.Thread(target=_realtime_worker, args=(session_id,), daemon=True)
        session['worker'] = t
        t.start()


def _realtime_worker(session_id):
    """Background worker that consumes latest frames and runs MediaPipe + inference."""
    try:
        session = SESSION_STORE.get(session_id)
        if session is None:
            return
        # Each worker keeps its own MediaPipe Pose instance (thread-safe)
        with mp_pose.Pose(static_image_mode=False,
                          model_complexity=pose_complexity,
                          smooth_landmarks=True,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as worker_pose:
            last_process_time = 0.0
            while session.get('running', False):
                try:
                    frame = session['in_queue'].get(timeout=0.5)
                except queue.Empty:
                    # check for inactivity
                    if time.time() - session.get('last_update_time', 0) > SESSION_TIMEOUT_SECONDS:
                        session['running'] = False
                        break
                    continue

                session['last_update_time'] = time.time()
                # Downscale same as main thread
                proc_frame = frame
                try:
                    h, w = proc_frame.shape[:2]
                    if REALTIME_PROCESS_WIDTH and w > REALTIME_PROCESS_WIDTH:
                        scale = REALTIME_PROCESS_WIDTH / float(w)
                        target_h = max(1, int(h * scale))
                        proc_frame = cv2.resize(proc_frame, (REALTIME_PROCESS_WIDTH, target_h), interpolation=cv2.INTER_LINEAR)
                    image_rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
                    image_rgb.flags.writeable = False
                    results = worker_pose.process(image_rgb)
                except Exception:
                    results = None

                if not results or not getattr(results, 'pose_landmarks', None):
                    session['last_center'] = None
                    # keep previous landmarks until they age out on main thread to avoid flicker
                    continue

                # compute simple center (use shoulders/hips);
                try:
                    lm = results.pose_landmarks.landmark
                    # prefer hips then shoulders
                    ids = []
                    try:
                        ids = [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value]
                    except Exception:
                        ids = []
                    if not ids:
                        try:
                            ids = [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                        except Exception:
                            ids = []
                    cx, cy, count = 0.0, 0.0, 0
                    for i in ids:
                        if i < len(lm):
                            cx += lm[i].x
                            cy += lm[i].y
                            count += 1
                    if count > 0:
                        cx /= count
                        cy /= count
                        session['last_center'] = (cx, cy)
                    else:
                        session['last_center'] = None
                except Exception:
                    session['last_center'] = None

                # store normalized landmarks list for main-thread drawing
                try:
                    session['last_landmarks'] = [(float(p.x), float(p.y), float(getattr(p, 'visibility', 1.0))) for p in results.pose_landmarks.landmark]
                    session['last_landmarks_time'] = time.time()
                except Exception:
                    session['last_landmarks'] = None
                    session['last_landmarks_time'] = session.get('last_landmarks_time', 0)

                # extract features and update sequence
                try:
                    feats = extract_and_normalize_features(results)
                    session['seq'].append(feats)
                except Exception:
                    pass

                # run inference when enough frames accumulated
                try:
                    if len(session['seq']) == INPUT_TIMESTEPS:
                        sample_input = np.array(list(session['seq']), dtype=np.float32)
                        sample_input = np.expand_dims(sample_input, axis=0)
                        interpreter.set_tensor(input_details[0]['index'], sample_input)
                        interpreter.invoke()
                        output_data = interpreter.get_tensor(output_details[0]['index'])
                        prob_fall = float(output_data[0][0])
                        if prob_fall > FALL_CONFIDENCE_THRESHOLD:
                            session['last_label'] = 'fall'
                            session['last_conf'] = prob_fall
                        else:
                            session['last_label'] = 'no_fall'
                            session['last_conf'] = 1.0 - prob_fall
                except Exception:
                    pass
    except Exception:
        pass


def _cleanup_sessions():
    now = time.time()
    for sid in list(SESSION_STORE.keys()):
        s = SESSION_STORE.get(sid)
        if not s:
            continue
        if not s.get('running', False) and now - s.get('last_update_time', 0) > SESSION_TIMEOUT_SECONDS:
            try:
                del SESSION_STORE[sid]
            except Exception:
                pass


# ensure cleanup on exit
atexit.register(_cleanup_sessions)

# --- CẤU HÌNH HỆ THỐNG (Tinh chỉnh cho phù hợp với Gradio) ---
_DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_DEPLOY_DIR, 'fall_detection_transformer.tflite')
INPUT_TIMESTEPS = 30 # Độ dài mỗi Sequence để nhét vào AI
FALL_CONFIDENCE_THRESHOLD = 0.90
_threshold_path = os.path.join(_DEPLOY_DIR, 'threshold.json')
if os.path.isfile(_threshold_path):
    try:
        with open(_threshold_path, 'r', encoding='utf-8') as _tf:
            FALL_CONFIDENCE_THRESHOLD = float(json.load(_tf).get('threshold', FALL_CONFIDENCE_THRESHOLD))
        print(f"Loaded threshold from {_threshold_path}: {FALL_CONFIDENCE_THRESHOLD}", flush=True)
    except Exception as _e:
        print(f"Warning: could not load threshold.json: {_e}", flush=True)
MIN_KEYPOINT_CONFIDENCE_FOR_NORMALIZATION = 0.3 # Ngưỡng tự tin thấp nhất để trích xuất điểm ảnh Landmarks
mp_pose = mp.solutions.pose
pose_complexity = 0 # Thu nhỏ độ phức tạp của Pose về nhỏ nhất (= 0) để chạy webcam không bị giật lag
use_static_image_mode = False # Tắt chế độ static mode khi bắt đầu

FALL_EVENT_COOLDOWN = 10 # Giới hạn lặp thông báo (Tối thiểu 10 seconds cảnh báo 1 lần)

# --- Realtime optimization settings ---
# Resize width for processing (maintain aspect ratio). Set to None to disable resizing.
REALTIME_PROCESS_WIDTH = 640
# Run TFLite inference only every N frames (>=1). Larger => less CPU.
REALTIME_INFERENCE_EVERY_N_FRAMES = 2
# Draw landmarks only every N frames to reduce drawing overhead.
REALTIME_DRAW_EVERY_N_FRAMES = 2
# Option: draw full skeleton overlay (from worker landmarks) on display frame
REALTIME_DRAW_FULL_SKELETON = True
# Draw full skeleton every N frames (main thread draws lightweight lines/circles)
REALTIME_FULL_DRAW_EVERY_N_FRAMES = 8

# How long to keep drawing last known landmarks when worker briefly misses detections (seconds)
LANDMARK_PERSIST_SECONDS = 1.5
# How often to update the textual log shown to the user (in frames)
REALTIME_LOG_EVERY_N_FRAMES = 10

# Persistent MediaPipe Pose instance for realtime (avoid recreating per-frame)
try:
    realtime_pose = mp_pose.Pose(static_image_mode=False,
                                 model_complexity=pose_complexity,
                                 smooth_landmarks=True,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)
    print("Initialized persistent realtime_pose for webcam (reused across frames).")
except Exception as e:
    realtime_pose = None
    print(f"Warning: Could not initialize persistent realtime_pose: {e}")

# ----- 0. ĐỊNH NGHĨA VÀ CHUẨN HÓA MỤC TIÊU CÁC KHỚP XƯƠNG (KEYPOINTS) -----
KEYPOINT_NAMES_ORIGINAL = [
    'Nose', 'Left Eye Inner', 'Left Eye', 'Left Eye Outer', 'Right Eye Inner', 'Right Eye', 'Right Eye Outer',
    'Left Ear', 'Right Ear', 'Mouth Left', 'Mouth Right',
    'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow', 'Left Wrist', 'Right Wrist',
    'Left Pinky', 'Right Pinky', 'Left Index', 'Right Index', 'Left Thumb', 'Right Thumb',
    'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle',
    'Left Heel', 'Right Heel', 'Left Foot Index', 'Right Foot Index'
]
MEDIAPIPE_TO_YOUR_KEYPOINTS_MAPPING = {
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
YOUR_KEYPOINT_NAMES_TRAINING = [
    'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
    'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
    'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
    'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'
]
# Sắp xếp lại thứ tự List các mảng theo chuẩn Alpha-B để Model TFLite đọc trơn tru
SORTED_YOUR_KEYPOINT_NAMES = sorted(YOUR_KEYPOINT_NAMES_TRAINING)
KEYPOINT_DICT_TRAINING = {name: i for i, name in enumerate(SORTED_YOUR_KEYPOINT_NAMES)}
NUM_KEYPOINTS_TRAINING = len(KEYPOINT_DICT_TRAINING) # Tổng Keypoints học được
NUM_FEATURES = NUM_KEYPOINTS_TRAINING * 3 # Tọa độ XYZ (3 chiều x Keypoints)

print("--- Đang khởi tạo Khởi động khung xương cho Gradio App ---")
print(f"NUM_FEATURES từ module hệ thống: {NUM_FEATURES}")
# ---------------------------------------------------------------

# --- Quá Trình Load Model Neural TFLite ---
try:
    # Nạp kiến trúc file tflite
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    # Phân tích cổng vào và cổng ra của AI
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"Đã tải thành công Model TFLite: {MODEL_PATH}")
    model_expected_shape = tuple(input_details[0]['shape'])
    # Bắt buộc input shape phải bằng chuẩn (ví dụ 30 timesteps và 51 tọa độ)
    if model_expected_shape[2] != NUM_FEATURES or model_expected_shape[1] != INPUT_TIMESTEPS:
        print(f"LỖI CHIẾN MẠNG: Hình dạng chiều đầu vào của model features/timesteps "
              f"({model_expected_shape[1]},{model_expected_shape[2]}) "
              f"KHÔNG TRÙNG KHỚP VỚI config hệ thống ({INPUT_TIMESTEPS},{NUM_FEATURES}).")
        exit()
except Exception as e:
    print(f"Lỗi không thể tải AI TFLite model: {e}")
    exit()

# --- Các hàm chức năng phục vụ Normalize Tọa độ (Chuẩn hóa) ---
def get_kpt_indices_training_order(keypoint_name):
    """Tìm lại vị trí thực tế của mảng tọa độ X, Y, Z."""
    if keypoint_name not in KEYPOINT_DICT_TRAINING:
        raise ValueError(f"Keypoint '{keypoint_name}' not found in KEYPOINT_DICT_TRAINING. Available: {list(KEYPOINT_DICT_TRAINING.keys())}")
    kp_idx = KEYPOINT_DICT_TRAINING[keypoint_name]
    return kp_idx * 3, kp_idx * 3 + 1, kp_idx * 3 + 2

def normalize_skeleton_frame(frame_features_sorted, min_confidence=MIN_KEYPOINT_CONFIDENCE_FOR_NORMALIZATION):
    """
    Tiến hành canh chuẩn tọa độ Khung xương để lọc bỏ yếu tố Camera xa hay gần.
    Tọa độ (X, Y) được chuyển hướng lấy Tâm gốc là trung điểm giữa 2 phần hông.
    """
    normalized_frame = np.copy(frame_features_sorted)
    ref_kp_names = {'ls': 'Left Shoulder', 'rs': 'Right Shoulder', 'lh': 'Left Hip', 'rh': 'Right Hip'}
    try:
        # Lấy Tọa độ Vai trái, Vai phải, Hông trái, Hông phải
        ls_x_idx, ls_y_idx, ls_c_idx = get_kpt_indices_training_order(ref_kp_names['ls'])
        rs_x_idx, rs_y_idx, rs_c_idx = get_kpt_indices_training_order(ref_kp_names['rs'])
        lh_x_idx, lh_y_idx, lh_c_idx = get_kpt_indices_training_order(ref_kp_names['lh'])
        rh_x_idx, rh_y_idx, rh_c_idx = get_kpt_indices_training_order(ref_kp_names['rh'])
    except ValueError as e:
        print(f"Cảnh báo trong hàm normalize_skeleton_frame (Mất tọa độ gốc): {e}")
        return frame_features_sorted

    ls_x, ls_y, ls_c = frame_features_sorted[ls_x_idx], frame_features_sorted[ls_y_idx], frame_features_sorted[ls_c_idx]
    rs_x, rs_y, rs_c = frame_features_sorted[rs_x_idx], frame_features_sorted[rs_y_idx], frame_features_sorted[rs_c_idx]
    lh_x, lh_y, lh_c = frame_features_sorted[lh_x_idx], frame_features_sorted[lh_y_idx], frame_features_sorted[lh_c_idx]
    rh_x, rh_y, rh_c = frame_features_sorted[rh_x_idx], frame_features_sorted[rh_y_idx], frame_features_sorted[rh_c_idx]

    # Tính điểm chính giữa (Center-point) của Vai
    mid_shoulder_x, mid_shoulder_y = np.nan, np.nan
    valid_ls, valid_rs = ls_c > min_confidence, rs_c > min_confidence
    if valid_ls and valid_rs: mid_shoulder_x, mid_shoulder_y = (ls_x + rs_x) / 2, (ls_y + rs_y) / 2
    elif valid_ls: mid_shoulder_x, mid_shoulder_y = ls_x, ls_y
    elif valid_rs: mid_shoulder_x, mid_shoulder_y = rs_x, rs_y

    # Tính điểm chính giữa (Center-point) của Vùng Xương Chậu (Hông)
    mid_hip_x, mid_hip_y = np.nan, np.nan
    valid_lh, valid_rh = lh_c > min_confidence, rh_c > min_confidence
    if valid_lh and valid_rh: mid_hip_x, mid_hip_y = (lh_x + rh_x) / 2, (lh_y + rh_y) / 2
    elif valid_lh: mid_hip_x, mid_hip_y = lh_x, lh_y
    elif valid_rh: mid_hip_x, mid_hip_y = rh_x, rh_y

    # Trả về nguyên bản nếu không bắt được Tọa độ hông (Do bị che)
    if np.isnan(mid_hip_x) or np.isnan(mid_hip_y):
        return frame_features_sorted

    # Thu phóng tỷ lệ để khớp với chiều cao chuẩn của người từ Vai xuống Hông
    reference_height = np.nan
    if not np.isnan(mid_shoulder_y) and not np.isnan(mid_hip_y):
        reference_height = np.abs(mid_shoulder_y - mid_hip_y)

    perform_scaling = not (np.isnan(reference_height) or reference_height < 1e-5)

    # Chạy vòng lập dời lại Tâm Gốc tọa độ vào phía Hông thay vì mép Màn hình ảnh
    for kp_name_sorted in SORTED_YOUR_KEYPOINT_NAMES:
        try:
            x_col, y_col, _ = get_kpt_indices_training_order(kp_name_sorted)
            normalized_frame[x_col] -= mid_hip_x
            normalized_frame[y_col] -= mid_hip_y
            if perform_scaling:
                normalized_frame[x_col] /= reference_height
                normalized_frame[y_col] /= reference_height
        except ValueError: # Bỏ qua nếu ko tìm ra khớp này
            pass
    return normalized_frame

def extract_and_normalize_features(pose_results):
    """
    Rút trích các Điểm Cột Chỉ (Landmarks) trong ảnh và chuẩn hóa Tọa độ (Scale).
    """
    frame_features_sorted = np.zeros(NUM_FEATURES, dtype=np.float32)
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        for mp_landmark_enum, your_kp_name in MEDIAPIPE_TO_YOUR_KEYPOINTS_MAPPING.items():
            if your_kp_name in KEYPOINT_DICT_TRAINING:
                try:
                    lm = landmarks[mp_landmark_enum.value]
                    x_idx, y_idx, c_idx = get_kpt_indices_training_order(your_kp_name)
                    # Gắn vào đúng dòng cột như khi huấn luyện Model
                    frame_features_sorted[x_idx], frame_features_sorted[y_idx], frame_features_sorted[c_idx] = lm.x, lm.y, lm.visibility
                except (IndexError, ValueError) as e:
                    print(f"Cảnh cáo trong lúc rút trích Landmark cho {your_kp_name}: {e}")
                    pass
    # Trả về tọa độ bộ Xương Người sau khi Canh Scale và Reset Tỉ lệ màn hình
    normalized_features = normalize_skeleton_frame(frame_features_sorted.copy())
    return normalized_features
# -------------------------------------------------------------------------------------------------------------------

# --- LUỒNG LOGIC TÍNH TOÁN CHO REALTIME TRÊN LƯỚI WEBCAM (TỐC ĐỘ CAO) ---

def normalize_keypoints(raw_keypoints):
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    min_z, max_z = float('inf'), float('-inf')
    for kp in raw_keypoints:
        if kp.visibility > 0.2:
            min_x, max_x = min(min_x, kp.x), max(max_x, kp.x)
            min_y, max_y = min(min_y, kp.y), max(max_y, kp.y)
            min_z, max_z = min(min_z, kp.z), max(max_z, kp.z)
    norm = []
    width = max(1e-6, max_x - min_x)
    height = max(1e-6, max_y - min_y)
    depth = max(1e-6, max_z - min_z)
    for kp in raw_keypoints:
        import types
        n = types.SimpleNamespace()
        n.x = (kp.x - min_x) / width
        n.y = (kp.y - min_y) / height
        n.z = (kp.z - min_z) / depth
        n.visibility = kp.visibility
        norm.append(n)
    return norm

def process_frame_for_realtime(frame, state, draw_full_skeleton=True):
    import cv2
    import numpy as np
    from collections import deque
    global SESSION_STORE

    try:
        if state is None:
            session_id = str(uuid.uuid4())
            SESSION_STORE[session_id] = {
                'seq': deque(maxlen=INPUT_TIMESTEPS),
                'last_fall_time': 0.0,
                'last_label': 'no_fall',
                'last_conf': 0.0,
                'last_center': None,
                'last_update_time': time.time(),
                'running': False,
                # local log cache to avoid updating UI every frame
                'last_log_text': '',
                'last_log_frame': -1,
                'last_log_label': None
            }
            frame_count = 0
            overall_status = "Trạng thái: Khởi chạy..."
            infer_counter = 0
            last_label = "no_fall"
            last_conf = 0.0
        else:
            if isinstance(state, (list, tuple)) and len(state) >= 6 and isinstance(state[0], str) and state[0] in SESSION_STORE:
                session_id = state[0]
                frame_count = int(state[1])
                overall_status = state[2]
                infer_counter = int(state[3])
                last_label = state[4]
                last_conf = float(state[5])
            else:
                if isinstance(state, (list, tuple)) and len(state) >= 2 and hasattr(state[1], 'append'):
                    session_id = str(uuid.uuid4())
                    SESSION_STORE[session_id] = {
                        'seq': state[1],
                        'last_fall_time': 0.0,
                        'last_label': 'no_fall',
                        'last_conf': 0.0,
                        'last_center': None,
                        'last_update_time': time.time(),
                        'running': False,
                        'last_log_text': '',
                        'last_log_frame': -1,
                        'last_log_label': None
                    }
                    frame_count = int(state[0]) if len(state) > 0 else 0
                    overall_status = state[2] if len(state) > 2 else "Trạng thái: Khởi chạy..."
                    infer_counter = int(state[3]) if len(state) > 3 else 0
                    last_label = state[4] if len(state) > 4 else "no_fall"
                    last_conf = float(state[5]) if len(state) > 5 else 0.0
                else:
                    session_id = str(uuid.uuid4())
                    SESSION_STORE[session_id] = {
                        'seq': deque(maxlen=INPUT_TIMESTEPS),
                        'last_fall_time': 0.0,
                        'last_label': 'no_fall',
                        'last_conf': 0.0,
                        'last_center': None,
                        'last_update_time': time.time(),
                        'running': False,
                        'last_log_text': '',
                        'last_log_frame': -1,
                        'last_log_label': None
                    }
                    frame_count = 0
                    overall_status = "Trạng thái: Khởi chạy..."
                    infer_counter = 0
                    last_label = "no_fall"
                    last_conf = 0.0

        # Ensure background worker is running for this session
        _ensure_session_worker(session_id)

        if frame is None:
            status_text = "Trạng thái nghỉ"
            return None, status_text, (session_id, frame_count, overall_status, infer_counter, last_label, last_conf)

        # Prepare display frame and an enqueue copy (downscaled)
        frame = np.ascontiguousarray(np.copy(frame))
        frame.flags.writeable = True
        orig_h, orig_w = frame.shape[:2]
        enqueue_frame = frame
        if REALTIME_PROCESS_WIDTH and orig_w > REALTIME_PROCESS_WIDTH:
            scale = REALTIME_PROCESS_WIDTH / float(orig_w)
            target_h = max(1, int(orig_h * scale))
            enqueue_frame = cv2.resize(frame, (REALTIME_PROCESS_WIDTH, target_h), interpolation=cv2.INTER_LINEAR)

        # Try to push latest frame to worker queue (non-blocking, replace if full)
        session = SESSION_STORE.get(session_id)
        try:
            if session is not None and 'in_queue' in session:
                try:
                    session['in_queue'].put_nowait(enqueue_frame.copy())
                except queue.Full:
                    try:
                        session['in_queue'].get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        session['in_queue'].put_nowait(enqueue_frame.copy())
                    except queue.Full:
                        pass
        except Exception:
            pass

        # Read latest lightweight results from session
        last_center = session.get('last_center') if session is not None else None
        last_label = session.get('last_label', last_label) if session is not None else last_label
        last_conf = session.get('last_conf', last_conf) if session is not None else last_conf

        # Draw lightweight overlay (center + label) to keep UI responsive
        display_frame = frame
        if last_center:
            try:
                cx = int(last_center[0] * orig_w)
                cy = int(last_center[1] * orig_h)
                color = (0, 255, 0) if last_label != 'fall' else (0, 0, 255)
                cv2.circle(display_frame, (cx, cy), 8, color, -1)
            except Exception:
                pass

        # Optionally draw full skeleton (lines + joints) using normalized landmarks from worker
        try:
            if draw_full_skeleton:
                now_ts = time.time()
                landmarks = session.get('last_landmarks') if session is not None else None
                landmarks_time = session.get('last_landmarks_time', 0) if session is not None else 0
                # only draw if landmarks are recent enough to avoid flicker
                if landmarks and (now_ts - landmarks_time) <= LANDMARK_PERSIST_SECONDS:
                    # draw connections
                    try:
                        for conn in mp_pose.POSE_CONNECTIONS:
                            try:
                                a, b = conn
                                idx_a = a.value if hasattr(a, 'value') else int(a)
                                idx_b = b.value if hasattr(b, 'value') else int(b)
                                if idx_a < len(landmarks) and idx_b < len(landmarks):
                                    x1 = int(landmarks[idx_a][0] * orig_w)
                                    y1 = int(landmarks[idx_a][1] * orig_h)
                                    x2 = int(landmarks[idx_b][0] * orig_w)
                                    y2 = int(landmarks[idx_b][1] * orig_h)
                                    cv2.line(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    # draw joints
                    for j, (lx, ly, lv) in enumerate(landmarks):
                        try:
                            if lv > 0.15:
                                px = int(lx * orig_w)
                                py = int(ly * orig_h)
                                cv2.circle(display_frame, (px, py), 3, (0, 0, 255), -1)
                        except Exception:
                            pass
        except Exception:
            pass

        # compute status text from model state (prefer model label/confidence)
        try:
            seq_len = len(session.get('seq', [])) if session is not None else 0
            if seq_len < INPUT_TIMESTEPS:
                base_text = f"COLLECTING ({seq_len}/{INPUT_TIMESTEPS})"
            else:
                base_text = f"{session.get('last_label', last_label).upper()} ({session.get('last_conf', last_conf):.2f})"
        except Exception:
            try:
                base_text = f"{last_label.upper()} ({last_conf:.2f})"
            except Exception:
                base_text = "Buffering..."

        # Update textual log only every REALTIME_LOG_EVERY_N_FRAMES frames, or immediately if label changed
        try:
            if session is None:
                status_text = base_text
            else:
                # ensure last_log fields exist
                if 'last_log_text' not in session:
                    session['last_log_text'] = base_text
                    session['last_log_frame'] = -1
                    session['last_log_label'] = session.get('last_label')

                update_log = False
                try:
                    if REALTIME_LOG_EVERY_N_FRAMES and (frame_count % REALTIME_LOG_EVERY_N_FRAMES == 0):
                        update_log = True
                except Exception:
                    update_log = True

                # immediate update if the model label changed
                if session.get('last_label') != session.get('last_log_label'):
                    update_log = True

                if update_log:
                    session['last_log_text'] = base_text
                    session['last_log_frame'] = frame_count
                    session['last_log_label'] = session.get('last_label')

                status_text = session.get('last_log_text', base_text)
        except Exception:
            status_text = base_text

        cv2.putText(display_frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # update timestamp
        session['last_update_time'] = time.time()
        frame_count += 1

        return display_frame, status_text, (session_id, frame_count, overall_status, infer_counter, last_label, last_conf)
    except Exception as e:
        try:
            cv2.putText(frame, f"Error: {str(e)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        except Exception:
            pass
        session_id = locals().get('session_id', str(uuid.uuid4()))
        if session_id not in SESSION_STORE:
            SESSION_STORE[session_id] = {
                'seq': deque(maxlen=INPUT_TIMESTEPS),
                'last_fall_time': 0.0,
                'last_label': 'no_fall',
                'last_conf': 0.0,
                'last_center': None,
                'last_update_time': time.time(),
                'running': False,
                'last_log_text': '',
                'last_log_frame': -1,
                'last_log_label': None
            }
        status_text = f"Error: {str(e)}"
        return frame, status_text, (session_id, locals().get('frame_count', 0), "Error", locals().get('infer_counter', 0), "no_fall", 0.0)

def process_video_for_gradio(uploaded_video):
    # Normalize different possible Gradio inputs: str path, list, dict, data URI, or URL
    if not uploaded_video:
        return None, "Please upload a video file."

    # Unwrap list/tuple (Gradio Examples sometimes send a list)
    if isinstance(uploaded_video, (list, tuple)):
        if len(uploaded_video) == 0:
            return None, "No video uploaded."
        uploaded_video = uploaded_video[0]

    # If Gradio passed a dict-like object, try common keys
    if isinstance(uploaded_video, dict):
        for key in ('name', 'tmp_path', 'tempfile', 'file', 'path', 'filename', 'tempfile_path', 'data'):
            if key in uploaded_video:
                uploaded_video = uploaded_video[key]
                break
        else:
            uploaded_video = str(uploaded_video)

    # If it's a file-like object with .name, use that
    if hasattr(uploaded_video, 'name') and isinstance(getattr(uploaded_video, 'name'), str):
        uploaded_video = getattr(uploaded_video, 'name')

    # At this point uploaded_video should be a string (path, data URI, or URL)
    uploaded_str = str(uploaded_video)
    print(f"Gradio provided input (normalized): {uploaded_str}")

    # Prepare unique local path
    timestamp_str = str(int(time.time() * 1000))
    local_video_path = None
    created_local_copy = False

    # Handle data URI (base64)
    try:
        import re, base64, urllib.request
        data_uri_match = re.match(r"data:(?P<mime>[^;]+);base64,(?P<data>.+)", uploaded_str)
        if data_uri_match:
            mime = data_uri_match.group('mime')
            data_b64 = data_uri_match.group('data')
            ext = 'mp4'
            if 'webm' in mime: ext = 'webm'
            elif 'quicktime' in mime or 'mov' in mime: ext = 'mov'
            local_video_path = os.path.join(os.getcwd(), f"{timestamp_str}_uploaded.{ext}")
            print(f"Decoding data URI to {local_video_path} (mime={mime})")
            with open(local_video_path, 'wb') as fw:
                fw.write(base64.b64decode(data_b64))
            created_local_copy = True
        elif re.match(r'https?://', uploaded_str):
            # Remote URL: download it
            local_video_path = os.path.join(os.getcwd(), f"{timestamp_str}_" + os.path.basename(uploaded_str))
            print(f"Downloading remote video URL to {local_video_path}")
            urllib.request.urlretrieve(uploaded_str, local_video_path)
            created_local_copy = True
        else:
            # Treat as local temp path provided by Gradio
            base_name = os.path.basename(uploaded_str)
            # If the provided path already exists on disk, use it directly (no extra copy)
            if os.path.exists(uploaded_str):
                local_video_path = uploaded_str
                created_local_copy = False
                print(f"Using provided local path without copying: {local_video_path}")
            else:
                local_video_path = os.path.join(os.getcwd(), f"{timestamp_str}_{base_name}")
                try:
                    print(f"Copying video from {uploaded_str} to {local_video_path}")
                    shutil.copy2(uploaded_str, local_video_path)
                    created_local_copy = True
                    print(f"Video copied successfully to {local_video_path}")
                except Exception as e:
                    # Fallback: try reading and writing as binary
                    try:
                        print(f"Copy failed with {e}; trying binary read/write fallback")
                        with open(uploaded_str, 'rb') as fr, open(local_video_path, 'wb') as fw:
                            fw.write(fr.read())
                        created_local_copy = True
                        print(f"Fallback copy succeeded to {local_video_path}")
                    except Exception as e2:
                        error_msg = f"Error copying video file: {e}; fallback error: {e2}\nInput: {uploaded_str}"
                        print(error_msg)
                        return None, error_msg
    except Exception as e:
        error_msg = f"Failed to prepare uploaded video: {e}\nInput: {uploaded_video}"
        print(error_msg)
        return None, error_msg

    local_feature_sequence = deque(maxlen=INPUT_TIMESTEPS)
    local_last_fall_event_time = 0 # ใช้ local_last_fall_event_time_sec เพื่อความชัดเจนว่าเป็นหน่วยวินาทีของวิดีโอ
    
    cap = cv2.VideoCapture(local_video_path)
    if not cap.isOpened():
        error_msg = f"Error: OpenCV cannot open video file at copied path: {local_video_path}"
        if os.path.exists(local_video_path): print(f"File size of '{local_video_path}': {os.path.getsize(local_video_path)} bytes")
        else: print(f"File '{local_video_path}' does not exist after copy attempt.")
        if created_local_copy and os.path.exists(local_video_path):
            try: os.remove(local_video_path)
            except Exception: pass
        return None, error_msg

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps) or fps < 1 or fps > 120: 
        fps = 25.0 # Default FPS, ensure it's float
    
    processed_frames_list = []
    overall_status_updates = []

    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=pose_complexity,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        
        frame_count = 0
        while cap.isOpened():
            success, original_bgr_frame = cap.read() # อ่าน frame มาเป็น BGR
            if not success:
                break
            
            frame_count += 1

            # *** START: การแก้ไขเรื่องสีและการวาด ***
            # สร้างสำเนาของ BGR frame สำหรับการวาดผลลัพธ์
            frame_for_display = original_bgr_frame.copy()

            # 1. แปลงเป็น RGB เฉพาะตอนส่งให้ MediaPipe
            image_rgb_for_mediapipe = cv2.cvtColor(original_bgr_frame, cv2.COLOR_BGR2RGB)
            image_rgb_for_mediapipe.flags.writeable = False
            results = pose.process(image_rgb_for_mediapipe)
            # image_rgb_for_mediapipe.flags.writeable = True # ไม่จำเป็นแล้ว

            # 2. Extract and Normalize Features
            current_features = extract_and_normalize_features(results)
            local_feature_sequence.append(current_features)
            
            # ... (ส่วนการทำนายผล prediction เหมือนเดิม) ...
            current_status_text_for_log = f"Frame {frame_count}: Collecting..." # สำหรับ log
            prediction_label = "no_fall"
            display_confidence_value = 0.0

            if len(local_feature_sequence) == INPUT_TIMESTEPS:
                model_input_data = np.array(local_feature_sequence, dtype=np.float32)
                model_input_data = np.expand_dims(model_input_data, axis=0)
                try:
                    interpreter.set_tensor(input_details[0]['index'], model_input_data)
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(output_details[0]['index'])
                    prediction_probability_fall = output_data[0][0]

                    if prediction_probability_fall >= FALL_CONFIDENCE_THRESHOLD:
                        prediction_label = "fall"
                        display_confidence_value = prediction_probability_fall
                    else:
                        prediction_label = "no_fall"
                        display_confidence_value = 1.0 - prediction_probability_fall
                    
                    current_status_text_for_log = f"Frame {frame_count}: {prediction_label.upper()} (Conf: {display_confidence_value:.2f})"

                    current_video_time_sec = frame_count / fps
                    if prediction_label == "fall":
                        if (current_video_time_sec - local_last_fall_event_time) > FALL_EVENT_COOLDOWN: # ใช้ local_last_fall_event_time
                            fall_message = f"Frame {frame_count} (~{current_video_time_sec:.1f}s): FALL DETECTED! (Conf: {prediction_probability_fall:.2f})"
                            print(fall_message)
                            overall_status_updates.append(fall_message)
                            local_last_fall_event_time = current_video_time_sec # อัปเดตเวลา
                except Exception as e:
                    print(f"Frame {frame_count}: Error during prediction: {e}")
                    current_status_text_for_log = f"Frame {frame_count}: Prediction Error"
                    display_confidence_value = 0.0
            
            # อัปเดต overall_status_updates โดยใช้ current_status_text_for_log
            if "FALL DETECTED" not in current_status_text_for_log and \
               (frame_count % int(fps*1) == 0 or (len(local_feature_sequence) == INPUT_TIMESTEPS and frame_count == INPUT_TIMESTEPS) or frame_count ==1) :
                 if "Collecting..." not in current_status_text_for_log or frame_count == 1 :
                    overall_status_updates.append(current_status_text_for_log)


            # 3. วาด Landmarks (ถ้ามี) บน frame_for_display (BGR)
            if results.pose_landmarks:
                # เพื่อให้ได้สี default ของ MediaPipe ที่ถูกต้องที่สุด, เราจะวาดบนสำเนา RGB ชั่วคราว
                # แล้วค่อยแปลงกลับมาเป็น BGR เพื่อใส่ใน frame_for_display
                temp_rgb_to_draw_landmarks = cv2.cvtColor(original_bgr_frame, cv2.COLOR_BGR2RGB).copy()
                mp.solutions.drawing_utils.draw_landmarks(
                    temp_rgb_to_draw_landmarks,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                )
                # ตอนนี้ frame_for_display ยังเป็น BGR ดั้งเดิม, เราจะเอา temp_rgb_to_draw_landmarks ที่วาดแล้ว
                # แปลงกลับเป็น BGR แล้วใช้เป็น frame_for_display ใหม่
                frame_for_display = cv2.cvtColor(temp_rgb_to_draw_landmarks, cv2.COLOR_RGB2BGR)
            # ถ้าไม่มี landmarks, frame_for_display จะยังคงเป็น original_bgr_frame.copy()

            # 4. วาด Text บน frame_for_display (BGR) ทางขวามือ
            font_face = cv2.FONT_HERSHEY_DUPLEX
            font_scale_status = 0.6
            thickness_status = 1
            font_scale_alert = 1
            thickness_alert = 2
            padding = 30 # ระยะห่างจากขอบ

            text_to_show_on_frame = f"{prediction_label.upper()} (Conf: {display_confidence_value:.2f})"
            if "Collecting" in current_status_text_for_log or "Error" in current_status_text_for_log: # ใช้ current_status_text_for_log
                 text_to_show_on_frame = current_status_text_for_log.split(': ')[-1]

            (text_w, text_h), _ = cv2.getTextSize(text_to_show_on_frame, font_face, font_scale_status, thickness_status)
            text_x_status = frame_for_display.shape[1] - text_w - padding
            text_y_status = padding + text_h

            status_color_bgr = (255, 255, 255) # เขียว (BGR)
            current_video_time_sec_for_alert_check = frame_count / fps
            if prediction_label == "fall" and not (current_video_time_sec_for_alert_check - local_last_fall_event_time < FALL_EVENT_COOLDOWN):
                status_color_bgr = (0, 165, 255) # สีส้ม (BGR)
            if "Error" in text_to_show_on_frame:
                status_color_bgr = (0,0,255) # สีแดง (BGR)

            cv2.putText(frame_for_display, text_to_show_on_frame, (text_x_status, text_y_status), font_face, font_scale_status, status_color_bgr, thickness_status, cv2.LINE_AA)
            
            if prediction_label == "fall" and (current_video_time_sec_for_alert_check - local_last_fall_event_time < FALL_EVENT_COOLDOWN):
                alert_text = "FALL DETECTED!"
                (alert_w, alert_h), _ = cv2.getTextSize(alert_text, font_face, font_scale_alert, thickness_alert)
                alert_x_pos = frame_for_display.shape[1] - alert_w - padding
                alert_y_pos = text_y_status + alert_h + padding // 2
                cv2.putText(frame_for_display, alert_text, (alert_x_pos, alert_y_pos), font_face, font_scale_alert, (0, 0, 255), thickness_alert, cv2.LINE_AA) # สีแดง (BGR)
            
            # *** END ***
            processed_frames_list.append(frame_for_display) # เพิ่ม BGR frame ที่วาดแล้ว

    cap.release()

    if not processed_frames_list:
        if created_local_copy and os.path.exists(local_video_path):
            try: os.remove(local_video_path); print(f"Cleaned up temp copied file: {local_video_path}")
            except Exception as e: print(f"Could not remove temp copied file {local_video_path} after no frames: {e}")
        return None, "No frames processed. Video might be empty or unreadable after copy."

    # Build absolute output path for Gradio to serve reliably
    output_temp_video_path = os.path.abspath(f"processed_gradio_output_{timestamp_str}.mp4")
    height, width, _ = processed_frames_list[0].shape

    # Helper to write frames using OpenCV and a given fourcc
    def _write_video(path, fourcc_code):
        fourcc_val = cv2.VideoWriter_fourcc(*fourcc_code)
        writer = cv2.VideoWriter(path, fourcc_val, fps, (width, height))
        if not writer.isOpened():
            return False
        for frame_out_bgr in processed_frames_list:
            writer.write(frame_out_bgr)
        writer.release()
        return True

    written = _write_video(output_temp_video_path, 'mp4v')
    # Fallback: try AVI/MJPEG if mp4v fails
    if not written:
        fallback_path = os.path.abspath(f"processed_gradio_output_{timestamp_str}.avi")
        written = _write_video(fallback_path, 'XVID')
        if written:
            output_temp_video_path = fallback_path

    # Verify that the written file contains frames
    def _video_has_frames(path):
        try:
            cap_check = cv2.VideoCapture(path)
            cnt = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap_check.release()
            return cnt > 0
        except Exception:
            return False

    # Re-encode output to a web-friendly H.264 MP4 so browsers can play it reliably
    try:
        import subprocess, imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        web_mp4 = os.path.abspath(f"web_processed_gradio_output_{timestamp_str}.mp4")
        # Run ffmpeg to convert to H.264 / yuv420p which is broadly supported by browsers
        res = subprocess.run([
            ffmpeg_exe, "-y", "-i", output_temp_video_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-crf", "23",
            web_mp4
        ], capture_output=True, text=True)
        if res.returncode == 0 and os.path.exists(web_mp4) and os.path.getsize(web_mp4) > 1024:
            try:
                os.remove(output_temp_video_path)
            except Exception:
                pass
            output_temp_video_path = web_mp4
        else:
            print(f"FFMPEG re-encode failed or produced invalid output: {res.stderr}")
    except Exception as e:
        print(f"Could not re-encode video with ffmpeg: {e}")

    # Final validation: file must exist and be non-trivial
    if not os.path.exists(output_temp_video_path) or os.path.getsize(output_temp_video_path) < 1024:
        print(f"Processed video invalid or missing: {output_temp_video_path} (size={os.path.getsize(output_temp_video_path) if os.path.exists(output_temp_video_path) else 'NA'})")
        if created_local_copy and os.path.exists(local_video_path):
            try: os.remove(local_video_path); print(f"Cleaned up temp copied file: {local_video_path}")
            except Exception as e: print(f"Could not remove temp copied file {local_video_path}: {e}")
        return None, "Processed video could not be created (write/encode failure). Check server logs."

    print(f"Processed video saved to: {output_temp_video_path}")
    
    summary_text = "Recent Events / Status:\n" + "\n".join(overall_status_updates[-15:])

    if created_local_copy and os.path.exists(local_video_path):
        try: os.remove(local_video_path); print(f"Cleaned up temp copied file: {local_video_path}")
        except Exception as e: print(f"Could not remove temp copied file {local_video_path}: {e}")

    return output_temp_video_path, summary_text


# --- สร้าง Gradio Interface ---

# กำหนด list ของชื่อไฟล์ตัวอย่างของคุณ
example_filenames = [
    "fall_example_1.mp4",     # <<<< แก้ไขชื่อไฟล์ตามที่คุณใช้
    "fall_example_2.mp4",     # <<<< แก้ไขชื่อไฟล์ตามที่คุณใช้
    "fall_example_3.mp4",  # <<<< แก้ไขชื่อไฟล์ตามที่คุณใช้
    "fall_example_4.mp4"   # <<<< แก้ไขชื่อไฟล์ตามที่คุณใช้
]

examples_list_for_gradio = []
for filename in example_filenames:
    # ตรวจสอบว่าไฟล์ example มีอยู่ใน root directory ของ repo จริงๆ
    if os.path.exists(filename): # Gradio examples ต้องการแค่ชื่อไฟล์ (ถ้าอยู่ใน root)
        examples_list_for_gradio.append([filename]) # Gradio ต้องการ list ของ list
        print(f"Info: Example file '{filename}' found and added.")
    else:
        print(f"Warning: Example file '{filename}' not found in the repository root. It will not be added to examples.")

# -- Frontend Refactored (Blocks) --
with gr.Blocks(title="Hệ thống Cảnh Báo Té Ngã v2", css="footer {display: none !important;}") as demo:
    gr.Markdown("# Hệ thống Cảnh Báo Té Ngã (ver2) — Transformer TFLite")
    gr.Markdown(
        f"Video-level split training · ngưỡng fall = **{FALL_CONFIDENCE_THRESHOLD:.2f}** (từ `threshold.json` trên val). "
        "Hỗ trợ upload video và webcam realtime."
    )
    
    with gr.Tab("Xử lý Video (Độ Chính Xác Chuẩn)"):
        with gr.Row():
            video_input = gr.Video(label="Nạp Video / Mở Webcam chụp File")
        with gr.Row():
            video_output = gr.Video(label="Trang Trực Quan Khung Xương (Skeleton View)")
            log_output = gr.Textbox(label="Hiển thị Log Tỉ Lệ Ngã")
        
        btn = gr.Button("Phân tích Video Tĩnh", variant="primary")
        
        gr.Examples(examples=examples_list_for_gradio, inputs=video_input, outputs=[video_output, log_output], fn=process_video_for_gradio, cache_examples=False)

        btn.click(fn=process_video_for_gradio, inputs=video_input, outputs=[video_output, log_output])
        
    with gr.Tab("Webcam Live Real-Time (Beta)"):
        gr.Markdown("*Vui lòng cấp quyền Camera cho trình duyệt và đứng cách camera tầm 2m thấy đủ toàn thân. Tab này được loại bỏ nút Flag gây phiền phức.*")
        with gr.Row():
            image_in = gr.Image(sources=["webcam"], streaming=True, label="Đầu Vào Webcam RGB")
            image_out = gr.Image(label="Live Camera Kết Quả (Pose Inference)")
        # Toggle to enable/disable full skeleton drawing (keeps UI responsive when disabled)
        draw_full_toggle = gr.Checkbox(label="Vẽ khung xương đầy đủ (Full skeleton)", value=True)
        
        rt_log = gr.Textbox(label="Logs Tình Trạng Web-cam Time Queue", max_lines=10)
        
        state_rt = gr.State()
        image_in.stream(fn=process_frame_for_realtime, inputs=[image_in, state_rt, draw_full_toggle], outputs=[image_out, rt_log, state_rt])

if __name__ == "__main__":
    print("Starting Gradio Web Server...")
    demo.launch()

