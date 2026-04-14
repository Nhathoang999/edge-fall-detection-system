import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
print("tf imported", flush=True)
tflite = tf.lite
import time
from collections import deque
import gradio as gr
import os
import shutil # Dùng để sao chép file thư mục

# --- CẤU HÌNH HỆ THỐNG (Tinh chỉnh cho phù hợp với Gradio) ---
MODEL_PATH = 'fall_detection_transformer_v1.tflite' # Đường dẫn tới Model AI
INPUT_TIMESTEPS = 60 # Độ dài mỗi Sequence để nhét vào AI
FALL_CONFIDENCE_THRESHOLD = 0.90 # Tỉ lệ rơi ngã (90%)
MIN_KEYPOINT_CONFIDENCE_FOR_NORMALIZATION = 0.3 # Ngưỡng tự tin thấp nhất để trích xuất điểm ảnh Landmarks
mp_pose = mp.solutions.pose
pose_complexity = 0 # Thu nhỏ độ phức tạp của Pose về nhỏ nhất (= 0) để chạy webcam không bị giật lag
use_static_image_mode = False # Tắt chế độ static mode khi bắt đầu

FALL_EVENT_COOLDOWN = 10 # Giới hạn lặp thông báo (Tối thiểu 10 seconds cảnh báo 1 lần)

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

def process_frame_for_realtime(frame, state):
    from collections import deque
    import cv2
    
    if state is None:
        state = (0, deque(maxlen=INPUT_TIMESTEPS), "Trạng thái: Khởi chạy...")
        
    if frame is None:
        return frame, state

    import numpy as np
    frame = np.ascontiguousarray(np.copy(frame))
    frame.flags.writeable = True
        
    frame_count, local_feature_sequence, overall_status = state
    frame_count += 1
    
    try:
        
        import numpy as np

        with mp_pose.Pose(static_image_mode=False, model_complexity=0, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as global_realtime_pose:
            results = global_realtime_pose.process(frame)
            
            if not getattr(results, 'pose_landmarks', None):
                cv2.putText(frame, "No Person (Buffering...)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)
                return frame, (frame_count, local_feature_sequence, overall_status)
                
            current_features = extract_and_normalize_features(results)
                
            local_feature_sequence.append(current_features)
            
            prediction_label = "no_fall"
            display_confidence_value = 0.0
            text_to_show_on_frame = "Buffering frames..."
            color_status = (0, 255, 0)
            
            if len(local_feature_sequence) == INPUT_TIMESTEPS:
                sample_input = np.array([local_feature_sequence], dtype=np.float32)
                interpreter.set_tensor(input_details[0]['index'], sample_input)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                prediction_probability_fall = float(output_data[0][0])
                
                if prediction_probability_fall > FALL_CONFIDENCE_THRESHOLD:
                    prediction_label = "fall"
                    display_confidence_value = prediction_probability_fall
                    text_to_show_on_frame = f"FALL DETECTED (Prob: {display_confidence_value:.2f})"
                    overall_status = f"CẢNH BÁO TÉ NGÃ - Frame {frame_count}\n" + str(overall_status)
                else:
                    prediction_label = "no_fall"
                    display_confidence_value = 1.0 - prediction_probability_fall
                    text_to_show_on_frame = f"NORMAL (Prob: {display_confidence_value:.2f})"
                    
            if prediction_label == "fall":
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), 6)
                color_status = (255, 0, 0)
                

            mp.solutions.drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=results.pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
            cv2.putText(frame, text_to_show_on_frame, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_status, 3)
            
    except Exception as e:
        cv2.putText(frame, f"Error: {str(e)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
    return frame, (frame_count, local_feature_sequence, str(overall_status)[:500])

def process_video_for_gradio(uploaded_video_path_temp):
    if uploaded_video_path_temp is None:
        return None, "Please upload a video file."

    print(f"Gradio provided temp video path: {uploaded_video_path_temp}")
    base_name = os.path.basename(uploaded_video_path_temp)
    # สร้าง path ที่ unique มากขึ้นสำหรับไฟล์ที่ copy มา
    timestamp_str = str(int(time.time() * 1000)) # เพิ่ม timestamp เพื่อความ unique
    local_video_path = os.path.join(os.getcwd(), f"{timestamp_str}_{base_name}") 

    try:
        print(f"Copying video from {uploaded_video_path_temp} to {local_video_path}")
        shutil.copy2(uploaded_video_path_temp, local_video_path)
        print(f"Video copied successfully to {local_video_path}")
        
    except Exception as e:
        error_msg = f"Error copying video file: {e}\nTemp path: {uploaded_video_path_temp}"
        print(error_msg); return None, error_msg

    local_feature_sequence = deque(maxlen=INPUT_TIMESTEPS)
    local_last_fall_event_time = 0 # ใช้ local_last_fall_event_time_sec เพื่อความชัดเจนว่าเป็นหน่วยวินาทีของวิดีโอ
    
    cap = cv2.VideoCapture(local_video_path)
    if not cap.isOpened():
        error_msg = f"Error: OpenCV cannot open video file at copied path: {local_video_path}"
        if os.path.exists(local_video_path): print(f"File size of '{local_video_path}': {os.path.getsize(local_video_path)} bytes")
        else: print(f"File '{local_video_path}' does not exist after copy attempt.")
        if os.path.exists(local_video_path): os.remove(local_video_path) # Cleanup
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
        if os.path.exists(local_video_path):
            try: os.remove(local_video_path); print(f"Cleaned up temp copied file: {local_video_path}")
            except Exception as e: print(f"Could not remove temp copied file {local_video_path} after no frames: {e}")
        return None, "No frames processed. Video might be empty or unreadable after copy."

    output_temp_video_path = f"processed_gradio_output_{timestamp_str}.mp4"
    height, width, _ = processed_frames_list[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_temp_video_path, fourcc, fps, (width, height))
    for frame_out_bgr in processed_frames_list:
        video_writer.write(frame_out_bgr)
    video_writer.release()
    try:
        import subprocess, imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        web_mp4 = "web_" + output_temp_video_path
        res = subprocess.run([ffmpeg_exe, "-y", "-i", output_temp_video_path, "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "ultrafast", web_mp4], capture_output=True, text=True)
        if res.returncode != 0:
            print(f"FFMPEG OUT ERROR: {res.stderr}")
        elif os.path.exists(web_mp4):
            os.remove(output_temp_video_path)
            output_temp_video_path = web_mp4
    except Exception as e:
        print(f"Không thể re-encode video H264: {e}")
    print(f"Processed video saved to: {output_temp_video_path}")
    
    summary_text = "Recent Events / Status:\n" + "\n".join(overall_status_updates[-15:])

    if os.path.exists(local_video_path):
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
with gr.Blocks(title="Hệ thống Cảnh Báo Té Ngã", css="footer {display: none !important;}") as demo:
    gr.Markdown("# Hệ thống Cảnh Báo Té Ngã Sử Dụng LSTM TFLite")
    gr.Markdown("Nạp mô hình nhận diện qua ảnh tĩnh chậm rãi (Video) hoặc Camera Live streaming liên tiếp tốc độ cao.")
    
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
        
        rt_log = gr.Textbox(label="Logs Tình Trạng Web-cam Time Queue", max_lines=10)
        
        state_rt = gr.State()
        image_in.stream(fn=process_frame_for_realtime, inputs=[image_in, state_rt], outputs=[image_out, state_rt])
        def ext_log(s):
            return s[2] if s else "Trạng thái nghỉ"
        state_rt.change(fn=ext_log, inputs=state_rt, outputs=rt_log)

if __name__ == "__main__":
    print("Starting Gradio Web Server...")
    demo.launch()

