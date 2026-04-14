import codecs, re
p = r'c:\KLTN\Fall-Detection\deployment\huggingface_space\app.py'
txt = codecs.open(p, 'r', 'utf-8').read()

new_function = '''
# ---------------------------------------------------------------------------------------------------
# --- Web-cam Realtime Logic ---

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
        
    frame_count, local_feature_sequence, overall_status = state
    frame_count += 1
    
    try:
        from common_keypoints import SELECTED_KEYPOINTS
        import numpy as np

        with mp_pose.Pose(static_image_mode=False, model_complexity=0, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as global_realtime_pose:
            results = global_realtime_pose.process(frame)
            if not getattr(results, 'pose_landmarks', None):
                cv2.putText(frame, "No Person (Buffering...)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)
                return frame, (frame_count, local_feature_sequence, overall_status)
                
            raw_keypoints = results.pose_landmarks.landmark
            normalized_keypoints = normalize_keypoints(raw_keypoints)
            
            current_features = []
            for i in SELECTED_KEYPOINTS:
                kp = normalized_keypoints[i]
                current_features.extend([kp.x, kp.y, kp.z, kp.visibility])
                
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
                    overall_status = f"CẢNH BÁO TÉ NGÃ - Frame {frame_count}\\n" + str(overall_status)
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
'''

frontend_pattern = r'iface\s*=\s*gr\.Interface\([\s\S]*?(?:if\s+__name__\s*==\s*"__main__":)'
frontend_replacement = '''# -- Frontend Refactored (Blocks) --
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
'''

if 'process_frame_for_realtime' not in txt:
    txt = txt.replace('def process_video_for_gradio', new_function + '\ndef process_video_for_gradio')

txt = re.sub(frontend_pattern, frontend_replacement, txt)
txt = txt.replace('iface.launch(', 'demo.launch(')

codecs.open(p, 'w', 'utf-8').write(txt)
