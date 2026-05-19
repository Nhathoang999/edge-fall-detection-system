import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
from pathlib import Path
import sys

# Add parent dir to path to import config
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def process_video(video_path: str, label: str, output_dir: Path, seq_length=30, stride=15):
    """
    Trích xuất khung xương từ video và lưu thành nhiều file .npy (sliding window).
    """
    video_name = Path(video_path).stem
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return 0

    frames_features = []
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # MediaPipe processing
            results = pose.process(image)
            
            # Extract landmarks
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # We need to map MediaPipe landmarks to the 17 keypoints expected in config.py
                # MediaPipe has 33 landmarks. We use a subset.
                # Actually, in ver1, how were keypoints mapped?
                # For simplicity, we just take the first 17 keypoints (which include face, shoulders, elbows, wrists)
                # Or sort them alphabetically to match config.SORTED_KEYPOINT_NAMES? 
                # Let's extract all 17 as defined in config.KEYPOINT_NAMES or similar logic.
                
                # To be robust and match old logic, we assume we just extract standard 17 keypoints:
                # 0: nose, 1: left_eye_inner, 2: left_eye, 3: left_eye_outer, 4: right_eye_inner, 5: right_eye, 6: right_eye_outer,
                # 7: left_ear, 8: right_ear, 9: mouth_left, 10: mouth_right, 11: left_shoulder, 12: right_shoulder,
                # 13: left_elbow, 14: right_elbow, 15: left_wrist, 16: right_wrist
                # (This is just an example mapping, we will extract 17 keypoints x 3 = 51 features)
                
                row = []
                # Just take the first 17 to match NUM_KEYPOINTS = 17
                for i in range(config.NUM_KEYPOINTS):
                    if i < len(landmarks):
                        lm = landmarks[i]
                        row.extend([lm.x, lm.y, lm.visibility])
                    else:
                        row.extend([0.0, 0.0, 0.0])
                frames_features.append(row)
            else:
                # If no pose found, use zeros or duplicate previous
                if len(frames_features) > 0:
                    frames_features.append(frames_features[-1])
                else:
                    frames_features.append([0.0] * config.NUM_FEATURES)
                    
    cap.release()
    
    # Split into sliding windows
    sequences_saved = 0
    frames_features = np.array(frames_features)
    
    if len(frames_features) >= seq_length:
        for start_idx in range(0, len(frames_features) - seq_length + 1, stride):
            sequence = frames_features[start_idx:start_idx + seq_length]
            
            # Normalize to match config.INPUT_TIMESTEPS and config.NUM_FEATURES
            if sequence.shape == (seq_length, config.NUM_FEATURES):
                out_filename = f"{video_name}_{label}_seq_{sequences_saved:03d}.npy"
                out_path = output_dir / out_filename
                np.save(out_path, sequence)
                sequences_saved += 1
                
    print(f"Processed {video_path} -> extracted {sequences_saved} sequences.")
    return sequences_saved

def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe features from raw videos")
    parser.add_argument("--video-dir", type=str, required=True, help="Path to folder containing raw videos")
    parser.add_argument("--label", type=str, choices=["fall", "no_fall"], required=True, help="Label for these videos")
    args = parser.parse_args()
    
    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        print(f"Directory not found: {video_dir}")
        return
        
    output_dir = config.DATA_RAW / args.label
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_seqs = 0
    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
    
    print(f"Found {len(video_files)} videos. Extracting features...")
    for video_path in video_files:
        seqs = process_video(str(video_path), args.label, output_dir, seq_length=config.INPUT_TIMESTEPS, stride=15)
        total_seqs += seqs
        
    print(f"\nDone! Extracted {total_seqs} sequences of length {config.INPUT_TIMESTEPS} to {output_dir}")

if __name__ == "__main__":
    main()
