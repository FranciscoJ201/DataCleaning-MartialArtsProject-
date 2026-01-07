import cv2
import numpy as np
import json
import os
from ultralytics import YOLO

# --- CONFIGURATION ---
INPUT_DIR = 'realsense_field_recordings'
DEPTH_DIR = os.path.join(INPUT_DIR, "depth_maps")
VIDEO_PATH = os.path.join(INPUT_DIR, 'final_sync_output.mp4')
INTRINSICS_PATH = os.path.join(INPUT_DIR, 'camera_intrinsics.json')
OUTPUT_JSON = os.path.join(INPUT_DIR, 'processed_3d_data.json')

# Load Model
model = YOLO('yolo11x-pose.pt') 

# Load Camera Settings
if not os.path.exists(INTRINSICS_PATH):
    print(f"Error: Could not find {INTRINSICS_PATH}")
    exit()

with open(INTRINSICS_PATH, 'r') as f:
    cam = json.load(f)

# --- PROCESSING LOOP ---
cap = cv2.VideoCapture(VIDEO_PATH)
frame_index = 0
all_results = []

print("Starting Post-Processing...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    depth_path = os.path.join(DEPTH_DIR, f"depth_{frame_index:05d}.png")
    
    frame_detections = [] 
    if os.path.exists(depth_path):
        depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        # Run YOLO with verbose=True so you can see progress
        results = model.predict(frame, verbose=True)

        if results and results[0].keypoints.data.numel() > 0:
            for pid, kpts in enumerate(results[0].keypoints.data):
                keypoints_3d_list = []
                for kp in kpts:
                    u, v, conf = int(kp[0]), int(kp[1]), float(kp[2])
                    
                    if 0 <= v < cam['height'] and 0 <= u < cam['width']:
                        z_mm = depth_map[v, u]
                        
                        if z_mm > 0:
                            z = float(z_mm) / 1000.0
                            x = float((u - cam['cx']) * (z / cam['fx']))
                            y = float((v - cam['cy']) * (z / cam['fy']))
                            # Format: [X, Y, Z, confidence]
                            keypoints_3d_list.append([x, y, z, conf])
                            continue
                    
                    keypoints_3d_list.append([None, None, None, conf])
                
                frame_detections.append({
                    "person_id": int(pid),
                    # CRITICAL: This key must match the visualizer's expected name
                    "keypoints_3d_m": keypoints_3d_list 
                })

    all_results.append({
        "frame_index": frame_index,
        "detections": frame_detections
    })

    frame_index += 1

cap.release()

# --- FINAL SAVE ---
with open(OUTPUT_JSON, 'w') as f:
    json.dump(all_results, f, indent=4)

print(f"Successfully processed {frame_index} frames.")
print(f"Data saved to: {OUTPUT_JSON}")