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

# Load Model (Using Engine for speed if available)
model = YOLO('yolo11x-pose.pt') 

# Load Camera Settings
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

    # 1. Load corresponding depth map
    depth_path = os.path.join(DEPTH_DIR, f"depth_{frame_index:05d}.png")
    if not os.path.exists(depth_path):
        frame_index += 1
        continue
    
    depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    # 2. Run YOLO on the color frame
    results = model.predict(frame, verbose=False)
    frame_detections = []

    if results and results[0].keypoints.data.numel() > 0:
        for pid, kpts in enumerate(results[0].keypoints.data):
            keypoints_3d = []
            for kp in kpts:
                u, v, conf = int(kp[0]), int(kp[1]), float(kp[2])
                
                # Check bounds and confidence
                if 0 <= v < cam['height'] and 0 <= u < cam['width']:
                    z_mm = depth_map[v, u]
                    if z_mm > 0:
                        # Convert to 3D Meters
                        z = z_mm / 1000.0
                        x = (u - cam['cx']) * (z / cam['fx'])
                        y = (v - cam['cy']) * (z / cam['fy'])
                        keypoints_3d.append([x, y, z, conf])
                        continue
                
                keypoints_3d.append([None, None, None, conf])
            
            frame_detections.append({
                "person_id": pid,
                "keypoints_3d": keypoints_3d
            })

    all_results.append({
        "frame": frame_index,
        "detections": frame_detections
    })

    if frame_index % 30 == 0:
        print(f"Processed {frame_index} frames...")
    
    frame_index += 1

cap.release()

# Save final 3D data
with open(OUTPUT_JSON, 'w') as f:
    json.dump(all_results, f, indent=4)

print(f"Processing complete! Data saved to {OUTPUT_JSON}")