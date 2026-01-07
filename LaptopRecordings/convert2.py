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
# If you have a GPU, yolo11x-pose.pt is recommended; otherwise use yolo11n-pose.pt
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

    # 1. Load corresponding depth map
    # The recording script saves depth as depth_00000.png, depth_00001.png, etc.
    depth_path = os.path.join(DEPTH_DIR, f"depth_{frame_index:05d}.png")
    
    if not os.path.exists(depth_path):
        # If a specific depth frame is missing, we skip the 3d calc but keep the index in sync
        frame_detections = [] 
    else:
        depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        # 2. Run YOLO on the color frame
        results = model.predict(frame, verbose=False)
        frame_detections = []

        if results and results[0].keypoints.data.numel() > 0:
            # keypoints.data shape is [num_people, 17, 3] -> (x, y, conf)
            for pid, kpts in enumerate(results[0].keypoints.data):
                keypoints_3d = []
                for kp in kpts:
                    # Convert tensors/numpy to standard Python types for JSON
                    u, v, conf = int(kp[0]), int(kp[1]), float(kp[2])
                    
                    # Check image bounds
                    if 0 <= v < cam['height'] and 0 <= u < cam['width']:
                        z_mm = depth_map[v, u]
                        
                        if z_mm > 0:
                            # Convert to 3D Meters using Pinhole Camera Model
                            # Equations: X = (u - cx) * Z / fx | Y = (v - cy) * Z / fy
                            z = float(z_mm) / 1000.0
                            x = float((u - cam['cx']) * (z / cam['fx']))
                            y = float((v - cam['cy']) * (z / cam['fy']))
                            keypoints_3d.append([x, y, z, conf])
                            continue
                    
                    # If depth is 0 or point is out of bounds
                    keypoints_3d.append([None, None, None, conf])
                
                frame_detections.append({
                    "person_id": int(pid),
                    "keypoints_3d": keypoints_3d
                })

    # 3. Store results for this frame
    all_results.append({
        "frame_index": frame_index,
        "detections": frame_detections
    })

    if frame_index % 30 == 0:
        print(f"Processed {frame_index} frames...")
    
    frame_index += 1

cap.release()

# --- FINAL SAVE ---
# This part failed previously due to ndarray types; now fixed by explicit casting above
with open(OUTPUT_JSON, 'w') as f:
    json.dump(all_results, f, indent=4)

print(f"Successfully processed {frame_index} frames.")
print(f"Data saved to: {OUTPUT_JSON}")