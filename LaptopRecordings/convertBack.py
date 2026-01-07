import cv2
import numpy as np
import os
import json
from ultralytics import YOLO

# --- CONFIGURATION ---
INPUT_DIR = 'realsense_field_recordings'  # Path to your field laptop data
OUTPUT_FILE = 'final_3d_pose_data.json'
MODEL_PATH = 'yolo11n-pose.pt'  # Run the "X" model for max accuracy on PC

# Load inputs
manifest_path = os.path.join(INPUT_DIR, "recording_manifest.json")
intrinsics_path = os.path.join(INPUT_DIR, "camera_intrinsics.json")
video_path = os.path.join(INPUT_DIR, "raw_color_input.mp4")
depth_dir = os.path.join(INPUT_DIR, "depth_maps")

with open(manifest_path, 'r') as f:
    manifest = json.load(f)

with open(intrinsics_path, 'r') as f:
    cam = json.load(f)

# Camera params from the field recording
fx, fy, cx, cy = cam['fx'], cam['fy'], cam['cx'], cam['cy']

# Initialize Model
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(video_path)

all_processed_data = []

print(f"Processing {len(manifest)} frames...")

for entry in manifest:
    frame_idx = entry['frame_index']
    timestamp = entry['timestamp']
    depth_file = entry['depth_file']
    
    ret, color_image = cap.read()
    if not ret:
        break
        
    # Load the matching 16-bit depth PNG
    depth_path = os.path.join(depth_dir, depth_file)
    depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
    # Run YOLOv11 Pose Inference
    results = model.predict(source=color_image, verbose=False)
    frame_detections = []

    if results and len(results[0].keypoints.data) > 0:
        for pid, keypoint_data in enumerate(results[0].keypoints.data):
            person_3d = []
            for kp in keypoint_data:
                u, v, conf = int(kp[0]), int(kp[1]), float(kp[2])
                
                # Verify bounds and depth
                if 0 <= v < cam['height'] and 0 <= u < cam['width']:
                    z_mm = depth_map[v, u]
                    if z_mm > 0:
                        Z = z_mm / 1000.0  # Convert mm to meters
                        X = (u - cx) * (Z / fx)
                        Y = (v - cy) * (Z / fy)
                        person_3d.append([X, Y, Z, conf])
                        continue
                person_3d.append([None, None, None, conf])
            
            frame_detections.append({
                "person_id": pid, 
                "keypoints_3d_m": person_3d
            })

    # Save to final list
    all_processed_data.append({
        "frame_index": frame_idx,
        "timestamp": timestamp,
        "detections": frame_detections
    })

    if frame_idx % 30 == 0:
        print(f"Processed frame {frame_idx}...")

# Final Export
with open(OUTPUT_FILE, 'w') as f:
    json.dump(all_processed_data, f, indent=4)

cap.release()
print(f"DONE! Data saved to {OUTPUT_FILE}")