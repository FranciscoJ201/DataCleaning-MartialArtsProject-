import cv2
import numpy as np
import os
import json
from ultralytics import YOLO

# --- CONFIGURATION ---
# Point this to the folder you copied from your field laptop
INPUT_DIR = 'realsense_field_recordings'  
OUTPUT_DIR = 'processed_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = 'yolo11x-pose.pt' # Use the high-accuracy model on your PC
CONFIDENCE_THRESHOLD = 0.1

# 1. Load the recorded data
manifest_path = os.path.join(INPUT_DIR, "recording_manifest.json")
intrinsics_path = os.path.join(INPUT_DIR, "camera_intrinsics.json")
video_path = os.path.join(INPUT_DIR, "temp_video.mp4") # Video file from fieldTrack.py
depth_dir = os.path.join(INPUT_DIR, "depth_maps")

with open(manifest_path, 'r') as f:
    manifest = json.load(f)

with open(intrinsics_path, 'r') as f:
    cam = json.load(f)

# Camera parameters for 3D reconstruction
fx, fy, cx, cy = cam['fx'], cam['fy'], cam['cx'], cam['cy']

# 2. Initialize Hardware
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(video_path)
all_processed_data = []

print(f"Starting processing for {len(manifest)} frames...")

for entry in manifest:
    # Use .get() to avoid crashing if a key is missing
    frame_idx = entry.get('frame_index')
    timestamp = entry.get('timestamp')
    
    # Read the next frame from the video
    ret, color_frame = cap.read()
    if not ret:
        break
    
    # Link to corresponding 16-bit depth PNG
    depth_filename = f"depth_{frame_idx:05d}.png"
    depth_path = os.path.join(depth_dir, depth_filename)
    
    if not os.path.exists(depth_path):
        print(f"Skipping frame {frame_idx}: Depth map not found.")
        continue
        
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    # 3. Run YOLO Inference
    results = model.predict(source=color_frame, verbose=False)
    frame_detections = []

    if results and results[0].keypoints.data.numel() > 0:
        result = results[0]
        # Use keypoints.data for raw values
        for pid, keypoint_data in enumerate(result.keypoints.data):
            person_3d = []
            for kp in keypoint_data:
                u, v, conf = int(kp[0]), int(kp[1]), float(kp[2])
                
                # Project 2D pixels to 3D meters using depth
                if conf >= CONFIDENCE_THRESHOLD and 0 <= v < depth_image.shape[0] and 0 <= u < depth_image.shape[1]:
                    z_mm = depth_image[v, u]
                    if z_mm > 0:
                        Z = z_mm / 1000.0 # Convert mm to meters
                        X = (u - cx) * (Z / fx)
                        Y = (v - cy) * (Z / fy)
                        person_3d.append([X, Y, Z, conf])
                        continue
                person_3d.append([None, None, None, conf])
            
            frame_detections.append({
                "person_id": pid,
                "keypoints_3d_m": person_3d
            })

    # 4. Store the final data
    all_processed_data.append({
        "frame_index": frame_idx,
        "timestamp": timestamp,
        "detections": frame_detections
    })

    if frame_idx % 50 == 0:
        print(f"Processed frame {frame_idx}...")

# 5. Export Final JSON
final_json_path = os.path.join(OUTPUT_DIR, "final_3d_pose_data.json")
with open(final_json_path, 'w') as f:
    json.dump(all_processed_data, f, indent=4)

cap.release()
print(f"SUCCESS! Final data saved to {final_json_path}")