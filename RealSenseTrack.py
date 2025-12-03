import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO 
import time
import os
import json 


# --- OUTPUT DIRECTORY CONFIGURATION ---
OUTPUT_DIR = "realsense_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory created: {OUTPUT_DIR}")

# --- CAMERA/YOLO CONFIG ---
model = YOLO('yolov8n-pose.pt') 
W, H = 640, 480
FPS = 10
CONFIDENCE_THRESHOLD = 0.5 # Added minimum confidence for a keypoint to be processed

# Start the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure streams 
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS) 
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)

# Start streaming and get profile
profile = pipeline.start(config)

# Get the intrinsic parameters (K matrix) for the COLOR stream
color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
fx = color_intrinsics.fx
fy = color_intrinsics.fy
cx = color_intrinsics.ppx
cy = color_intrinsics.ppy

print(f"RealSense Intrinsics loaded: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

# Create an align object for mapping Depth to Color pixels
align_to = rs.stream.color
align = rs.align(align_to)

# --- JSON DATA COLLECTOR ---
all_pose_data_for_json = []

try:
    frame_index = 0
    while True:
        start_time = time.time()
        
        # 1. CAPTURE AND ALIGN FRAME-SET
        frames = pipeline.wait_for_frames(10000)
        
        # Get frame timestamp immediately after capture
        # RealSense timestamp is in milliseconds since the epoch, convert to seconds
        frame_timestamp = frames.get_timestamp() / 1000.0 
        
        aligned_frames = align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # 2. SAVE RAW DEPTH MAP
        depth_filename = os.path.join(OUTPUT_DIR, f"depth_frame_{frame_index:05d}.png")
        cv2.imwrite(depth_filename, depth_image)

        # 3. YOLOv8 2D POSE ESTIMATION
        results = model.predict(source=color_image, verbose=False)
        
        frame_detections = [] # Holds 3D data for all people in this frame

        # Check for detections
        if not results or results[0].keypoints.data.numel() == 0:
            frame_index += 1
            continue
            
        result = results[0] # All persons are in this single result object

        # 4. PROCESS DETECTIONS AND PERFORM 3D CONVERSION (Multi-person handling)
        for pid, keypoint_data in enumerate(result.keypoints.data):
            
            # keypoint_data has shape (17, 3) -> [u, v, confidence] for one person
            person_3d_keypoints = [] 

            # Loop over the 17 keypoints for the current person
            for kp in keypoint_data:
                u, v = int(kp[0]), int(kp[1])
                confidence = float(kp[2])

                # --- NEW/IMPROVED LOGIC START ---
                
                # 1. Check Confidence
                if confidence >= CONFIDENCE_THRESHOLD:
                    
                    # 2. Check 2D Bounds
                    if 0 <= v < H and 0 <= u < W:
                        Z_mm = depth_image[v, u]
                        
                        # 3. Check Depth Validity
                        if Z_mm > 0:
                            # SUCCESS: Calculate 3D point
                            Z = Z_mm / 1000.0
                            X_m = (u - cx) * (Z / fx)
                            Y_m = (v - cy) * (Z / fy)
                            
                            # Append valid 3D coordinates
                            person_3d_keypoints.append([float(X_m), float(Y_m), float(Z), confidence])
                            continue # Skip the fallback

                # FALLBACK: Confidence too low, out of bounds, or invalid depth (Z=0).
                # Append placeholder to preserve the 17-point fixed structure.
                person_3d_keypoints.append([None, None, None, confidence])
                # --- NEW/IMPROVED LOGIC END ---

            
            # Only append the detection if we have 17 points (which we now always do)
            if len(person_3d_keypoints) == 17:
                frame_detections.append({
                    "person_id": pid, # Index order from YOLO output
                    "keypoints_3d_m": person_3d_keypoints
                })
        
        # 6. COLLECT FRAME DATA FOR JSON
        if frame_detections:
            all_pose_data_for_json.append({
                "frame_index": frame_index,
                "timestamp": frame_timestamp,
                "detections": frame_detections 
            })
        
        frame_index += 1
        
        # --- Visualization ---
        color_image = result.plot() 

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(color_image, f"Frames: {frame_index} | FPS: {fps:.2f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Live Pose Estimation (D455f)", color_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 7. STOP STREAMING AND SAVE FINAL JSON
    pipeline.stop()
    cv2.destroyAllWindows()
    
    json_output_path = os.path.join(OUTPUT_DIR, "3d_pose_reconstruction.json")
    with open(json_output_path, 'w') as f:
        json.dump(all_pose_data_for_json, f, indent=4)

    print(f"\nLive 3D processing stopped after {frame_index} frames.")
    print(f"Raw depth maps saved to: {OUTPUT_DIR}")
    print(f"Final 3D pose data saved to: {json_output_path}")