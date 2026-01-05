import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO 
import time
import os
import json 

# --- DIRECTORY CONFIGURATION ---
OUTPUT_DIR = 'realsense_recordings'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- MODEL LOADING ---
ENGINE_PATH = 'yolo11x-pose.engine' 

if not os.path.exists(ENGINE_PATH):
    print(f"Engine file not found. Exporting to {ENGINE_PATH}...")
    model = YOLO('yolo11x-pose.pt')
    model.export(format='engine', half=True)
else:
    print(f"Loading existing engine: {ENGINE_PATH}")
    model = YOLO(ENGINE_PATH)

# --- CAMERA PARAMETERS ---
W, H = 640, 480
FPS = 30
CONFIDENCE_THRESHOLD = 0.1

# --- VIDEO WRITER CONFIG ---
# This creates the actual video file of your session
video_path = os.path.join(OUTPUT_DIR, "skeleton_tracking_output.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(video_path, fourcc, FPS, (W, H))

# Start the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS) 
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)

profile = pipeline.start(config)

# Get intrinsics for 3D mapping
color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
fx, fy, cx, cy = color_intrinsics.fx, color_intrinsics.fy, color_intrinsics.ppx, color_intrinsics.ppy

align = rs.align(rs.stream.color)

all_pose_data_for_json = []

try:
    frame_index = 0
    print("Recording started... Press 'q' to stop.")
    
    while True:
        start_time = time.time()
        
        frames = pipeline.wait_for_frames(10000)
        frame_timestamp = frames.get_timestamp() / 1000.0 
        
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # YOLOv11 Pose Estimation
        results = model.predict(source=color_image, verbose=False)
        frame_detections = [] 

        if results and results[0].keypoints.data.numel() > 0:
            result = results[0]
            
            # Draw skeletons on the image
            color_image = result.plot() 

            for pid, keypoint_data in enumerate(result.keypoints.data):
                person_3d_keypoints = [] 
                for kp in keypoint_data:
                    u, v, conf = int(kp[0]), int(kp[1]), float(kp[2])

                    if conf >= CONFIDENCE_THRESHOLD and 0 <= v < H and 0 <= u < W:
                        Z_mm = depth_image[v, u]
                        if Z_mm > 0:
                            Z = Z_mm / 1000.0
                            X = (u - cx) * (Z / fx)
                            Y = (v - cy) * (Z / fy)
                            person_3d_keypoints.append([X, Y, Z, conf])
                            continue
                    
                    person_3d_keypoints.append([None, None, None, conf])

                if len(person_3d_keypoints) == 17:
                    frame_detections.append({
                        "person_id": pid,
                        "keypoints_3d_m": person_3d_keypoints
                    })
        
        # Add metadata to the visual frame
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(color_image, f"FPS: {fps:.1f} | Frame: {frame_index}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # --- SAVE FRAME TO VIDEO ---
        out.write(color_image)

        # Collect data for JSON
        if frame_detections:
            all_pose_data_for_json.append({
                "frame_index": frame_index,
                "timestamp": frame_timestamp,
                "detections": frame_detections 
            })
        
        cv2.imshow("RealSense Tracking", color_image)
        frame_index += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # --- CLEANUP ---
    pipeline.stop()
    out.release() # Finalizes the MP4 file
    cv2.destroyAllWindows()
    
    # Save numeric data
    json_path = os.path.join(OUTPUT_DIR, "3d_pose_data.json")
    with open(json_path, 'w') as f:
        json.dump(all_pose_data_for_json, f, indent=4)

    print(f"\nProcessing Complete:")
    print(f"- Video saved: {video_path}")
    print(f"- Data saved: {json_path}")