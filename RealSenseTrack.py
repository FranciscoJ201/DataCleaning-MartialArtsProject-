import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO 
import time
import os
import json 

# --- CONFIGURATION ---
OUTPUT_DIR = 'realsense_recordings'
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEPTH_DIR = os.path.join(OUTPUT_DIR, "depth_maps")
os.makedirs(DEPTH_DIR, exist_ok=True)

ENGINE_PATH = 'yolo11x-pose.engine' 
W, H = 640, 480
TARGET_FPS = 29.97
FRAME_DURATION = 1.0 / TARGET_FPS 
CONFIDENCE_THRESHOLD = 0.1

# --- MODEL LOADING ---
if not os.path.exists(ENGINE_PATH):
    print(f"Exporting engine...")
    model = YOLO('yolo11x-pose.pt')
    model.export(format='engine', half=True)
else:
    model = YOLO(ENGINE_PATH)

# --- VIDEO WRITER CONFIG ---
video_path = os.path.join(OUTPUT_DIR, "skeleton_tracking_output.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(video_path, fourcc, TARGET_FPS, (W, H))

# --- REALSENSE SETUP ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30) 
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
profile = pipeline.start(config)

color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
fx, fy, cx, cy = color_intrinsics.fx, color_intrinsics.fy, color_intrinsics.ppx, color_intrinsics.ppy
align = rs.align(rs.stream.color)

all_pose_data_for_json = []

try:
    frame_index = 0
    print(f"Recording/Processing started at {TARGET_FPS} FPS...")
    
    while True:
        start_time = time.perf_counter() # Precision timing for 5080
        
        frames = pipeline.wait_for_frames(10000)
        frame_timestamp = frames.get_timestamp() / 1000.0 
        aligned_frames = align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # 1. Save Raw Depth PNG (Capability from RealSenseTrack)
        depth_filename = os.path.join(DEPTH_DIR, f"depth_{frame_index:05d}.png")
        cv2.imwrite(depth_filename, depth_image)

        # 2. YOLO Inference
        results = model.predict(source=color_image, verbose=False)
        frame_detections = [] 

        if results and results[0].keypoints.data.numel() > 0:
            result = results[0]
            color_image = result.plot() # Draw skeletons for video

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
                    frame_detections.append({"person_id": pid, "keypoints_3d_m": person_3d_keypoints})
        
        # 3. Save Frame to Video (Capability from VideoTest)
        out.write(color_image)

        # 4. Data Collection
        if frame_detections:
            all_pose_data_for_json.append({
                "frame_index": frame_index,
                "timestamp": frame_timestamp,
                "detections": frame_detections 
            })
        
        # Visual Overlay
        cv2.putText(color_image, f"CAP: {TARGET_FPS} | Frame: {frame_index}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("RealSense Tracking Plus", color_image)
        
        frame_index += 1
        
        # --- FPS CAPPER ---
        elapsed = time.perf_counter() - start_time
        sleep_time = FRAME_DURATION - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    out.release() 
    cv2.destroyAllWindows()
    
    with open(os.path.join(OUTPUT_DIR, "3d_pose_data.json"), 'w') as f:
        json.dump(all_pose_data_for_json, f, indent=4)
    print(f"Data saved to {OUTPUT_DIR}")