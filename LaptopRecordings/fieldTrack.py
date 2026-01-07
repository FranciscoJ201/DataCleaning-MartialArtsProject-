import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
import json

# --- CONFIGURATION ---
OUTPUT_DIR = 'realsense_field_recordings'
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEPTH_DIR = os.path.join(OUTPUT_DIR, "depth_maps")
os.makedirs(DEPTH_DIR, exist_ok=True)

W, H = 640, 480
TARGET_FPS = 30  # RealSense native 30fps is more stable than 29.97
FRAME_DURATION = 1.0 / TARGET_FPS 

# --- VIDEO WRITER CONFIG ---
# Saving raw BGR video for later processing
video_path = os.path.join(OUTPUT_DIR, "raw_color_input.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(video_path, fourcc, TARGET_FPS, (W, H))

# --- REALSENSE SETUP ---
pipeline = rs.pipeline()
config = rs.config()
# Ensure depth and color are requested at the same rate
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30) 
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
profile = pipeline.start(config)

# Save intrinsics to a JSON file - You will need these for 3D projection later!
color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
intrinsics_data = {
    "fx": color_intrinsics.fx,
    "fy": color_intrinsics.fy,
    "cx": color_intrinsics.ppx,
    "cy": color_intrinsics.ppy,
    "width": color_intrinsics.width,
    "height": color_intrinsics.height
}
with open(os.path.join(OUTPUT_DIR, "camera_intrinsics.json"), 'w') as f:
    json.dump(intrinsics_data, f)

align = rs.align(rs.stream.color)

# Placeholder list to keep index/timestamp mapping
metadata_log = []

try:
    frame_index = 0
    print("RECORDING STARTED. Press 'q' to stop.")
    
    while True:
        start_time = time.perf_counter()
        
        # Wait for frames (timeout at 10s)
        frames = pipeline.wait_for_frames(10000)
        frame_timestamp = frames.get_timestamp() 
        
        # Align depth to color to ensure pixels match 1:1
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # 1. Save Raw Depth (16-bit PNG preserves millimeter precision)
        depth_filename = f"depth_{frame_index:05d}.png"
        cv2.imwrite(os.path.join(DEPTH_DIR, depth_filename), depth_image)

        # 2. Save Color Frame to Video
        out.write(color_image)

        # 3. Log Metadata (Placeholders for tracking data)
        metadata_log.append({
            "frame_index": frame_index,
            "timestamp": frame_timestamp,
            "depth_file": depth_filename,
            "detections": None  # Placeholder for your PC processing later
        })
        
        # Lightweight Preview
        cv2.putText(color_image, f"REC | Frame: {frame_index}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Field Recorder (Low Power)", color_image)
        
        frame_index += 1
        
        # Maintain FPS timing
        elapsed = time.perf_counter() - start_time
        sleep_time = FRAME_DURATION - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("\nWrapping up...")
    pipeline.stop()
    out.release() 
    cv2.destroyAllWindows()
    
    # Save the manifest
    with open(os.path.join(OUTPUT_DIR, "recording_manifest.json"), 'w') as f:
        json.dump(metadata_log, f, indent=4)
        
    print(f"Recording saved to {OUTPUT_DIR}")
    print(f"Total frames captured: {frame_index}")