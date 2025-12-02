import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO # Import your YOLO library here
import time
import os

# --- Configuration ---
# Your YOLO model must be loaded here.
model = YOLO('yolov8n-pose.pt') 

# --- OUTPUT DIRECTORY CONFIGURATION ---
OUTPUT_DIR = "realsense_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory created: {OUTPUT_DIR}")

# Start the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure streams 
W, H = 640, 480
FPS = 30
# Note: Depth is enabled and will be saved
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

# Create an align object: crucial for mapping Depth to Color pixels
align_to = rs.stream.color
align = rs.align(align_to)

try:
    frame_index = 0
    while True:
        start_time = time.time()
        
        # 1. Capture and Align Frame-Set
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # --- NEW: SAVE DEPTH DATA ---
        depth_filename = os.path.join(OUTPUT_DIR, f"depth_frame_{frame_index:05d}.png")
        # Save the 16-bit depth map directly to a PNG file
        cv2.imwrite(depth_filename, depth_image)
        # --- END NEW ---

        # 2. YOLOv8 2D Pose Estimation
        results = model.predict(source=color_image, verbose=False)
        
        all_3d_keypoints = []

        # 3. Process Detections and Perform 3D Conversion
        for result in results:
            if result.keypoints.data.numel() == 0:
                continue

            keypoints_tensor = result.keypoints.data
            
            for keypoint_data in keypoints_tensor:
                for kp in keypoint_data:
                    u, v = int(kp[0]), int(kp[1])
                    confidence = float(kp[2])

                    if 0 <= v < H and 0 <= u < W:
                        # Depth Lookup (Z distance in millimeters)
                        Z_mm = depth_image[v, u]
                        
                        if Z_mm > 0:
                            # 4. 3D Conversion 
                            Z = Z_mm / 1000.0 # Convert to meters
                            
                            X_m = (u - cx) * (Z / fx) # Left/Right
                            Y_m = (v - cy) * (Z / fy) # Up/Down
                            
                            all_3d_keypoints.append([X_m, Y_m, Z, confidence])
        
        frame_index += 1
        
        # --- Visualization/Output ---
        if results:
             color_image = results[0].plot()

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(color_image, f"3D KPs: {len(all_3d_keypoints) // 17} | FPS: {fps:.2f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Live Pose Estimation (D455f)", color_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

print(f"\nLive 3D processing stopped after {frame_index} frames.")
print(f"Depth maps saved to: {OUTPUT_DIR}")