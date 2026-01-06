import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO 
import tkinter as tk
from tkinter import colorchooser
import colorsys
import os
import json
import time

# --- CONFIGURATION ---
W, H = 640, 480
ENGINE_PATH = 'yolo11x-pose.engine'
CONFIDENCE_THRESHOLD = 0.1
OUTPUT_DIR = 'realsense_recordings'
os.makedirs(OUTPUT_DIR, exist_ok=True)

class ColorPickerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Setup: Tracking Color")
        self.hsv_target = None
        
        tk.Label(root, text="Select Color to Track", font=("Arial", 12)).pack(pady=10)
        self.btn = tk.Button(root, text="Open Color Wheel", command=self.pick_color)
        self.btn.pack(pady=5)
        
        self.preview = tk.Label(root, text="None", width=20, height=2, bg="gray")
        self.preview.pack(pady=10)
        
        self.ready_btn = tk.Button(root, text="READY", state="disabled", 
                                   command=self.finish, bg="green", fg="white")
        self.ready_btn.pack(pady=10)

    def pick_color(self):
        color = colorchooser.askcolor()
        if color[1]:
            self.preview.config(bg=color[1], text=color[1])
            r, g, b = [x/255.0 for x in color[0]]
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            self.hsv_target = np.array([int(h*179), int(s*255), int(v*255)])
            self.ready_btn.config(state="normal")

    def finish(self):
        self.root.destroy()

def run_main_system(target_hsv):
    # --- MODEL LOADING ---
    if not os.path.exists(ENGINE_PATH):
        model = YOLO('yolo11x-pose.pt')
        model.export(format='engine', half=True)
    else:
        model = YOLO(ENGINE_PATH)

    # --- REALSENSE SETUP ---
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    # Align depth to color
    align = rs.align(rs.stream.color)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    
    all_pose_data = []
    frame_index = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # 1. COLOR TRACKING (RED DOT)
            hsv_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            lower = np.array([max(0, target_hsv[0]-15), 100, 100])
            upper = np.array([min(179, target_hsv[0]+15), 255, 255])
            mask = cv2.inRange(hsv_frame, lower, upper)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > 500:
                    M = cv2.moments(largest)
                    if M["m00"] != 0:
                        cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                        cv2.circle(color_image, (cX, cY), 10, (0,0,255), -1)

            # 2. YOLO POSE + 3D DATA COLLECTION
            results = model.predict(source=color_image, verbose=False)
            frame_detections = []

            if results and results[0].keypoints.data.numel() > 0:
                color_image = results[0].plot()
                for pid, kpts in enumerate(results[0].keypoints.data):
                    kp_3d = []
                    for kp in kpts:
                        u, v, conf = int(kp[0]), int(kp[1]), float(kp[2])
                        if conf >= CONFIDENCE_THRESHOLD and 0 <= v < H and 0 <= u < W:
                            z_mm = depth_image[v, u]
                            if z_mm > 0:
                                z = z_mm / 1000.0
                                x = (u - intr.ppx) * (z / intr.fx)
                                y = (v - intr.ppy) * (z / intr.fy)
                                kp_3d.append([x, y, z, conf])
                                continue
                        kp_3d.append([None, None, None, conf])
                    frame_detections.append({"person_id": pid, "keypoints_3d_m": kp_3d})

            # 3. SAVE DATA FOR JSON
            all_pose_data.append({
                "frame_index": frame_index,
                "timestamp": time.time(),
                "detections": frame_detections
            })

            # 4. DISPLAY
            cv2.imshow("Tracking & Skeleton", color_image)
            cv2.imshow("Mask", mask)
            
            frame_index += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        
        # Save JSON
        with open(os.path.join(OUTPUT_DIR, "3d_pose_and_color_data.json"), 'w') as f:
            json.dump(all_pose_data, f, indent=4)
        print(f"Session saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ColorPickerGUI(root)
    root.mainloop()

    if app.hsv_target is not None:
        run_main_system(app.hsv_target)