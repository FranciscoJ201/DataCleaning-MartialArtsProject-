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
TARGET_FPS = 29.97  #
FRAME_DURATION = 1.0 / TARGET_FPS 
CONFIDENCE_THRESHOLD = 0.1
OUTPUT_DIR = 'realsense_recordings'
os.makedirs(OUTPUT_DIR, exist_ok=True)
ENGINE_PATH = 'yolo11x-pose.engine'  #

class FullSetupGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("System Configuration")
        self.hsv_target = None
        self.sensitivity = 20 

        # 1. Color Selection
        tk.Label(root, text="Step 1: Tracking Color", font=("Arial", 10, "bold")).pack(pady=5)
        self.color_btn = tk.Button(root, text="Pick Color", command=self.pick_color)
        self.color_btn.pack()
        self.preview = tk.Label(root, text="None Selected", width=20, bg="gray")
        self.preview.pack(pady=5)

        # 2. Sensitivity Slider
        tk.Label(root, text="Step 2: Hue Sensitivity (Threshold)", font=("Arial", 10, "bold")).pack(pady=5)
        self.sens_slider = tk.Scale(root, from_=5, to=50, orient="horizontal")
        self.sens_slider.set(20)
        self.sens_slider.pack(fill="x", padx=20)

        # 3. Ready Button
        self.ready_btn = tk.Button(root, text="START TRACKING", state="disabled", 
                                   command=self.finish, bg="green", fg="white", height=2)
        self.ready_btn.pack(pady=20, fill="x", padx=20)

    def pick_color(self):
        color = colorchooser.askcolor() #
        if color[1]:
            self.preview.config(bg=color[1], text=color[1])
            r, g, b = [x/255.0 for x in color[0]]
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            self.hsv_target = np.array([int(h*179), int(s*255), int(v*255)])
            self.ready_btn.config(state="normal")

    def finish(self):
        self.sensitivity = self.sens_slider.get()
        self.root.destroy()

def run_system(target_hsv, sensitivity):
    # --- MODEL LOADING ---
    if not os.path.exists(ENGINE_PATH):
        model = YOLO('yolo11x-pose.pt')
        model.export(format='engine', half=True)
    else:
        model = YOLO(ENGINE_PATH)

    # --- VIDEO WRITER ---
    video_path = os.path.join(OUTPUT_DIR, "skeleton_color_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(video_path, fourcc, TARGET_FPS, (W, H))

    # --- REALSENSE SETUP ---
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30) 
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    
    align = rs.align(rs.stream.color)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    
    all_pose_data = []
    frame_index = 0

    try:
        print(f"Recording/Processing at {TARGET_FPS} FPS...")
        while True:
            start_time = time.perf_counter() # Precision timing
            
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # --- COLOR TRACKING ---
            hsv_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            lower = np.array([max(0, target_hsv[0] - sensitivity), 100, 100])
            upper = np.array([min(179, target_hsv[0] + sensitivity), 255, 255])
            mask = cv2.inRange(hsv_frame, lower, upper)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > 400:
                    M = cv2.moments(largest)
                    if M["m00"] != 0:
                        cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                        cv2.circle(color_image, (cX, cY), 10, (0,0,255), -1)

            # --- YOLO POSE & 3D CALCULATION ---
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

            # --- DATA LOGGING ---
            all_pose_data.append({
                "frame_index": frame_index,
                "timestamp": frames.get_timestamp(),
                "detections": frame_detections
            })
            
            out.write(color_image)

            # --- DISPLAY & FPS CAPPER ---
            cv2.putText(color_image, f"CAP: {TARGET_FPS} | Frame: {frame_index}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Main Tracking Feed", color_image)
            cv2.imshow("Threshold Mask", mask)

            frame_index += 1

            # Precision FPS Capping logic
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
        with open(os.path.join(OUTPUT_DIR, "3d_combined_data.json"), 'w') as f:
            json.dump(all_pose_data, f, indent=4)
        print(f"Data saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FullSetupGUI(root)
    root.mainloop()

    if app.hsv_target is not None:
        run_system(app.hsv_target, app.sensitivity)