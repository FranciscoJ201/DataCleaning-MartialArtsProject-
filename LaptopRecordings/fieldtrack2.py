import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
import json
import sounddevice as sd
import soundfile as sf
import threading
import subprocess

# --- CONFIGURATION ---
OUTPUT_DIR = 'realsense_field_recordings'
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEPTH_DIR = os.path.join(OUTPUT_DIR, "depth_maps")
COLOR_DIR = os.path.join(OUTPUT_DIR, "color_frames") # For post-process YOLO
os.makedirs(DEPTH_DIR, exist_ok=True)
os.makedirs(COLOR_DIR, exist_ok=True)

W, H = 1280, 720
TARGET_FPS = 30 
FRAME_DURATION = 1.0 / TARGET_FPS 

# Audio Config
TEMP_AUDIO = os.path.join(OUTPUT_DIR, 'temp_audio.wav')
TEMP_VIDEO = os.path.join(OUTPUT_DIR, 'temp_video.mp4')
FINAL_OUTPUT = os.path.join(OUTPUT_DIR, 'final_sync_output.mp4')

audio_frames = []
recording_active = True
fs = 44100

def audio_record_loop():
    global audio_frames
    with sd.InputStream(samplerate=fs, channels=1) as stream:
        while recording_active:
            data, _ = stream.read(1024)
            audio_frames.append(data.copy())
    
    full_audio = np.concatenate(audio_frames, axis=0)
    sf.write(TEMP_AUDIO, full_audio, fs)

# --- START AUDIO THREAD ---
audio_thread = threading.Thread(target=audio_record_loop)
audio_thread.start()

# --- VIDEO WRITER CONFIG ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(TEMP_VIDEO, fourcc, TARGET_FPS, (W, H))

# --- REALSENSE SETUP ---
pipeline = rs.pipeline()
config = rs.config()
# Use Z16 for 16-bit raw depth data
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30) 
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
profile = pipeline.start(config)

# Save intrinsics - VITAL for later YOLO 3D calculations
color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
intrinsics_data = {
    "fx": color_intrinsics.fx, 
    "fy": color_intrinsics.fy,
    "cx": color_intrinsics.ppx, 
    "cy": color_intrinsics.ppy,
    "width": W,
    "height": H
}
with open(os.path.join(OUTPUT_DIR, "camera_intrinsics.json"), 'w') as f:
    json.dump(intrinsics_data, f, indent=4)

# Create alignment object (Align depth to color)
align = rs.align(rs.stream.color)
metadata_log = []

try:
    frame_index = 0
    print("RECORDING... Press 'q' to stop.")
    
    while True:
        start_time = time.perf_counter()
        frames = pipeline.wait_for_frames(10000)
        
        # Align frames
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # 1. Save Video Frame (for easy viewing)
        out.write(color_image)
        
        # 2. Save Raw Color PNG (Optional - better for YOLO than compressed MP4)
        # cv2.imwrite(os.path.join(COLOR_DIR, f"color_{frame_index:05d}.jpg"), color_image)

        # 3. Save Raw Depth PNG (16-bit)
        depth_path = os.path.join(DEPTH_DIR, f"depth_{frame_index:05d}.png")
        cv2.imwrite(depth_path, depth_image)

        metadata_log.append({
            "frame_index": frame_index,
            "timestamp": frames.get_timestamp(),
            "depth_file": f"depth_{frame_index:05d}.png"
        })
        
        # Live Preview (Visual feedback)
        cv2.putText(color_image, f"REC: {frame_index}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Field Recorder", color_image)
        
        frame_index += 1
        
        # Timing control to keep file size/FPS stable
        elapsed = time.perf_counter() - start_time
        wait = FRAME_DURATION - elapsed
        if wait > 0:
            time.sleep(wait)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("\nStopping streams and finalizing data...")
    recording_active = False 
    audio_thread.join()
    pipeline.stop()
    out.release() 
    cv2.destroyAllWindows()
    
    # Save the manifest which links frames to timestamps
    with open(os.path.join(OUTPUT_DIR, "recording_manifest.json"), 'w') as f:
        json.dump(metadata_log, f, indent=4)

    # --- FINAL STEP: COMBINE AUDIO AND VIDEO ---
    print("Muxing audio and video...")
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', TEMP_VIDEO,
            '-i', TEMP_AUDIO,
            '-c:v', 'copy',
            '-c:a', 'aac',
            FINAL_OUTPUT
        ]
        subprocess.run(cmd, check=True)
        print(f"Success! Final file: {FINAL_OUTPUT}")
    except Exception as e:
        print(f"Muxing failed. Error: {e}")