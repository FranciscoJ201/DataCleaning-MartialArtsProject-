import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO 
import time
import os
import json 
import sounddevice as sd
import soundfile as sf
import threading
import subprocess

# --- CONFIGURATION ---
OUTPUT_DIR = 'realsense_recordings'
os.makedirs(OUTPUT_DIR, exist_ok=True)

W, H, FPS = 640, 480, 30
CONFIDENCE_THRESHOLD = 0.1
TEMP_VIDEO = os.path.join(OUTPUT_DIR, "temp_video.mp4")
TEMP_AUDIO = os.path.join(OUTPUT_DIR, "temp_audio.wav")
FINAL_OUTPUT = os.path.join(OUTPUT_DIR, "final_fight_sync.mp4")

# --- AUDIO RECORDING LOGIC ---
audio_frames = []
recording_active = True

def audio_record_loop():
    """Background thread to record audio from default microphone."""
    global audio_frames
    fs = 44100  # Sample rate
    with sd.InputStream(samplerate=fs, channels=1) as stream:
        while recording_active:
            data, overflowed = stream.read(1024)
            audio_frames.append(data.copy())
    # Save to temp file once stopped
    full_audio = np.concatenate(audio_frames, axis=0)
    sf.write(TEMP_AUDIO, full_audio, fs)

# --- YOLO & CAMERA SETUP ---
model = YOLO('yolo11x-pose.engine') if os.path.exists('yolo11x-pose.engine') else YOLO('yolo11x-pose.pt')
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

# Video Writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(TEMP_VIDEO, fourcc, FPS, (W, H))

# Start Audio Thread
audio_thread = threading.Thread(target=audio_record_loop)
audio_thread.start()

all_pose_data = []

try:
    print("Recording... Press 'q' to stop.")
    frame_index = 0
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if not color_frame: continue

        color_image = np.asanyarray(color_frame.get_data())
        
        # YOLO Prediction
        results = model.predict(color_image, verbose=False)
        annotated_image = results[0].plot()
        
        # Save frame to video
        out.write(annotated_image)
        
        cv2.imshow("Fighter Tracking", annotated_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_index += 1

finally:
    # 1. Stop everything
    recording_active = False
    audio_thread.join()
    pipeline.stop()
    out.release()
    cv2.destroyAllWindows()

    # 2. Merge Audio and Video using FFmpeg
    print("Merging audio and video...")
    try:
        # This command combines the two temp files into one final mp4
        cmd = f'ffmpeg -y -i "{TEMP_VIDEO}" -i "{TEMP_AUDIO}" -c:v copy -c:a aac "{FINAL_OUTPUT}"'
        subprocess.run(cmd, shell=True, check=True)
        print(f"Success! Final file: {FINAL_OUTPUT}")
        
        # Optional: Clean up temp files
        os.remove(TEMP_VIDEO)
        os.remove(TEMP_AUDIO)
    except Exception as e:
        print(f"Merge failed. You still have {TEMP_VIDEO} and {TEMP_AUDIO}. Error: {e}")