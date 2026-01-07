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
os.makedirs(DEPTH_DIR, exist_ok=True)

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

# SAFETY: This class prevents the "ndarray is not JSON serializable" error
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)

def audio_record_loop():
    global audio_frames
    try:
        with sd.InputStream(samplerate=fs, channels=1) as stream:
            while recording_active:
                data, _ = stream.read(1024)
                audio_frames.append(data.copy())
        
        full_audio = np.concatenate(audio_frames, axis=0)
        sf.write(TEMP_AUDIO, full_audio, fs)
    except Exception as e:
        print(f"Audio Error: {e}")

# --- START AUDIO THREAD ---
audio_thread = threading.Thread(target=audio_record_loop)
audio_thread.start()

# --- REALSENSE SETUP ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30) 
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
profile = pipeline.start(config)

# Save intrinsics (Casting to float to avoid JSON errors)
color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
intrinsics_data = {
    "fx": float(color_intrinsics.fx), "fy": float(color_intrinsics.fy),
    "cx": float(color_intrinsics.ppx), "cy": float(color_intrinsics.ppy),
    "width": int(W), "height": int(H)
}
with open(os.path.join(OUTPUT_DIR, "camera_intrinsics.json"), 'w') as f:
    json.dump(intrinsics_data, f)

# Video Writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(TEMP_VIDEO, fourcc, TARGET_FPS, (W, H))
align = rs.align(rs.stream.color)
metadata_log = []

try:
    frame_index = 0
    print("RECORDING... Press 'q' to stop.")
    
    while True:
        start_time = time.perf_counter()
        frames = pipeline.wait_for_frames(10000)
        
        # Hardware timestamp from RealSense
        frame_timestamp = frames.get_timestamp() 
        
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # Save Depth Image (16-bit PNG)
        depth_filename = f"depth_{frame_index:05d}.png"
        cv2.imwrite(os.path.join(DEPTH_DIR, depth_filename), depth_image)
        
        # Save Color Frame
        out.write(color_image)

        # Logging (Ensuring types are native Python types)
        metadata_log.append({
            "frame_index": int(frame_index),
            "timestamp": float(frame_timestamp),
            "depth_file": depth_filename
        })
        
        cv2.imshow("Field Recorder (Low Power)", color_image)
        frame_index += 1
        
        # Timing Control
        elapsed = time.perf_counter() - start_time
        wait = FRAME_DURATION - elapsed
        if wait > 0:
            time.sleep(wait)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("\nStopping and saving data...")
    recording_active = False
    audio_thread.join()
    pipeline.stop()
    out.release() 
    cv2.destroyAllWindows()
    
    # Save the manifest using the Custom Encoder as a safety net
    with open(os.path.join(OUTPUT_DIR, "recording_manifest.json"), 'w') as f:
        json.dump(metadata_log, f, indent=4, cls=NumpyEncoder)

    print("Muxing audio and video...")
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', TEMP_VIDEO,
            '-i', TEMP_AUDIO,
            '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental',
            FINAL_OUTPUT
        ]
        subprocess.run(cmd, check=True)
        print(f"SUCCESS! Final file: {FINAL_OUTPUT}")
    except Exception as e:
        print(f"FFmpeg Error: {e}")