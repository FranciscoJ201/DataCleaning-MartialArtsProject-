import cv2
import numpy as np
from ultralytics import YOLO
import sounddevice as sd
import soundfile as sf
import threading
import time
import os
import subprocess

# --- CONFIGURATION ---
OUTPUT_DIR = 'webcam_recordings'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEMP_VIDEO = os.path.join(OUTPUT_DIR, "temp_video.mp4")
TEMP_AUDIO = os.path.join(OUTPUT_DIR, "temp_audio.wav")
FINAL_OUTPUT = os.path.join(OUTPUT_DIR, "final_synced_video.mp4")

# --- AUDIO RECORDING ---
audio_frames = []
recording_active = True

def audio_record_loop():
    global audio_frames
    fs = 44100
    with sd.InputStream(samplerate=fs, channels=1) as stream:
        while recording_active:
            data, _ = stream.read(1024)
            audio_frames.append(data.copy())
    full_audio = np.concatenate(audio_frames, axis=0)
    sf.write(TEMP_AUDIO, full_audio, fs)

# --- INITIALIZE HARDWARE ---
cap = cv2.VideoCapture(0)
model = YOLO('yolo11n-pose.pt') 

# Get actual frame size from camera
ret, first_frame = cap.read()
if not ret:
    print("Failed to access camera")
    exit()

H, W, _ = first_frame.shape
print(f"Detected Resolution: {W}x{H}")

# Use 'avc1' or 'mp4v' - avc1 is often more compatible with Mac
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter(TEMP_VIDEO, fourcc, 30.0, (W, H))

audio_thread = threading.Thread(target=audio_record_loop)
audio_thread.start()

print("Recording... Press 'q' to stop.")
frame_count = 0
start_time = time.time()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = model.predict(frame, verbose=False)
        annotated_frame = results[0].plot()
        
        # CRITICAL FIX: Ensure the frame is resized exactly to what the writer expects
        annotated_frame = cv2.resize(annotated_frame, (W, H))
        
        out.write(annotated_frame)
        frame_count += 1
        
        cv2.imshow("Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    end_time = time.time()
    total_duration = end_time - start_time
    recording_active = False
    audio_thread.join()
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Calculate Effective FPS
    effective_fps = frame_count / total_duration
    print(f"\nStats: {frame_count} frames in {total_duration:.2f}s ({effective_fps:.2f} FPS)")

    # --- FINAL MERGE ---
    if os.path.exists(TEMP_AUDIO):
        print("Merging with FFmpeg...")
        try:
            # We use the calculated effective_fps to stretch/shrink video to match audio
            cmd = (
                f'ffmpeg -y -r {effective_fps} -i "{TEMP_VIDEO}" -i "{TEMP_AUDIO}" '
                f'-c:v libx264 -pix_fmt yuv420p -c:a aac -b:a 192k "{FINAL_OUTPUT}"'
            )
            subprocess.run(cmd, shell=True, check=True)
            print(f" FINAL SYNCED VIDEO: {FINAL_OUTPUT}")
            
            # Clean up
            os.remove(TEMP_VIDEO)
            os.remove(TEMP_AUDIO)
        except Exception as e:
            print(f" Merge failed. Temp files saved in {OUTPUT_DIR}. Error: {e}")