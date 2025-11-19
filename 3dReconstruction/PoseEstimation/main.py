from poseestimation import poseestimate
from reid import recycle
from Plot2d import plot


Source = '/Users/franciscojimenez/Desktop/main.mp4' 
# RE-ID parameters (MFGFR, MBJF, MBOS, MBC)
# 50 frames gap, 25% screen diagonal jump factor, 1.0 bbox overlap score, 0.3 min confidence
REID_PARAMS = (50, 0.25, 1.0, 0.3) 
# ---------------------

print("--- STEP 1: Starting Pose Estimation and Data Extraction ---")
json_out = poseestimate(Source)

print("\n--- STEP 2: Starting Re-Identification (ID Recycling) ---")
clean_json_out = recycle(json_out, *REID_PARAMS)

print("\n--- STEP 3: Starting 2D Visualization ---")
plot(clean_json_out)

