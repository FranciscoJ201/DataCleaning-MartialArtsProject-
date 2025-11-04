import json
from ultralytics import YOLO 
#8x or 8m for faster likely just use 8m its quicker
model = YOLO('yolov8m-pose.pt')
sor= '/Users/franciscojimenez/Desktop/saved.mp4'
#source 0 for camera or sor for video
results = model(source=sor, show=True, conf=0.3, save=True)

# --- NEW CODE TO EXTRACT AND OUTPUT DATA ---
for i, result in enumerate(results):
    # result is a Results object for a single frame
    
    # 1. Get the raw tensor data for all detections in the current frame
    keypoints_tensor = result.keypoints.data 
    
    # 2. Iterate through each person (detection) in the frame
    for j, keypoint_data in enumerate(keypoints_tensor):
        # keypoint_data shape is (N, 3), where N=17 is the number of keypoints (x, y, confidence)
        
        # Convert the tensor to a NumPy array for easier processing
        keypoints_array = keypoint_data.cpu().numpy()
        
        # 3. Print or save the extracted data
        print(f"--- Frame {i + 1}, Person {j + 1} ---")
        
        # Bounding box coordinates (x, y, w, h format)
        box = result.boxes.xywh[j].cpu().numpy()
        print(f"BBox (xywh): {box.round(1)}")

        # Print all 17 keypoints (x, y, conf)
        print("Keypoints (x, y, confidence):")
        
        # You can access specific keypoints by index (e.g., keypoints_array[0] is the nose)
        for k, (x, y, conf) in enumerate(keypoints_array):
            # Print index (k) of the keypoint, its (x, y) coordinates, and confidence
            print(f"  {k:2d}: ({x:5.1f}, {y:5.1f}) | Conf: {conf:.4f}")

# Optional: Print a summary
output_file = 'test.json'
with open(output_file, 'w') as f:
        # Use json.dump to save the entire dictionary
        json.dump(calibration_data, f, indent=4) 
print("\nData extraction complete.")

