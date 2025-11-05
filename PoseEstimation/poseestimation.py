import json
from ultralytics import YOLO 
import numpy as np 


def poseestimate(source):
    # The pose model is still the base model
    model = YOLO('yolov8m-pose.pt') 

    sor= source
    # --- THE KEY CHANGE: USE .track() INSTEAD OF .predict() or model() ---
    results = model.track(
        source=sor, 
        tracker='botsort.yaml', 
        show=True, 
        conf=0.3, 
        save=True 
    )

    # --- NEW CODE TO EXTRACT AND OUTPUT DATA (modified to include confidence) ---
    all_detection_data = []

    for i, result in enumerate(results):
        
        # Safety checks
        if result.keypoints.data.numel() == 0 or result.boxes.data.numel() == 0:
            continue

        track_ids = result.boxes.id
        keypoints_tensor = result.keypoints.data
        
        # box_data usually shape (N, 7) or (N, 6). 
        # Confidence is typically index 4.
        box_data = result.boxes.data.cpu().numpy() 

        # Prepare Track IDs
        if track_ids is None:
            track_ids = [-1] * len(keypoints_tensor)
        else:
            track_ids = track_ids.cpu().numpy().astype(int).tolist()


        # Iterate through each person (detection) in the frame
        for j, keypoint_data in enumerate(keypoints_tensor):
            
            # Get the track ID safely
            track_id = track_ids[j] if j < len(track_ids) else -1
            
            # Bounding box coordinates (xywh format)
            box_xywh = result.boxes.xywh[j].cpu().numpy().round(1).tolist()
            
            # --- CRITICAL: Extract Confidence ---
            # Confidence is at index 4 of the box_data array.
            confidence = 0.0 
            if box_data.shape[1] > 4:
                # Index 4 is the confidence score
                confidence = float(box_data[j, 4])
            else:
                # Fallback for unexpected box format (shouldn't happen)
                confidence = 1.0 


            # Convert keypoints to standard Python list format
            keypoints_array = keypoint_data.cpu().numpy().tolist()

            # Create the digestable dictionary for JSON
            detection_record = {
                "frame_index": i,
                "track_id_native": track_id, # Native ID from YOLO (inconsistent)
                "bbox_xywh": box_xywh,
                "conf": confidence,          # <--- CONFIDENCE IS NOW GUARANTEED
                "keypoints_xyz": keypoints_array 
            }
            all_detection_data.append(detection_record)


    # Optional: Print a summary and save the JSON
    output_file = 'pose_detection_results.json' # Using the file name the tracker expects
    with open(output_file, 'w') as f:
            json.dump(all_detection_data, f, indent=4) 
    print(f"\nData extraction complete. Saved {len(all_detection_data)} detections to {output_file}")
    return output_file