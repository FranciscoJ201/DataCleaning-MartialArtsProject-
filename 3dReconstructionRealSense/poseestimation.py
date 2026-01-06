import json
from ultralytics import YOLO
import numpy as np
import os

def poseestimateCPU(source):
    model = YOLO('yolov8n-pose.pt') 
    sor=source
    
    results = model.track(
        source=sor, 
        tracker='botsort.yaml', 
        show=True, 
        conf=0.3, 
        save=False 
    )
    base_name = os.path.basename(sor)
    video_name, _ = os.path.splitext(base_name)

    # --- CODE TO EXTRACT AND OUTPUT DATA  ---
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
    output_file = f'{video_name}_pose_detection.json' # Using the file name the tracker expects
    with open(output_file, 'w') as f:
            json.dump(all_detection_data, f, indent=4) 
    print(f"\nData extraction complete. Saved {len(all_detection_data)} detections to {output_file}")
    return output_file

def poseestimateGPU(source):
    # --- 1. OPTIMIZED MODEL LOADING ---
    # Use the .engine file for 5x speedup on GPU
    # If the .engine doesn't exist, it will export from the .pt automatically
    engine_path = 'yolo11x-pose.engine'
    if not os.path.exists(engine_path):
        print(f"Exporting optimized GPU engine: {engine_path}...")
        model = YOLO('yolo11x-pose.pt')
        model.export(format='engine', half=True, device=0) 
    
    model = YOLO(engine_path)

    base_name = os.path.basename(source)
    video_name, _ = os.path.splitext(base_name)
    all_detection_data = []

    # --- 2. OPTIMIZED TRACKING PARAMETERS ---
    # stream=True is critical for speed as it yields results one by one
    # device=0 and half=True match your realsensetrack.py
    results_generator = model.track(
        source=source, 
        tracker='botsort.yaml', 
        show=True, 
        conf=0.3, 
        save=False,
        device=0,      # Explicitly use RTX 5080
        half=True,    # Use FP16 precision
        stream=True,   # Process frame-by-frame instead of loading all at once
        persist=True   # Required for consistent tracking IDs
    )

    print("GPU Processing started...")
    
    for i, result in enumerate(results_generator):
        # 3. SAFETY & BATCH TRANSFER
        # Moving all result data to CPU at once is much faster than index-by-index
        if result.keypoints is None or result.keypoints.data.numel() == 0:
            continue

        # Convert entire tensors to CPU/Numpy once per frame
        keypoints_tensor = result.keypoints.data.cpu().numpy()
        box_data = result.boxes.data.cpu().numpy() 
        track_ids = result.boxes.id
        
        # Prepare track IDs safely
        if track_ids is None:
            track_ids_list = [-1] * len(keypoints_tensor)
        else:
            track_ids_list = track_ids.cpu().numpy().astype(int).tolist()

        # 4. DATA EXTRACTION
        for j, keypoint_array in enumerate(keypoints_tensor):
            track_id = track_ids_list[j] if j < len(track_ids_list) else -1
            
            # Confidence is at index 4 of box_data
            confidence = float(box_data[j, 4]) if box_data.shape[1] > 4 else 1.0
            
            # Get xywh from boxes (already moved to CPU efficiently)
            box_xywh = result.boxes.xywh[j].cpu().numpy().round(1).tolist()

            all_detection_data.append({
                "frame_index": i,
                "track_id_native": track_id,
                "bbox_xywh": box_xywh,
                "conf": confidence,
                "keypoints_xyz": keypoint_array.tolist() 
            })

    # Save output
    output_file = f'{video_name}_gpu_pose.json'
    with open(output_file, 'w') as f:
        json.dump(all_detection_data, f, indent=4) 
            
    print(f"\nGPU Finish: {len(all_detection_data)} detections saved to {output_file}")
    return output_file