#RE-ID tracking conditions
#The first index is 1 not 0 so keep that in mind. 
# use bounding boxes as a reference point, they should not be far off from their previous bounding box (check previous frames)
#check the similarity in pose from previous frames... (might not be needed and may even screw up floor tracking)
#calculate max pixels a person can move

import json
import numpy as np
from collections import defaultdict, deque
import math

# ----------------------------
# Tunables (Adjust these values for better recycling performance)
# ----------------------------
# MAX_FRAME_GAP_FOR_REAPPEARANCE: How many frames an ID can be missing before it's permanently retired.
MAX_FRAME_GAP_FOR_REAPPEARANCE = 50 

# MAX_BBOX_JUMP_FACTOR: Max distance a center can jump, relative to the screen diagonal.
# Example: 0.25 means the person cannot jump more than 1/4 of the screen diagonal.
MAX_BBOX_JUMP_FACTOR = 0.25          

# MAX_BBOX_OVERLAP_SCORE: Weight given to normalized distance score (lower is stricter).
MAX_BBOX_OVERLAP_SCORE = 1.0        

# MIN_BBOX_CONFIDENCE: Detections below this score are ignored.
MIN_BBOX_CONFIDENCE = 0.3          

INPUT_JSON = "pose_detection_results.json"
OUTPUT_JSON = "recycled_tracked_pose_data.json" # New output file name
INITIAL_TRACK_ID = 1 # IDs start from 1

# ----------------------------
# Helpers
# ----------------------------

def get_bbox_center(bbox_xywh):
    """Calculates the (cx, cy) center from [x, y, w, h]."""
    if not bbox_xywh or len(bbox_xywh) < 4:
        return None
    x, y, w, h = bbox_xywh
    return np.array([x + w / 2, y + h / 2])

def center_distance(center_a, center_b):
    """Calculates the L2 distance between two (cx, cy) centers."""
    if center_a is None or center_b is None:
        return np.inf
    return float(np.linalg.norm(center_a - center_b))

# ----------------------------
# Main Tracking Logic
# ----------------------------

def track_poses_with_fixed_ids(input_json_path: str, output_json_path: str = OUTPUT_JSON):
    
    print(f"Loading raw YOLOv8 data from: {input_json_path}")
    try:
        with open(input_json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input JSON file not found at {input_json_path}. Run YOLO track first.")
        return

    # 1. Group detections by frame index and find overall screen size
    frames_by_index = defaultdict(list)
    max_x, max_y = 0, 0
    
    for entry in data:
        # **CRITICAL SAFETY CHECK:** Ensure the entry has the minimum required data.
        if entry.get('bbox_xywh') and len(entry['bbox_xywh']) == 4 and 'frame_index' in entry:
            frames_by_index[entry["frame_index"]].append(entry)
            
            x, y, w, h = entry['bbox_xywh']
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)

    sorted_frame_indices = sorted(frames_by_index.keys())
    if not sorted_frame_indices:
        print("No valid pose data found in the input file. Outputting empty JSON.")
        with open(output_json_path, "w") as f:
            json.dump([], f)
        return

    # --- INITIALIZATION (ID Lock) ---
    first_frame_idx = sorted_frame_indices[0]
    initial_detections = frames_by_index[first_frame_idx]
    
    # Lock the total number of IDs to the count in the first frame
    MAX_TRACK_IDS = len(initial_detections)
    if MAX_TRACK_IDS == 0:
        print("First frame has no detections. Exiting.")
        return

    print(f"--- ID LOCK: The maximum number of unique IDs will be {MAX_TRACK_IDS}. ---")

    # Screen dimensions for distance normalization
    screen_diag = math.sqrt(max_x**2 + max_y**2) + 1e-6
    max_jump_pixels = MAX_BBOX_JUMP_FACTOR * screen_diag

    # --- Tracking State ---
    # Key: track_id (1 to MAX_TRACK_IDS), Value: { 'last_center': (cx, cy), 'last_frame': int }
    track_state = {} 
    tracked_entries = []
    
    # 2. Initialize state with the first frame
    for j, det in enumerate(initial_detections):
        tid = INITIAL_TRACK_ID + j # IDs: 1, 2, 3, ... MAX_TRACK_IDS
        
        track_state[tid] = {
            'last_center': get_bbox_center(det['bbox_xywh']),
            'last_frame': first_frame_idx,
            'is_active': True # New flag to indicate if track was just matched
        }
        
        # Write the first frame entries with assigned fixed IDs
        fixed = dict(det)
        fixed["track_id"] = tid
        tracked_entries.append(fixed)

    # 3. Process remaining frames sequentially
    for current_frame_idx in sorted_frame_indices[1:]:
        current_detections = frames_by_index[current_frame_idx]
            
        det_used = [False] * len(current_detections)
        
        # Reset active status for all tracks
        for state in track_state.values():
            state['is_active'] = False
        
        
        # --- PHASE A: Match to ACTIVE and LOST (Recyclable) Tracks ---
        
        recycling_candidates = []
        
        # Combine currently tracked IDs (1 to MAX_TRACK_IDS)
        for tid in range(INITIAL_TRACK_ID, INITIAL_TRACK_ID + MAX_TRACK_IDS):
            
            # 1. Check if the ID state exists and is not too old to match
            state = track_state.get(tid)
            if state is None:
                continue # Should not happen after initialization

            frame_gap = current_frame_idx - state['last_frame']
            
            # If the gap is too large, the track is permanently retired (though we still recycle the ID)
            if frame_gap > MAX_FRAME_GAP_FOR_REAPPEARANCE:
                continue 

            last_center = state['last_center']
            
            # Check for matches against ALL UNUSED detections
            for j, det in enumerate(current_detections):
                if det_used[j] or det.get('conf', 0) < MIN_BBOX_CONFIDENCE: 
                    continue
                
                det_center = get_bbox_center(det['bbox_xywh'])
                if det_center is None:
                    continue 
                
                # Check 1: Distance (Primary RE-ID criteria)
                dist = center_distance(det_center, last_center)
                
                if dist <= max_jump_pixels:
                    # Score is based on normalized distance (lower score is better)
                    normalized_dist_score = dist / max_jump_pixels
                    score = MAX_BBOX_OVERLAP_SCORE * normalized_dist_score
                    recycling_candidates.append((score, tid, j))

        # Sort by best score (lowest distance)
        recycling_candidates.sort(key=lambda x: x[0])
        
        # Assign matches (Greedy assignment based on best score)
        assigned_recycling_ids = set()
        
        for score, tid, j in recycling_candidates:
            if tid not in assigned_recycling_ids and not det_used[j]:
                assigned_recycling_ids.add(tid)
                det_used[j] = True
                
                # RE-ASSIGNMENT SUCCESS
                
                # Update track state
                track_state[tid]['last_center'] = get_bbox_center(current_detections[j]['bbox_xywh'])
                track_state[tid]['last_frame'] = current_frame_idx
                track_state[tid]['is_active'] = True 
                
                # Write to output with recycled ID
                fixed = dict(current_detections[j])
                fixed["track_id"] = tid
                tracked_entries.append(fixed)

        # --- PHASE B: Handle unmatched new detections (They are dropped) ---
        
        for j, det in enumerate(current_detections):
            if not det_used[j] and det.get('conf', 0) >= MIN_BBOX_CONFIDENCE:
                # If we made it here, this is a new person who appeared *after* the first frame, 
                # but we have no available ID to assign (since all MAX_TRACK_IDS are either active 
                # or reserved/lost). They must be dropped to maintain the fixed ID set size.
                pass # Explicitly drop unmatched detections
        
    # Final count
    print(f"Tracking complete. Total unique IDs used (locked set size): {MAX_TRACK_IDS}")
    
    try:
        with open(output_json_path, "w") as f:
            json.dump(tracked_entries, f, indent=4)
        print(f"Wrote {len(tracked_entries)} tracked entries to: {output_json_path}")
    except Exception as e:
        print(f"Error writing output file: {e}")

# --- 4. EXECUTION ---

if __name__ == "__main__":
    track_poses_with_fixed_ids(INPUT_JSON, OUTPUT_JSON)