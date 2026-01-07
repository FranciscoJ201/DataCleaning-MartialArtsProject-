import cv2
import numpy as np
import json
import os
from ultralytics import YOLO

class JudoColorTracker:
    def __init__(self):
        # Initialize Background Subtractor to remove mat color
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
        # Store fingerprints: {track_id: [hist_top, hist_mid, hist_bottom]}
        self.fingerprints = {}

    def get_hsv_fingerprint(self, frame, bbox):
        """Extracts a 3-part HSV histogram fingerprint from a masked crop."""
        x, y, w, h = map(int, bbox)
        # Ensure bbox is within frame boundaries
        x, y = max(0, x), max(0, y)
        
        # 1. Background Masking
        fgMask = self.backSub.apply(frame)
        roi_mask = fgMask[y:y+h, x:x+w]
        roi_img = frame[y:y+h, x:x+w]
        
        if roi_img.size == 0: return None

        # Mask the ROI so we only look at the fighter, not the mat
        masked_roi = cv2.bitwise_and(roi_img, roi_img, mask=roi_mask)
        hsv_roi = cv2.cvtColor(masked_roi, cv2.COLOR_BGR2HSV)

        # 2. Vertical 3-Part Split (Head, Torso, Legs)
        sections = []
        h_part = h // 3
        for i in range(3):
            section = hsv_roi[i*h_part : (i+1)*h_part, :]
            # We use Hue and Saturation (H=180, S=256)
            hist = cv2.calcHist([section], [0, 1], None, [30, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            sections.append(hist)
        return sections

    def compare_fingerprints(self, current_fp, target_id):
        """Compares current detection against a stored ID using Bhattacharyya distance."""
        if target_id not in self.fingerprints or current_fp is None:
            return 1.0 # Max difference
        
        scores = []
        for i in range(3):
            score = cv2.compareHist(current_fp[i], self.fingerprints[target_id][i], cv2.HISTCMP_BHATTACHARYYA)
            scores.append(score)
        return np.mean(scores) # Lower is better match

def poseestimateColorFix(source):
    model = YOLO('yolo11n-pose.pt') 
    color_manager = JudoColorTracker()
    
    # stream=True allows us to process and display frame-by-frame
    results = model.track(source=source, tracker='botsort.yaml', stream=True, conf=0.3)
    
    all_detection_data = []

    for i, result in enumerate(results):
        if result.boxes is None or result.boxes.id is None:
            continue

        frame = result.orig_img
        # Create a copy for drawing so we don't mess up the original pixel data
        display_frame = frame.copy() 
        
        track_ids = result.boxes.id.cpu().numpy().astype(int)
        bboxes = result.boxes.xyxy.cpu().numpy() 
        
        for j, track_id in enumerate(track_ids):
            x1, y1, x2, y2 = map(int, bboxes[j])
            
            # --- Draw on the frame ---
            # Box color (Green)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Label with ID
            cv2.putText(display_frame, f"ID: {track_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- Display Windows ---
        cv2.imshow("Judo ID Tracking", display_frame)
        
        # Optional: Uncomment to see what the color tracker 'sees' (the mask)
        # fgMask = color_manager.backSub.apply(frame)
        # cv2.imshow("Background Mask", fgMask)

        # BREAK LOGIC: Press 'q' to stop the video or 'space' to pause
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '): # Pause feature
            cv2.waitKey(0)

    cv2.destroyAllWindows()
    return "Processing complete"

if __name__ == "__main__":
    # Path to your Judo video
    video_path = "/Users/franciscojimenez/Desktop/down.MP4" 
    
    # 1. Verification of assets
    if not os.path.exists(video_path):
        print(f"Error: Could not find {video_path}")
    else:
        print(f"Starting ID Tracking & Color Analysis on: {video_path}")
        
        # 2. Execute the tracking
        # This will process frame-by-frame, extract HSV histograms, 
        # and save the results to a JSON.
        json_output = poseestimateColorFix(video_path)
        
        print("-" * 30)
        print(f"SUCCESS: Analysis finished.")
        print(f"JSON Data saved to: {json_output}")
        print("-" * 30)

    # Optional: Quick check to see if Background Subtraction is working
    # Press 'q' to close the preview window if you use cv2.imshow inside the loop