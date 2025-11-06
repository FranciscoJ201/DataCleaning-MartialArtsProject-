import cv2
import numpy as np
import json
import os
def triangulate(calb1,calb2,stereocalb,posedata1,posedata2):
    # --- Configuration ---
    # IMPORTANT: Update these file paths to match your generated files
    CALIB_FILE_1 = "cam1_intrinsic_data.json"   # Output from intrinsic.py for Camera 1
    CALIB_FILE_2 = "cam2_intrinsic_data.json"   # Output from intrinsic.py for Camera 2
    STEREO_CALIB_FILE = "stereo_calibration.json" # Output from extrinsic.py

    # IMPORTANT: Update these file paths to match your generated pose estimation data
    POSE_DATA_FILE_1 = "cam1_pose_detection_results.json" # Output from poseestimation.py (Cam 1)
    POSE_DATA_FILE_2 = "cam2_pose_detection_results.json" # Output from poseestimation.py (Cam 2)

    OUTPUT_FILE = "3d_reconstruction_results.json"

    def load_all_calibration_data():
        """Loads all necessary intrinsic and extrinsic parameters."""
        
        def load_intrinsics(file_path):
            """Loads K and D from a single intrinsic calibration file."""
            with open(file_path, 'r') as f:
                data = json.load(f)
            return {
                "K": np.array(data["camera_matrix"], dtype=np.float64),
                "D": np.array(data["distortion_coefficients"], dtype=np.float64),
                "image_size": tuple(data["image_size"])
            }
            
        def load_extrinsics(file_path):
            """Loads R and T from the stereo calibration file."""
            with open(file_path, 'r') as f:
                data = json.load(f)
            return {
                "R": np.array(data["rotation_matrix_R1_to_R2"], dtype=np.float64),
                "T": np.array(data["translation_vector_T1_to_T2"], dtype=np.float64)
            }

        # Load Intrinsics
        intrinsics1 = load_intrinsics(CALIB_FILE_1)
        intrinsics2 = load_intrinsics(CALIB_FILE_2)
        
        # Load Extrinsics (R and T from Cam1 to Cam2)
        extrinsics = load_extrinsics(STEREO_CALIB_FILE)
        
        return intrinsics1, intrinsics2, extrinsics

    def load_pose_data(file_path):
        """
        Loads and organizes pose data into a dictionary structure (frame -> track_id -> keypoints).
        ***UPDATED***: Now uses 'keypoints_xyc' to match poseestimation.py output.
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Structure: {frame_index: {track_id: keypoints_list}}
        organized_data = {}
        for record in data:
            frame = record["frame_index"]
            track_id = record["track_id_native"]
            
            if frame not in organized_data:
                organized_data[frame] = {}
                
            # Keypoints are stored as a list of lists: [[x1, y1, c1], [x2, y2, c2], ...]
            # NOTE: Using 'keypoints_xyc' to match the latest poseestimation.py file
            organized_data[frame][track_id] = record["keypoints_xyc"] 
            
        return organized_data

    def get_projection_matrices(K1, D1, K2, D2, R, T, image_size):
        """
        Computes the Projection matrices P1 and P2 using the R and T found
        during stereo calibration, while fixing the K and D from the intrinsic step.
        """
        # 1. Stereo Rectification
        # P1/P2 are the 3x4 Projection Matrices which encapsulate K and the R/T of the camera system.
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
            cameraMatrix1=K1, distCoeffs1=D1,
            cameraMatrix2=K2, distCoeffs2=D2,
            imageSize=image_size,
            R=R, T=T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=-1 # Fully zoom in
        )
        return P1, P2


    def triangulate_points(P1, P2, pose1_data, pose2_data, K1, D1, K2, D2):
        """
        Performs Undistortion and Triangulation for all matched keypoints.
        """
        final_3d_data = {}
        
        # Find all common frames
        common_frames = sorted(list(set(pose1_data.keys()) & set(pose2_data.keys())))
        
        for frame in common_frames:
            frame_data = {}
            
            # Find all common track IDs in this frame
            tracks1 = pose1_data[frame]
            tracks2 = pose2_data[frame]
            common_tracks = sorted(list(set(tracks1.keys()) & set(tracks2.keys())))
            
            for track_id in common_tracks:
                # Get keypoints for the same person in the same frame from both cameras
                kps1_raw = tracks1[track_id]
                kps2_raw = tracks2[track_id]
                
                # Ensure both lists are the same length (17 keypoints from YOLO)
                if len(kps1_raw) != len(kps2_raw) or not kps1_raw:
                    continue

                # Separate coordinates and confidences
                points1_xy = np.array([kp[:2] for kp in kps1_raw], dtype=np.float64)
                points2_xy = np.array([kp[:2] for kp in kps2_raw], dtype=np.float64)
                confidences = np.array([kp[2] for kp in kps1_raw]) # Use Cam 1 confidence

                # 1. Undistort the 2D points (CRITICAL STEP - Integrated Here!)
                # cv2.undistortPoints converts distorted pixels to normalized coordinates.
                points1_norm = cv2.undistortPoints(points1_xy.reshape(-1, 1, 2), K1, D1, P=None, R=None)
                points2_norm = cv2.undistortPoints(points2_xy.reshape(-1, 1, 2), K2, D2, P=None, R=None)
                
                # Reshape from (N, 1, 2) to (2, N) for triangulation
                points1_norm = points1_norm.reshape(-1, 2).T 
                points2_norm = points2_norm.reshape(-1, 2).T 
                
                # 2. Triangulation
                points_4d_homogenous = cv2.triangulatePoints(P1, P2, points1_norm, points2_norm)
                
                # Convert homogeneous coordinates (X, Y, Z, W) to non-homogeneous (X, Y, Z)
                points_3d = (points_4d_homogenous[:3] / points_4d_homogenous[3]).T 
                
                # 3. Store Results
                keypoints_3d = []
                for i in range(len(points_3d)):
                    X = points_3d[i, 0]
                    Y = points_3d[i, 1]
                    Z = points_3d[i, 2]
                    C = confidences[i]
                    
                    # Filter low-confidence points (recommended)
                    if C > 0.5:
                        keypoints_3d.append([float(X), float(Y), float(Z), float(C)])
                    else:
                        # Append NaNs or zeros if filtered out, for consistency
                        keypoints_3d.append([0.0, 0.0, 0.0, 0.0])

                frame_data[track_id] = keypoints_3d
                
            final_3d_data[frame] = frame_data
            
        return final_3d_data


    def main():
        # 1. Load Calibration Data
        try:
            intrinsics1, intrinsics2, extrinsics = load_all_calibration_data()
        except Exception as e:
            print(f"FATAL ERROR: Could not load all calibration files. Please ensure you have run intrinsic.py (for both cams) and extrinsic.py. Error: {e}")
            return

        K1, D1 = intrinsics1["K"], intrinsics1["D"]
        K2, D2 = intrinsics2["K"], intrinsics2["D"]
        R, T = extrinsics["R"], extrinsics["T"]

        # Use Cam 1 Image Size (assuming both cameras have the same resolution)
        image_size = intrinsics1["image_size"]

        # 2. Get Projection Matrices
        P1, P2 = get_projection_matrices(K1, D1, K2, D2, R, T, image_size)
        print("\nProjection Matrices (P1, P2) successfully computed.")

        # 3. Load Pose Data
        try:
            pose1_data = load_pose_data(POSE_DATA_FILE_1)
            pose2_data = load_pose_data(POSE_DATA_FILE_2)
        except Exception as e:
            print(f"FATAL ERROR: Could not load pose data files. Error: {e}")
            return

        # 4. Triangulate and Save (Pass all K/D to the triangulation function)
        print(f"Attempting to triangulate across {len(pose1_data.keys())} frames...")
        
        final_3d_data = triangulate_points(P1, P2, pose1_data, pose2_data, K1, D1, K2, D2)

        with open(OUTPUT_FILE, 'w') as f:
            json.dump(final_3d_data, f, indent=4)
        
        print(f"\n--- 3D RECONSTRUCTION SUCCESS ---")
        print(f"3D world coordinates saved to {OUTPUT_FILE}")
        print(f"Data is structured as: {{frame_index: {{track_id: [[X, Y, Z, Confidence], ...]}}}}\n")
        print("Coordinates are relative to Camera 1's coordinate system.")

 