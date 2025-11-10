import numpy as np
import cv2
import json
import os

def load_calibration_data(file_path):
    """Loads Camera Matrix (K) and Distortion Coeffs (D) from an intrinsic JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    K = np.array(data["camera_matrix"], dtype=np.float64)
    D = np.array(data["distortion_coefficients"], dtype=np.float64)
    return K, D

def load_stereo_data(file_path):
    """Loads Rotation (R) and Translation (T) from the extrinsic JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    R = np.array(data["rotation_matrix_R1_to_R2"], dtype=np.float64)
    T = np.array(data["translation_vector_T1_to_T2"], dtype=np.float64)
    return R, T

def triangulate_3d_pose(intrinsic_1_file, intrinsic_2_file, stereo_file, pose_data_1_file, pose_data_2_file, output_file):
    """
    Performs 3D reconstruction of keypoints using stereo triangulation.

    Args:
        intrinsic_1_file (str): JSON file for Camera 1 (left).
        intrinsic_2_file (str): JSON file for Camera 2 (right).
        stereo_file (str): JSON file from extrinsic calibration.
        pose_data_1_file (str): JSON file with normalized 2D points from Camera 1.
        pose_data_2_file (str): JSON file with normalized 2D points from Camera 2.
        output_file (str): Path to save the final 3D coordinates.
    """
    try:
        # 1. Load Calibration Data
        K1, _ = load_calibration_data(intrinsic_1_file)
        K2, _ = load_calibration_data(intrinsic_2_file)
        R, T = load_stereo_data(stereo_file)
    except Exception as e:
        print(f"Error loading calibration files: {e}")
        return

    # 2. Calculate Projection Matrices (P = K * [R|T])
    
    # P1: Camera 1 is the world origin: Extrinsics = [I | 0]
    R1 = np.eye(3)
    T1 = np.zeros((3, 1))
    Extrinsic1 = np.hstack((R1, T1)) # 3x4 matrix
    P1 = K1 @ Extrinsic1
    
    # P2: Camera 2's extrinsics are R and T relative to Camera 1: Extrinsics = [R | T]
    Extrinsic2 = np.hstack((R, T)) # 3x4 matrix
    P2 = K2 @ Extrinsic2
    
    print("Calculated Projection Matrices P1 and P2.")

    # 3. Load Normalized 2D Keypoints
    try:
        with open(pose_data_1_file, 'r') as f:
            detections_cam1 = json.load(f)
        with open(pose_data_2_file, 'r') as f:
            detections_cam2 = json.load(f)
    except Exception as e:
        print(f"Error loading pose data files: {e}")
        return

    # Assuming a one-to-one match between the people detected in each frame/file.
    if len(detections_cam1) != len(detections_cam2):
        print("Warning: Mismatched number of detected poses. Only processing the minimum count.")
        num_poses = min(len(detections_cam1), len(detections_cam2))
    else:
        num_poses = len(detections_cam1)
        
    # 4. Triangulate Points
    final_3d_poses = []
    
    for record1, record2 in zip(detections_cam1[:num_poses], detections_cam2[:num_poses]):
        
        # Extract normalized (x, y) coordinates and confidence/z-value
        keypoints_norm_1 = np.array(record1["keypoints_undistorted_normalized"], dtype=np.float32)
        keypoints_norm_2 = np.array(record2["keypoints_undistorted_normalized"], dtype=np.float32)

        points1 = keypoints_norm_1[:, :2].T # (2, N)
        points2 = keypoints_norm_2[:, :2].T # (2, N)
        
        # Perform cv2.triangulatePoints: returns (4, N) homogeneous points
        points_4d_homogeneous = cv2.triangulatePoints(P1, P2, points1, points2)
        
        # Convert to Euclidean (3D) coordinates: (x/w, y/w, z/w)
        # Result is (N, 3) matrix
        points_3d_euclidean = (points_4d_homogeneous[:3] / points_4d_homogeneous[3]).T
        
        # 5. Compile Results
        triangulated_keypoints = []
        for i, (x, y, z) in enumerate(points_3d_euclidean):
            # Retain the confidence/z-value from the original data (e.g., Cam 1)
            conf = float(keypoints_norm_1[i, 2])
            # The 3D point is [X, Y, Z, Confidence]
            triangulated_keypoints.append([float(x), float(y), float(z), conf])
            
        new_record = {
            # Use 'frame_index' or 'person_id' if available in your pose data
            "pose_index": record1.get("pose_index", len(final_3d_poses)), 
            "keypoints_3d_mm": triangulated_keypoints 
        }
        final_3d_poses.append(new_record)

    # 6. Save Final Output
    with open(output_file, 'w') as f:
        json.dump(final_3d_poses, f, indent=4) 
    
    print(f"\n3D Triangulation complete. Saved {len(final_3d_poses)} 3D poses to {output_file}")

