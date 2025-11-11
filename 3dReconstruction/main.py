from intrinsic import intrinsic_calb
from videosplit import split_video_into_frames # Assuming this exists
from extrinsic import extrinstic_calb
from normalize import undistort_pose_data
from triangulate import triangulate_3d_pose # <-- NEW IMPORT

video_file = '/Users/franciscojimenez/Desktop/saved.mp4' 
video_file2 = '/Users/franciscojimenez/Desktop/saved.mp4' 
output_folder_1 = 'calibration_frames_phone' # Left Camera
output_folder_2 = 'calibration_frames_laptop' # Right Camera

# --- 1. Video Splitting (Assuming you rename images to left_*.jpg and right_*.jpg)
split_video_into_frames('left', video_file, output_folder_1)
split_video_into_frames('right', video_file2, output_folder_2) # Use the same video_file if it's a stereo video, or a second if using two cameras

# --- 2. Intrinsic Calibration
intrinsic_calb(output_folder_1)
intrinsic_calb(output_folder_2)

# --- 3. Extrinsic Calibration
extrinstic_calb(output_folder_1, output_folder_2)



#--- 4. Setup for Pose Estimation and Undistortion (You must replace the 'pose_detection_results' files)

# File paths based on your current intrinsic_calb outputs
CALIBRATION_OUTPUT_FILE_1 = f"{output_folder_1}.json" 
CALIBRATION_OUTPUT_FILE_2 = f"{output_folder_2}.json" 
STEREO_CALIBRATION_FILE = "stereo_calibration.json"

# **NOTE: These two files must be created by your pose estimator!**
POSE_ESTIMATION_OUTPUT_FILE_1 = "pose_detection_results_cam1.json" 
POSE_ESTIMATION_OUTPUT_FILE_2 = "pose_detection_results_cam2.json" 

# Final outputs for the undistorted 2D points
UNDISTORTED_OUTPUT_FILE_1 = "undistorted_pose_data_1.json" 
UNDISTORTED_OUTPUT_FILE_2 = "undistorted_pose_data_2.json"

# --- 5. Undistort 2D Pose Data for Each Camera
print("\n--- Starting 2D Pose Undistortion for Camera 1 ---")
undistort_pose_data(
    CALIBRATION_OUTPUT_FILE_1, 
    POSE_ESTIMATION_OUTPUT_FILE_1, 
    UNDISTORTED_OUTPUT_FILE_1
)

print("\n--- Starting 2D Pose Undistortion for Camera 2 ---")
undistort_pose_data(
    CALIBRATION_OUTPUT_FILE_2, 
    POSE_ESTIMATION_OUTPUT_FILE_2, 
    UNDISTORTED_OUTPUT_FILE_2
)

# --- 6. Triangulate to 3D
FINAL_3D_OUTPUT_FILE = "3d_pose_reconstruction.json"

print("\n--- Starting 3D Triangulation ---")
triangulate_3d_pose(
    CALIBRATION_OUTPUT_FILE_1,
    CALIBRATION_OUTPUT_FILE_2,
    STEREO_CALIBRATION_FILE,
    UNDISTORTED_OUTPUT_FILE_1,
    UNDISTORTED_OUTPUT_FILE_2,
    FINAL_3D_OUTPUT_FILE
)