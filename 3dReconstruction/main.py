from intrinsic import intrinsic_calb
from videosplit import split_video_into_frames # Assuming this exists
from extrinsic import extrinstic_calb
from normalize import undistort_pose_data
from triangulate import triangulate_3d_pose # <-- NEW IMPORT

video_file = '/Users/franciscojimenez/Desktop/cam1test.mp4' 
video_file2 = '/Users/franciscojimenez/Desktop/cam2test.mp4' 
output_folder_1 = 'calbFRAMEScam1' # Left Camera
output_folder_2 = 'calbFRAMEScam2' # Right Camera

# --- 1. Video Splitting )
split_video_into_frames('main', video_file, output_folder_1,300) #if you change the string at the start u must update path in extrinsic.py
split_video_into_frames('helper', video_file2, output_folder_2,300) 

# --- 2. Intrinsic Calibration
intrinsic_calb(output_folder_1,False) #set value to true if you want to show images being calibrated but this slows down runtime by 500ms each image
intrinsic_calb(output_folder_2,False)

# --- 3. Extrinsic Calibration
extrinstic_calb(output_folder_1, output_folder_2)



#--- 4. Setup for Pose Estimation and Undistortion (You must replace the 'pose_detection_results' files)

# File paths based on your current intrinsic_calb outputs
CALIBRATION_OUTPUT_FILE_1 = f"{output_folder_1}.json" 
CALIBRATION_OUTPUT_FILE_2 = f"{output_folder_2}.json" 
STEREO_CALIBRATION_FILE = "stereo_calibration.json"

# **NOTE: These two files must be created by your pose estimator!**
POSE_ESTIMATION_OUTPUT_FILE_1 = "recycledcam1.json" 
POSE_ESTIMATION_OUTPUT_FILE_2 = "recycledcam2.json" 

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