from intrinsic import intrinsic_calb
from videosplit import split_video_into_frames
from extrinsic import extrinstic_calb
from normalize import undistort_pose_data

video_file = '/Users/franciscojimenez/Desktop/saved.mp4' 
output_folder = 'calibration_frames'

video_file2 = video_file 
output_folder2 = 'calibration_frames2'

split_video_into_frames('left',video_file,output_folder)
split_video_into_frames('right',video_file2,output_folder2)
intrinsic_calb(output_folder)
intrinsic_calb(output_folder2)
extrinstic_calb(output_folder,output_folder2)


    # 1. Output from your intrinsic.py script (e.g., 'calib_data.json' if you named the folder 'calib_data')
CALIBRATION_OUTPUT_FILE = "calibration_frames.json" 
    
    # 2. Output from your poseestimation.py script
POSE_ESTIMATION_OUTPUT_FILE = "pose_detection_results.json" 
    
    # 3. Where you want to save the final corrected coordinates
UNDISTORTED_OUTPUT_FILE = "undistorted_pose_data.json"
    
undistort_pose_data(
        CALIBRATION_OUTPUT_FILE, 
        POSE_ESTIMATION_OUTPUT_FILE, 
        UNDISTORTED_OUTPUT_FILE
    )