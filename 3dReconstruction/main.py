from intrinsic import intrinsic_calb
from videosplit import split_video_into_frames
from extrinsic import extrinstic_calb
from NormalizeINTR import undistort_pose_data

video_file = '/Users/franciscojimenez/Desktop/saved.mp4' 
output_folder = 'calibration_frames'

video_file2 = video_file 
output_folder2 = 'calibration_frames2'

#output from pose estimation
POSE_ESTIMATION_OUTPUT_FILE = "/Users/franciscojimenez/Desktop/DataCleaning-MartialArtsProject-/pose_detection_results.json"
UNDISTORTED_OUTPUT_FILE = "Undistored_data.json"


undistort_pose_data(output_folder, POSE_ESTIMATION_OUTPUT_FILE, UNDISTORTED_OUTPUT_FILE)


# split_video_into_frames('left',video_file,output_folder)
# split_video_into_frames('right',video_file2,output_folder2)
# intrinsic_calb(video_file,output_folder)
# intrinsic_calb(video_file2,output_folder2)
# extrinstic_calb(output_folder,output_folder2)

  
    
    
    
   
    
   
