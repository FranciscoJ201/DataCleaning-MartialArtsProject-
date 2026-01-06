#from intrinsic import intrinsic_calb
from videosplit import split_video_into_frames 
#from extrinsic import extrinstic_calb
#from normalize import undistort_pose_data
#from triangulate import triangulate_3d_pose
from poseestimation import poseestimateCPU,poseestimateGPU

#from PoseEstimation.reid import recycle


RS_video_file = 'test' 
NORM_video_file = 'test'
output_folder_1 = 'calbFRAMEScam1' # Left Camera
output_folder_2 = 'calbFRAMEScam2' # Right Camera

REID_PARAMS = (50, 0.25, 1.0, 0.3) 

print("STARTING POSE ESTIMATION PROCESS")
#take in the PoseEstimations from RealSenseTrack

realsenseINPUT = 'realsense_recordings/3d_pose_data.json'

#json_out2 = poseestimateCPU(video_file2)
print("\n STARTING RE-ID PROCESS")
# clean_json_out = recycle(json_out, *REID_PARAMS)
# clean_json_out2 = recycle(json_out2, *REID_PARAMS)