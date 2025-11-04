import cv2
import os

def split_video_into_frames(type,video_path, output_dir="frames_output"):
    """
    Splits a video file into individual frames and saves them as images.

    Args:
        video_path (str): The path to the input video file.
        output_dir (str): The name of the directory to save the frames in.
    """
    # 1. Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    else:
        print(f"Saving frames to existing directory: {output_dir}")

    # 2. Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_count = 0

    # 3. Loop through and read frames
    while True:
        # ret is a boolean, frame is the image array
        ret, frame = cap.read()

        if ret:
            # 4. Construct the output filename
            frame_filename = os.path.join(output_dir, f"{type}{frame_count:05d}.jpg")
            
            # 5. Save the frame
            cv2.imwrite(frame_filename, frame)
            
            frame_count += 1
        else:
            # Break the loop when no more frames are returned
            break

    # 6. Release the video capture object and clean up
    cap.release()
    print(f"\nFinished extracting frames. Total frames saved: {frame_count}")

# --- Example Usage ---
# # NOTE: Replace 'your_video.mp4' with the actual path to your video file.
# video_file = '/Users/franciscojimenez/Desktop/Movie on 10-30-25 at 4.31 PM.mov' 
# output_folder = 'calibration_frames'

# split_video_into_frames(video_file, output_folder)