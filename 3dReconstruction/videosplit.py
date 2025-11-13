import cv2
import os

def split_video_into_frames(type, video_path, output_dir="frames_output", max_frames=None):
    """
    Splits a video file into individual frames and saves them as images,
    up to a specified maximum number of frames.

    Args:
        type (str): Prefix for the frame filenames.
        video_path (str): The path to the input video file.
        output_dir (str): The name of the directory to save the frames in.
        max_frames (int, optional): The maximum number of frames to save.
                                    If None, all frames will be saved.
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
        # Check if the maximum number of frames has been reached
        if max_frames is not None and frame_count >= max_frames:
            print(f"\nLimit of {max_frames} frames reached.")
            break

        # ret is a boolean, frame is the image array
        ret, frame = cap.read()

        if ret:
            # 4. Construct the output filename
            frame_filename = os.path.join(output_dir, f"{type}{frame_count:05d}.jpg")

            # 5. Save the frame
            cv2.imwrite(frame_filename, frame)

            frame_count += 1
        else:
            # Break the loop when no more frames are returned (end of video)
            break

    # 6. Release the video capture object and clean up
    cap.release()
    print(f"\nFinished extracting frames. Total frames saved: {frame_count}")